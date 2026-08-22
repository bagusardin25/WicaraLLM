"""Training loop dengan bf16 autocast, gradient accumulation, clipping, dan selective weight decay."""

import math
import time
from pathlib import Path

import torch

from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
from src.train.checkpoint import (
    append_jsonl,
    find_latest,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from src.train.config import TrainConfig
from src.train.data import TokenDataset
from src.train.lr_schedule import get_lr
from src.tokenizer.chat_template import BOS_ID, EOS_ID

GB = 1024**3


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        tokenizer=None,
    ):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        # Tokenizer hanya dipakai untuk sample_probe.
        self.tokenizer = tokenizer

        self.device = torch.device(train_cfg.device)
        torch.manual_seed(train_cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(train_cfg.seed)
            torch.set_float32_matmul_precision("high")

        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.jsonl"
        self.sample_path = self.out_dir / "samples.txt"

        self.model = MiniLLM(model_cfg).to(self.device)
        self.optimizer = self._build_optimizer()

        self.train_data = TokenDataset(train_cfg.train_bin,
                                       model_cfg.max_seq_len, seed=train_cfg.seed)
        self.val_data = TokenDataset(train_cfg.val_bin, model_cfg.max_seq_len)

        self.total_steps = train_cfg.total_steps(model_cfg.max_seq_len)
        self.warmup_steps = train_cfg.warmup_steps(model_cfg.max_seq_len)
        self.tokens_per_step = train_cfg.tokens_per_step(model_cfg.max_seq_len)

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self._last_ckpt_time = time.time()

        # Generator terpisah berbenih tetap untuk evaluasi konsisten.
        self._eval_gen = torch.Generator().manual_seed(train_cfg.seed + 1)

    def close(self) -> None:
        """Lepaskan memory map dataset."""
        self.train_data.close()
        self.val_data.close()

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- optimizer --------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Pisahkan parameter yang kena weight decay dan yang tidak."""
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "tok_emb" in name or "lm_head" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [
            {"params": decay, "weight_decay": self.tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # Pakai fused AdamW di CUDA.
        use_fused = self.device.type == "cuda"
        opt = torch.optim.AdamW(
            groups,
            lr=self.tcfg.lr,
            betas=(self.tcfg.beta1, self.tcfg.beta2),
            fused=use_fused,
        )
        self._n_decay = sum(p.numel() for p in decay)
        self._n_no_decay = sum(p.numel() for p in no_decay)
        return opt

    # -- satu langkah -----------------------------------------------------

    def _train_step(self) -> tuple[float, float]:
        """Satu update parameter penuh dengan gradient accumulation."""
        self.model.train()
        total_loss = 0.0

        for micro in range(self.tcfg.grad_accum_steps):
            x, y = self.train_data.get_batch(self.tcfg.batch_size, self.device)

            with torch.autocast(
                self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                _, loss = self.model(x, y)

            # Bagi loss dengan grad_accum_steps untuk mendapatkan rata-rata gradient.
            (loss / self.tcfg.grad_accum_steps).backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tcfg.grad_clip
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.tokens_seen += self.tokens_per_step
        return total_loss / self.tcfg.grad_accum_steps, float(grad_norm)

    def _set_lr(self) -> float:
        lr = get_lr(
            self.step,
            lr=self.tcfg.lr,
            min_lr=self.tcfg.min_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- evaluasi ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, n_batches: int | None = None) -> float:
        """Hitung rata-rata loss di data validasi."""
        n_batches = n_batches or self.tcfg.eval_batches
        self.model.eval()
        gen = torch.Generator().manual_seed(self.tcfg.seed + 1)

        total = 0.0
        for _ in range(n_batches):
            x, y = self.val_data.get_batch(
                self.tcfg.batch_size, self.device, generator=gen
            )
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                _, loss = self.model(x, y)
            total += loss.item()

        self.model.train()
        return total / n_batches

    # -- probe sampling ---------------------------------------------------

    @torch.no_grad()
    def sample_probe(
        self, max_new_tokens: int = 60, temperature: float = 0.8, top_k: int = 40
    ) -> str:
        """Hasilkan teks sampel dari prompt tetap untuk evaluasi visual."""
        self.model.eval()
        prompts = self.tcfg.sample_prompts or [""]
        potongan = []

        for prompt_txt in prompts:
            if self.tokenizer is not None and prompt_txt:
                awal = [BOS_ID] + self.tokenizer.encode(prompt_txt).ids
            else:
                awal = [BOS_ID]

            idx = torch.tensor([awal], dtype=torch.long, device=self.device)
            caches = self.model.make_caches(1, self.mcfg.max_seq_len)
            keluar: list[int] = []

            for _ in range(max_new_tokens):
                with torch.autocast(
                    self.device.type, dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    logits, _ = self.model(idx, caches=caches, last_token_only=True)

                logit = logits[:, -1].float() / max(temperature, 1e-6)
                if top_k:
                    # Terapkan top-k filtering.
                    ambang = torch.topk(logit, min(top_k, logit.size(-1)))[0][..., -1:]
                    logit = logit.masked_fill(logit < ambang, float("-inf"))

                nxt = torch.multinomial(torch.softmax(logit, dim=-1), num_samples=1)
                tok = int(nxt.item())
                if tok == EOS_ID:
                    break
                keluar.append(tok)
                idx = nxt

            if self.tokenizer is not None:
                teks = self.tokenizer.decode(keluar, skip_special_tokens=False)
                label = f"[{prompt_txt}]" if prompt_txt else "[<bos>]"
                potongan.append(f"{label} {teks}")
            else:
                potongan.append(" ".join(str(t) for t in keluar[:32]))

        self.model.train()
        return "\n".join(potongan)

    # -- checkpoint -------------------------------------------------------

    def save(self, name: str) -> Path:
        return save_checkpoint(
            self.out_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            tokens_seen=self.tokens_seen,
            best_val_loss=self.best_val_loss,
            model_config=self.mcfg.__dict__,
            train_config=self.tcfg.to_dict(),
            data_state=self.train_data.state_dict(),
        )

    def resume(self, path: str | Path | None = None) -> bool:
        """Lanjutkan training dari checkpoint."""
        path = path or find_latest(self.out_dir)
        if path is None:
            return False

        meta = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer,
            device=str(self.device),
        )
        self.step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        self.best_val_loss = meta["best_val_loss"]
        # Restore data state agar melanjutkan di epoch yang benar.
        if meta.get("data_state"):
            self.train_data.load_state_dict(meta["data_state"])
        print(f"  Dilanjutkan dari {Path(path).name} "
              f"(langkah {self.step:,}, {self.tokens_seen / 1e6:.1f}M token)")
        return True

    # -- loop utama -------------------------------------------------------

    def train(self, max_steps: int | None = None, verbose: bool = True) -> dict:
        target = min(max_steps or self.total_steps, self.total_steps)
        flops_per_token = 6 * self.mcfg.n_params_non_embedding

        if verbose:
            self._print_header(target)

        t0 = time.time()
        tokens_at_mark = self.tokens_seen
        loss_terakhir = float("nan")

        while self.step < target:
            lr = self._set_lr()
            loss, grad_norm = self._train_step()
            self.step += 1
            loss_terakhir = loss

            if self.step % self.tcfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = (self.tokens_seen - tokens_at_mark) / max(dt, 1e-9)
                tflops = tok_s * flops_per_token / 1e12
                vram = (
                    torch.cuda.max_memory_allocated() / GB
                    if self.device.type == "cuda" else 0.0
                )
                rec = {
                    "step": self.step, "loss": round(loss, 4),
                    "lr": lr, "grad_norm": round(grad_norm, 3),
                    "tokens": self.tokens_seen, "tok_per_s": round(tok_s),
                    "tflops": round(tflops, 2), "vram_gb": round(vram, 2),
                }
                append_jsonl(self.log_path, rec)
                if verbose:
                    sisa = (target - self.step) * self.tokens_per_step / max(tok_s, 1)
                    print(
                        f"  step {self.step:>7,}/{target:,}  "
                        f"loss {loss:6.3f}  lr {lr:.2e}  "
                        f"gnorm {grad_norm:5.2f}  "
                        f"{tok_s:>7,.0f} tok/s  "
                        f"{vram:4.2f}GB  sisa {sisa / 3600:5.1f}j"
                    )
                t0, tokens_at_mark = time.time(), self.tokens_seen

            if self.step % self.tcfg.eval_interval == 0:
                val = self.evaluate()
                append_jsonl(self.log_path, {"step": self.step, "val_loss": val})
                tanda = ""
                if val < self.best_val_loss:
                    self.best_val_loss = val
                    self.save("best.pt")
                    tanda = "  <- terbaik"
                if verbose:
                    print(f"  {'':>7}  val loss {val:6.3f}  "
                          f"ppl {math.exp(min(val, 20)):8.1f}{tanda}")
                t0, tokens_at_mark = time.time(), self.tokens_seen

            if self.step % self.tcfg.sample_interval == 0:
                contoh = self.sample_probe()
                with self.sample_path.open("a", encoding="utf-8") as f:
                    f.write(f"--- step {self.step} ---\n{contoh}\n\n")
                t0, tokens_at_mark = time.time(), self.tokens_seen

            # Simpan checkpoint berbasis waktu.
            if (time.time() - self._last_ckpt_time
                    > self.tcfg.checkpoint_interval_minutes * 60):
                self.save(f"step_{self.step}.pt")
                rotate_checkpoints(self.out_dir, self.tcfg.keep_last_n)
                self._last_ckpt_time = time.time()
                t0, tokens_at_mark = time.time(), self.tokens_seen

        self.save("final.pt")
        val = self.evaluate()
        if verbose:
            print(f"\n  Selesai di langkah {self.step:,} — "
                  f"{self.tokens_seen / 1e6:.1f}M token, val loss {val:.4f}")

        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "train_loss": loss_terakhir,
            "val_loss": val,
            "best_val_loss": self.best_val_loss,
        }

    def _print_header(self, target: int) -> None:
        print(f"  Model            : {self.mcfg.name} "
              f"({self.mcfg.n_params / 1e6:.1f}M parameter)")
        print(f"  Data latih       : {self.train_data.n_tokens / 1e6:.1f}M token "
              f"({self.train_data.n_chunks:,} potongan)")
        print(f"  Data validasi    : {self.val_data.n_tokens / 1e6:.1f}M token")
        print(f"  Batch            : {self.tcfg.batch_size} x "
              f"{self.tcfg.grad_accum_steps} akumulasi = "
              f"{self.tcfg.batch_size * self.tcfg.grad_accum_steps} sekuens efektif")
        print(f"  Token per langkah: {self.tokens_per_step:,}")
        print(f"  Langkah          : {target:,} "
              f"(warmup {self.warmup_steps:,})")
        print(f"  LR               : {self.tcfg.lr:.2e} -> {self.tcfg.min_lr:.2e}")
        print(f"  Weight decay     : {self._n_decay / 1e6:.1f}M kena, "
              f"{self._n_no_decay / 1e6:.1f}M dikecualikan")
        print(f"  Loss awal target : {self.mcfg.expected_init_loss:.4f}")
        print()
