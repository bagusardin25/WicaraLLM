"""Fase 6 — SFT: ubah wicara-56m-base menjadi wicara-56m-chat.

    .venv\\Scripts\\python.exe scripts\\train_sft.py

Perbedaan pokok dari pretrain:

    Titik awal    : bobot dari checkpoint base, bukan acak
    Optimizer     : BARU, tidak melanjutkan momentum pretrain
    Learning rate : ~10x lebih kecil
    Loss          : hanya pada token jawaban asisten
    Durasi        : hitungan menit, bukan jam

Kenapa optimizer dimulai dari nol: momentum AdamW dari pretrain menyimpan
arah gradient untuk tugas yang berbeda (melanjutkan teks). Membawanya ke SFT
justru menarik bobot ke arah lama.

Kenapa LR jauh lebih kecil: model sudah punya kemampuan bahasa yang mahal
didapat. LR besar akan menimpanya — catastrophic forgetting. Yang diinginkan
hanya menggeser PERILAKU, bukan melatih ulang bahasanya.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.sft_dataset import SFTDataset  # noqa: E402
from src.model.config import CONFIGS, ModelConfig  # noqa: E402
from src.train.checkpoint import load_checkpoint  # noqa: E402
from src.train.config import TrainConfig  # noqa: E402
from src.train.lr_schedule import describe_schedule  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402
from src.tokenizer.bpe import load as load_tokenizer  # noqa: E402
from src.tokenizer.chat_template import DEFAULT_SYSTEM_PROMPT  # noqa: E402

BASE = REPO_ROOT / "checkpoints" / "wicara-56m-base" / "best.pt"
SFT_DIR = REPO_ROOT / "data" / "sft"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"

PROMPT_PANTAU = [
    "Halo, apa kabar?",
    "Siapa namamu?",
    "Terima kasih banyak!",
    "Apa ibu kota Zimbabwe?",
    "Jelaskan cara membuat nasi goreng.",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(BASE))
    p.add_argument("--sft-dir", default=str(SFT_DIR))
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    p.add_argument("--out-dir", default="checkpoints/wicara-56m-chat")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=8e-5,
                   help="~10x lebih kecil dari LR pretrain")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"Checkpoint base tidak ada: {base_path}")
        print("Jalankan pretrain dulu (scripts/train.py).")
        return 1

    sft_dir = Path(args.sft_dir)
    if not (sft_dir / "train.jsonl.gz").exists():
        print(f"Data SFT tidak ada di {sft_dir}")
        print("Jalankan scripts/build_sft_data.py dulu.")
        return 1

    print("=" * 70)
    print("  SFT — wicara-56m-base -> wicara-56m-chat")
    print("=" * 70)

    tok = load_tokenizer(args.tokenizer)
    sysprompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT

    print("  Memuat data SFT...")
    t0 = time.time()
    train_data = SFTDataset(sft_dir / "train.jsonl.gz", tok,
                            max_len=args.max_len, system_prompt=sysprompt,
                            seed=args.seed)
    val_data = SFTDataset(sft_dir / "val.jsonl.gz", tok,
                          max_len=args.max_len, system_prompt=sysprompt,
                          seed=args.seed)
    print(f"    train : {len(train_data):,} contoh "
          f"({train_data.n_dibuang:,} dibuang karena terlalu panjang)")
    print(f"    val   : {len(val_data):,} contoh")
    print(f"    token yang dilatih: {train_data.n_token_dilatih:,} "
          f"(hanya jawaban asisten)")
    print(f"    dimuat dalam {time.time() - t0:.0f}s")

    # Arsitektur dibaca dari checkpoint, bukan ditebak — mustahil salah pasang.
    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                          if k in ModelConfig.__dataclass_fields__})
    del ckpt

    langkah_per_epoch = len(train_data) / (args.batch_size * args.grad_accum)
    total_langkah = max(1, int(langkah_per_epoch * args.epochs))
    # TrainConfig menghitung langkah dari total_tokens; di sini panjangnya
    # bervariasi per contoh, jadi angka token dihitung mundur dari jumlah
    # langkah yang diinginkan supaya jadwal LR tetap pas.
    tokens_per_step = args.batch_size * mcfg.max_seq_len * args.grad_accum

    tcfg = TrainConfig(
        model_config="56m",
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        total_tokens=total_langkah * tokens_per_step,
        eval_interval=max(50, total_langkah // 20),
        eval_batches=20,
        log_interval=max(10, total_langkah // 100),
        sample_interval=max(50, total_langkah // 10),
        sample_prompts=PROMPT_PANTAU,
        out_dir=args.out_dir,
        checkpoint_interval_minutes=15.0,
        seed=args.seed,
        device=args.device,
    )

    trainer = Trainer(mcfg, tcfg, tokenizer=tok,
                      train_data=train_data, val_data=val_data,
                      chat_mode=True)

    if args.resume and trainer.resume():
        pass
    else:
        # Muat BOBOT saja, bukan optimizer. Momentum AdamW dari pretrain
        # menyimpan arah untuk tugas berbeda; membawanya ke SFT justru
        # menarik bobot kembali ke perilaku lama.
        meta = load_checkpoint(base_path, model=trainer.model,
                               optimizer=None, device=args.device,
                               restore_rng=False)
        print(f"\n  Bobot dimuat dari {base_path.name} "
              f"(langkah {meta['step']:,}, val loss {meta['best_val_loss']:.4f})")
        print("  Optimizer dimulai dari nol (momentum pretrain tidak dibawa)")

    print(f"\n  Contoh/epoch  : {len(train_data):,}")
    print(f"  Langkah/epoch : {langkah_per_epoch:,.0f}")
    print(f"  Epoch         : {args.epochs}")
    print(f"  Total langkah : {total_langkah:,}")
    print(f"  System prompt : {'tidak' if sysprompt is None else 'ya'}")
    print(f"\n  Jadwal learning rate:")
    print(describe_schedule(lr=tcfg.lr, min_lr=tcfg.min_lr,
                            warmup_steps=trainer.warmup_steps,
                            total_steps=trainer.total_steps))
    print()

    try:
        hasil = trainer.train(max_steps=total_langkah)
        print("\n" + "=" * 70)
        for k, v in hasil.items():
            print(f"  {k:<16}: {v}")
        print(f"\n  Checkpoint : {tcfg.out_dir}")
        print(f"  Samples    : {tcfg.out_dir}/samples.txt")
        print("\n  Uji dengan:")
        print(f"    .venv\\Scripts\\python.exe scripts\\chat.py")
    finally:
        trainer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
