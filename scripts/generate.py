"""Uji model hasil pretrain dengan membangkitkan teks.

    .venv\\Scripts\\python.exe scripts\\generate.py --prompt "Halo, apa kabar?"
    .venv\\Scripts\\python.exe scripts\\generate.py --interactive
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.infer.generate import generate  # noqa: E402
from src.model.config import CONFIGS, ModelConfig  # noqa: E402
from src.model.transformer import MiniLLM  # noqa: E402
from src.tokenizer.bpe import load as load_tokenizer  # noqa: E402

CKPT = REPO_ROOT / "checkpoints" / "wicara-56m-base" / "best.pt"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"

PROMPT_UJI = [
    "",
    "Halo, apa kabar?",
    "Selamat pagi,",
    "Saya ingin bertanya tentang",
    "Menurut saya,",
    "Indonesia adalah negara",
]


def muat_model(ckpt_path: Path, device: str):
    """Muat model dari checkpoint beserta konfigurasinya."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    cfg = ModelConfig(**{k: v for k, v in mc.items()
                         if k in ModelConfig.__dataclass_fields__})
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint tidak ada: {ckpt_path}")
        print("Jalankan pretrain dulu (scripts/train.py), atau tunjuk "
              "checkpoint lain dengan --ckpt.")
        return 1

    tok = load_tokenizer(args.tokenizer)
    model, cfg, ckpt = muat_model(ckpt_path, args.device)

    print("=" * 72)
    print(f"  {cfg.name} — {model.num_params() / 1e6:.1f}M parameter")
    print(f"  checkpoint : {ckpt_path.name} "
          f"(langkah {ckpt['step']:,}, {ckpt['tokens_seen'] / 1e9:.2f}B token)")
    print(f"  val loss   : {ckpt['best_val_loss']:.4f}")
    print(f"  sampling   : suhu {args.temperature} · top-k {args.top_k} · "
          f"top-p {args.top_p} · penalti ulang {args.repetition_penalty}")
    print("=" * 72)

    opsi = dict(max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device, seed=args.seed)

    if args.interactive:
        print("\n  Ketik prompt, Enter untuk kirim. Kosongkan lalu Enter "
              "untuk keluar.")
        print("  Ingat: model base MELANJUTKAN teks, bukan menjawab.\n")
        while True:
            try:
                prompt = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            print(f"\n  {prompt}{generate(model, tok, prompt, **opsi)}\n")
        return 0

    prompts = [args.prompt] if args.prompt is not None else PROMPT_UJI
    for prompt in prompts:
        hasil = generate(model, tok, prompt, **opsi)
        label = f"[{prompt}]" if prompt else "[tanpa prompt]"
        print(f"\n  {label}")
        print(f"  {prompt}{hasil}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
