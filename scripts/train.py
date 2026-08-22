"""Entry point training.

Contoh:
    # smoke run di data sintetis
    .venv\\Scripts\\python.exe scripts\\train.py --model main ^
        --train-bin data/tokens/synth_train.bin ^
        --val-bin data/tokens/synth_val.bin ^
        --steps 300 --out-dir checkpoints/smoke

    # pretrain sungguhan (Fase 5)
    .venv\\Scripts\\python.exe scripts\\train.py --model main --resume
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.config import CONFIGS  # noqa: E402
from src.train.config import TrainConfig, scaled_lr  # noqa: E402
from src.train.lr_schedule import describe_schedule  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402
from src.tokenizer.bpe import load as load_tokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="56m", choices=list(CONFIGS))
    p.add_argument("--train-bin", default="data/tokens/train.bin")
    p.add_argument("--val-bin", default="data/tokens/val.bin")
    p.add_argument("--out-dir", default="checkpoints/run1")
    p.add_argument("--steps", type=int, default=None,
                   help="batasi jumlah langkah (untuk smoke run)")
    p.add_argument("--total-tokens", type=int, default=1_300_000_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: diskalakan otomatis dari d_model")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true",
                   help="lanjutkan dari checkpoint terbaru di out-dir")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sample-interval", type=int, default=500)
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--tokenizer",
                   default="data/tokenizer/wicara-bpe-16k.json",
                   help="dipakai agar samples.txt berisi teks terbaca")
    args = p.parse_args()

    mcfg = CONFIGS[args.model]

    # LR diskalakan terhadap lebar model kalau tidak ditentukan manual.
    lr = args.lr if args.lr is not None else scaled_lr(mcfg.d_model)
    if args.lr is None:
        print(f"  LR otomatis untuk d_model={mcfg.d_model}: {lr:.2e}")

    # Kalau --steps diberikan, jadwal LR harus MENGIKUTI panjang run itu.
    #
    # Tanpa ini, warmup tetap dihitung 2% dari total_tokens (1,3B token =
    # 19.836 langkah = warmup 396 langkah). Run pendek 400 langkah jadi
    # habis di warmup saja dan tidak pernah masuk cosine decay -- LR-nya
    # naik terus sampai langkah terakhir. Diam-diam salah, tanpa error.
    total_tokens = args.total_tokens
    if args.steps is not None and not args.resume:
        total_tokens = args.steps * args.batch_size * mcfg.max_seq_len * args.grad_accum
        print(f"  --steps {args.steps} diberikan: jadwal LR disesuaikan ke "
              f"{total_tokens / 1e6:.1f}M token.")
        print("  (untuk pretrain sungguhan, pakai --total-tokens tanpa --steps)")
        print()

    tcfg = TrainConfig(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        model_config=args.model,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=lr,
        total_tokens=total_tokens,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        checkpoint_interval_minutes=args.ckpt_minutes,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
    )

    print("=" * 66)
    print("  TRAINING")
    print("=" * 66)

    # Tokenizer hanya untuk sample_probe. Kalau tidak ada, training tetap
    # jalan tapi samples.txt berisi ID token mentah yang tak terbaca.
    tok = None
    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tok = load_tokenizer(tok_path)
        print(f"  Tokenizer  : {tok_path.name} "
              f"(vocab {tok.get_vocab_size():,}) -- samples.txt terbaca")
    else:
        print(f"  [!] Tokenizer tidak ada di {tok_path}")
        print("      samples.txt akan berisi ID token, bukan teks.")
    print()

    trainer = Trainer(mcfg, tcfg, tokenizer=tok)
    try:
        if args.resume:
            trainer.resume()

        # Cetak kurva LR SEBELUM training dimulai. Jauh lebih murah menemukan
        # jadwal yang salah sekarang daripada setelah 9 jam berjalan.
        print("  Jadwal learning rate:")
        print(describe_schedule(
            lr=tcfg.lr, min_lr=tcfg.min_lr,
            warmup_steps=trainer.warmup_steps,
            total_steps=trainer.total_steps,
        ))
        print()

        # Kalau ada metadata korpus sintetis, tampilkan target entropinya.
        meta_path = Path(args.train_bin).with_name(
            Path(args.train_bin).stem.replace("_train", "_meta") + ".json"
        )
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"  Korpus sintetis terdeteksi — entropi teoretis "
                  f"{meta['theoretical_entropy']:.4f} "
                  f"(perplexity {meta['theoretical_perplexity']})")
            print("  Loss yang benar akan konvergen ke angka itu, tidak di bawahnya.")
            print()

        hasil = trainer.train(max_steps=args.steps)

        print()
        print("=" * 66)
        for k, v in hasil.items():
            print(f"  {k:<16}: {v}")
        print(f"  checkpoint      : {tcfg.out_dir}")
        print(f"  log             : {tcfg.out_dir}/log.jsonl")
    finally:
        trainer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
