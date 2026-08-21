"""Buat korpus sintetis dengan ENTROPI YANG DIKETAHUI PERSIS.

Kenapa perlu, padahal Fase 4 katanya tidak butuh data:
    Training loop memang bisa diuji dengan token acak murni. Tapi token acak
    tidak punya pola sama sekali, jadi loss akan diam di ln(vocab) = 9,70
    selamanya. Mekanismenya terverifikasi, tapi tidak terbukti bahwa loop-nya
    benar-benar BISA BELAJAR.

Solusinya: bahasa buatan yang polanya sederhana tapi entropinya bisa dihitung
di atas kertas.

Prosesnya bigram: tiap token punya tepat `branching` kemungkinan token
berikutnya, dipilih seragam. Karena hanya token sebelumnya yang menentukan,
entropi kondisionalnya tepat:

    H = ln(branching)

Untuk branching=8:  H = ln(8) = 2,0794

Jadi target verifikasinya presisi:
    * Loss HARUS turun dari 9,70 menuju ~2,08
    * Loss TIDAK BOLEH turun jauh di bawah 2,08 — kalau itu terjadi, model
      menghafal urutan spesifik dan bukan mempelajari aturannya. Ini juga
      cara mendeteksi kebocoran data validasi ke data latih.

Contoh:
    python scripts/make_synthetic_data.py --tokens 20000000 --branching 8
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.train.data import write_tokens  # noqa: E402
from src.train.synthetic import generate_bigram_corpus  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--n-active", type=int, default=2048,
                   help="berapa token dari vocab yang benar-benar dipakai")
    p.add_argument("--branching", type=int, default=8,
                   help="kemungkinan lanjutan per token; entropi = ln(nilai ini)")
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="data/tokens")
    p.add_argument("--prefix", type=str, default="synth")
    args = p.parse_args()

    entropi = math.log(args.branching)
    out = Path(args.out_dir)

    print(f"  Token target   : {args.tokens:,}")
    print(f"  Token aktif    : {args.n_active:,} (dari vocab 16.384)")
    print(f"  Branching      : {args.branching}")
    print(f"  ENTROPI TEORI  : {entropi:.4f}  <- target konvergensi loss")
    print()

    print("  Membuat rantai...")
    tokens = generate_bigram_corpus(
        args.tokens, args.n_active, args.branching, args.seed
    )

    # Split dilakukan dengan MEMOTONG di satu titik, bukan mengambil acak
    # tersebar. Kalau tersebar, potongan validasi akan bertetangga langsung
    # dengan potongan latih dan hasilnya bocor.
    n_val = int(len(tokens) * args.val_ratio)
    train, val = tokens[:-n_val], tokens[-n_val:]

    write_tokens(out / f"{args.prefix}_train.bin", train)
    write_tokens(out / f"{args.prefix}_val.bin", val)

    meta = {
        "kind": "synthetic_bigram",
        "n_tokens": int(len(tokens)),
        "n_active": args.n_active,
        "branching": args.branching,
        "theoretical_entropy": entropi,
        "theoretical_perplexity": args.branching,
        "seed": args.seed,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
    }
    (out / f"{args.prefix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"  train : {len(train):>12,} token -> {args.prefix}_train.bin "
          f"({train.nbytes / 2 / 1024**2:.0f} MB)")
    print(f"  val   : {len(val):>12,} token -> {args.prefix}_val.bin")
    print(f"  meta  : {args.prefix}_meta.json")
    print()
    print(f"  Loss yang benar akan konvergen ke {entropi:.4f} "
          f"(perplexity {args.branching}).")
    print("  Jauh di bawah itu = model menghafal, bukan belajar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
