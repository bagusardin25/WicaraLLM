"""Latih tokenizer BPE 16k dari korpus bersih (data/clean/).

    .venv\\Scripts\\python.exe scripts\\train_tokenizer.py
    .venv\\Scripts\\python.exe scripts\\train_tokenizer.py --sample-ratio 0.05
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tokenizer.bpe import (  # noqa: E402
    VOCAB_SIZE,
    build_tokenizer,
    build_trainer,
    compression_ratio,
    iter_jsonl_gz_text,
    save,
    verify_special_tokens,
)
from src.tokenizer.chat_template import ALL_SPECIAL_TOKENS  # noqa: E402

CLEAN = REPO_ROOT / "data" / "clean"
OUT = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"

# Urutan sumber untuk pelaporan.
SUMBER = ["opensubtitles", "fineweb2", "wikipedia", "aya", "cendol", "ted"]


def iter_sampel(rasio: float):
    """Gabungan teks dari semua sumber, disampel proporsional."""
    setiap_ke = max(1, round(1 / rasio))
    for nama in SUMBER:
        path = CLEAN / f"{nama}.jsonl.gz"
        if not path.exists():
            print(f"    lewati {nama} (belum ada)")
            continue
        yield from iter_jsonl_gz_text(path, setiap_ke=setiap_ke)


def ambil_contoh_uji(n_per_sumber: int = 300) -> dict[str, list[str]]:
    """Dokumen untuk mengukur rasio kompresi per sumber, setelah training."""
    contoh = {}
    for nama in SUMBER:
        path = CLEAN / f"{nama}.jsonl.gz"
        if path.exists():
            contoh[nama] = list(
                itertools.islice(iter_jsonl_gz_text(path, setiap_ke=997),
                                 n_per_sumber)
            )
    return contoh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--sample-ratio", type=float, default=0.15,
                   help="porsi dokumen yang dipakai melatih BPE")
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    if not CLEAN.exists() or not list(CLEAN.glob("*.jsonl.gz")):
        print("data/clean/ kosong — jalankan scripts/clean_corpus.py dulu.")
        return 1

    print("=" * 72)
    print("  LATIH TOKENIZER BPE")
    print("=" * 72)
    print(f"  Vocab        : {args.vocab_size:,}")
    print(f"  Special token: {len(ALL_SPECIAL_TOKENS)} (ID 0-"
          f"{len(ALL_SPECIAL_TOKENS) - 1})")
    print(f"  Sampel       : {args.sample_ratio:.2%} dokumen "
          f"(tiap dokumen ke-{max(1, round(1 / args.sample_ratio))})")
    print(f"  Keluaran     : {args.out}")
    print()

    tok = build_tokenizer()
    trainer = build_trainer(args.vocab_size)

    print("  Melatih... (progress bar dari tokenizers, bisa beberapa menit)")
    t0 = time.time()
    tok.train_from_iterator(iter_sampel(args.sample_ratio), trainer=trainer)
    dt = time.time() - t0
    print(f"\n  Selesai dalam {dt / 60:.1f} menit")

    # Verifikasi ID special token.
    verify_special_tokens(tok)
    print(f"  Verifikasi ID special token: OK")

    vocab_nyata = tok.get_vocab_size()
    print(f"  Ukuran vocab akhir: {vocab_nyata:,}")
    if vocab_nyata < args.vocab_size:
        print(f"  CATATAN: kurang dari target — korpus sampel kehabisan "
              f"pasangan yang cukup sering. Naikkan --sample-ratio.")

    path = save(tok, args.out)

    # -- mutu tokenizer -------------------------------------------------
    print("\n" + "=" * 72)
    print("  RASIO KOMPRESI (karakter per token)")
    print("=" * 72)
    print("  Target bahasa Indonesia: 3,5-4,0. Makin tinggi makin efisien —")
    print("  dengan anggaran token sama, model melihat lebih banyak isi.\n")

    contoh = ambil_contoh_uji()
    print(f"  {'sumber':<16}{'dokumen':>9}{'karakter':>11}"
          f"{'token':>10}{'char/token':>12}")
    print("  " + "-" * 58)

    total_c = total_t = 0
    per_sumber = {}
    for nama, teks in contoh.items():
        if not teks:
            continue
        r = compression_ratio(tok, teks)
        per_sumber[nama] = r["char_per_token"]
        total_c += r["karakter"]
        total_t += r["token"]
        print(f"  {nama:<16}{len(teks):>9,}{r['karakter']:>11,}"
              f"{r['token']:>10,}{r['char_per_token']:>12.2f}")

    rata = total_c / max(total_t, 1)
    print("  " + "-" * 58)
    print(f"  {'GABUNGAN':<16}{'':<9}{total_c:>11,}{total_t:>10,}{rata:>12.2f}")

    # -- perkiraan ulang jumlah token korpus ----------------------------
    stats_path = CLEAN / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        korpus_char = sum(r["karakter"] for r in stats["hasil"])
        perkiraan_lama = korpus_char / 3.7
        perkiraan_baru = korpus_char / rata

        print("\n" + "=" * 72)
        print("  PERKIRAAN ULANG JUMLAH TOKEN KORPUS")
        print("=" * 72)
        print(f"  Karakter korpus     : {korpus_char / 1e6:,.0f} juta")
        print(f"  Asumsi lama (3,70)  : {perkiraan_lama / 1e6:,.0f} juta token")
        print(f"  Terukur ({rata:.2f})      : "
              f"{perkiraan_baru / 1e6:,.0f} juta token")
        selisih = (perkiraan_baru - perkiraan_lama) / perkiraan_lama
        print(f"  Selisih             : {selisih:+.1%}")
        print(f"\n  Target training 1,3 miliar token -> "
              f"{'CUKUP' if perkiraan_baru >= 1.3e9 else 'KURANG'} "
              f"untuk 1 epoch")

    meta = {
        "vocab_size": vocab_nyata,
        "special_tokens": len(ALL_SPECIAL_TOKENS),
        "sample_ratio": args.sample_ratio,
        "train_seconds": round(dt, 1),
        "char_per_token": round(rata, 4),
        "char_per_token_per_source": {k: round(v, 4)
                                      for k, v in per_sumber.items()},
    }
    meta_path = Path(args.out).with_name("tokenizer_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"\n  Tokenizer : {path}")
    print(f"  Metadata  : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
