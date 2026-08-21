"""Jalankan pipeline pembersihan ke seluruh korpus.

Murni Python deterministik — regex, hash, dan bit array. Tidak ada model,
tidak menyentuh GPU. Input sama selalu menghasilkan output sama.

    .venv\\Scripts\\python.exe scripts\\clean_corpus.py
    .venv\\Scripts\\python.exe scripts\\clean_corpus.py --only wikipedia,ted
    .venv\\Scripts\\python.exe scripts\\clean_corpus.py --budget 200000000

Tiap sumber berhenti dibaca begitu kuota tokennya tercapai. Jadi walaupun
OpenSubtitles punya 169 juta baris, yang benar-benar diproses hanya sebanyak
yang dibutuhkan.
"""

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.clean import (  # noqa: E402
    Alasan,
    clean_document_verbose,
    clean_subtitle_line_verbose,
    group_subtitle_lines,
)
from src.data.dedup import BloomDedup  # noqa: E402
from src.data.readers import LINE_SOURCES, build_reader  # noqa: E402

RAW = REPO_ROOT / "data" / "raw"
CLEAN = REPO_ROOT / "data" / "clean"
CHARS_PER_TOKEN = 3.7

# Anggaran total token untuk pretrain (1 epoch, tanpa pengulangan).
BUDGET_DEFAULT = 1_300_000_000

# Komposisi akhir korpus.
#
# OpenSubtitles diberi porsi terbesar sesuai fokus percakapan, tapi sengaja
# TIDAK mendominasi. Subtitle punya kekhasan yang menyesatkan kalau berlebihan:
# kalimat pendek, konteks visual yang hilang, banyak seruan. Model yang hanya
# makan subtitle akan bicara terpotong-potong seperti orang di film.
#
# Wikipedia menyeimbangkan dengan kalimat utuh dan pengetahuan dunia; FineWeb
# menambah ragam tulis informal; Aya dan Cendol memberi pola tanya-jawab.
KOMPOSISI = {
    "opensubtitles": 0.40,
    "fineweb2": 0.20,
    "wikipedia": 0.15,
    "aya": 0.12,
    "cendol": 0.12,
    "ted": 0.01,
}

# Panjang blok percakapan untuk sumber berbasis baris.
BLOK_CHARS = 1200


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def proses_sumber(key: str, kuota_token: int, verbose: bool = True) -> dict:
    """Bersihkan satu sumber sampai kuota token tercapai."""
    kuota_chars = int(kuota_token * CHARS_PER_TOKEN)
    out_path = CLEAN / f"{key}.jsonl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = build_reader(key, RAW)
    is_line = key in LINE_SOURCES

    # Bloom di-ukur dari perkiraan jumlah item, bukan jumlah token.
    perkiraan_item = kuota_chars // (BLOK_CHARS if is_line else 800) + 1000
    bloom = BloomDedup(expected_items=max(perkiraan_item * 2, 100_000),
                       fp_rate=0.01)

    alasan = Counter()
    n_masuk = n_tulis = n_dup = 0
    chars_tulis = 0
    buffer: list[str] = []
    t0 = time.time()

    with gzip.open(out_path, "wt", encoding="utf-8") as out:

        def tulis(teks: str) -> bool:
            """Tulis satu dokumen. False kalau kuota sudah penuh."""
            nonlocal n_tulis, n_dup, chars_tulis
            if bloom.is_duplicate(teks):
                n_dup += 1
                return True
            out.write(json.dumps({"text": teks}, ensure_ascii=False) + "\n")
            n_tulis += 1
            chars_tulis += len(teks)
            return chars_tulis < kuota_chars

        for item in reader:
            n_masuk += 1

            if is_line:
                hasil, sebab = clean_subtitle_line_verbose(item)
                alasan[sebab] += 1
                if hasil:
                    buffer.append(hasil)
                    # Dikelompokkan jadi blok DULU, baru dedup. Dedup per
                    # baris akan membuang frasa umum ("terima kasih",
                    # "baiklah") yang memang wajar sering muncul.
                    if sum(len(x) + 1 for x in buffer) >= BLOK_CHARS:
                        if not tulis("\n".join(buffer)):
                            buffer = []
                            break
                        buffer = []
            else:
                hasil, sebab = clean_document_verbose(item)
                alasan[sebab] += 1
                if hasil and not tulis(hasil):
                    break

            if verbose and n_masuk % 200_000 == 0:
                pct = 100 * chars_tulis / kuota_chars
                print(f"      {n_masuk:>11,} dibaca  {n_tulis:>9,} ditulis  "
                      f"{pct:5.1f}% kuota  {time.time() - t0:5.0f}s", flush=True)

        # Sisa buffer, kalau cukup panjang untuk jadi blok bermakna.
        if is_line and len(buffer) >= 4:
            tulis("\n".join(buffer))

    dt = time.time() - t0
    return {
        "sumber": key,
        "dibaca": n_masuk,
        "ditulis": n_tulis,
        "duplikat": n_dup,
        "karakter": chars_tulis,
        "token_perkiraan": int(chars_tulis / CHARS_PER_TOKEN),
        "kuota_token": kuota_token,
        "bytes": out_path.stat().st_size,
        "detik": round(dt, 1),
        "alasan": dict(alasan.most_common()),
        "lolos_filter": alasan[Alasan.OK] / max(n_masuk, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--budget", type=int, default=BUDGET_DEFAULT,
                   help="total token target untuk seluruh korpus")
    args = p.parse_args()

    pilih = args.only.split(",") if args.only else list(KOMPOSISI)
    CLEAN.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  PEMBERSIHAN KORPUS")
    print("=" * 74)
    print(f"  Anggaran total : {args.budget / 1e6:.0f} juta token")
    print(f"  Sumber         : {', '.join(pilih)}")
    print(f"\n  {'sumber':<16}{'porsi':>7}{'kuota token':>14}")
    print("  " + "-" * 37)
    for k in pilih:
        print(f"  {k:<16}{KOMPOSISI[k]:>6.0%}{int(args.budget * KOMPOSISI[k]):>14,}")

    hasil = []
    t_mulai = time.time()

    for key in pilih:
        kuota = int(args.budget * KOMPOSISI[key])
        print(f"\n  [{key}] kuota {kuota / 1e6:.0f}M token")
        try:
            r = proses_sumber(key, kuota)
        except Exception as exc:
            print(f"    GAGAL: {type(exc).__name__}: {exc}")
            continue

        hasil.append(r)
        pct_kuota = 100 * r["token_perkiraan"] / kuota
        print(f"    dibaca {r['dibaca']:,} -> ditulis {r['ditulis']:,} "
              f"({r['lolos_filter']:.1%} lolos filter, "
              f"{r['duplikat']:,} duplikat)")
        print(f"    {r['token_perkiraan'] / 1e6:.0f}M token "
              f"({pct_kuota:.0f}% kuota) | {human(r['bytes'])} | {r['detik']:.0f}s")
        if pct_kuota < 90:
            print(f"    CATATAN: kuota tidak terpenuhi — sumber kehabisan data")

    stats_path = CLEAN / "stats.json"
    stats_path.write_text(
        json.dumps({"budget": args.budget, "komposisi": KOMPOSISI,
                    "hasil": hasil}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 74)
    print("  RINGKASAN")
    print("=" * 74)
    print(f"  {'sumber':<16}{'ditulis':>11}{'token':>10}{'porsi':>8}"
          f"{'lolos':>8}{'ukuran':>10}")
    print("  " + "-" * 63)
    total_tok = sum(r["token_perkiraan"] for r in hasil)
    for r in hasil:
        porsi = r["token_perkiraan"] / max(total_tok, 1)
        print(f"  {r['sumber']:<16}{r['ditulis']:>11,}"
              f"{r['token_perkiraan'] / 1e6:>9.0f}M{porsi:>8.1%}"
              f"{r['lolos_filter']:>8.1%}{human(r['bytes']):>10}")
    print("  " + "-" * 63)
    print(f"  {'TOTAL':<16}{sum(r['ditulis'] for r in hasil):>11,}"
          f"{total_tok / 1e6:>9.0f}M")
    print(f"\n  Waktu total : {(time.time() - t_mulai) / 60:.1f} menit")
    print(f"  Keluaran    : {CLEAN}")
    print(f"  Statistik   : {stats_path}")
    print(f"\n  Cukup untuk {total_tok / 1e9:.2f} miliar token training "
          f"(1 epoch, tanpa pengulangan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
