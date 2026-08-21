"""Verifikasi integritas korpus yang sudah diunduh.

Bukan sekadar mengecek berkas ada. Yang diperiksa:

  1. Ukuran berkas cocok dengan manifest
  2. SHA256 dihitung ULANG dan dibandingkan  -> mendeteksi berkas rusak
  3. Berkas benar-benar bisa dibaca sampai habis -> mendeteksi berkas
     terpotong akibat unduhan putus (gzip akan melempar error di akhir)
  4. Jumlah dokumen dan karakter nyata -> perkiraan token yang jujur

Poin 3 penting: unduhan yang putus di tengah menghasilkan berkas yang
UKURANNYA terlihat wajar dan bisa dibuka, tapi meledak saat dibaca sampai
akhir. Kalau baru ketahuan di tengah training, sia-sia berjam-jam.
"""

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.sources import SOURCES_BY_KEY  # noqa: E402

RAW = REPO_ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
MB = 1024**2

# Perkiraan kompresi tokenizer BPE 16k untuk bahasa Indonesia.
# Target plan: 3,5-4 karakter per token. Dipakai 3,7 sebagai tengah.
CHARS_PER_TOKEN = 3.7


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(8 * MB):
            h.update(blk)
    return h.hexdigest()


def scan_parquet(path: Path, fields: list[str]) -> tuple[int, int]:
    """Baca seluruh isi kolom teks. Mengembalikan (baris, karakter)."""
    pf = pq.ParquetFile(path)
    ada = [f for f in fields if f in pf.schema_arrow.names]
    if not ada:
        raise ValueError(f"kolom {fields} tidak ada; tersedia "
                         f"{pf.schema_arrow.names}")
    baris = chars = 0
    for batch in pf.iter_batches(batch_size=4096, columns=ada):
        cols = [batch.column(f).to_pylist() for f in ada]
        for row in zip(*cols):
            baris += 1
            chars += sum(len(v) for v in row if v)
    return baris, chars


def scan_gz_lines(path: Path) -> tuple[int, int]:
    """Dekompresi SAMPAI HABIS. Berkas terpotong akan melempar error di sini."""
    baris = chars = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            baris += 1
            chars += len(line)
    return baris, chars


def scan_jsonl_gz(path: Path, field: str = "text") -> tuple[int, int]:
    baris = chars = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                v = json.loads(line).get(field) or ""
            except json.JSONDecodeError:
                continue
            baris += 1
            chars += len(v)
    return baris, chars


SCANNER = {
    "wikipedia": lambda p: scan_parquet(p, ["text"]),
    "aya": lambda p: scan_parquet(p, ["inputs", "targets"]),
    "cendol": lambda p: scan_parquet(p, ["input", "output"]),
    "fineweb2": lambda p: scan_jsonl_gz(p),
    "opensubtitles": scan_gz_lines,
    "ted": scan_gz_lines,
}


def main() -> int:
    if not MANIFEST.exists():
        print("manifest.json tidak ada — jalankan download_corpus.py dulu.")
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("=" * 76)
    print("  VERIFIKASI KORPUS")
    print("=" * 76)

    semua_ok = True
    ringkas = []

    for key, src in SOURCES_BY_KEY.items():
        entri = manifest.get("sources", {}).get(key)
        print(f"\n  [{key}] {src.name}")

        if not entri:
            print("    TIDAK ADA di manifest")
            semua_ok = False
            continue

        berkas = entri.get("files", [])
        if not berkas:
            print("    tidak ada berkas tercatat")
            semua_ok = False
            continue

        tot_baris = tot_chars = tot_bytes = 0
        src_ok = True

        for f in berkas:
            path = RAW / key / f["file"]
            nama = f["file"]

            if not path.exists():
                print(f"    [X] {nama}: HILANG")
                src_ok = semua_ok = False
                continue

            size = path.stat().st_size
            tot_bytes += size

            if f.get("bytes") and size != f["bytes"]:
                print(f"    [X] {nama}: ukuran beda "
                      f"({human(size)} vs manifest {human(f['bytes'])})")
                src_ok = semua_ok = False
                continue

            if f.get("sha256"):
                print(f"      {nama}: cek SHA256...", end="", flush=True)
                nyata = sha256_file(path)
                if nyata != f["sha256"]:
                    print(" TIDAK COCOK — berkas rusak")
                    src_ok = semua_ok = False
                    continue
                print(" cocok", end="", flush=True)

            print(f" | baca isi...", end="", flush=True)
            t0 = time.time()
            try:
                baris, chars = SCANNER[key](path)
            except Exception as exc:
                print(f" GAGAL: {type(exc).__name__}: {exc}")
                print("      -> kemungkinan berkas terpotong; unduh ulang")
                src_ok = semua_ok = False
                continue

            tot_baris += baris
            tot_chars += chars
            print(f" OK ({baris:,} baris, {time.time() - t0:.0f}s)")

        tok = tot_chars / CHARS_PER_TOKEN
        status = "OK" if src_ok else "BERMASALAH"
        print(f"    {status} — {human(tot_bytes)} | {tot_baris:,} baris | "
              f"{tot_chars / 1e6:.0f}M karakter | ~{tok / 1e6:.0f}M token")
        ringkas.append((key, src_ok, tot_bytes, tot_baris, tot_chars, tok))

    print("\n" + "=" * 76)
    print("  RINGKASAN")
    print("=" * 76)
    print(f"  {'sumber':<16}{'status':<8}{'ukuran':>9}{'baris':>14}"
          f"{'karakter':>12}{'~token':>11}")
    print("  " + "-" * 70)

    g_bytes = g_baris = g_chars = g_tok = 0
    for key, ok, b, ln, ch, tk in ringkas:
        print(f"  {key:<16}{'OK' if ok else 'RUSAK':<8}{human(b):>9}"
              f"{ln:>14,}{ch / 1e6:>10.0f}M{tk / 1e6:>10.0f}M")
        g_bytes += b; g_baris += ln; g_chars += ch; g_tok += tk

    print("  " + "-" * 70)
    print(f"  {'TOTAL':<16}{'':<8}{human(g_bytes):>9}{g_baris:>14,}"
          f"{g_chars / 1e6:>10.0f}M{g_tok / 1e6:>10.0f}M")

    print(f"\n  Perkiraan token MENTAH   : {g_tok / 1e6:.0f} juta")
    print(f"  Setelah bersih+dedup ~65%: {g_tok * 0.65 / 1e6:.0f} juta token unik")
    print(f"  Dengan 3 epoch           : "
          f"{g_tok * 0.65 * 3 / 1e9:.2f} miliar token training")
    print(f"  Target plan              : 1,30 miliar token training")

    print(f"\n  Hasil akhir: {'SEMUA BERKAS UTUH' if semua_ok else 'ADA MASALAH'}")
    return 0 if semua_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
