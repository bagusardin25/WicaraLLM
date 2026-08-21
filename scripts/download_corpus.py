"""Unduh korpus bahasa Indonesia dari sumber publik.

Semua sumber didefinisikan di src/data/sources.py. Tidak ada data sintetis.

Fitur:
  * Resume — unduhan besar (OpenSubtitles 1,5 GB) tahan koneksi putus
  * SHA256 — dicatat untuk tiap berkas, supaya asal-usul data bisa dibuktikan
  * Manifest — data/raw/manifest.json + SOURCES.md dibuat otomatis

Contoh:
    .venv\\Scripts\\python.exe scripts\\download_corpus.py --list
    .venv\\Scripts\\python.exe scripts\\download_corpus.py --only wikipedia,ted
    .venv\\Scripts\\python.exe scripts\\download_corpus.py
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.sources import (  # noqa: E402
    DITOLAK,
    SOURCES,
    SOURCES_BY_KEY,
    TOTAL_TARGET_TOKENS,
    Kind,
    Source,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
UA = {"User-Agent": "wicara-corpus-downloader/1.0"}
MB = 1024**2


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sha256_file(path: Path, chunk: int = 8 * MB) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, retries: int = 5) -> dict:
    """Unduh satu berkas, lanjutkan kalau sudah ada sebagian.

    Server OPUS dan HuggingFace sama-sama mendukung HTTP Range, jadi unduhan
    1,5 GB yang putus di tengah tidak perlu diulang dari nol.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        sudah = dest.stat().st_size if dest.exists() else 0
        headers = dict(UA)
        if sudah:
            headers["Range"] = f"bytes={sudah}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers.get("Content-Length", 0)) + sudah
                mode = "ab" if (sudah and r.status == 206) else "wb"
                if mode == "wb":
                    sudah = 0

                t0, terakhir = time.time(), 0.0
                with dest.open(mode) as f:
                    while chunk := r.read(4 * MB):
                        f.write(chunk)
                        sudah += len(chunk)
                        now = time.time()
                        if now - terakhir > 3:
                            laju = sudah / max(now - t0, 1e-9) / MB
                            pct = f"{100 * sudah / total:5.1f}%" if total else "  ?  "
                            print(f"      {pct}  {human(sudah):>9} / "
                                  f"{human(total) if total else '?':>9}  "
                                  f"{laju:5.1f} MB/s", flush=True)
                            terakhir = now

            return {"ok": True, "bytes": dest.stat().st_size}

        except Exception as exc:
            if attempt == retries:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            tunggu = 2**attempt
            print(f"      gagal ({type(exc).__name__}), coba lagi "
                  f"{attempt}/{retries} dalam {tunggu}s", flush=True)
            time.sleep(tunggu)

    return {"ok": False, "error": "kehabisan percobaan"}


def fetch_http_gz(src: Source, out_dir: Path) -> list[dict]:
    hasil = []
    for url in src.download:
        dest = out_dir / Path(url).name
        print(f"    -> {dest.name}")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"       sudah ada ({human(dest.stat().st_size)}), dilewati")
            hasil.append({"url": url, "file": dest.name,
                          "bytes": dest.stat().st_size, "skipped": True})
            continue
        r = download_with_resume(url, dest)
        r.update({"url": url, "file": dest.name})
        hasil.append(r)
    return hasil


def fetch_hf_parquet(src: Source, out_dir: Path) -> list[dict]:
    hasil = []
    for i, url in enumerate(src.download):
        dest = out_dir / f"{src.key}_{i:03d}.parquet"
        print(f"    -> {dest.name}")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"       sudah ada ({human(dest.stat().st_size)}), dilewati")
            hasil.append({"url": url, "file": dest.name,
                          "bytes": dest.stat().st_size, "skipped": True})
            continue
        r = download_with_resume(url, dest)
        r.update({"url": url, "file": dest.name})
        hasil.append(r)
    return hasil


def fetch_hf_stream(src: Source, out_dir: Path) -> list[dict]:
    """Streaming untuk dataset yang terlalu besar diunduh utuh.

    FineWeb-2 ind_Latn ukurannya 141 GB. Kebutuhan pretrain hanya ~110 juta token,
    jadi baris ditarik satu per satu dan dihentikan setelah kuota tercapai.
    Hanya bagian yang benar-benar dibaca yang diunduh.
    """
    from datasets import load_dataset

    dest = out_dir / f"{src.key}.jsonl.gz"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"       sudah ada ({human(dest.stat().st_size)}), dilewati")
        return [{"url": src.download[0], "file": dest.name,
                 "bytes": dest.stat().st_size, "skipped": True}]

    print(f"    -> {dest.name}  (streaming, maks {src.max_docs:,} dokumen)")
    ds = load_dataset(
        src.hf_dataset, src.hf_config, split=src.hf_split, streaming=True
    )

    n_doc = n_char = 0
    t0 = time.time()
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        for row in ds:
            teks = (row.get(src.text_field) or "").strip()
            if not teks:
                continue
            f.write(json.dumps({"text": teks}, ensure_ascii=False) + "\n")
            n_doc += 1
            n_char += len(teks)
            if n_doc % 20_000 == 0:
                dt = time.time() - t0
                print(f"      {n_doc:>8,} dok  {n_char / 1e6:7.1f}M karakter  "
                      f"{n_doc / max(dt, 1e-9):6.0f} dok/s", flush=True)
            if src.max_docs and n_doc >= src.max_docs:
                break

    print(f"      selesai: {n_doc:,} dokumen, {n_char / 1e6:.1f}M karakter")
    return [{"url": src.download[0], "file": dest.name,
             "bytes": dest.stat().st_size, "docs": n_doc, "chars": n_char,
             "ok": True}]


FETCHERS = {
    Kind.HTTP_GZ: fetch_http_gz,
    Kind.HF_PARQUET: fetch_hf_parquet,
    Kind.HF_STREAM: fetch_hf_stream,
}


def tulis_sources_md(manifest: dict) -> Path:
    """Dokumentasi asal-usul korpus dengan tautan langsung."""
    path = RAW_DIR / "SOURCES.md"
    L = [
        "# Sumber Korpus — Wicara",
        "",
        "Semua data di bawah adalah **teks nyata dari repositori publik**.",
        "Tidak ada satu pun data sintetis.",
        "",
        f"Diunduh: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}",
        f"Kuota unduhan: **{TOTAL_TARGET_TOKENS / 1e6:.0f} juta token** "
        "(target per sumber saat mengunduh; jumlah akhir setelah "
        "pembersihan tercatat di data/clean/stats.json)",
        "",
        "---",
        "",
    ]

    for src in SOURCES:
        entri = manifest.get("sources", {}).get(src.key, {})
        berkas = entri.get("files", [])
        total_b = sum(f.get("bytes", 0) for f in berkas)
        status = entri.get("status", "belum diunduh")

        L += [
            f"## {src.name}",
            "",
            f"- **Peran**: {src.role}",
            f"- **Target**: {src.target_tokens / 1e6:.0f} juta token",
            f"- **Lisensi**: {src.license}",
            f"- **Status**: {status}"
            + (f" — {human(total_b)} di disk" if total_b else ""),
            f"- **Halaman sumber**: <{src.homepage}>",
            "",
            "**Tautan unduhan langsung:**",
            "",
        ]
        for u in src.download:
            L.append(f"- <{u}>")
        L.append("")

        if berkas:
            L += ["| Berkas | Ukuran | SHA256 (16 char awal) |",
                  "|---|---|---|"]
            for f in berkas:
                sh = f.get("sha256", "")[:16] or "-"
                L.append(f"| `{f.get('file','?')}` | "
                         f"{human(f.get('bytes', 0))} | `{sh}` |")
            L.append("")

        if src.citation:
            L += [f"**Sitasi**: {src.citation}", ""]
        if src.catatan:
            L += [f"**Catatan**: {src.catatan}", ""]
        L += ["---", ""]

    L += [
        "## Sumber yang dipertimbangkan tapi tidak dipakai",
        "",
        "Dicatat supaya keputusannya bisa ditelusuri dan tidak diteliti ulang.",
        "",
        "| Dataset | Alasan |",
        "|---|---|",
    ]
    for nama, url, alasan in DITOLAK:
        L.append(f"| [{nama}]({url}) | {alasan} |")
    L += [
        "",
        "> Dua sumber paling sering direkomendasikan untuk bahasa Indonesia — ",
        "> CulturaX dan OSCAR — ternyata **ter-gate** dan butuh akun serta ",
        "> persetujuan. FineWeb-2 dipakai sebagai gantinya: terbuka, dan ",
        "> penyaringan serta deduplikasinya lebih ketat.",
        "",
    ]

    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   help="daftar key dipisah koma, mis. wikipedia,ted")
    p.add_argument("--list", action="store_true", help="tampilkan sumber lalu keluar")
    p.add_argument("--skip-hash", action="store_true",
                   help="lewati SHA256 (cepat, tapi provenance tidak tercatat)")
    args = p.parse_args()

    if args.list:
        print(f"\n  {'key':<16}{'target':>10}  peran")
        print("  " + "-" * 74)
        for s in SOURCES:
            print(f"  {s.key:<16}{s.target_tokens / 1e6:>8.0f}M  {s.role}")
        print(f"\n  TOTAL {TOTAL_TARGET_TOKENS / 1e6:.0f} juta token target\n")
        return 0

    pilih = args.only.split(",") if args.only else list(SOURCES_BY_KEY)
    tidak_dikenal = [k for k in pilih if k not in SOURCES_BY_KEY]
    if tidak_dikenal:
        print(f"key tidak dikenal: {tidak_dikenal}")
        print(f"tersedia: {list(SOURCES_BY_KEY)}")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(MANIFEST.read_text(encoding="utf-8"))
        if MANIFEST.exists() else {"sources": {}}
    )

    print("=" * 72)
    print("  UNDUH KORPUS BAHASA INDONESIA")
    print("=" * 72)

    for key in pilih:
        src = SOURCES_BY_KEY[key]
        out_dir = RAW_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{key}] {src.name}")
        print(f"    peran  : {src.role}")
        print(f"    lisensi: {src.license}")

        try:
            berkas = FETCHERS[src.kind](src, out_dir)
        except Exception as exc:
            print(f"    GAGAL: {type(exc).__name__}: {exc}")
            manifest["sources"][key] = {"status": f"gagal: {exc}", "files": []}
            MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            continue

        if not args.skip_hash:
            for f in berkas:
                path = out_dir / f["file"]
                if path.exists() and "sha256" not in f:
                    print(f"      hashing {f['file']}...", flush=True)
                    f["sha256"] = sha256_file(path)

        gagal = [f for f in berkas if f.get("ok") is False]
        manifest["sources"][key] = {
            "name": src.name,
            "homepage": src.homepage,
            "license": src.license,
            "role": src.role,
            "target_tokens": src.target_tokens,
            "status": "gagal" if gagal else "selesai",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": berkas,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        total = sum(f.get("bytes", 0) for f in berkas)
        print(f"    {'GAGAL' if gagal else 'OK'} — {human(total)}")

    md = tulis_sources_md(manifest)

    print("\n" + "=" * 72)
    print("  RINGKASAN")
    print("=" * 72)
    grand = 0
    for key in pilih:
        e = manifest["sources"].get(key, {})
        b = sum(f.get("bytes", 0) for f in e.get("files", []))
        grand += b
        print(f"  {key:<16}{e.get('status','-'):<12}{human(b):>10}")
    print(f"  {'TOTAL':<16}{'':<12}{human(grand):>10}")
    print(f"\n  manifest : {MANIFEST}")
    print(f"  dokumen  : {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
