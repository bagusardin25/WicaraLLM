"""Peragaan pembersihan korpus di data NYATA yang sudah diunduh.

Bukan contoh karangan — semua yang ditampilkan diambil langsung dari berkas
di data/raw/. Tujuannya supaya keputusan tiap filter bisa dilihat, dan kalau
ada yang terlalu galak atau terlalu longgar, ketahuan SEBELUM pipeline
dijalankan ke 3 miliar token.

    .venv\\Scripts\\python.exe scripts\\demo_cleaning.py
"""

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Konsol Windows memakai cp1252 dan akan melempar UnicodeEncodeError begitu
# bertemu karakter di luar Latin-1 — padahal korpus ini penuh dengannya.
# errors="replace" dipilih supaya peragaan tidak mati hanya karena satu
# karakter aneh tidak bisa digambar di terminal.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.clean import (  # noqa: E402
    clean_document_verbose,
    clean_subtitle_line_verbose,
    group_subtitle_lines,
    normalize,
)
from src.data.dedup import BloomDedup, ExactDedup, normalize_for_hash  # noqa: E402
from src.data.readers import build_reader  # noqa: E402

RAW = REPO_ROOT / "data" / "raw"


def judul(s: str) -> None:
    print(f"\n{'=' * 74}\n  {s}\n{'=' * 74}")


def potong(s: str, n: int = 96) -> str:
    s = s.replace("\n", " ⏎ ")
    return s[:n] + ("…" if len(s) > n else "")


# ==========================================================================
def tahap_1_normalisasi() -> None:
    judul("TAHAP 1 — NORMALISASI: menyatukan yang terlihat sama")

    print("""
  Masalahnya: karakter yang terlihat identik bisa punya kode berbeda.
  Tokenizer memperlakukannya sebagai token BERBEDA, jadi slot vocab
  terbuang untuk varian penulisan yang sebenarnya sama.
""")
    contoh = [
        ("Huruf lebar", "Ｈａｌｏ ａｐａ ｋａｂａｒ"),
        ("Kutip tipografis", "Dia bilang “halo” dan ‘pergi’"),
        ("Spasi ganda", "halo     apa      kabar"),
        ("Karakter diulang", "haloooooooooooo!!!!!!!!!!"),
        ("Spasi tak terlihat", "halo​apa​kabar"),
    ]
    for label, teks in contoh:
        print(f"  {label:<20} {teks!r}")
        print(f"  {'':<20} -> {normalize(teks)!r}\n")


# ==========================================================================
def tahap_2_filter(n_sampel: int = 3000) -> None:
    judul("TAHAP 2 — FILTER: membuang yang bukan bahasa Indonesia layak")

    print("""
  Tiap dokumen diuji beberapa saringan. Model 56M kapasitasnya terbatas —
  setiap pola sampah yang lolos akan ikut dipelajari dan memakan kapasitas
  yang seharusnya dipakai untuk bahasa.
""")

    for key in ("wikipedia", "fineweb2", "cendol"):
        reader = build_reader(key, RAW)
        stat = Counter()
        contoh_tolak: dict[str, str] = {}
        n = 0

        for doc in reader:
            hasil, alasan = clean_document_verbose(doc)
            stat[alasan] += 1
            if alasan != "ok" and alasan not in contoh_tolak:
                contoh_tolak[alasan] = doc
            n += 1
            if n >= n_sampel:
                break

        lolos = stat["ok"]
        print(f"\n  [{key}] {n:,} dokumen sampel — lolos {lolos:,} "
              f"({100 * lolos / n:.1f}%)")
        for alasan, jml in stat.most_common():
            if alasan == "ok":
                continue
            print(f"      dibuang {jml:>5,} ({100 * jml / n:4.1f}%)  {alasan}")

        for alasan, teks in list(contoh_tolak.items())[:2]:
            print(f"\n      contoh '{alasan}':")
            print(f"        {potong(teks)!r}")


# ==========================================================================
def tahap_2b_subtitle(n_sampel: int = 20000) -> None:
    judul("TAHAP 2b — SUBTITLE: aturan berbeda untuk baris dialog")

    print("""
  Subtitle tidak bisa diperlakukan seperti prosa. Barisnya sangat pendek,
  penuh artefak format, dan yang paling merepotkan: KREDIT PENERJEMAH yang
  identik muncul di ribuan berkas. Kalau lolos, model akan hafal nama
  penerjemah alih-alih belajar bahasa.
""")
    reader = build_reader("opensubtitles", RAW)
    stat = Counter()
    contoh_tolak: dict[str, str] = {}
    diubah: list[tuple[str, str]] = []
    lolos_teks: list[str] = []
    n = 0

    for line in reader:
        hasil, alasan = clean_subtitle_line_verbose(line)
        stat[alasan] += 1
        if alasan != "ok" and alasan not in contoh_tolak:
            contoh_tolak[alasan] = line
        if hasil:
            lolos_teks.append(hasil)
            if hasil != line.strip() and len(diubah) < 4:
                diubah.append((line, hasil))
        n += 1
        if n >= n_sampel:
            break

    print(f"  {n:,} baris sampel — lolos {stat['ok']:,} "
          f"({100 * stat['ok'] / n:.1f}%)")
    for alasan, jml in stat.most_common():
        if alasan == "ok":
            continue
        print(f"      dibuang {jml:>6,} ({100 * jml / n:4.1f}%)  {alasan}")

    if diubah:
        print("\n  Contoh baris yang DIPERBAIKI (bukan dibuang):")
        for asli, bersih in diubah:
            print(f"      {asli.strip()!r}")
            print(f"      -> {bersih!r}\n")

    for alasan in ("kredit_subtitle", "terlalu_pendek", "kurang_dari_3_kata"):
        if alasan in contoh_tolak:
            print(f"  Contoh dibuang '{alasan}': {contoh_tolak[alasan].strip()!r}")

    return lolos_teks


# ==========================================================================
def tahap_3_dedup(n_sampel: int = 200000) -> None:
    judul("TAHAP 3 — DEDUPLIKASI: membuang yang kembar")

    print("""
  Langkah dengan hasil terbesar per usaha. Data kembar membuat model
  MENGHAFAL alih-alih belajar, dan membuat val loss berbohong (potongan
  yang sama muncul di data latih DAN validasi).

  Pembandingan dilakukan pada bentuk kanonik — huruf kecil, tanpa tanda
  baca — supaya beda sepele tidak membuat duplikat lolos:
""")
    for a, b in [("Halo, apa kabar?", "halo apa kabar"),
                 ("Aku  TIDAK   tahu!!", "aku tidak tahu")]:
        sama = normalize_for_hash(a) == normalize_for_hash(b)
        print(f"    {a!r:<26} vs {b!r:<22} -> "
              f"{'KEMBAR' if sama else 'beda'}")

    print(f"""
  TAPI ADA JEBAKAN. Dedup di level BARIS justru merusak.

  Menguji {n_sampel:,} baris subtitle nyata, dedup per baris:""")

    reader = build_reader("opensubtitles", RAW)
    bloom_baris = BloomDedup(expected_items=n_sampel * 2, fp_rate=0.01)
    ulang = Counter()
    bersih: list[str] = []

    for line in reader:
        hasil, _ = clean_subtitle_line_verbose(line)
        if hasil is None:
            continue
        bersih.append(hasil)
        if bloom_baris.is_duplicate(hasil):
            ulang[normalize_for_hash(hasil)[:52]] += 1
        if len(bersih) >= n_sampel:
            break

    s = bloom_baris.stats
    print(f"    duplikat {s['duplikat']:,} dari {s['total']:,} "
          f"({100 * s['rasio_duplikat']:.1f}%)")
    print(f"    RAM Bloom {s['bit_mb']} MB, {s['k_hash']} hash\n")

    print("  Yang paling sering berulang:")
    for teks, jml in ulang.most_common(6):
        print(f"    {jml:>5}x  {teks!r}")

    print("""
  Perhatikan: itu BUKAN sampah. Itu frasa percakapan Indonesia yang paling
  umum. Bahasa nyata memang Zipfian — "aku tak tahu" MEMANG sering muncul.
  Membuangnya justru menghapus paparan model pada pola paling dasar.

  Jadi dedup dilakukan di level BLOK, bukan baris. Blok yang identik berarti
  adegan atau berkas yang sama diproses dua kali — itu duplikasi sungguhan.
""")

    blok = group_subtitle_lines(bersih, target_chars=1200)
    bloom_blok = BloomDedup(expected_items=len(blok) * 2, fp_rate=0.01)
    dup_blok = sum(1 for b in blok if bloom_blok.is_duplicate(b))

    print(f"  Dedup per BLOK ({len(blok):,} blok dari {len(bersih):,} baris):")
    print(f"    duplikat {dup_blok:,} ({100 * dup_blok / max(len(blok), 1):.1f}%)")
    print(f"\n    Level baris: {100 * s['rasio_duplikat']:5.1f}% dibuang  "
          f"-> frekuensi alami bahasa ikut hilang")
    print(f"    Level blok : {100 * dup_blok / max(len(blok), 1):5.1f}% dibuang  "
          f"-> hanya duplikasi sungguhan")


# ==========================================================================
def tahap_4_pengelompokan() -> None:
    judul("TAHAP 4 — PENGELOMPOKAN: dari baris jadi percakapan")

    print("""
  Satu baris subtitle terlalu pendek untuk jadi contoh training. Model tidak
  akan pernah melihat konteks lebih dari satu kalimat, dan tidak belajar
  bahwa percakapan itu BERGILIRAN.

  Beberapa baris berurutan digabung jadi satu blok, supaya model melihat
  alur tanya-jawab. Inilah yang membuat korpus subtitle berharga untuk
  model percakapan.
""")
    reader = build_reader("opensubtitles", RAW)
    bersih = []
    for line in reader:
        hasil, _ = clean_subtitle_line_verbose(line)
        if hasil:
            bersih.append(hasil)
        if len(bersih) >= 400:
            break

    blok = group_subtitle_lines(bersih, target_chars=420)
    print(f"  {len(bersih)} baris bersih -> {len(blok)} blok percakapan\n")
    print("  Contoh satu blok (inilah bentuk yang dilihat model):")
    print("  " + "-" * 66)
    for ln in blok[1].split("\n")[:12]:
        print(f"    {ln}")
    print("  " + "-" * 66)


# ==========================================================================
def main() -> int:
    if not (RAW / "opensubtitles").exists():
        print("Korpus belum diunduh — jalankan scripts/download_corpus.py")
        return 1

    tahap_1_normalisasi()
    tahap_2_filter()
    tahap_2b_subtitle()
    tahap_3_dedup()
    tahap_4_pengelompokan()

    judul("RINGKAS")
    print("""
  Urutan pipeline:
      1. Normalisasi   — satukan bentuk yang terlihat sama
      2. Filter        — buang yang bukan bahasa Indonesia layak
      3. Deduplikasi   — buang yang kembar
      4. Pengelompokan — baris pendek jadi blok percakapan

  Karena ketersediaan korpus berlebih (~3 miliar token untuk kebutuhan 1,3 miliar),
  filter sengaja dibuat AGRESIF. Membuang teks bersih secara berlebihan
  jauh lebih murah daripada menyimpan yang kotor.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
