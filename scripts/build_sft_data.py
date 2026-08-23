"""Bangun data SFT percakapan ke data/sft/{train,val}.jsonl.gz

    .venv\\Scripts\\python.exe scripts\\build_sft_data.py

Keluaran: satu objek JSON per baris,
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
     "sumber": "...", "kategori": "..."}

Hampir semua contoh ditambang dari data NYATA. Lihat src/data/sft.py untuk
daftar subset yang dipakai dan alasan tiap subset lain dibuang.

Satu pengecualian: data/sft/tulis_tangan.jsonl berisi 400 contoh yang DITULIS
TANGAN, bukan ditambang. Lihat kumpulkan_tulis_tangan() untuk alasannya.
"""

import argparse
import glob
import gzip
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.dedup import ExactDedup  # noqa: E402
from src.data.sft import (  # noqa: E402
    AYA_DIPAKAI,
    CENDOL_DIPAKAI,
    layak_instruksi,
    tambang_pasangan_subtitle,
)
from src.tokenizer.bpe import iter_jsonl_gz_text  # noqa: E402

RAW = REPO_ROOT / "data" / "raw"
CLEAN = REPO_ROOT / "data" / "clean"
OUT = REPO_ROOT / "data" / "sft"
TULIS_TANGAN = REPO_ROOT / "data" / "sft" / "tulis_tangan.jsonl"

# Kuota sengaja KECIL. SFT berlebihan membuat model kehilangan kemampuan
# bahasa yang didapat susah payah di pretrain (catastrophic forgetting), dan
# mutu contoh jauh lebih menentukan daripada jumlahnya.
KUOTA_SUBTITLE = 16_000

# Kuota per SUBSET, bukan per sumber.
#
# Versi pertama memakai satu kuota untuk seluruh Aya, dan akibatnya Dolly-v2
# tidak kebagian sama sekali: kuota habis diisi SODA-inst yang kebetulan
# dibaca lebih dulu. Padahal Dolly-v2 justru subset bermutu tertinggi —
# ditulis manusia, bukan hasil terjemahan mesin.
KUOTA_SUBSET = {
    "SODA-inst (T)": 12_000,   # dialog sosial & instruksi naratif
    "Dolly-v2 (T)": 10_000,    # ditulis manusia, mutu terbaik
    "wikihow": 8_000,          # instruksi prosedural
}

# Batas per kategori sapaan, supaya satu pola tidak mendominasi hanya karena
# lebih sering muncul di film.
# tidak_tahu SENGAJA dibatasi kecil.
#
# Versi pertama memberinya 3.000 slot (7,6% data). Hasilnya model menjawab
# "aku tidak tahu" untuk 30% prompt uji -- termasuk sapaan biasa seperti
# "Kamu lagi apa?" dan "Ceritakan sesuatu". Penyebabnya: contoh yang ditambang
# memasangkan pertanyaan APA PUN dengan jawaban tidak tahu, sehingga model
# menyimpulkan aturan yang terlalu luas.
#
# Dan tujuan awalnya toh tidak tercapai: pertanyaan pengetahuan seperti
# "Apa ibu kota Zimbabwe?" tetap dikarang, karena contoh yang ditambang
# berupa pertanyaan percakapan, bukan ensiklopedis. Jadi kategori ini
# merugikan obrolan biasa tanpa memberi manfaat yang diharapkan.
#
# Disisakan sedikit karena "aku tidak tahu" memang respons percakapan
# yang sah -- hanya tidak boleh jadi jawaban default.
#
# ---
#
# perkenalan DIMATIKAN (0), bukan sekadar dikecilkan.
#
# Kategori ini menambang pasangan "Siapa namamu?" -> "Namaku X" dari subtitle,
# dan X selalu nama tokoh film. Itu persis pertanyaan yang sekarang harus
# dijawab dengan persona Wicara, tapi dengan jawaban yang salah.
#
# Dengan 1.200 contoh tambang melawan 138 contoh persona tulis tangan,
# perbandingannya 9:1 dan model akan ikut yang mayoritas. Inilah sebab model
# menjawab "Namaku Kittan" dan "Aku Daphne" — bukan kebocoran dari pretrain,
# melainkan data SFT sendiri yang mengajarkannya.
#
# Menyisakan sebagian pun tidak ada gunanya: seluruh isi kategori ini
# mengajarkan perilaku yang justru ingin diganti.
BATAS_KATEGORI = {
    "sapaan": 3000, "kabar": 2000, "perkenalan": 0, "terima_kasih": 1500,
    "perpisahan": 1500, "maaf": 1200, "tidak_tahu": 600,
}


def tulis(f, user: str, asisten: str, sumber: str, kategori: str) -> None:
    f.write(json.dumps({
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": asisten}],
        "sumber": sumber,
        "kategori": kategori,
    }, ensure_ascii=False) + "\n")


def kumpulkan_subtitle(kuota: int, dedup: ExactDedup) -> list[tuple]:
    """Tambang pasangan sapaan, dengan dedup di tingkat PASANGAN.

    Tanpa dedup, "Terima kasih" -> "Sama-sama" muncul ribuan kali: itu memang
    pertukaran paling umum di film. Untuk SFT yang dibutuhkan RAGAM, bukan
    frekuensi — model cukup melihat satu pola beberapa kali, dan sisa kuota
    lebih berguna diisi variasi lain.
    """
    path = CLEAN / "opensubtitles.jsonl.gz"
    if not path.exists():
        print("    opensubtitles.jsonl.gz tidak ada, dilewati")
        return []

    per_kategori = Counter()
    hasil = []
    n_blok = n_dup = 0
    t0 = time.time()

    for blok in iter_jsonl_gz_text(path):
        n_blok += 1
        for user, asisten, kat in tambang_pasangan_subtitle(blok):
            if per_kategori[kat] >= BATAS_KATEGORI.get(kat, 0):
                continue
            if dedup.is_duplicate(f"{user}||{asisten}"):
                n_dup += 1
                continue
            per_kategori[kat] += 1
            hasil.append((user, asisten, "subtitle", kat))

        if len(hasil) >= kuota:
            break
        if n_blok % 300_000 == 0:
            print(f"      {n_blok:,} blok -> {len(hasil):,} unik "
                  f"({n_dup:,} duplikat, {time.time() - t0:.0f}s)", flush=True)

    print(f"    {n_blok:,} blok dibaca, {n_dup:,} duplikat dibuang")
    for kat, n in per_kategori.most_common():
        print(f"      {n:>6,}  {kat}")
    return hasil


def kumpulkan_parquet(pola: str, kolom_in: str, kolom_out: str,
                      kolom_filter: str, kuota_subset: dict,
                      sumber: str, dedup: ExactDedup) -> list[tuple]:
    """Kumpulkan contoh instruksi, tiap subset dengan kuota sendiri."""
    hasil = []
    per_subset = Counter()
    n_dup = 0

    def penuh() -> bool:
        return all(per_subset[k] >= v for k, v in kuota_subset.items())

    for f in sorted(glob.glob(pola)):
        for b in pq.ParquetFile(f).iter_batches(
            batch_size=4096, columns=[kolom_in, kolom_out, kolom_filter]
        ):
            for r in b.to_pylist():
                subset = r[kolom_filter]
                if subset not in kuota_subset:
                    continue
                if per_subset[subset] >= kuota_subset[subset]:
                    continue
                inp, out = r[kolom_in], r[kolom_out]
                if not layak_instruksi(inp, out):
                    continue
                if dedup.is_duplicate(f"{inp}||{out}"):
                    n_dup += 1
                    continue
                per_subset[subset] += 1
                hasil.append((inp.strip(), out.strip(), sumber, subset))
            if penuh():
                break
        if penuh():
            break

    for k, n in per_subset.most_common():
        print(f"      {n:>6,}  {k}")
    if n_dup:
        print(f"      {n_dup:>6,}  (duplikat dibuang)")
    return hasil


def kumpulkan_tulis_tangan(path: Path, dedup: ExactDedup) -> list[tuple]:
    """Baca contoh yang DITULIS TANGAN, bukan ditambang dari korpus.

    Ini satu-satunya bagian data SFT yang tidak berasal dari data nyata, dan
    itu disengaja. Dua pola di bawah tidak ada di sumber mana pun, jadi
    memang harus ditulis:

        persona  Tidak ada korpus yang memuat "Saya Wicara". Tanpa contoh
                 ini model mengambil nama tokoh film dari subtitle dan
                 menjawab "Namaku Kittan".

        terbuka  Permintaan tanpa topik ("ceritakan sesuatu", "bosen nih").
                 Dialog film tidak punya bentuk ini — di film selalu ada
                 adegan yang memberi konteks, jadi tidak pernah ada giliran
                 yang benar-benar kosong topik.

    Jumlahnya sengaja kecil, sekitar 1% dari total. Tujuannya menambah dua
    pola yang hilang, bukan menggeser gaya bahasa yang didapat dari pretrain.

    Berkasnya JSONL polos (bukan gzip) supaya bisa dibaca dan disunting
    langsung, dan supaya perubahannya terlihat utuh di git diff.
    """
    if not path.exists():
        print(f"    {path.name} tidak ada, dilewati")
        return []

    hasil = []
    per_kategori = Counter()
    n_dup = n_tolak = 0

    with open(path, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            d = json.loads(baris)
            msgs = d["messages"]
            user = msgs[0]["content"]
            asisten = msgs[1]["content"]

            # Disaring dengan aturan yang sama seperti sumber lain, supaya
            # berkas yang disunting tangan tidak bisa menyelundupkan contoh
            # yang tidak akan pernah lolos dari jalur tambang.
            if not layak_instruksi(user, asisten):
                n_tolak += 1
                continue
            if dedup.is_duplicate(f"{user}||{asisten}"):
                n_dup += 1
                continue

            per_kategori[d["kategori"]] += 1
            hasil.append((user, asisten, "tulis_tangan", d["kategori"]))

    for kat, n in per_kategori.most_common():
        print(f"      {n:>6,}  {kat}")
    if n_dup or n_tolak:
        print(f"      {n_dup:>6,}  (duplikat)   {n_tolak:,} ditolak saringan")
    return hasil


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  BANGUN DATA SFT")
    print("=" * 74)

    semua = []
    # Satu dedup dipakai bersama lintas sumber, supaya contoh yang sama
    # tidak lolos dua kali lewat jalur berbeda.
    dedup = ExactDedup()

    # Tulis tangan dikumpulkan PALING AWAL supaya ia yang memenangkan dedup
    # kalau ada tabrakan. Contohnya cuma 400 dan tidak tergantikan; contoh
    # tambang ada puluhan ribu dan mudah dicari penggantinya.
    print(f"\n  [tulis tangan] {TULIS_TANGAN.name} — persona & permintaan terbuka")
    semua += kumpulkan_tulis_tangan(TULIS_TANGAN, dedup)

    print(f"\n  [subtitle] kuota {KUOTA_SUBTITLE:,} — sapaan & basa-basi")
    semua += kumpulkan_subtitle(KUOTA_SUBTITLE, dedup)

    kuota_aya = {k: v for k, v in KUOTA_SUBSET.items() if k in AYA_DIPAKAI}
    print(f"\n  [aya] kuota per subset: {kuota_aya}")
    semua += kumpulkan_parquet(
        str(RAW / "aya" / "*.parquet"), "inputs", "targets",
        "dataset_name", kuota_aya, "aya", dedup)

    kuota_cendol = {k: v for k, v in KUOTA_SUBSET.items() if k in CENDOL_DIPAKAI}
    print(f"\n  [cendol] kuota per subset: {kuota_cendol}")
    semua += kumpulkan_parquet(
        str(RAW / "cendol" / "*.parquet"), "input", "output",
        "subset_name", kuota_cendol, "cendol", dedup)

    # Diacak supaya batch tidak berisi satu sumber saja — kalau tidak,
    # gradient tiap langkah bias ke satu gaya bahasa.
    rng = random.Random(args.seed)
    rng.shuffle(semua)

    n_val = max(1, int(len(semua) * args.val_ratio))
    val, train = semua[:n_val], semua[n_val:]

    for nama, data in (("train", train), ("val", val)):
        path = out / f"{nama}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for u, a, s, k in data:
                tulis(f, u, a, s, k)
        print(f"\n  {nama:<6}: {len(data):>7,} contoh -> {path.name} "
              f"({path.stat().st_size / 1024**2:.1f} MB)")

    stat = Counter(s for _, _, s, _ in semua)
    kat = Counter(k for _, _, _, k in semua)

    print("\n" + "=" * 74)
    print("  RINGKASAN")
    print("=" * 74)
    print(f"  {'sumber':<14}{'contoh':>9}{'porsi':>8}")
    print("  " + "-" * 31)
    for s, n in stat.most_common():
        print(f"  {s:<14}{n:>9,}{n / len(semua):>8.1%}")
    print("  " + "-" * 31)
    print(f"  {'TOTAL':<14}{len(semua):>9,}")

    print(f"\n  {'kategori':<18}{'contoh':>9}")
    print("  " + "-" * 27)
    for k, n in kat.most_common(15):
        print(f"  {k:<18}{n:>9,}")

    meta = {"total": len(semua), "train": len(train), "val": len(val),
            "per_sumber": dict(stat), "per_kategori": dict(kat),
            "kuota_subtitle": KUOTA_SUBTITLE, "kuota_subset": KUOTA_SUBSET,
            "seed": args.seed}
    (out / "stats.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Statistik: {out / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
