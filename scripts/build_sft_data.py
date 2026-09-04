"""Bangun data/sft/{train,val}.jsonl.gz

    .venv\\Scripts\\python.exe scripts\\build_sft_data.py

Mayoritas contoh ditambang dari korpus (lihat src/data/sft.py). Sisanya
data/sft/tulis_tangan.jsonl — kita tulis sendiri untuk perilaku yang tidak
ada di korpus mana pun. Lihat kumpulkan_tulis_tangan().
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

# Kuota ditahan kecil. SFT berlebihan menggerus kemampuan bahasa
# yang didapat susah payah di pretrain.
KUOTA_SUBTITLE = 16_000

# Faktor pengulangan contoh persona. Alasannya di kumpulkan_tulis_tangan().
ULANG_KATEGORI = {"persona", "batas"}
ULANG_PERSONA = 2

# Per subset, bukan per sumber. Kuota tunggal untuk seluruh Aya membuat
# Dolly-v2 tidak kebagian — kuota habis di SODA-inst yang kebetulan dibaca dulu.
KUOTA_SUBSET = {
    "SODA-inst (T)": 12_000,   # dialog sosial
    "Dolly-v2 (T)": 10_000,    # ditulis manusia
    "wikihow": 8_000,          # prosedural
}

# Batas per kategori sapaan supaya satu pola film tidak mendominasi.
#
# perkenalan dan tidak_tahu dikunci 0: keduanya mengajarkan lawan dari
# perilaku yang kita bangun. perkenalan menambang "Siapa namamu?" ->
# nama tokoh film; 55 dari 587 contoh tidak_tahu diawali "Siapa" sehingga
# membajak pertanyaan identitas. Jangan dihidupkan lagi tanpa mengukur
# ulang persona lewat scripts/eval_perilaku.py.
BATAS_KATEGORI = {
    "sapaan": 3000, "kabar": 2000, "perkenalan": 0, "terima_kasih": 1500,
    "perpisahan": 1500, "maaf": 1200, "tidak_tahu": 0,
}


def tulis(f, user: str, asisten: str, sumber: str, kategori: str) -> None:
    f.write(json.dumps({
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": asisten}],
        "sumber": sumber,
        "kategori": kategori,
    }, ensure_ascii=False) + "\n")


def kumpulkan_subtitle(kuota: int, dedup: ExactDedup) -> list[tuple]:
    """Tambang sapaan, dedup di tingkat pasangan.

    Tanpa dedup, "Terima kasih" -> "Sama-sama" muncul ribuan kali. SFT
    butuh ragam, bukan frekuensi.
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
    """Contoh yang kita tulis sendiri, bukan hasil tambang.

    Dua perilaku produk yang tidak ada di korpus mana pun:
        persona  tidak ada teks Indonesia yang memuat "Saya Wicara"
        terbuka  permintaan tanpa topik; dialog film selalu punya adegan

    Ditahan di ~2% dari total. JSONL polos supaya bisa disunting langsung
    dan perubahannya terbaca di git diff.
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

            # Saringan yang sama dengan jalur tambang. Berkas yang disunting
            # tangan tidak boleh jadi pintu belakang.
            if not layak_instruksi(user, asisten):
                n_tolak += 1
                continue
            if dedup.is_duplicate(f"{user}||{asisten}"):
                n_dup += 1
                continue

            per_kategori[d["kategori"]] += 1
            hasil.append((user, asisten, "tulis_tangan", d["kategori"]))

    # Pada 161 instance, persona masih kalah dari prior nama subtitle dan
    # model menjawab "Aku Kawis!". Diulang sampai porsinya menang.
    # Pengulangan aman khusus di sini: menghafal "Saya Wicara" memang target,
    # beda dari kategori lain yang justru rusak kalau dihafal.
    ulang = [h for h in hasil if h[3] in ULANG_KATEGORI]
    hasil += ulang * (ULANG_PERSONA - 1)

    for kat, n in per_kategori.most_common():
        n_akhir = n * ULANG_PERSONA if kat in ULANG_KATEGORI else n
        tanda = f"  (x{ULANG_PERSONA})" if kat in ULANG_KATEGORI else ""
        print(f"      {n_akhir:>6,}  {kat}{tanda}")
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
    # Satu ExactDedup lintas sumber.
    dedup = ExactDedup()

    # Tulis tangan didahulukan supaya menang dedup. Contohnya tidak
    # tergantikan; contoh tambang selalu ada penggantinya.
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

    # Acak supaya batch tidak satu sumber.
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
