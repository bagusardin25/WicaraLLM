"""Pembaca dokumen dari tiap format sumber.

Tiap sumber punya bentuk berbeda, dan perbedaannya bukan sekadar teknis —
menentukan apa yang sebenarnya dipelajari model:

    wikipedia / fineweb2  -> satu baris = satu dokumen prosa utuh
    aya / cendol          -> pasangan instruksi-jawaban, digabung jadi dialog
    opensubtitles / ted   -> satu baris = satu kalimat, perlu dikelompokkan

Semua reader menghasilkan string mentah. Pembersihan dilakukan terpisah di
src/data/clean.py supaya aturannya bisa diuji sendiri.
"""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

BATCH = 2048


def iter_parquet_field(
    paths: list[Path], field: str, batch_size: int = BATCH
) -> Iterator[str]:
    """Baca satu kolom teks dari beberapa berkas parquet, per batch.

    iter_batches dipakai (bukan read_table) supaya berkas 800 MB tidak
    dimuat sekaligus ke RAM.
    """
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[field]):
            for v in batch.column(field).to_pylist():
                if v:
                    yield v


def iter_parquet_pair(
    paths: list[Path],
    field_a: str,
    field_b: str,
    batch_size: int = BATCH,
) -> Iterator[str]:
    """Gabungkan dua kolom (pertanyaan + jawaban) jadi satu blok dialog.

    Kenapa digabung dan bukan diambil salah satu: kalau hanya `inputs` yang
    dipakai, model cuma melihat cara ORANG BERTANYA dan tidak pernah melihat
    bentuk jawaban. Kalau hanya `targets`, sebaliknya. Yang berharga justru
    PASANGANNYA — di situlah pola tanya-jawab muncul.

    Formatnya sengaja polos, bukan chat template. Chat template baru dipasang
    di Fase 6 (SFT); di pretrain, model cukup melihat bahwa teks sering
    berbentuk giliran bicara.
    """
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size,
                                     columns=[field_a, field_b]):
            a_col = batch.column(field_a).to_pylist()
            b_col = batch.column(field_b).to_pylist()
            for a, b in zip(a_col, b_col):
                a = (a or "").strip()
                b = (b or "").strip()
                if a and b:
                    yield f"{a}\n{b}"
                elif a or b:
                    yield a or b


def iter_jsonl_gz(path: Path, field: str = "text") -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                v = json.loads(line).get(field)
            except json.JSONDecodeError:
                continue
            if v:
                yield v


def iter_lines_gz(path: Path) -> Iterator[str]:
    """Satu baris teks per baris berkas (OpenSubtitles, TED)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


# --------------------------------------------------------------------------

def build_reader(key: str, raw_dir: Path):
    """Kembalikan iterator dokumen untuk satu sumber."""
    d = raw_dir / key

    if key == "wikipedia":
        return iter_parquet_field(list(d.glob("*.parquet")), "text")

    if key == "fineweb2":
        return iter_jsonl_gz(d / "fineweb2.jsonl.gz", "text")

    if key == "aya":
        return iter_parquet_pair(list(d.glob("*.parquet")), "inputs", "targets")

    if key == "cendol":
        return iter_parquet_pair(list(d.glob("*.parquet")), "input", "output")

    if key in ("opensubtitles", "ted"):
        return iter_lines_gz(d / "id.txt.gz")

    raise ValueError(f"Tidak tahu cara membaca sumber: {key}")


# Sumber yang isinya baris pendek dan perlu dikelompokkan jadi blok,
# bukan diperlakukan sebagai dokumen per baris.
LINE_SOURCES = {"opensubtitles", "ted"}
