"""Registry sumber korpus bahasa Indonesia.

SEMUA sumber di sini adalah teks nyata dari repositori publik terpercaya.
Tidak ada data sintetis.

Ketersediaan tiap sumber diverifikasi langsung lewat HuggingFace API dan
HTTP HEAD pada 2026-08-21, bukan dari ingatan. Dua sumber populer sengaja
TIDAK dipakai karena ter-gate (butuh login + persetujuan manual):

    uonlp/CulturaX          gated=auto    -> butuh akun HF
    oscar-corpus/OSCAR-2301 gated=manual  -> butuh persetujuan pemilik

Penggantinya HuggingFaceFW/fineweb-2, yang terbuka DAN kualitasnya lebih
baik karena sudah melalui penyaringan dan deduplikasi FineWeb.
"""

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    HF_PARQUET = "hf_parquet"  # unduh shard parquet langsung
    HF_STREAM = "hf_stream"    # streaming, berhenti setelah N dokumen
    HTTP_GZ = "http_gz"        # file .gz langsung dari server sumber


@dataclass
class Source:
    key: str
    name: str
    kind: Kind

    # Peran dalam korpus dan porsi yang ditargetkan
    role: str
    target_tokens: int

    # Tautan — ini yang jadi bukti asal-usul data
    homepage: str
    download: list[str] = field(default_factory=list)

    # Metadata HuggingFace
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    n_shards: int = 1
    max_docs: int | None = None  # untuk HF_STREAM

    # Legal & akademik
    license: str = "?"
    citation: str = ""
    catatan: str = ""


# --------------------------------------------------------------------------
# Komposisi korpus.
#
# Ditekankan ke teks PERCAKAPAN sesuai permintaan. OpenSubtitles jadi tulang
# punggung karena isinya benar-benar dialog manusia — bukan prosa yang
# kebetulan berisi kutipan.
#
# Tapi tidak 100% dialog: subtitle punya kekhasan sendiri (kalimat pendek,
# banyak seruan, konteks visual yang hilang). Wikipedia menambah kalimat utuh
# dan pengetahuan dunia; FineWeb menambah ragam tulis informal. Model yang
# hanya makan subtitle akan bicara seperti orang di film, terpotong-potong.
# --------------------------------------------------------------------------

SOURCES: list[Source] = [
    Source(
        key="opensubtitles",
        name="OpenSubtitles v2024 (Indonesia, monolingual)",
        kind=Kind.HTTP_GZ,
        role="Dialog percakapan — tulang punggung korpus",
        target_tokens=180_000_000,
        homepage="https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles",
        download=[
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/id.txt.gz"
        ],
        license="Lihat opus.nlpl.eu — subtitle dari opensubtitles.org",
        citation=(
            "Lison & Tiedemann (2016), OpenSubtitles2016: Extracting Large "
            "Parallel Corpora from Movie and TV Subtitles, LREC. "
            "Korpus OPUS: Tiedemann (2012), LREC."
        ),
        catatan=(
            "1,5 GB terkompresi. Satu baris = satu baris subtitle, jadi "
            "sangat pendek. Perlu digabung jadi blok percakapan saat "
            "pembersihan, bukan diperlakukan sebagai dokumen per baris."
        ),
    ),
    Source(
        key="wikipedia",
        name="Wikipedia Bahasa Indonesia (dump 2023-11-01)",
        kind=Kind.HF_PARQUET,
        role="Faktual, kalimat formal utuh, pengetahuan dunia",
        target_tokens=90_000_000,
        homepage="https://huggingface.co/datasets/wikimedia/wikipedia",
        download=[
            f"https://huggingface.co/api/datasets/wikimedia/wikipedia/"
            f"parquet/20231101.id/train/{i}.parquet"
            for i in range(3)
        ],
        hf_dataset="wikimedia/wikipedia",
        hf_config="20231101.id",
        n_shards=3,
        license="CC-BY-SA-3.0 / GFDL",
        citation="Wikimedia Foundation. Dump 20231101, 665.622 artikel.",
        catatan="Sudah bersih dari markup wiki oleh Wikimedia.",
    ),
    Source(
        key="fineweb2",
        name="FineWeb-2 (ind_Latn)",
        kind=Kind.HF_STREAM,
        role="Ragam tulis informal dari web — penyeimbang register",
        target_tokens=110_000_000,
        homepage="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        download=[
            "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/"
            "parquet/ind_Latn/train/0.parquet"
        ],
        hf_dataset="HuggingFaceFW/fineweb-2",
        hf_config="ind_Latn",
        max_docs=400_000,
        license="ODC-BY 1.0",
        citation=(
            "Penedo et al., FineWeb-2: A sparser, better, multilingual "
            "corpus. HuggingFace."
        ),
        catatan=(
            "Total 141 GB / 100 juta dokumen — jauh melebihi kebutuhan, jadi "
            "diambil lewat streaming dan dihentikan setelah kuota tercapai. "
            "Dipakai menggantikan CulturaX/OSCAR yang ter-gate."
        ),
    ),
    Source(
        key="aya",
        name="Aya Collection (indonesian)",
        kind=Kind.HF_PARQUET,
        role="Instruksi & tanya-jawab beranotasi manusia",
        target_tokens=40_000_000,
        homepage=(
            "https://huggingface.co/datasets/CohereLabs/"
            "aya_collection_language_split"
        ),
        download=[
            "https://huggingface.co/api/datasets/CohereLabs/"
            "aya_collection_language_split/parquet/indonesian/train/0.parquet"
        ],
        hf_dataset="CohereLabs/aya_collection_language_split",
        hf_config="indonesian",
        text_field="inputs",
        license="Apache-2.0",
        citation=(
            "Singh et al. (2024), Aya Dataset: An Open-Access Collection "
            "for Multilingual Instruction Tuning. Cohere For AI."
        ),
        catatan="3,6 juta baris. Kolom inputs + targets digabung jadi dialog.",
    ),
    Source(
        key="cendol",
        name="Cendol Collection v2 (IndoNLP)",
        kind=Kind.HF_PARQUET,
        role="Instruksi bahasa Indonesia & bahasa daerah",
        target_tokens=30_000_000,
        homepage="https://huggingface.co/datasets/indonlp/cendol_collection_v2",
        download=[
            f"https://huggingface.co/api/datasets/indonlp/"
            f"cendol_collection_v2/parquet/default/train/{i}.parquet"
            for i in range(3)
        ],
        hf_dataset="indonlp/cendol_collection_v2",
        hf_config="default",
        text_field="input",
        n_shards=3,
        license="Apache-2.0",
        citation=(
            "Cahyawijaya et al. (2024), Cendol: Open Instruction-tuned "
            "Generative Large Language Models for Indonesian Languages."
        ),
        catatan="12,8 juta baris total; diambil 3 shard dari 23.",
    ),
    Source(
        key="ted",
        name="TED2020 (Indonesia, monolingual)",
        kind=Kind.HTTP_GZ,
        role="Transkrip pidato lisan — bahasa lisan yang tertata",
        target_tokens=5_000_000,
        homepage="https://opus.nlpl.eu/TED2020/id&id/v1/TED2020",
        download=["https://object.pouta.csc.fi/OPUS-TED2020/v1/mono/id.txt.gz"],
        license="CC-BY-NC-ND 4.0 (TED)",
        citation=(
            "Reimers & Gurevych (2020), Making Monolingual Sentence "
            "Embeddings Multilingual using Knowledge Distillation, EMNLP."
        ),
        catatan="Kecil (5 MB) tapi kualitas tinggi: lisan tapi terstruktur.",
    ),
]

SOURCES_BY_KEY = {s.key: s for s in SOURCES}
TOTAL_TARGET_TOKENS = sum(s.target_tokens for s in SOURCES)


# --------------------------------------------------------------------------
# Sumber yang DIPERTIMBANGKAN tapi tidak dipakai — dicatat supaya keputusannya
# bisa ditelusuri ulang, dan supaya tidak diteliti dua kali.
# --------------------------------------------------------------------------

DITOLAK = [
    ("uonlp/CulturaX", "https://huggingface.co/datasets/uonlp/CulturaX",
     "gated=auto — butuh akun HuggingFace dan menyetujui syarat"),
    ("oscar-corpus/OSCAR-2301",
     "https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
     "gated=manual — butuh persetujuan pemilik dataset"),
    ("HPLT/HPLT2.0_cleaned",
     "https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned",
     "terbuka dan bagus (212 GB, ind_Latn), tapi tumpang tindih dengan "
     "FineWeb-2; disimpan sebagai cadangan kalau perlu tambahan"),
    ("MBZUAI/Bactrian-X", "https://huggingface.co/datasets/MBZUAI/Bactrian-X",
     "CC-BY-NC-4.0 dan hasil terjemahan mesin — kualitas bahasanya di bawah "
     "Aya/Cendol yang beranotasi manusia"),
    ("allenai/c4 (multilingual id)", "https://huggingface.co/datasets/allenai/c4",
     "mC4 disaring jauh lebih longgar daripada FineWeb-2; banyak boilerplate"),
]
