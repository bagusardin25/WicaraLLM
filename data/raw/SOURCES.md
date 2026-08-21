# Sumber Korpus — Wicara

Semua data di bawah adalah **teks nyata dari repositori publik**.
Tidak ada satu pun data sintetis.

Diunduh: 2026-08-22 02:12
Kuota unduhan: **455 juta token** (target per sumber saat mengunduh; jumlah akhir setelah pembersihan tercatat di data/clean/stats.json)

---

## OpenSubtitles v2024 (Indonesia, monolingual)

- **Peran**: Dialog percakapan — tulang punggung korpus
- **Target**: 180 juta token
- **Lisensi**: Lihat opus.nlpl.eu — subtitle dari opensubtitles.org
- **Status**: selesai — 1.5GB di disk
- **Halaman sumber**: <https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles>

**Tautan unduhan langsung:**

- <https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/id.txt.gz>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `id.txt.gz` | 1.5GB | `96aa12bf674d9fcf` |

**Sitasi**: Lison & Tiedemann (2016), OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles, LREC. Korpus OPUS: Tiedemann (2012), LREC.

**Catatan**: 1,5 GB terkompresi. Satu baris = satu baris subtitle, jadi sangat pendek. Perlu digabung jadi blok percakapan saat pembersihan, bukan diperlakukan sebagai dokumen per baris.

---

## Wikipedia Bahasa Indonesia (dump 2023-11-01)

- **Peran**: Faktual, kalimat formal utuh, pengetahuan dunia
- **Target**: 90 juta token
- **Lisensi**: CC-BY-SA-3.0 / GFDL
- **Status**: selesai — 556.8MB di disk
- **Halaman sumber**: <https://huggingface.co/datasets/wikimedia/wikipedia>

**Tautan unduhan langsung:**

- <https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.id/train/0.parquet>
- <https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.id/train/1.parquet>
- <https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.id/train/2.parquet>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `wikipedia_000.parquet` | 254.5MB | `ff54e23415c8e62c` |
| `wikipedia_001.parquet` | 139.7MB | `2c0912b6e653251c` |
| `wikipedia_002.parquet` | 162.6MB | `d0894394c5d73624` |

**Sitasi**: Wikimedia Foundation. Dump 20231101, 665.622 artikel.

**Catatan**: Sudah bersih dari markup wiki oleh Wikimedia.

---

## FineWeb-2 (ind_Latn)

- **Peran**: Ragam tulis informal dari web — penyeimbang register
- **Target**: 110 juta token
- **Lisensi**: ODC-BY 1.0
- **Status**: selesai — 537.2MB di disk
- **Halaman sumber**: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>

**Tautan unduhan langsung:**

- <https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/parquet/ind_Latn/train/0.parquet>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `fineweb2.jsonl.gz` | 537.2MB | `d03c54d82c6a15f4` |

**Sitasi**: Penedo et al., FineWeb-2: A sparser, better, multilingual corpus. HuggingFace.

**Catatan**: Total 141 GB / 100 juta dokumen — jauh melebihi kebutuhan, jadi diambil lewat streaming dan dihentikan setelah kuota tercapai. Dipakai menggantikan CulturaX/OSCAR yang ter-gate.

---

## Aya Collection (indonesian)

- **Peran**: Instruksi & tanya-jawab beranotasi manusia
- **Target**: 40 juta token
- **Lisensi**: Apache-2.0
- **Status**: selesai — 816.7MB di disk
- **Halaman sumber**: <https://huggingface.co/datasets/CohereLabs/aya_collection_language_split>

**Tautan unduhan langsung:**

- <https://huggingface.co/api/datasets/CohereLabs/aya_collection_language_split/parquet/indonesian/train/0.parquet>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `aya_000.parquet` | 816.7MB | `7fc4113923e69630` |

**Sitasi**: Singh et al. (2024), Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning. Cohere For AI.

**Catatan**: 3,6 juta baris. Kolom inputs + targets digabung jadi dialog.

---

## Cendol Collection v2 (IndoNLP)

- **Peran**: Instruksi bahasa Indonesia & bahasa daerah
- **Target**: 30 juta token
- **Lisensi**: Apache-2.0
- **Status**: selesai — 274.9MB di disk
- **Halaman sumber**: <https://huggingface.co/datasets/indonlp/cendol_collection_v2>

**Tautan unduhan langsung:**

- <https://huggingface.co/api/datasets/indonlp/cendol_collection_v2/parquet/default/train/0.parquet>
- <https://huggingface.co/api/datasets/indonlp/cendol_collection_v2/parquet/default/train/1.parquet>
- <https://huggingface.co/api/datasets/indonlp/cendol_collection_v2/parquet/default/train/2.parquet>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `cendol_000.parquet` | 108.7MB | `cce8d988c147acdf` |
| `cendol_001.parquet` | 88.8MB | `d560d73abe729437` |
| `cendol_002.parquet` | 77.5MB | `da2d8f531f8924cb` |

**Sitasi**: Cahyawijaya et al. (2024), Cendol: Open Instruction-tuned Generative Large Language Models for Indonesian Languages.

**Catatan**: 12,8 juta baris total; diambil 3 shard dari 23.

---

## TED2020 (Indonesia, monolingual)

- **Peran**: Transkrip pidato lisan — bahasa lisan yang tertata
- **Target**: 5 juta token
- **Lisensi**: CC-BY-NC-ND 4.0 (TED)
- **Status**: selesai — 5.1MB di disk
- **Halaman sumber**: <https://opus.nlpl.eu/TED2020/id&id/v1/TED2020>

**Tautan unduhan langsung:**

- <https://object.pouta.csc.fi/OPUS-TED2020/v1/mono/id.txt.gz>

| Berkas | Ukuran | SHA256 (16 char awal) |
|---|---|---|
| `id.txt.gz` | 5.1MB | `52951257124f465c` |

**Sitasi**: Reimers & Gurevych (2020), Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation, EMNLP.

**Catatan**: Kecil (5 MB) tapi kualitas tinggi: lisan tapi terstruktur.

---

## Sumber yang dipertimbangkan tapi tidak dipakai

Dicatat supaya keputusannya bisa ditelusuri dan tidak diteliti ulang.

| Dataset | Alasan |
|---|---|
| [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | gated=auto — butuh akun HuggingFace dan menyetujui syarat |
| [oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | gated=manual — butuh persetujuan pemilik dataset |
| [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | terbuka dan bagus (212 GB, ind_Latn), tapi tumpang tindih dengan FineWeb-2; disimpan sebagai cadangan kalau perlu tambahan |
| [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) | CC-BY-NC-4.0 dan hasil terjemahan mesin — kualitas bahasanya di bawah Aya/Cendol yang beranotasi manusia |
| [allenai/c4 (multilingual id)](https://huggingface.co/datasets/allenai/c4) | mC4 disaring jauh lebih longgar daripada FineWeb-2; banyak boilerplate |

> Dua sumber paling sering direkomendasikan untuk bahasa Indonesia — 
> CulturaX dan OSCAR — ternyata **ter-gate** dan butuh akun serta 
> persetujuan. FineWeb-2 dipakai sebagai gantinya: terbuka, dan 
> penyaringan serta deduplikasinya lebih ketat.
