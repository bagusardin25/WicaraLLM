**🇮🇩 Bahasa Indonesia** | [🇬🇧 English](README_en.md)

# Wicara

**W**eight-efficient **I**ndonesian **C**onversational **A**rchitecture, **R**esearch **A**rtifact

Model bahasa Indonesia yang dibangun dari nol — data, arsitektur, training,
sampai inference — di **satu laptop dengan RTX 4050 6 GB**.

*Wicara* (dari Sanskerta *vicāra*) berarti tutur atau bicara.

Tujuannya belajar cara kerja LLM secara utuh dengan membangunnya sendiri, bukan
memakai model jadi. Target fungsionalnya sengaja sederhana: merespons
percakapan dasar bahasa Indonesia ("halo", "apa kabar?").

> *Building an Indonesian language model from scratch on a single 6 GB laptop
> GPU. Every component — RMSNorm, RoPE, GQA, SwiGLU, KV-cache, training loop —
> written by hand in PyTorch. Documentation and code comments are in Indonesian.*

![tests](https://github.com/bagusardin25/WicaraLLM/actions/workflows/tests.yml/badge.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

---

## Status

| Fase | Isi | Status |
|---|---|---|
| 0 | Environment & benchmark GPU | ✅ Selesai |
| 1 | Pipeline data (unduh, bersih, dedup) | ✅ Selesai |
| 2 | Tokenizer BPE 16k + packing | ✅ Selesai |
| 3 | Arsitektur model | ✅ Selesai |
| 3b | Uji sanity | ✅ Selesai |
| 4 | Training loop | ✅ Selesai |
| 5 | Pretrain | ⬜ Siap dijalankan — lihat [panduan](notes/04-panduan-pretrain.md) |
| 6 | SFT percakapan | ⬜ |
| 7 | Inference engine & CLI | ⬜ |
| 8 | Evaluasi | ⬜ |

**149 uji lolos.** Model, trainer, checkpoint, korpus, dan tokenizer sudah
siap. `train.bin` berisi **1.115.298.306 token** — pretrain tinggal dijalankan.

---

## Penamaan

```
wicara-{ukuran}-{tipe}
```

| Bagian | Aturan | Contoh |
|---|---|---|
| `ukuran` | jumlah parameter; `m` di bawah 1 miliar, `b` di atasnya | `56m`, `350m`, `1b` |
| `tipe` | tahap pelatihan | `base`, `chat` |

| Tipe | Arti |
|---|---|
| `base` | Hasil pretrain saja — lancar berbahasa, belum bisa diajak tanya-jawab |
| `chat` | Sudah melewati SFT dengan chat template |

Rilis pertama: **`wicara-56m-base`** lalu **`wicara-56m-chat`**.

Kalau nanti arsitekturnya berubah mendasar, nomor generasi disisipkan setelah
nama — `wicara-2-350m-base` — mengikuti pola Llama. Selama masih generasi
pertama, nomornya dihilangkan.

Di dalam kode, config dipilih lewat kunci berbasis ukuran: `7m`, `19m`, `32m`,
`56m`, `88m`.

---

## Spesifikasi

Arsitektur gaya Llama: pre-norm RMSNorm, RoPE, GQA, SwiGLU, tanpa bias,
*tied embeddings*.

| | `wicara-56m` |
|---|---|
| Parameter | **56,0 juta** (45,5 juta non-embedding) |
| `d_model` / `n_layer` | 640 / 10 |
| Attention head | 10 query / 5 key-value (GQA 2:1) |
| `head_dim` / `d_ffn` | 64 / 1728 |
| Vocab | 16.384 (BPE byte-level custom, 4,26 karakter/token) |
| Panjang konteks | 512 token |
| Presisi | bf16 autocast |
| Learning rate | 8,0e-4 (diskalakan 1/`d_model`, muP) |

### Kenapa 56M

Ukuran model optimal ditentukan **anggaran waktu**, bukan kapasitas VRAM.
Semua angka di bawah diukur langsung di RTX 4050 Laptop, bukan diperkirakan:

| Config | Parameter | tok/detik | VRAM | MFU | 1,3B token |
|---|---|---|---|---|---|
| `32m` | 32,0M | 40.615 | 2,19 GB | 22,8% | 8,9 jam |
| **`56m`** | **56,0M** | **20.733** | **2,96 GB** | 22,5% | **17,4 jam** |
| `88m` | 88,1M | 14.276 | 4,10 GB | 25,7% | 25,3 jam |

Untuk anggaran compute ~3,6×10¹⁷ FLOPs (≈17 jam di GPU ini), ukuran
Chinchilla-optimal adalah √(C/120) ≈ **55 juta parameter**. Pada anggaran waktu
yang sama, 32M jadi kelewat latih (79 token/parameter) dan 88M jadi kurang
latih (10 token/parameter). VRAM baru menjadi pengikat di sekitar 100M.

Config `7m` dan `19m` dipakai untuk iterasi cepat dan uji, bukan untuk rilis.

---

## Korpus

**1,12 miliar token** dari 3.985.535 dokumen. Seluruhnya teks nyata dari
repositori publik — **tidak ada data sintetis**.

| Sumber | Peran | Token | Porsi | Lolos filter | char/token |
|---|---|---|---|---|---|
| [OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles) | Dialog percakapan | 484M | 43,2% | 93,9% | 3,98 |
| [FineWeb-2 `ind_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Ragam tulis informal | 210M | 18,7% | 99,4% | 4,58 |
| [Wikipedia Indonesia](https://huggingface.co/datasets/wikimedia/wikipedia) | Faktual, kalimat utuh | 174M | 15,5% | 83,1% | 4,16 |
| [Cendol v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | Instruksi Indonesia | 129M | 11,5% | 73,3% | 4,46 |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Tanya-jawab | 121M | 10,8% | 79,7% | 4,78 |
| [TED2020](https://opus.nlpl.eu/TED2020/id&id/v1/TED2020) | Lisan tertata | 3M | 0,3% | 98,7% | 4,92 |

Hasil akhir: `train.bin` **1.115.298.306 token**, `val.bin` 5.683.512 token.
Validasi dipisahkan di tingkat **dokumen**, bukan dengan mengiris larik akhir —
mengiris akan membelah dokumen antara latih dan validasi, dan itu kebocoran.

**19,9 token per parameter** — nyaris tepat di titik Chinchilla-optimal (20)
untuk model 56M, tanpa direncanakan.

### Tokenizer

BPE byte-level, vocab 16.384, dilatih dari korpus **bersih** (bukan mentah,
supaya tidak ada slot vocab terbuang untuk sampah yang sudah disaring).

Rasio kompresi **4,26 karakter/token** — di atas target plan 3,5-4,0.
Tokenizer berbasis Inggris hanya mencapai ~2,2 untuk teks Indonesia. Sapaan
yang jadi target proyek ini masing-masing satu token: `" halo"`, `" hai"`,
`" iya"`, `" tidak"`.

32 special token menempati ID 0-31 (8 aktif + 24 cadangan), diverifikasi
otomatis setelah training. Slot cadangan memungkinkan menambah token baru
nanti tanpa melatih ulang tokenizer dan tanpa me-resize embedding.

Tautan unduhan langsung, lisensi, sitasi, dan SHA256 tiap berkas ada di
[`data/raw/SOURCES.md`](data/raw/SOURCES.md). Statistik pembersihan lengkap —
termasuk rincian per alasan penolakan — di `data/clean/stats.json`.

Subtitle diberi porsi terbesar sesuai fokus percakapan, tapi sengaja **tidak**
mendominasi: model yang hanya makan subtitle akan bicara terpotong-potong
seperti dialog film. Wikipedia dan FineWeb menyeimbangkannya dengan kalimat
utuh.

CulturaX dan OSCAR — dua sumber yang paling sering direkomendasikan untuk
bahasa Indonesia — ternyata **ter-*gate*** dan butuh akun. FineWeb-2 dipakai
sebagai gantinya: terbuka, dan penyaringan serta deduplikasinya lebih ketat.

---

## Mulai

Butuh Python **3.11** dari python.org (bukan MSYS2) dan GPU NVIDIA.

```bash
# 1. Environment
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Verifikasi GPU + ukur TFLOPS efektif mesin sendiri
.venv\Scripts\python.exe scripts\check_env.py

# 3. Unduh korpus (~3,6 GB) lalu verifikasi integritasnya
.venv\Scripts\python.exe scripts\download_corpus.py
.venv\Scripts\python.exe scripts\verify_corpus.py

# 4. Bersihkan korpus (~40 menit, CPU saja)
.venv\Scripts\python.exe scripts\clean_corpus.py

# 5. Latih tokenizer BPE 16k lalu pack ke .bin (~8 menit)
.venv\Scripts\python.exe scripts	rain_tokenizer.py
.venv\Scripts\python.exe scripts\pack_tokens.py

# 6. Uji
.venv\Scripts\python.exe -m pytest tests\ -q
```

Perintah lain yang berguna:

```bash
.venv\Scripts\python.exe src\model\config.py                 # parameter & VRAM tiap config
.venv\Scripts\python.exe scripts\bench_model.py --model 56m  # ukur throughput
.venv\Scripts\python.exe scripts\demo_cleaning.py            # peragaan pembersihan
```

---

## Struktur

```
src/
  model/       rmsnorm · rope · attention (GQA+KV-cache) · ffn · transformer
  data/        sources · readers · clean · dedup
  train/       trainer · data (memmap, epoch penuh) · lr_schedule · checkpoint
  infer/       generate (KV-cache, top-k/top-p, repetition penalty)
  tokenizer/   bpe · chat_template
scripts/       check_env · download_corpus · verify_corpus · clean_corpus ·
               demo_cleaning · train_tokenizer · pack_tokens · bench_model ·
               train · generate
tests/         149 uji
notes/         catatan per fase + glosarium istilah LLM (PDF)
```

---

## Catatan teknis yang mungkin berguna

Beberapa temuan dari membangun ini di GPU 6 GB, lengkap di [`notes/`](notes/):

**VRAM logits sangat besar dan mudah terlupakan.** Di model kecil ber-vocab
besar, tensor logits + cross-entropy bisa memakan hampir sebanyak seluruh
lapisan Transformer (768 MB vs 880 MB pada batch 8). Estimasi awal meleset 2×
karena mengabaikannya.

**Spill VRAM di Windows tidak memunculkan error.** Di batch 32, PyTorch
mengalokasikan 7,1 GB pada GPU 6,0 GB tanpa `OutOfMemoryError` — Windows
diam-diam meluapkannya ke RAM sistem, dan throughput anjlok 4×. Batas aman
praktis ~5,5 GB, bukan 6,0 GB.

**FlashAttention tidak tersedia di wheel PyTorch Windows.** Bukan masalah:
kernel `mem_efficient` juga tidak membentuk matriks seq×seq di memori.

**memmap mengunci berkas di Windows.** Berkas yang sedang di-*map* tidak bisa
ditimpa, sehingga pipeline data gagal selama masih ada `TokenDataset` hidup.
Tidak terjadi di Linux, jadi mudah lolos.

**Dedup di level yang salah lebih merusak daripada tidak dedup.** Dedup per
baris pada subtitle membuang 39,7% data — tapi yang terbuang adalah frasa
percakapan paling umum (`terima kasih` 845×, `halo` 458×). Bahasa nyata itu
Zipfian; frasa itu memang *seharusnya* sering muncul. Dedup per blok hanya
membuang 3,2%, dan itulah duplikasi sungguhan.

**`loss awal = ln(vocab)` adalah alat uji, bukan sekadar info.** Loss di bawah
angka itu berarti ada kebocoran label — dan kebocoran membuat angka terlihat
*bagus*, sehingga tidak akan ketahuan dari kurva loss.

---

## Batas kemampuan

Perlu dinyatakan terus terang, karena nama proyek tidak menjanjikannya:
`wicara-56m` **tidak menalar**. Model sebesar ini bisa menghasilkan bahasa
Indonesia yang gramatikal, merespons sapaan dengan nada yang tepat, dan
menjaga koherensi satu sampai tiga kalimat. Ia **tidak** akan akurat secara
faktual, tidak bisa berhitung, tidak bisa menalar bertahap, dan akan
berhalusinasi dengan penuh percaya diri.

Itu bukan bug — memang begitu kapasitas 56 juta parameter. Jarak ke model 7B
adalah sekitar 125× parameter dan 1000× compute.

---

## Lisensi

Di proyek LLM ada **tiga hal berbeda** yang lisensinya tidak sama, dan
mencampuradukkannya adalah kesalahan yang umum:

### 1. Kode — Apache-2.0

Seluruh isi `src/`, `scripts/`, dan `tests/` berlisensi
[Apache-2.0](LICENSE). Bebas dipakai, dimodifikasi, dan didistribusikan
termasuk untuk keperluan komersial, dengan atribusi. Apache dipilih (bukan
MIT) karena menyertakan hibah paten eksplisit.

### 2. Korpus — tidak didistribusikan ulang

Repo ini **tidak** memuat satu byte pun teks korpus. Yang disertakan hanya
resep untuk membangunnya: `scripts/download_corpus.py` mengunduh langsung
dari sumber aslinya, dan [`data/raw/SOURCES.md`](data/raw/SOURCES.md) mencatat
tautan, lisensi, sitasi, serta SHA256 tiap berkas.

Lisensi tiap sumber berbeda-beda dan harus dipatuhi masing-masing:

| Sumber | Lisensi | Catatan |
|---|---|---|
| Wikipedia Indonesia | CC-BY-SA-3.0 | *share-alike* |
| FineWeb-2 | ODC-BY-1.0 | atribusi |
| Aya Collection | Apache-2.0 | bebas |
| Cendol v2 | Apache-2.0 | bebas |
| TED2020 | CC-BY-NC-ND-4.0 | **non-komersial** |
| OpenSubtitles | lihat opus.nlpl.eu | status paling tidak jelas |

### 3. Bobot model — belum ada, dan perlu dipikirkan sebelum dirilis

Belum ada bobot yang dirilis. Saat nanti `wicara-56m-base` dipublikasikan,
dua hal berikut perlu dinyatakan terus terang di *model card*-nya:

**OpenSubtitles menyumbang 40% korpus.** Isinya dialog film yang diunggah
pengguna ke opensubtitles.org. OPUS mendistribusikannya untuk keperluan riset,
dan melatih model di atasnya adalah praktik yang lazim di penelitian. Tapi
status hukum *bobot model* sebagai karya turunan belum tuntas di mana pun, dan
repo ini tidak berpura-pura tahu jawabannya.

**TED2020 berlisensi non-komersial** (CC-BY-NC-ND). Porsinya cuma 0,3%, tapi
kalau bobot model hendak dipakai komersial, sumber ini sebaiknya dikeluarkan
dan model dilatih ulang tanpanya — komposisi korpus bisa diatur di
`KOMPOSISI` pada `scripts/clean_corpus.py`.

Untuk penggunaan sebagai bahan belajar dan riset — tujuan proyek ini — kedua
hal di atas tidak menjadi masalah. Keduanya baru penting kalau arahnya berubah
ke komersial.

## Sitasi

Kalau memakai kode atau model ini untuk riset, silakan sitasi:

```bibtex
@misc{ardin2026wicara,
  title={Wicara: A Weight-efficient Indonesian Conversational Architecture Built from Scratch},
  author={Bagus Ardin Prayoga},
  year={2026},
  url={https://github.com/bagusardin25/WicaraLLM}
}
```

## Atribusi

Kalau memakai kode ini, cukup sertakan atribusi sesuai Apache-2.0. Korpus
harus disitasi ke sumber aslinya masing-masing — daftar sitasi lengkapnya
ada di [`data/raw/SOURCES.md`](data/raw/SOURCES.md).
