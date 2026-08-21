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

---

## Status

| Fase | Isi | Status |
|---|---|---|
| 0 | Environment & benchmark GPU | ✅ Selesai |
| 1 | Pipeline data (unduh, bersih, dedup) | ✅ Selesai |
| 2 | Tokenizer BPE 16k + packing | 🟨 Special token siap, BPE belum dilatih |
| 3 | Arsitektur model | ✅ Selesai |
| 3b | Uji sanity | ✅ Selesai |
| 4 | Training loop | ✅ Selesai |
| 5 | Pretrain | ⬜ Menunggu Fase 2 |
| 6 | SFT percakapan | ⬜ |
| 7 | Inference engine & CLI | ⬜ |
| 8 | Evaluasi | ⬜ |

**105 uji lolos.** Model, trainer, checkpoint, dan korpus sudah siap; tinggal
tokenizer sebelum pretrain bisa dijalankan.

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
| Vocab | 16.384 (BPE custom Indonesia) |
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

**1,29 miliar token** dari 3.985.535 dokumen. Seluruhnya teks nyata dari
repositori publik — **tidak ada data sintetis**.

| Sumber | Peran | Token | Porsi | Lolos filter |
|---|---|---|---|---|
| [OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles) | Dialog percakapan | 520M | 40,3% | 93,9% |
| [FineWeb-2 `ind_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Ragam tulis informal | 260M | 20,1% | 99,4% |
| [Wikipedia Indonesia](https://huggingface.co/datasets/wikimedia/wikipedia) | Faktual, kalimat utuh | 195M | 15,1% | 83,1% |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Tanya-jawab | 156M | 12,1% | 79,7% |
| [Cendol v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | Instruksi Indonesia | 156M | 12,1% | 73,3% |
| [TED2020](https://opus.nlpl.eu/TED2020/id&id/v1/TED2020) | Lisan tertata | 4M | 0,3% | 98,7% |

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

# 5. Uji
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
  train/       trainer · data (memmap) · lr_schedule · checkpoint · synthetic
  tokenizer/   chat_template
scripts/       check_env · download_corpus · verify_corpus · clean_corpus ·
               demo_cleaning · bench_model · train
tests/         105 uji
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

## Lisensi & atribusi

Kode: bebas dipakai untuk belajar.

Korpus **tidak** disertakan di repo ini. Tiap sumber punya lisensinya sendiri
(CC-BY-SA, ODC-BY, Apache-2.0, dsb.) — lihat
[`data/raw/SOURCES.md`](data/raw/SOURCES.md). Jalankan
`scripts/download_corpus.py` untuk membangunnya ulang dari sumber aslinya.
