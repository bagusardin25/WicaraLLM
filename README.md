**🇮🇩 Bahasa Indonesia** | [🇬🇧 English](README_en.md)

<img src="assets/wicarallm-logomark.png" alt="Logo Wicara" width="150">

# Wicara

**W**eight-efficient **I**ndonesian **C**onversational **A**rchitecture, **R**esearch **A**rtifact

Model bahasa Indonesia 56 juta parameter. Dilatih dari nol di satu laptop
RTX 4050 6 GB.

*Wicara* (Sanskerta *vicāra*): tutur, bicara. Itu yang dikerjakannya.

Data, tokenizer, arsitektur, training, inference — kami tulis sendiri di
PyTorch. Tidak ada model jadi, tidak ada API.

![tests](https://github.com/bagusardin25/WicaraLLM/actions/workflows/tests.yml/badge.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/num1notsvn/wicara-56m-chat)
[![GGUF Quantized](https://img.shields.io/badge/%F0%9F%A4%97%20GGUF-Quantized-green)](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF)
[![Ollama](https://img.shields.io/badge/Ollama-hf.co%2Fnum1notsvn%2Fwicara--56m--chat--GGUF-black)](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-orange)](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

```bash
# Opsi 1: Jalankan langsung via Ollama (hanya ~70 MB, tanpa clone/setup)
ollama run hf.co/num1notsvn/wicara-56m-chat-GGUF:Q8_0

# Opsi 2: Jalankan lokal dengan PyTorch
.venv\Scripts\python.exe scripts\chat.py
```

| Anda | Wicara |
|---|---|
| Halo! | Halo, Tn. |
| Apa kabar? | Baik, baik. |
| Siapa namamu? | Aku tidak punya nama. |
| Terima kasih banyak! | Tidak masalah, tidak masalah. |
| Sampai jumpa! | Sampai jumpa! |

`wicara-56m-chat` menjawab satu giliran lalu berhenti.
`wicara-56m-base` melanjutkan teks — itu hasil pretrain, sebelum SFT.

Pretrain selesai. SFT selesai. Evaluasi perilaku selesai (Fase 8). **217 uji.**

---

## Model

```
wicara-{ukuran}-{tipe}
```

`56m` / `350m` / `1b` — jumlah parameter. `base` pretrain, `chat` sudah SFT.
Rilis pertama: **`wicara-56m-base`**, lalu **`wicara-56m-chat`**. Generasi
berikutnya baru dapat nomor (`wicara-2-…`) kalau arsitekturnya berubah
mendasar.

Arsitektur gaya Llama: pre-norm RMSNorm, RoPE, GQA, SwiGLU, tanpa bias,
*tied embeddings*.

| | `wicara-56m` |
|---|---|
| Parameter | **56,0 juta** (45,5 juta non-embedding) |
| `d_model` / `n_layer` | 640 / 10 |
| Attention | 10 query / 5 KV (GQA 2:1) |
| `head_dim` / `d_ffn` | 64 / 1728 |
| Vocab | 16.384 BPE byte-level (4,26 karakter/token) |
| Konteks | 512 token |
| Presisi | bf16 autocast |

56 juta karena itu yang muat di anggaran ~11 jam GPU ini. 1,1 miliar token
menempatkannya di titik Chinchilla (~20 token/parameter). Config `7m` dan
`19m` hanya untuk iterasi cepat.

### Pretrain — `wicara-56m-base`

| | Nilai |
|---|---|
| Val loss | **3,0505** (perplexity 21,2) |
| Train loss | 2,937 — jarak ke val 0,12 |
| Langkah | 17.010 (1 epoch, cakupan 100%) |
| Throughput | 28.017 token/detik |
| VRAM | 3,12 GB dari 6,00 GB |
| Durasi | 11,1 jam |

Pada langkah 17.000, prompt `"Halo, apa kabar?"` tanpa SFT sama sekali:

```
Halo, ini aku, ada yang bisa kubantu?
Aku segera ke sana.
Halo, ini aku Anna.
```

Itu diserap dari percakapan telepon di subtitle. Sapaan memicu *aku/kau*,
pertanyaan formal memicu *Saya/Anda*. Fakta tidak bisa diandalkan:
`"Indonesia adalah negara"` dilanjutkan `"...yang paling maju di dunia."`

Rincian: [`notes/05-fase-5-pretrain.md`](notes/05-fase-5-pretrain.md).

### SFT — `wicara-56m-chat`

3.619 langkah, ~20 menit, val loss **1,37**. Loss SFT tidak sebanding dengan
pretrain: yang dihitung hanya token jawaban asisten.

39.400 contoh, ditambang dari Aya, Cendol, dan subtitle. Ditambah 591 contoh
tulis tangan untuk mengunci persona ("Wicara") dan membatasi halusinasi.

Rincian: [`notes/06-fase-6-sft.md`](notes/06-fase-6-sft.md) dan [`notes/06b-fase-6b-tulis-tangan.md`](notes/06b-fase-6b-tulis-tangan.md).

---

## Korpus

**1,12 miliar token** dari 3.985.535 dokumen. Teks nyata, tidak ada sintetis.

| Sumber | Peran | Token | Porsi | Lolos filter | char/token |
|---|---|---|---|---|---|
| [OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles) | Dialog | 484M | 43,2% | 93,9% | 3,98 |
| [FineWeb-2 `ind_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Tulis informal | 210M | 18,7% | 99,4% | 4,58 |
| [Wikipedia Indonesia](https://huggingface.co/datasets/wikimedia/wikipedia) | Kalimat utuh | 174M | 15,5% | 83,1% | 4,16 |
| [Cendol v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | Instruksi | 129M | 11,5% | 73,3% | 4,46 |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Tanya-jawab | 121M | 10,8% | 79,7% | 4,78 |
| [TED2020](https://opus.nlpl.eu/TED2020/id&id/v1/TED2020) | Lisan tertata | 3M | 0,3% | 98,7% | 4,92 |

`train.bin` 1.115.298.306 token, `val.bin` 5.683.512. Split di tingkat
dokumen — mengiris larik akan membelah dokumen ke latih dan validasi.

Subtitle porsi terbesar karena fokusnya percakapan, tapi tidak dibiarkan
mendominasi: model yang hanya makan film bicara terpotong-potong. Wikipedia
dan FineWeb menahan itu. CulturaX dan OSCAR ter-*gate*; FineWeb-2 yang
terpakai.

Tokenizer BPE 16.384 dilatih dari korpus bersih. 4,26 karakter/token —
tokenizer Inggris ~2,2 pada teks Indonesia. `" halo"`, `" hai"`, `" iya"`,
`" tidak"` masing-masing satu token. 32 special token di ID 0–31 (8 aktif,
24 cadangan) supaya eksperimen berikutnya tidak perlu resize embedding.

Unduhan, lisensi, sitasi, SHA256: [`data/raw/SOURCES.md`](data/raw/SOURCES.md).
Statistik saringan: `data/clean/stats.json`.

---

## Isi repo

```
src/
  model/       rmsnorm · rope · attention (GQA+KV-cache) · ffn · transformer
  data/        sources · readers · clean · dedup · sft · sft_dataset
  train/       trainer · data (memmap) · lr_schedule · checkpoint
  infer/       generate · pencegat
  tokenizer/   bpe · chat_template
scripts/       check_env · download_corpus · verify_corpus · clean_corpus ·
               train_tokenizer · pack_tokens · train · train_sft ·
               build_sft_data · eval_perilaku · banding_eval ·
               export_to_hf · upload_to_hf · deploy_space · generate · chat
tests/         217 uji
notes/         catatan per fase (01 s.d. 08) + HANDOFF.md
space/         aplikasi web demo Gradio untuk Hugging Face Spaces
```

---

## Menjalankan

### Demo daring (tanpa GPU lokal)
- **Hugging Face Space:** [num1notsvn/wicara-56m-chat-demo](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
- **Google Colab:** [Wicara_56M_Chat_Demo.ipynb](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

### Di komputer lokal
Python **3.11** dari python.org (bukan MSYS2). GPU NVIDIA. PowerShell, bukan
Git Bash.

Ngobrol, kalau checkpoint chat sudah ada:

```powershell
.venv\Scripts\python.exe scripts\chat.py
.venv\Scripts\python.exe scripts\chat.py --uji
```

Dari nol:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
.venv\Scripts\python.exe scripts\download_corpus.py
.venv\Scripts\python.exe scripts\verify_corpus.py
.venv\Scripts\python.exe scripts\clean_corpus.py
.venv\Scripts\python.exe scripts\train_tokenizer.py
.venv\Scripts\python.exe scripts\pack_tokens.py
.venv\Scripts\python.exe -m pytest tests\ -q
```

Yang lain:

```powershell
.venv\Scripts\python.exe src\model\config.py
.venv\Scripts\python.exe scripts\bench_model.py --model 56m
.venv\Scripts\python.exe scripts\train.py --model 56m
.venv\Scripts\python.exe scripts\train_sft.py
```

---

## Yang kami tabrak di 6 GB

Lengkapnya di [`notes/`](notes/). Yang sering menjebak:

- **Logits memakan VRAM hampir sebanyak seluruh Transformer.** Estimasi awal
  meleset 2× karena mengabaikannya (768 MB vs 880 MB pada batch 8).
- **Windows tidak melempar OOM.** Batch 32 mengalokasikan 7,1 GB di GPU 6,0 GB
  — spill ke RAM, throughput anjlok 4×. Batas aman ~5,5 GB.
- **FlashAttention tidak ada di wheel PyTorch Windows.** `mem_efficient` cukup.
- **memmap mengunci berkas di Windows.** `TokenDataset.close()` wajib sebelum
  menimpa `.bin`. Tidak terjadi di Linux.
- **Dedup per baris subtitle membuang 39,7%** — termasuk `terima kasih` 845×
  dan `halo` 458×, yang memang Zipfian. Dedup per blok hanya 3,2%.
- **Loss di bawah ln(vocab) di langkah awal = kebocoran label.** Angkanya
  terlihat bagus, jadi tidak ketahuan dari kurva.

---

## Batas

`wicara-56m` lancar berbahasa. Ia tidak menalar, tidak berhitung, dan tidak
bisa diandalkan soal fakta. Halusinasi-nya terdengar yakin. Itu kapasitas
56 juta parameter, bukan bug. Jarak ke model 7B kira-kira 125× parameter.

---

## Lisensi

Tiga benda, tiga aturan.

**Kode** (`src/`, `scripts/`, `tests/`) — [Apache-2.0](LICENSE). Termasuk
hibah paten.

**Korpus** tidak ikut di repo. Yang ada hanya resep unduhan. Tiap sumber
punya lisensi sendiri; daftarnya di [`data/raw/SOURCES.md`](data/raw/SOURCES.md).

| Sumber | Lisensi |
|---|---|
| Wikipedia Indonesia | CC-BY-SA-3.0 |
| FineWeb-2 | ODC-BY-1.0 |
| Aya Collection | Apache-2.0 |
| Cendol v2 | Apache-2.0 |
| TED2020 | CC-BY-NC-ND-4.0 (non-komersial) |
| OpenSubtitles | lihat opus.nlpl.eu |

**Bobot model:**
- **Model Chat (PyTorch / Safetensors):** [`num1notsvn/wicara-56m-chat`](https://huggingface.co/num1notsvn/wicara-56m-chat)
- **Format GGUF (Ollama & LM Studio):** [`num1notsvn/wicara-56m-chat-GGUF`](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF) (versi `Q8_0` ~70 MB & `F16` ~133 MB)
- **Demo Web Interaktif:** [Hugging Face Space](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
- **Notebook Google Colab:** [Open In Colab](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

Bobot base (`wicara-56m-base`) belum kami rilis publik. Kalau nanti dipublikasikan:
OpenSubtitles ~43% korpus, status hukum bobot sebagai karya turunan belum
tuntas di mana pun; TED2020 0,3% dan non-komersial — untuk pemakaian
komersial sumber itu harus dikeluarkan di `KOMPOSISI` pada
`scripts/clean_corpus.py`, lalu dilatih ulang.

Untuk riset dan belajar, keduanya tidak menjadi masalah.

## Sitasi

```bibtex
@misc{ardin2026wicara,
  title={Wicara: A Weight-efficient Indonesian Conversational Architecture Built from Scratch},
  author={Bagus Ardin Prayoga},
  year={2026},
  url={https://github.com/bagusardin25/WicaraLLM}
}
```

Kode cukup atribusi Apache-2.0. Korpus disitasi ke sumber aslinya —
daftar lengkap di [`data/raw/SOURCES.md`](data/raw/SOURCES.md).

hi
