# Wicara-1B — Rencana Persiapan & Spesifikasi Teknis (Internal)

Dokumen perencanaan dan panduan teknis internal kami untuk meningkatkan skala model Wicara dari baseline 56 juta parameter menuju **1 Miliar parameter (1B)** saat fasilitas komputasi yang lebih besar sudah siap kami gunakan.

---

## 1. Latar Belakang & Rasional Kami

Model baseline kami saat ini (**`wicara-56m`**) berhasil membuktikan bahwa seluruh pipeline yang kami bangun dari nol sudah bekerja solid: arsitektur LLaMA-style kustom, tokenisasi BPE 16k, pretrain 1,1 miliar token, SFT multi-turn, hingga sistem evaluasi perilaku kami (skor perilaku 73%, persona 90%).

Namun, dari evaluasi kami di [notes/HANDOFF.md](HANDOFF.md#L24) dan [notes/08-fase-8-evaluasi.md](08-fase-8-evaluasi.md), kami menemukan batas fisik kapasitas model 56M:
* **Pengetahuan Faktual 0%:** Model kami tidak punya ruang memori parametrik yang cukup untuk menyimpan fakta dunia tanpa merusak kelancaran tata bahasanya.
* **Konteks Masih Sempit (512 token):** Hanya cukup untuk percakapan pendek; belum bisa kami pakai untuk memproses dokumen panjang, rangkuman teks, atau RAG.
* **Penalaran Skenario:** Model kami masih mudah terdistraksi jika pertanyaan dibungkus dalam skenario tidak langsung.

**Target kami untuk `wicara-1b`:**
1. Memberikan kapasitas memori parametrik yang cukup agar model kami mampu menyimpan pengetahuan faktual Indonesia.
2. Memperluas jendela konteks ke standar modern: **8.192 (8k) hingga 32.768 (32k) token**.
3. Menjaga responsivitas dan gaya tutur santun/natural khas Wicara yang sudah kami capai di 56M.

---

## 2. Spesifikasi Arsitektur yang Kami Rencanakan

Kami tetap mempertahankan fondasi arsitektur bersih yang sudah kami bangun di [src/model/](file:///d:/Project/llmgusz/src/model/): RMSNorm (*pre-norm*), RoPE, SwiGLU, GQA (*Grouped-Query Attention*), dan *tied embeddings*.

| Komponen | `wicara-56m` (Baseline Kami) | `wicara-1b` (Target Kami) | Rasional Keputusan Kami |
|---|---|---|---|
| **Total Parameter** | **56,0 Juta** | **~1,05 Miliar** | Kapasitas penalaran dan penyimpanan memori fakta |
| **Non-embedding** | 45,5 Juta | ~1,02 Miliar | Parameter murni komputasi |
| `d_model` | 640 | **2.048** | Lebar representasi standar untuk kelas 1B |
| `n_layer` | 10 | **18** (atau 20) | Kedalaman hierarki ekstraksi fitur |
| `n_head` / `n_kv_head` | 10 / 5 (GQA 2:1) | **16 / 4** (GQA 4:1) | Menghemat VRAM KV-cache kami 4x lipat saat inferensi |
| `head_dim` | 64 | **128** | Kapasitas pencocokan attention per head |
| `d_ffn` | 1.728 | **5.504** | $\approx \frac{8}{3} \times d_{model}$ (kelipatan 128) |
| `vocab_size` | 16.384 | **16.384** (atau 32.768) | Tetap 16k untuk hemat memori, atau 32k untuk kompresi lebih tinggi |
| `max_seq_len` | 512 | **8.192 (8k) s.d. 32.768 (32k)** | Standar modern (dokumen PDF, perundang-undangan, dialog panjang) |
| `rope_theta` | 10.000,0 | **500.000,0** | Kami adopsi teknik Llama-3 agar RoPE kami stabil di jarak jauh |
| *Stability Feature* | Standar RoPE | **+ QK-Norm** | Mencegah ledakan attention logit di model $\ge 1\text{B}$ |

---

## 3. Kebutuhan Data Kami (The Data Hunger)

Menurut hukum penskalaan Chinchilla, rasio komputasi optimal minimal berada di **20 token per parameter**. Sedangkan standar industri modern untuk model kecil (*inference-optimal*) mendorong rasio ini ke **50:1 s.d. 100:1**.

```
   Target Minimal Kami (Chinchilla 20:1)  : 20 Miliar Token
   Target Optimal Kami (50:1)            : 50 Miliar Token
```

Korpus yang kami miliki saat ini (~1,12 miliar token) baru mencakup **2–5%** dari kebutuhan model 1B kami. Jika kami paksakan melatih 1B hanya dengan 1,1B token, model kami akan mengalami *under-fitting* parah. Kami harus mengumpulkan korpus baru hingga mencapai minimal 20B token.

### Resep Komposisi Data yang Kami Targetkan (50B Token)

| Sumber Korpus | Target Token | Porsi | Peran untuk Model Kami |
|---|---|---|---|
| **FineWeb-2 (`ind_Latn`) Full** | 20 Miliar | 40% | Kualitas web teks umum Indonesia terkini |
| **OSCAR ID / CC-100 ID / Madlad** | 10 Miliar | 20% | Diversitas kosakata dan register bahasa daerah/ragam bahasa |
| **Wikipedia ID + Wikibooks + Ensiklopedia** | 3 Miliar | 6% | Sumber fakta padat (memori deklaratif kami) |
| **Buku / Dokumen Formal / Berita** | 5 Miliar | 10% | Struktur kalimat panjang dan tata bahasa baku |
| **OpenSubtitles ID + Percakapan** | 2 Miliar | 4% | Menjaga keluwesan percakapan natural khas Wicara |
| **Kode Pemrograman (The Stack / StarCoder subset)** | 5 Miliar | 10% | **Kritis:** Melatih logika penalaran (*reasoning*) dan keteraturan |
| **Teks Bilingual Berkualitas (EN)** | 5 Miliar | 10% | Pengetahuan sains & matematika global |

---

## 4. Analisis Memori VRAM & Kebutuhan Hardware Kami

### Rincian Kebutuhan Memori Training (1B Parameter)

1. **Bobot Model (BF16):** $1\times 10^9 \times 2\text{ byte} = \mathbf{2,0\text{ GB}}$
2. **Optimizer State (AdamW standar):**
   * Master weights FP32: 4 byte
   * Momentum $m$ FP32: 4 byte
   * Variance $v$ FP32: 4 byte
   * Total: $1\times 10^9 \times 12\text{ byte} = \mathbf{12,0\text{ GB}}$ (atau 16 GB jika gradien di FP32).
3. **Aktivasi (pada seq_len 8.192, batch_size 2 per GPU):**
   * Tanpa Activation Checkpointing: $\approx \mathbf{22\text{–}28\text{ GB}}$ (akan langsung OOM di GPU 24GB).
   * **Dengan Activation Checkpointing (recomputation):** Turun drastis menjadi $\approx \mathbf{4\text{–}6\text{ GB}}$.
4. **KV-Cache saat Inferensi (Berkat GQA 4:1 kami):**
   * Pada **8.192 token (8k):** Hanya butuh **~301 MB** per stream.
   * Pada **32.768 token (32k):** Hanya butuh **~1,2 GB** per stream.

> **Ringkasan Kebutuhan VRAM:**
> * Jika kami pakai **FP32 AdamW + Activation Checkpointing**: Butuh GPU dengan VRAM **$\ge 24\text{ GB}$**.
> * Jika kami pakai **8-bit AdamW (`bitsandbytes`)**: Kebutuhan optimizer turun ke **3 GB**, sehingga total VRAM yang kami butuhkan hanya **~14–18 GB**, sangat aman di GPU 24GB.

---

### Strategi Pelatihan Konteks Panjang Kami: Dua Tahap (*Two-Stage Training*)

Kami tidak akan melatih 100% token langsung di jendela 32k sejak hari pertama karena akan memboroskan komputasi kami (teks umum Indonesia rata-rata hanya 300–1.200 token).

**Strategi yang akan kami terapkan:**
1. **Tahap 1 — Pretraining Utama (90% Komputasi @ 4k / 8k):**
   * Pasang `rope_theta = 500000.0` sejak awal di [src/model/rope.py](file:///d:/Project/llmgusz/src/model/rope.py).
   * Latih model kami pada konteks 4.096 atau 8.192 hingga ~90% total token.
2. **Tahap 2 — Long-Context Annealing (10% Terakhir @ 32k):**
   * Naikkan `max_seq_len` ke **32.768 (32k)** di fase pendinginan (*decaying* LR).
   * Masukkan korpus teks panjang: buku, perundang-undangan, tesis, dan transkrip panjang.
   * Model kami hanya butuh 1–2 miliar token data panjang di tahap ini untuk langsung menguasai penalaran dokumen panjang.

---

### Skenario Fasilitas Hardware yang Realistis untuk Kami

#### Opsi A: 1x GPU Workstation Lokal (RTX 3090 / RTX 4090 24GB)
* **Setup Kami:** 8-bit AdamW + Gradient Checkpointing + FlashAttention.
* **Batch Size per Step:** 1–2 (seq 4096/8192), gradient accumulation 64–128.
* **Estimasi Throughput:** ~12.000–18.000 token/detik.
* **Waktu Latih Kami:** $\approx 14\text{–}18\text{ hari}$ non-stop untuk 20 Miliar token.

#### Opsi B: Sewa Klaster Cloud Mini (4x atau 8x A100/H100 80GB)
* **Platform Pilihan:** RunPod, Vast.ai, atau Lambda Labs.
* **Setup Kami:** PyTorch FSDP / DDP, BF16 penuh.
* **Estimasi Throughput:** ~120.000–250.000 token/detik.
* **Waktu Latih Kami:** **2 s.d. 3 hari** selesai. Estimasi biaya: ~\$100 – \$250 USD.

---

## 5. Modifikasi Kode yang Perlu Kami Kerjakan

Codebase kami sudah sangat rapi, tapi ada 5 hal yang wajib kami tambahkan sebelum memulai run 1B:

### 1. Tambahkan Preset `1b` di [src/model/config.py](file:///d:/Project/llmgusz/src/model/config.py)
```python
"1b": ModelConfig(
    name="wicara-1b",
    vocab_size=16384,     # atau 32768
    d_model=2048,
    n_layer=18,
    n_head=16,
    n_kv_head=4,         # GQA 4:1
    d_ffn=5504,
    max_seq_len=8192,    # Tahap 1: 8.192 -> Tahap 2: 32.768
    rope_theta=500000.0, # Standar Llama-3 untuk ekstrapolasi panjang
    norm_eps=1e-5,
    tie_embeddings=True,
)
```

### 2. Pasang QK-Norm di [src/model/attention.py](file:///d:/Project/llmgusz/src/model/attention.py)
Mencegah ledakan nilai logit attention sebelum softmax pada dimensi besar:
```python
self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
```

### 3. Pasang Activation Checkpointing di [src/model/transformer.py](file:///d:/Project/llmgusz/src/model/transformer.py)
Menghemat memori aktivasi hingga 75% saat training di sekuens 8k:
```python
from torch.utils.checkpoint import checkpoint

if self.gradient_checkpointing and self.training:
    x = checkpoint(block, x, cos, sin, use_reentrant=False)
else:
    x = block(x, cos, sin)
```

### 4. Upgrade Training Loop ke Multi-GPU di [src/train/trainer.py](file:///d:/Project/llmgusz/src/train/trainer.py)
Siapkan integrasi PyTorch DDP atau Hugging Face `accelerate` agar script kami langsung siap berjalan di multi-GPU tanpa rombak total:
```bash
accelerate launch scripts/train.py --model 1b ...
```

### 5. Dukungan Dataset Memmap Berkas Terpisah (*Sharding*) di [src/train/data.py](file:///d:/Project/llmgusz/src/train/data.py)
Untuk 20B token, file `.bin` kami akan berukuran 40 GB. Kami perlu memperbarui pembaca dataset agar mendukung berkas pecahan (`train_01.bin`, `train_02.bin`, dst.) supaya I/O tetap ringan dan stabil.

---

## 6. Roadmap Langkah Kami

Kami memutuskan untuk tidak melompat langsung dari 56M ke 1B dalam satu kali jalan agar risiko dan anggaran komputasi tetap terkendali:

```
┌────────────────────────────────────────────────────────┐
│  wicara-56m (Baseline Kami — Selesai)                  │
│  • 1,1B token, 512 seq_len, 1x RTX 4050 6GB            │
│  • Validasi: arsitektur & pipeline utuh terbukti jalan │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  wicara-350m (Batu Loncatan Validasi Kami)             │
│  • d_model=1024, layer=14, seq_len=1024                │
│  • ~5 Miliar token                                     │
│  • Uji: Apakah masalah pengetahuan faktual teratasi?   │
│  • Biaya komputasi: ~1-2 hari di 1x GPU 24GB           │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  wicara-1b (Target Skala Penuh Kami)                   │
│  • d_model=2048, layer=18, seq_len=8192 (s.d. 32k)     │
│  • 20 - 50 Miliar token                                │
│  • Multi-GPU / 8-bit AdamW / FSDP                      │
│  • Target: Model bahasa Indonesia mandiri & cerdas     │
└────────────────────────────────────────────────────────┘
```

---

## 7. Kesimpulan Kesiapan Kami

Kondisi proyek Wicara kami saat ini berada di posisi yang sangat menguntungkan:
1. **Arsitektur kami bersih:** Kami membangunnya di atas pure PyTorch tanpa beban dependensi framework eksternal yang rumit.
2. **Sistem evaluasi kami sudah siap:** Script evaluasi perilaku di [scripts/eval_perilaku.py](file:///d:/Project/llmgusz/scripts/eval_perilaku.py) bisa langsung kami pakai untuk menguji model 1B setelah pretrain dan SFT selesai.
3. **Langkah kami selanjutnya:** Menyiapkan pengumpulan korpus 20B+ token dan menunggu kesiapan fasilitas komputasi yang memadai.
