# Fase 3 — Arsitektur Model (SELESAI)

Tanggal: 2026-08-21

## Yang dibangun

| File | Isi |
|---|---|
| `src/model/rmsnorm.py` | RMSNorm, hitung internal fp32 |
| `src/model/rope.py` | Cache cos/sin, `rotate_half`, `apply_rope` dengan offset |
| `src/model/attention.py` | GQA + KV-cache pre-alokasi + 3 jalur masking |
| `src/model/ffn.py` | SwiGLU |
| `src/model/transformer.py` | `Block` (pre-norm) + `MiniLLM` |
| `tests/test_primitives.py` | 9 uji RMSNorm & RoPE |
| `tests/test_model.py` | 14 uji sanity model |
| `scripts/bench_model.py` | Verifikasi GPU + kalibrasi throughput |

**23/23 uji lolos.**

## Hasil verifikasi model nyata (config 32m, di GPU)

| Item | Hasil |
|---|---|
| Parameter nyata | **31.990.272** — sama persis dengan prediksi `config.py` |
| Non-embedding | 23.601.664 |
| Loss saat init | **9,80** (target `ln(16384)` = 9,70) ✅ |
| Batch disarankan | **8** |
| Throughput | **40.615 token/detik** |
| VRAM puncak | 2,19 GB dari 6,00 GB |
| MFU | 22,8% (dari puncak 25,2 TFLOPS) |
| **Perkiraan pretrain 1,3B token** | **8,9 jam** |

Sesuai perkiraan plan (8–11 jam).

## Tabel batch size terukur

| Batch | tok/detik | VRAM puncak | Catatan |
|---|---|---|---|
| **8** | **40.615** | 2,19 GB | ← pilihan terbaik |
| 16 | 39.833 | 3,81 GB | throughput sama, VRAM 1,7x |
| 24 | 33.996 | 5,48 GB | mulai melambat |
| 32 | 8.027 | 7,10 GB | **spill ke RAM** |

**Batch 8 dipilih**, bukan 16. Throughput keduanya praktis sama (40,6k vs 39,8k)
tapi batch 8 hanya makan separuh VRAM. Sisa ruang itu berharga: bisa dipakai
menaikkan `seq_len` nanti, dan memberi bantalan kalau ada aplikasi lain
memakai GPU. Batch efektif tetap dinaikkan lewat gradient accumulation di
Fase 4 — hasil matematisnya identik.

## Temuan 1: jebakan spill VRAM di Windows

Di batch 32, PyTorch mengalokasikan **7,10 GB pada GPU 6,00 GB** tanpa melempar
error apa pun. Windows/WDDM diam-diam meluapkan kelebihannya ke RAM sistem.

Akibatnya throughput anjlok dari 34.000 ke 8.000 token/detik — **4x lebih
lambat, tanpa satu pun pesan error**. Ini jauh lebih berbahaya daripada OOM:
OOM langsung terlihat, sedangkan spill hanya membuat training terasa "lambat"
dan mudah disalahartikan sebagai hal normal.

Efeknya juga bertahan: setelah satu run yang spill, pengukuran berikutnya
ikut melambat sampai sistem pulih. Sempat membuat satu ronde benchmark
memberi angka 4x lebih rendah di seluruh baris.

`bench_model.py` sekarang mendeteksi peak > 92% VRAM fisik, menandainya, dan
berhenti di situ.

**Aturan untuk Fase 5:** jangan pernah menaikkan batch sampai "hampir OOM".
Batas amannya adalah ~5,5 GB peak, bukan 6,0 GB.

## Temuan 2: biaya VRAM logits sangat besar dan mudah terlupakan

Estimasi aktivasi awal di plan meleset ~2x terlalu rendah. Penyebab utamanya:
saya sama sekali tidak menghitung tensor **logits dan cross-entropy**.

Rincian nyata di batch 8:

| Komponen | VRAM |
|---|---|
| Optimizer state | 488 MB |
| Aktivasi badan (8 layer) | 880 MB |
| **Aktivasi logits + loss** | **768 MB** |
| CUDA overhead | 500 MB |

**Kepala keluaran memakan hampir sebanyak seluruh 8 layer Transformer.**

Alasannya: vocab (16.384) itu **32x lebih lebar** dari `d_model` (512). Satu
tensor logits berukuran `batch x seq x 16384`, dan cross-entropy menaikkannya
ke fp32 lalu menyimpan gradient-nya. Di model besar hal ini tidak terasa
karena `d_model` jauh lebih besar; di model kecil, justru dominan.

Ini memberi satu alasan tambahan untuk vocab 16k (bukan 32k atau 50k), di luar
alasan penghematan parameter yang sudah dibahas di plan §4.

`config.py` sekarang memodelkan keduanya terpisah (`activation_breakdown()`),
dengan koefisien dikalibrasi dari pengukuran nyata. Prediksinya kini meleset
+0,3 GB ke arah **konservatif** — arah yang aman.

## Temuan 3: dua bug yang tertangkap uji

**Bug A — RMSNorm memutus rantai dtype.** `self.weight` selalu tersimpan fp32,
dan `bf16 * fp32` otomatis dipromosikan PyTorch kembali ke fp32. Tidak ada
error, hanya keluaran yang diam-diam berubah tipe. Tertangkap oleh
`test_rmsnorm_mempertahankan_dtype_bf16`. Perbaikan: `self.weight.to(dtype)`.

**Bug B — target tidak digeser di benchmark.** Saya menulis `model(idx, idx)`,
membuat target sama persis dengan input. Loss awal keluar **7,91** padahal
seharusnya 9,70.

Penyebabnya menarik dan layak diingat: residual stream membawa embedding token
langsung ke lapisan akhir, dan karena embedding di-*tie* dengan `lm_head`,
model sejak inisialisasi sudah condong memprediksi **token dirinya sendiri**.
Jadi kalau target tidak digeser, loss terlihat lebih baik dari seharusnya.

Ini persis kenapa uji "loss awal harus ln(vocab)" berharga: ia menangkap
kebocoran label yang membuat angka terlihat *bagus*.

## Keputusan arsitektur yang tercatat di kode

| Komponen | Pilihan | Catatan implementasi |
|---|---|---|
| Norm | RMSNorm pre-norm | fp32 internal, gain di-cast ke dtype input |
| Posisi | RoPE, konvensi Llama | `offset` wajib diisi saat pakai KV-cache |
| Attention | GQA 8 query / 4 KV | 3 jalur mask: penuh, decode, prefill-atas-cache |
| Kernel | `F.scaled_dot_product_attention` | backend dibiarkan dipilih PyTorch (flash tidak ada di Windows) |
| FFN | SwiGLU, `d_ffn=1408` | 3 matriks, `~(8/3) x d_model` |
| Bias | tidak ada | di semua Linear |
| Embedding | tied | hemat 8,4M parameter |
| Init | `std=0.02` | proyeksi residual (`wo`, `w_down`) diskalakan `1/sqrt(2L)` |

## Cara menjalankan

```
.venv\Scripts\python.exe -m pytest tests\ -q          # 23 uji
.venv\Scripts\python.exe scripts\bench_model.py       # verifikasi GPU
.venv\Scripts\python.exe src\model\config.py          # rincian parameter/VRAM
```

## Berikutnya

Fase 4 — training loop (AMP bf16, gradient accumulation, cosine LR + warmup,
grad clipping, checkpoint/resume, logging). Parameter yang sudah terkunci dari
fase ini: **batch 8**, seq_len 512, target ~40.000 token/detik.
