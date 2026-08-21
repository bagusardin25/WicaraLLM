# Fase 0 — Environment (SELESAI)

Tanggal: 2026-08-12

## Yang dipasang

| Item | Nilai |
|---|---|
| Interpreter | Python 3.11.9 (CPython dari python.org) |
| Lokasi venv | `D:\Project\llmgusz\.venv` |
| PyTorch | 2.13.0+cu130 |
| CUDA build | 13.0 (driver 581.86) |
| NumPy | 2.4.6 |

**Jangan pakai `python` dari PATH.** Itu MSYS2 Python 3.14.3 di
`C:\msys64\ucrt64\bin\python.exe`, dan wheel PyTorch CUDA Windows tidak
kompatibel dengannya. Selalu panggil `.venv\Scripts\python.exe` secara eksplisit.

## Hasil pengukuran mesin

| Metrik | Hasil terukur |
|---|---|
| GPU | RTX 4050 Laptop, sm_89, 20 SM |
| VRAM total | 6,00 GB |
| **VRAM bebas** | **4,95 GB** |
| bf16 native | Ya |
| **Puncak matmul bf16** | **25,2 TFLOPS** (di 8192x8192) |

Skala TFLOPS terhadap ukuran matriks — ini alasan model kecil tidak akan
mencapai puncak:

| Ukuran matmul | TFLOPS |
|---|---|
| 1024 | 16,9 |
| 2048 | 20,0 |
| 4096 | 21,4 |
| 8192 | 25,2 |

## Temuan penting: FlashAttention tidak tersedia di Windows

Plan §8 mengasumsikan `F.scaled_dot_product_attention` akan memanggil
FlashAttention. Ternyata tidak:

```
flash              TIDAK tersedia   -> "Torch was not compiled with flash attention"
mem_efficient      tersedia
math (fallback)    tersedia
```

Wheel PyTorch resmi untuk Windows memang tidak dikompilasi dengan FlashAttention.

**Dampaknya kecil, dan anggaran VRAM di plan tetap berlaku.** Alasannya: yang
menentukan bukan nama kernelnya, melainkan apakah kernel itu membentuk matriks
`seq_len x seq_len` di memori. Kernel `mem_efficient` juga tidak membentuknya —
ia memproses attention per blok, sama seperti flash. Yang berbeda hanya
kecepatan (perkiraan 10-20% lebih lambat), dan pada `seq_len=512` selisihnya
kecil karena matriks 512x512 memang belum besar.

**Konsekuensi untuk Fase 3:** tetap pakai `F.scaled_dot_product_attention`.
Jangan memaksa `SDPBackend.FLASH_ATTENTION` — biarkan PyTorch memilih sendiri,
supaya kode tetap jalan kalau nanti dipindah ke Linux (di sana flash tersedia).

## Kalibrasi ulang estimasi waktu

Dengan puncak 25,2 TFLOPS terukur, untuk config `32m` (32,0M) dan 1,3B token
(1,84e17 FLOPs):

| MFU | TFLOPS efektif | token/detik | Waktu |
|---|---|---|---|
| 15% | 3,8 | 26.700 | 13,5 jam |
| **25%** | **6,3** | **44.600** | **8,1 jam** |
| **35%** | **8,8** | **62.400** | **5,8 jam** |
| 45% | 11,4 | 80.200 | 4,5 jam |

Perkiraan realistis **5,8–8,1 jam** — sedikit lebih baik dari tebakan awal plan
(8–11 jam). Angka final ditetapkan dari throughput nyata 200 step di Fase 5.

## Anggaran VRAM: semua config muat

VRAM bebas terukur 4,95 GB (lebih longgar dari asumsi awal 4,6 GB).

| Config | Param | Batch | Perkiraan | Verdict |
|---|---|---|---|---|
| 7m | 7,1M | 64 | 1,13 GB | OK |
| 19m | 18,9M | 32 | 2,36 GB | OK |
| **32m** | **32,0M** | **16** | **2,03 GB** | **OK** |
| 88m | 88,1M | 8 | 3,00 GB | OK |

Catatan: `88m` ternyata **muat** di batch 8. Jadi alasan untuk tetap
memilih 32M bukan lagi soal memori, melainkan murni soal waktu — 88M butuh
~1,8B token (Chinchilla) yang berarti 20+ jam per run. Kecepatan iterasi tetap
lebih berharga daripada kualitas absolut untuk proyek belajar.

## Koreksi kecil jumlah parameter vs plan

Angka di plan dihitung dengan asumsi MHA. Setelah GQA diterapkan, attention
lebih ramping:

| Config | Plan | Aktual (GQA) |
|---|---|---|
| 7m | ~7,4M | 7,1M |
| 19m | ~20M | 18,9M |
| **32m** | **~32M** | **32,0M** (tepat) |
| 88m | ~97M | 88,1M |

Tidak ada yang mengubah keputusan.

## Cara menjalankan ulang

```
.venv\Scripts\python.exe scripts\check_env.py     # verifikasi + benchmark
.venv\Scripts\python.exe src\model\config.py      # rincian parameter & VRAM
```

## Berikutnya

Fase 1 — pipeline data Indonesia.
