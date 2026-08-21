# Fase 4 — Training Loop (SELESAI)

Tanggal: 2026-08-21

## Yang dibangun

| File | Isi |
|---|---|
| `src/train/config.py` | `TrainConfig` — hyperparameter training, terpisah dari arsitektur |
| `src/train/data.py` | Pembaca memmap uint16, batch acak, `close()` |
| `src/train/lr_schedule.py` | Warmup linear + peluruhan cosine |
| `src/train/checkpoint.py` | Simpan/resume atomik, rotasi, log JSONL |
| `src/train/trainer.py` | Loop utama |
| `src/train/synthetic.py` | Korpus bigram entropi diketahui |
| `scripts/make_synthetic_data.py` | CLI generator korpus uji |
| `scripts/train.py` | Entry point training |
| `tests/test_train.py` | 21 uji |

**44/44 uji lolos** (9 primitif + 14 model + 21 training).

## Smoke run: 400 langkah, config 32m, GPU

| Langkah | Train loss | Val loss | tok/detik |
|---|---|---|---|
| 25 | 9,119 | | 39.346 |
| 50 | 7,033 | | 38.480 |
| 100 | 3,819 | 3,778 | 45.152 |
| 150 | 2,473 | | 45.209 |
| 200 | 2,251 | 2,256 | 45.191 |
| 300 | 2,195 | 2,200 | 43.829 |
| **400** | **2,193** | **2,189** | 38.525 |

**Entropi teoretis korpus: 2,0794.** Model berhenti di **2,189** — yaitu 0,11
**di atas** lantai teoretis, persis perilaku yang benar. Perplexity 8,9 vs 8,0
teoretis.

Kalau loss tembus di bawah 2,0794, itu tanda model menghafal urutan spesifik
atau ada kebocoran data validasi. Tidak terjadi.

Ini verifikasi yang jauh lebih kuat daripada sekadar "loss turun": tersedia
angka target yang bisa dihitung di atas kertas, jadi konvergensi bisa dinilai
benar atau salah, bukan sekadar "kelihatannya membaik".

## Angka terukur

| Metrik | Hasil |
|---|---|
| Throughput | **38.500–45.200 token/detik** |
| VRAM | **2,25 GB** stabil (prediksi 2,57 GB — konservatif, benar) |
| Grad norm | puncak 2,04 di awal, lalu tenang di 0,32–0,56 |
| Batch efektif | 8 × 16 akumulasi = 128 sekuens |
| Token per langkah | 65.536 |

**Estimasi pretrain 1,3B token: 8,0–9,4 jam.** Konsisten dengan plan (8–11 jam)
dan dengan benchmark Fase 3 (8,9 jam).

## Temuan 1: jadwal LR tidak mengikuti panjang run

Terlihat dari kolom `lr` di smoke run: dari `2,53e-06` sampai `1,00e-03` — LR
**naik terus sepanjang 400 langkah** dan tidak pernah masuk cosine decay.

Penyebabnya: `warmup_steps` dihitung 2% dari `total_tokens` (1,3B token =
19.836 langkah), jadi warmup = 396 langkah. Run yang dibatasi `--steps 400`
habis di warmup saja.

Ini kategori bug yang berbahaya: **tidak ada error, hasilnya tetap masuk akal,
tapi diam-diam salah**. Setiap eksperimen pendek akan memakai jadwal LR yang
keliru, dan kesimpulan yang diambil darinya ikut keliru.

Perbaikan di `scripts/train.py`: kalau `--steps` diberikan, `total_tokens`
dihitung ulang dari panjang run itu, dan penyesuaiannya dicetak. Untuk
pretrain sungguhan, pakai `--total-tokens` **tanpa** `--steps`.

Setelah diperbaiki, run 20 langkah menghasilkan: warmup 1 langkah, lalu cosine
decay penuh 1e-3 → 1e-4. Benar.

Catatan menarik: model tetap konvergen ke 2,189 walaupun LR masih menanjak
sepanjang run. Dengan jadwal yang benar, hasilnya kemungkinan lebih rapat lagi
ke 2,079.

## Temuan 2: memmap mengunci file di Windows

`TokenDataset` menahan `np.memmap` terbuka. Windows **melarang menimpa atau
menghapus file yang sedang ter-map** — `write_tokens` ke file yang sama gagal
dengan `OSError: [Errno 22] Invalid argument`.

Ini akan menggigit di Fase 1 nanti: pipeline data yang menulis ulang
`train.bin` akan gagal selama masih ada `TokenDataset` hidup yang menunjuk ke
sana. Di Linux tidak terjadi, jadi mudah lolos kalau dikembangkan di sana.

Perbaikan: `TokenDataset.close()` + dukungan context manager, dan
`Trainer.close()` yang menutup keduanya. `scripts/train.py` memanggilnya di
blok `finally`.

## Keputusan yang tercatat di kode

| Aspek | Nilai | Alasan |
|---|---|---|
| Batch | 8 × 16 akumulasi | dari ukur Fase 3; VRAM separuh batch 16, throughput sama |
| Presisi | bf16 autocast | tanpa GradScaler; jangkauan eksponen selebar fp32 |
| LR | 1e-3 → 1e-4 | model ~32M menoleransi LR lebih tinggi dari GPT-2 124M (6e-4) |
| Warmup | 2% langkah | gradient di awal sangat liar |
| Grad clip | 1.0 | sabuk pengaman terhadap batch aneh |
| AdamW | β=(0.9, 0.95) | 0.95 standar LLM, bukan 0.999 |
| Weight decay | 0.1, **kecuali** norm & embedding | 23,6M kena, 8,4M dikecualikan |
| Optimizer | fused AdamW | satu kernel CUDA, bukan satu per tensor |
| Checkpoint | berbasis waktu (30 menit) | yang dibatasi adalah berapa lama kerja yang hilang |
| Eval | generator berbenih tetap | supaya val loss bisa dibandingkan lintas langkah |

## Uji yang paling menentukan

**`test_gradient_accumulation_setara_dengan_batch_besar`** — seluruh strategi
VRAM proyek ini bergantung pada janji bahwa 16 batch kecil = 1 batch besar.
Uji ini membandingkan gradient kedua jalur langsung; selisih maksimum < 1e-5.

**`test_belajar_sampai_mendekati_entropi_teoretis`** — dua arah sekaligus:
loss harus turun jauh dari ln(vocab), tapi tidak boleh tembus di bawah
entropi teoretis. Arah kedua itu yang menangkap kebocoran data.

**`test_close_melepaskan_kunci_file_windows`** — mengunci perilaku yang baru
saja jadi bug.

## Cara menjalankan

```
.venv\Scripts\python.exe scripts\make_synthetic_data.py --tokens 30000000
.venv\Scripts\python.exe scripts\train.py --model 32m ^
    --train-bin data/tokens/synth_train.bin ^
    --val-bin data/tokens/synth_val.bin ^
    --steps 400 --out-dir checkpoints/smoke
.venv\Scripts\python.exe -m pytest tests\ -q
```

## Berikutnya

Semua yang bisa dibangun tanpa data sungguhan sudah selesai. **Fase 1
(pipeline data) dan Fase 2 (tokenizer) sekarang menjadi penghalang** untuk
Fase 5.

Yang sudah siap menunggu: model 32M teruji, training loop terverifikasi,
checkpoint/resume berfungsi, throughput terukur ~40.000 token/detik.
