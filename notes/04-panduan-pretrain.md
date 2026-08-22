# Panduan Pretrain `wicara-56m-base`

Dokumen mandiri untuk menjalankan dan memantau pretrain tanpa bantuan lain.

---

## 1. Perintah

Jalankan dari folder `D:\Project\llmgusz`:

```
.venv\Scripts\python.exe scripts\train.py --model 56m --total-tokens 1115298306 --out-dir checkpoints/wicara-56m-base
```

Kalau berhenti di tengah (Ctrl+C, mati listrik, restart), lanjutkan dengan
perintah **yang sama persis** ditambah `--resume`:

```
.venv\Scripts\python.exe scripts\train.py --model 56m --total-tokens 1115298306 --out-dir checkpoints/wicara-56m-base --resume
```

Posisi baca dataset ikut tersimpan, jadi potongan yang sudah dilihat tidak
akan diulang.

---

## 2. Periksa sebelum mulai

- [ ] **Colokkan listrik.** Baterai tidak akan cukup, dan GPU akan turun daya
- [ ] **Matikan sleep.** Settings → System → Power → Screen and sleep →
      "When plugged in, put my device to sleep" = **Never**.
      Laptop yang tidur mematikan CUDA context dan training hilang sampai
      checkpoint terakhir
- [ ] **Tunda Windows Update.** Restart otomatis di jam ke-8 menyakitkan
- [ ] **Tutup aplikasi berat GPU** (game, Blender, editor video, model AI lain)
- [ ] **Ventilasi baik.** GPU akan di ~80W selama belasan jam
- [ ] **Sisakan ~5 GB disk** untuk checkpoint (5 berkas × ~670 MB)

**Internet tidak dibutuhkan sama sekali.** Semua sudah lokal — `train.bin`,
tokenizer, PyTorch. Mati internet tidak berpengaruh apa pun.

**Aplikasi ringan tetap boleh:** browser, VS Code, Office, terminal. Yang
menentukan adalah VRAM, dan sisa ~2,9 GB cukup untuk pemakaian normal.

---

## 3. Yang akan Anda lihat

Setiap 10 langkah:

```
step   1,250/17,018  loss  4.821  lr 7.94e-04  gnorm  0.43   27,540 tok/s  3.12GB  sisa 10.2j
```

| Kolom | Sehat | Bermasalah |
|---|---|---|
| `loss` | Mulai **9,70**, turun terus | `nan` → meledak. Macet di 9,70 setelah 500 langkah → ada yang rusak |
| `lr` | Naik sampai langkah 340, lalu turun mulus | Naik terus sampai akhir → jadwal salah |
| `gnorm` | 0,2–1,0 | Melonjak liar atau `nan` |
| `tok/s` | ~27.000 stabil | Anjlok ke ~7.000 → **spill VRAM** |
| VRAM | ~3,12 GB tetap | Naik terus → kebocoran memori |

Setiap 250 langkah muncul baris val loss:

```
         val loss  4.795  ppl    120.9  <- terbaik
```

`<- terbaik` berarti checkpoint `best.pt` baru saja diperbarui.

### Kisaran loss yang diharapkan

Perkiraan kasar, bukan target keras. Yang penting **arahnya turun**.

| Langkah | Loss |
|---|---|
| 0 | 9,70 |
| ~340 (akhir warmup) | 6,5–7,0 |
| ~1.000 | 5,5–6,0 |
| ~3.000 | 4,5–5,0 |
| ~8.000 | 4,0–4,4 |
| 17.018 (selesai) | 3,3–3,8 |

---

## 4. Berkas yang bisa dipantau

```
checkpoints\wicara-56m-base\
    log.jsonl      metrik tiap 10 langkah, satu baris JSON per event
    samples.txt    contoh keluaran model tiap 500 langkah
    best.pt        val loss terbaik sejauh ini
    step_*.pt      checkpoint berkala (3 terbaru disimpan)
    final.pt       hasil akhir
```

**`samples.txt` adalah yang paling menarik.** Buka sesekali:

```
type checkpoints\wicara-56m-base\samples.txt
```

Bacalah dari atas ke bawah setelah selesai. Keluarannya akan bergerak dari
kata acak → frasa → kalimat. Contoh nyata dari uji asap di langkah 20:

```
[<bos>]  yang ini-...
 ini yang-.
 tahun.
```

Model sudah memuntahkan kata fungsi Indonesia (`yang`, `ini`, `tahun`) tapi
belum punya tata bahasa. Itu tepat untuk loss ~7.

---

## 5. Kalau ada masalah

| Gejala | Sebab | Tindakan |
|---|---|---|
| `tok/s` anjlok 27k → 7k | Spill VRAM ke RAM sistem | Tutup aplikasi lain. Tidak perlu restart training |
| `loss` jadi `nan` | Training divergen | Ctrl+C, lalu `--resume` + `--lr 4e-4` |
| `CUDA out of memory` | VRAM habis | Ctrl+C, `--resume --batch-size 4 --grad-accum 32` (batch efektif tetap sama) |
| Laptop restart | Update/mati listrik | `--resume` saja, kehilangan maksimal 30 menit |
| `gnorm` melonjak terus | LR terlalu tinggi | `--resume --lr 4e-4` |
| Loss turun tapi val loss naik | Overfitting | Tidak akan terjadi di 1 epoch; kalau muncul, hentikan dan pakai `best.pt` |

Ctrl+C aman kapan saja — checkpoint terakhir maksimal berumur 30 menit.

---

## 6. Setelah selesai

### Menguji model

```
.venv\Scripts\python.exe scripts\generate.py --ckpt checkpoints/wicara-56m-base/best.pt
```

Mode tanya-jawab bebas:

```
.venv\Scripts\python.exe scripts\generate.py --ckpt checkpoints/wicara-56m-base/best.pt --interactive
```

Opsi yang bisa diatur: `--temperature` (0 = greedy, 0,8 = wajar, 1,2 = liar),
`--top-k`, `--top-p`, `--repetition-penalty`, `--max-new-tokens`, `--seed`.

### PENTING: yang akan Anda dapat

`wicara-56m-base` adalah **model base**, bukan chatbot. Ia **melanjutkan
teks**, bukan menjawab pertanyaan.

Diberi `"Halo, apa kabar?"`, keluarannya bukan *"Halo! Ada yang bisa saya
bantu?"* melainkan lanjutan teks — mungkin baris dialog berikutnya, mungkin
melantur jadi paragraf artikel.

**Itu perilaku yang benar.** Kemampuan menjawab datang dari SFT di Fase 6.

Yang wajar diharapkan dari model base 56M:
- Bahasa Indonesia yang gramatikal
- Koherensi 1–3 kalimat
- Gaya menyesuaikan prompt (sapaan → dialog, kalimat formal → prosa)

Yang **tidak** akan ada:
- Akurasi fakta — akan berhalusinasi dengan yakin
- Berhitung, menalar bertahap, konsisten di teks panjang
- Menjawab pertanyaan sebagai asisten

### Ukur hasilnya

```
.venv\Scripts\python.exe -c "import json; L=[json.loads(l) for l in open('checkpoints/wicara-56m-base/log.jsonl',encoding='utf-8')]; v=[x for x in L if 'val_loss' in x]; print('val loss akhir:', v[-1]['val_loss']); print('terbaik:', min(x['val_loss'] for x in v))"
```

Val loss 3,3–3,8 berarti perplexity 27–45. Untuk model 56M pada bahasa
Indonesia, itu hasil yang wajar.

---

## 7. Berikutnya

| Fase | Isi | Perlu |
|---|---|---|
| 6 | SFT percakapan | Data dialog + chat template (sudah ada di `src/tokenizer/chat_template.py`), loss masking hanya pada token asisten |
| 7 | Inference engine & CLI | Perluasan `src/infer/generate.py`: streaming, stop sequence, riwayat percakapan |
| 8 | Evaluasi | Kurva loss, eval manual 60 prompt, perbandingan checkpoint |

Setelah Fase 6, barulah model bisa diajak bicara sungguhan — dan namanya
berubah jadi `wicara-56m-chat`.

---

## 8. Ringkasan keadaan proyek

| | Nilai |
|---|---|
| Model | `wicara-56m` — 56.0M parameter (45,5M non-embedding) |
| Arsitektur | RMSNorm pre-norm · RoPE · GQA 10q/5kv · SwiGLU · tied embeddings |
| `d_model`/`n_layer` | 640 / 10, head_dim 64, d_ffn 1728 |
| Tokenizer | BPE byte-level 16.384, 4,26 karakter/token |
| Korpus | 1.115.298.306 token train + 5.683.512 val |
| Token/parameter | 19,9 (Chinchilla-optimal: 20) |
| Batch | 8 × 16 akumulasi = 65.536 token/langkah |
| Langkah | 17.018 (1 epoch penuh, cakupan 100%) |
| LR | 8,00e-04 → 8,00e-05, warmup 340 langkah |
| VRAM | ~3,12 GB dari 6,00 GB |
| Throughput | ~27.800 token/detik |
| **Perkiraan waktu** | **11–13 jam** |
| Uji | 149 lolos |

Semua perintah lain ada di [`../README.md`](../README.md).
