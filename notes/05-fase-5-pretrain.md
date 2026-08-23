# Fase 5 — Pretrain `wicara-56m-base` (SELESAI)

Tanggal: 2026-08-22

## Hasil

| | Nilai |
|---|---|
| Langkah | 17.010 dari 17.018 (1 epoch penuh) |
| Token diproses | 1.114.767.360 |
| **Val loss terbaik** | **3,0505** |
| **Perplexity** | **21,2** |
| Train loss akhir | 2,937 |
| Throughput rata-rata | 28.017 token/detik |
| VRAM puncak | 3,12 GB dari 6,00 GB |
| Grad norm akhir | 0,275 |
| Durasi | ~11,1 jam |

Perkiraan sebelum run adalah val loss 3,3–3,8 dan waktu 11–13 jam. Loss-nya
**lebih baik dari perkiraan**, waktunya di ujung cepat rentang.

## Kurva loss

| Langkah | Train | Val | PPL | LR |
|---|---|---|---|---|
| 250 | 4,945 | 4,906 | 135,1 | 5,65e-04 |
| 1.250 | 3,542 | 3,714 | 41,0 | 7,95e-04 |
| 3.250 | 3,359 | 3,446 | 31,4 | 7,48e-04 |
| 6.250 | 3,193 | 3,297 | 27,0 | 6,00e-04 |
| 10.250 | 3,177 | 3,175 | 23,9 | 3,36e-04 |
| 14.250 | 2,944 | 3,084 | 21,8 | 1,28e-04 |
| **17.000** | **2,937** | **3,053** | **21,2** | 8,00e-05 |

Turun mulus dan monoton, tanpa satu pun lonjakan. Grad norm tenang di
0,27–0,5 sepanjang run.

**Jarak train-val hanya 0,12** (2,937 vs 3,053) — tidak ada overfitting sama
sekali, persis yang diharapkan pada 1 epoch tanpa pengulangan. Ini sekaligus
membenarkan keputusan memperbaiki cakupan sampling dari 63% ke 100%:
kalau 37% korpus terlewat, jaraknya akan jauh lebih lebar.

## Perkembangan keluaran

Prompt tetap `"Halo, apa kabar?"`, diambil dari `samples.txt`:

**Langkah 500** — gramatikal tapi macet mengulang:
```
Halo, itu yang kamu tunggu.
Apa ini benar-benar lucu, kan?
Itu tak terlihat seperti ini.
Itu bukan begitu.
```

**Langkah 3.000** — variasi meningkat, kalimat lebih panjang:
```
Kau sedang apa?
Aku mencari sesuatu untuk dimakan.
Bagaimana kau tahu?
Aku hanya ingin tahu apakah ada cara untuk membuat sesuatu.
```

**Langkah 17.000** — giliran percakapan yang wajar:
```
Halo, ini aku, ada yang bisa kubantu?
Aku segera ke sana.
Halo, ini aku Anna.
Aku sudah menelepon untuk memberitahumu
```

Kalimat *"Halo, ini aku, ada yang bisa kubantu?"* muncul dari **pretraining
saja, tanpa SFT**. Model menyerapnya dari ribuan percakapan telepon di korpus
subtitle. Ini bukti langsung bahwa keputusan memberi OpenSubtitles porsi
terbesar (43%) memang tepat untuk tujuan percakapan.

## Kemampuan yang muncul

**Penyesuaian ragam bahasa.** Model mengganti register mengikuti prompt:

| Prompt | Ragam keluaran |
|---|---|
| `"Halo, apa kabar?"` | kasual — *aku, kau, kubantu* |
| `"Saya ingin bertanya tentang"` | formal — *Saya, Anda* |
| `"Selamat pagi,"` | hormat — *Tuan, Pak* |

**Struktur wacana.** Prompt `"Menurut saya,"` menghasilkan argumen bertahap:

```
Menurut saya, ada satu hal yang harus dilakukan untuk mencegah terjadinya
hal ini. Pertama, Anda harus memastikan bahwa ada bukti ...
Kedua, Anda harus membuat ...
```

Penanda *Pertama/Kedua* diserap dari Wikipedia dan teks web — bagian korpus
yang sengaja dipertahankan agar model tidak hanya bisa bicara terpotong-potong
seperti dialog film.

## Batas yang terkonfirmasi

```
Prompt : "Indonesia adalah negara"
Output : "Indonesia adalah negara yang paling maju di dunia."
```

Gramatikal sempurna, faktual salah total. Ini **bukan bug** — memang begitu
kapasitas 56 juta parameter. Model belajar bentuk bahasa, bukan fakta dunia.

Model base juga masih **melanjutkan teks**, bukan menjawab. Diberi sapaan ia
menghasilkan lanjutan dialog, bukan respons terarah kepada penanya.
Kemampuan menjawab sebagai asisten baru datang dari SFT di Fase 6.

## Verifikasi rencana vs hasil

| Yang diperkirakan | Yang terjadi |
|---|---|
| Val loss 3,3–3,8 | **3,05** — lebih baik |
| 11–13 jam | **11,1 jam** — sesuai |
| ~27.800 tok/s | **28.017** — sesuai |
| VRAM ~3,12 GB | **3,12 GB** — tepat |
| Loss awal 9,70 | Terkonfirmasi di langkah awal |

Tidak ada satu pun kejadian tak terduga sepanjang run: tidak ada NaN, tidak
ada spill VRAM, tidak ada divergensi, tidak perlu resume.

## Artefak

```
checkpoints/wicara-56m-base/
    best.pt        671 MB  val loss 3,0505
    final.pt       671 MB  langkah 17.010
    step_*.pt      3 checkpoint berkala terakhir
    log.jsonl      1.769 entri
    samples.txt    35 contoh keluaran sepanjang training
```

## Cara memakai

```
.venv\Scripts\python.exe scripts\generate.py --ckpt checkpoints/wicara-56m-base/best.pt
.venv\Scripts\python.exe scripts\generate.py --ckpt checkpoints/wicara-56m-base/best.pt --interactive
```

## Berikutnya

Fase 6 — SFT. Mengubah `wicara-56m-base` menjadi `wicara-56m-chat` supaya
model **menjawab**, bukan sekadar melanjutkan teks. Bahan yang sudah siap:

- Chat template dan 32 special token di `src/tokenizer/chat_template.py`
- Fungsi `assistant_spans()` untuk loss masking — loss hanya dihitung pada
  token jawaban asisten, tidak pada pertanyaan user
- Data instruksi di korpus: Aya (121M token) dan Cendol (129M token)
