# Fase 6 — SFT: `wicara-56m-base` → `wicara-56m-chat` (SELESAI)

Tanggal: 2026-08-23

## Hasil

| | Nilai |
|---|---|
| Langkah | 3.619 (3 epoch) |
| Contoh SFT | 38.612 train / 788 val |
| **Val loss** | **1,3738** (perplexity 4,0) |
| Token dilatih | 59,3 juta |
| LR | 8e-5 → 8e-6 (10x lebih kecil dari pretrain) |
| Durasi | ~20 menit |

Val loss SFT **tidak sebanding** dengan val loss pretrain (3,05). Keduanya
mengukur hal berbeda: pretrain menghitung loss di semua token teks umum, SFT
hanya di token jawaban asisten pada data percakapan.

## Perubahan perilaku

Inilah yang sebenarnya diukur — bukan angka loss.

| Prompt | Model base (Fase 5) | Model chat (Fase 6) |
|---|---|---|
| Halo! | melanjutkan dialog film berbaris-baris | **Halo, Tn.** |
| Apa kabar? | menulis adegan percakapan | **Baik, baik.** |
| Siapa namamu? | mengarang narasi | **Aku tidak punya nama.** |
| Terima kasih banyak! | melantur | **Tidak masalah, tidak masalah.** |
| Sampai jumpa! | terus menulis | **Sampai jumpa!** |

Model kini menjawab **satu giliran lalu berhenti**. Sebelumnya ia terus
menulis sampai batas panjang, termasuk mengarang giliran user berikutnya.

## Data SFT

39.400 contoh, seluruhnya dari data NYATA — tidak ada yang ditulis sintetis.

| Sumber | Contoh | Peran |
|---|---|---|
| Aya / SODA-inst | 10.000 | Dialog sosial & instruksi naratif |
| Aya / Dolly-v2 | 8.000 | Ditulis manusia, mutu tertinggi |
| Cendol / wikihow | 8.000 | Instruksi prosedural |
| Subtitle / sapaan | 3.000 | Halo, selamat pagi |
| Subtitle / tidak_tahu | 3.000 | Pertanyaan → mengaku tidak tahu |
| Subtitle / kabar | 2.000 | Apa kabar → Aku baik |
| Subtitle / perpisahan | 1.500 | Sampai jumpa |
| Subtitle / terima_kasih | 1.500 | Terima kasih → Sama-sama |
| Subtitle / perkenalan | 1.200 | Siapa namamu → Namaku X |
| Subtitle / maaf | 1.200 | Maaf → Tidak apa-apa |

### Yang sengaja dibuang

**HotpotQA + NQ-Open (531.326 baris).** Tanya-jawab faktual berjawaban pendek
dan tegas. Model 56M tidak punya kapasitas menyimpan fakta itu, jadi yang ia
pelajari bukan jawabannya melainkan KEBIASAAN menjawab yakin tanpa tahu —
memperparah halusinasi, bukan memperbaikinya.

Juga dibuang: Flan-CoT & PIQA (pilihan ganda), Flan-Coqa (keluaran rusak),
indo_puisi (banyak keluaran harfiah `nan`), wikipedia bahasa daerah
(ban/gor/bjn/ace/bug — bukan bahasa Indonesia).

## Tiga kesalahan yang tertangkap

### 1. Penambangan subtitle awalnya menghasilkan sampah

Asumsi pertama: dua baris subtitle berurutan membentuk pasangan tanya-jawab.
**Salah di sebagian besar kasus** — baris berurutan sering diucapkan penutur
yang sama, atau terpisah pergantian adegan:

```
"Halo, nak."      -> "Dia tak hanya takut, dia muak."
"Apa kabarmu?"    -> "Tuan Dalton."
"Terima kasih."   -> "Jangan hubungi kebun binatang."
```

Melatih model dengan itu mengajarkan menjawab dengan kalimat tak nyambung —
lebih merusak daripada tidak SFT sama sekali.

Satu kategori kebetulan lolos: `tidak_tahu`, karena dicocokkan lewat pola
RESPONS, bukan pola pertanyaan. Itu petunjuk perbaikannya — **kedua sisi**
harus cocok pola. Sesudah diperbaiki:

```
"Selamat malam."       -> "Selamat malam!"
"Kau baik-baik saja?"  -> "Aku baik."
"Terima kasih."        -> "Sama-sama."
"Maaf, jalan sesak."   -> "Tak apa."
```

### 2. Dolly-v2 tidak kebagian kuota sama sekali

Satu kuota global untuk seluruh Aya habis diisi SODA-inst yang kebetulan
dibaca lebih dulu. Padahal Dolly-v2 justru subset bermutu tertinggi — ditulis
manusia, bukan terjemahan mesin. Diperbaiki jadi kuota per-subset.

### 3. Model tidak berhenti di `<|end|>`

Sampler hanya berhenti di `<|eos|>`, sehingga model mencetak penanda akhir
giliran lalu lanjut mengarang percakapan sendiri:

```
[Halo, apa kabar?] Aku baik-baik saja.<|end|>Baik-baik saja.<|end|>Dale, apa
yang kamu lakukan?<|end|>Aku baik-baik saja.<|end|>...
```

Untuk model chat, `<|end|>` justru penanda berhenti yang utama. Diperbaiki di
`sample_probe` dan `src/infer/generate.py`.

## Loss masking — bagian paling rawan

Loss HANYA dihitung pada token jawaban asisten. Kalau bocor ke teks user,
model belajar MENIRU cara bertanya alih-alih menjawab — dan bug ini senyap:
training tetap mulus, loss tetap turun, baru ketahuan saat model dipakai.

Karena itu `periksa_masking()` mendekode ulang posisi yang dilatih:

```
teks penuh : <|bos|><|system|>Kamu adalah asisten...<|end|><|user|>Halo, apa
             kabar?<|end|><|assistant|>Baik, terima kasih!...<|end|><|eos|>

dilatih    : 'Baik, terima kasih! Ada yang bisa saya bantu?<|end|>'
             12 dari 33 token (36,4%)
```

System prompt dan teks user diabaikan; `<|end|>` ikut dilatih supaya model
belajar kapan berhenti. Dikunci oleh 21 uji di `tests/test_sft.py`.

## Batas yang terkonfirmasi

```
Anda   : Apa ibu kota Zimbabwe?
Wicara : Zimbabwe adalah sebuah negara di Eropa Barat Laut dengan
         populasi 3,7 juta orang.
```

Salah tiga kali sekaligus: Zimbabwe ada di Afrika bagian selatan, populasinya
~16 juta, dan pertanyaannya (ibu kota) bahkan tidak dijawab.

**Latihan `tidak_tahu` TIDAK menular ke pertanyaan pengetahuan.** 3.000 contoh
yang ditambang berbentuk *"Di mana dia?" → "Aku tak tahu"* — pertanyaan
percakapan, bukan ensiklopedis. Model belajar mengaku tidak tahu dalam konteks
dialog, tapi tidak menggeneralisasi ke pertanyaan fakta.

Ini kegagalan rancangan datanya, bukan kegagalan model. Untuk benar-benar
mengajarkan penolakan pada pertanyaan pengetahuan, dibutuhkan contoh yang
memang berbentuk *"Apa ibu kota X?" → "Maaf, saya tidak tahu"* — dan bentuk
seperti itu tidak ada di dialog film.

Resep nasi goreng menunjukkan pola serupa: format benar (langkah bernomor,
dipelajari dari wikihow), isi karangan (*"Masak nasi di dalam oven"*,
*"topping stroberi"*).

## Catatan lingkungan

Setelah SFT selesai, `import torch` gagal di Bash (MSYS2) dengan
`OSError: [WinError 126] ... torch_python.dll or one of its dependencies`,
padahal **berhasil normal di PowerShell**. Penyebabnya PATH MSYS2 yang
menaungi DLL sistem.

**Jalankan skrip Python lewat PowerShell**, bukan Git Bash.

## Cara memakai

```
.venv\Scripts\python.exe scripts\chat.py            # ngobrol
.venv\Scripts\python.exe scripts\chat.py --uji      # 8 prompt uji
```

Perintah saat mengobrol: `/baru` reset percakapan, `/suhu N` ubah temperature,
`/keluar` selesai.

## Berikutnya

Fase 7 — inference engine: streaming keluaran, penanganan riwayat percakapan
yang lebih baik, dan mungkin kuantisasi.

Perbaikan data yang layak dicoba lebih dulu: contoh penolakan untuk pertanyaan
pengetahuan, supaya `"Apa ibu kota Zimbabwe?"` dijawab *"Maaf, saya tidak
tahu"* alih-alih dikarang.
