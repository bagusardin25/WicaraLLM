# Fase 2 — Tokenizer & Packing (SELESAI)

Tanggal: 2026-08-22

## Yang dibangun

| File | Isi |
|---|---|
| `src/tokenizer/bpe.py` | BPE byte-level, pola pra-tokenisasi, verifikasi ID |
| `scripts/train_tokenizer.py` | Latih BPE 16k dari korpus bersih |
| `scripts/pack_tokens.py` | Encode + tulis train.bin / val.bin uint16 |
| `tests/test_tokenizer.py` | 39 uji |

**144/144 uji lolos** (9 primitif + 21 model + 21 training + 37 clean + 39 tokenizer + 17 lintas-config).

## Hasil

| | Nilai |
|---|---|
| Vocab | 16.384 |
| Special token | 32 (ID 0-31, terverifikasi cocok `chat_template.py`) |
| Waktu latih | 0,8 menit (sampel 15%) |
| **Rasio kompresi** | **4,26 karakter/token** |
| `train.bin` | **1.115.298.306 token** (2,08 GB) |
| `val.bin` | 5.683.512 token (11 MB) |
| Waktu packing | 7,6 menit |

## Rasio kompresi per sumber

| Sumber | char/token | Token |
|---|---|---|
| ted | 4,92 | 3M |
| aya | 4,78 | 121M |
| fineweb2 | 4,58 | 210M |
| cendol | 4,46 | 129M |
| wikipedia | 4,16 | 174M |
| opensubtitles | 3,98 | 484M |
| **gabungan** | **4,26** | **1.121M** |

Target di plan 3,5-4,0; hasilnya **4,26**, di atas target. Sebagai pembanding,
tokenizer berbasis bahasa Inggris hanya mencapai ~2,2 untuk teks Indonesia.

Subtitle paling rendah (3,98) karena banyak baris pendek dan ganti baris.
Aya tertinggi (4,78) karena prosa tanya-jawabnya panjang dan baku.

Sapaan yang jadi target proyek ini ter-encode efisien:

```
' halo'  -> 1 token      ' apa kabar'    -> 2 token
' hai'   -> 1 token      ' terima kasih' -> 2 token
' iya'   -> 1 token      ' selamat pagi' -> 2 token
' tidak' -> 1 token
```

## Konsekuensi: token korpus berubah dari perkiraan

Kompresi lebih baik berarti teks yang sama menghasilkan **lebih sedikit**
token:

| | Token |
|---|---|
| Perkiraan lama (asumsi 3,70) | 1.291 juta |
| **Nyata (terukur 4,26)** | **1.121 juta** |

Sekilas terdengar merugikan, tapi tidak: rasio token per parameter justru
mendarat nyaris tepat di titik Chinchilla-optimal.

| | Nilai |
|---|---|
| Token train | 1.115 juta |
| Parameter | 56,0 juta |
| **Token/parameter** | **19,9** (Chinchilla-optimal: 20) |

Dan mutu per token naik: satu token kini membawa 4,26 karakter makna, bukan
3,70.

## Temuan: bug akuntansi di packing

Laporan per-sumber run pertama memberi TED **2,01 char/token**, padahal
tokenizer mengukurnya 4,90. Semua angka juga tampak kelipatan 8 juta —
persis nilai `FLUSH_TOKEN`.

Penyebabnya: `PenulisBin.n_token` hanya bertambah di dalam `_flush()`, yang
jalan tiap 8 juta token. Saat rekap per-sumber membaca `n_token` di batas
antar sumber, sampai 8 juta token masih mengendap di buffer dan belum
terhitung — sehingga teratribusi ke sumber berikutnya.

**Berkas `.bin`-nya sendiri selalu benar**; yang salah hanya angka laporan.
Terbukti dari run ulang setelah perbaikan: total token **identik persis**
(1.115.298.306), hanya rincian per-sumbernya yang berubah jadi masuk akal.

Perbaikan: hitung token saat `tulis()` menerima, bukan saat `_flush()`
menulis.

Pelajarannya: angka yang "terlalu bulat" layak dicurigai. Kalau TED tidak
kebetulan jadi sumber terkecil dan terakhir, penyimpangannya tidak akan
sejelas itu dan bisa lolos ke dokumentasi.

## Keputusan yang tercatat di kode

| Aspek | Pilihan | Alasan |
|---|---|---|
| Algoritma | BPE byte-level | Tidak mungkin ada `<unk>`; emoji, aksara asing, dan artefak OCR tetap terwakili |
| Vocab | 16.384 | Embedding 18,7% dari model 56M; vocab 50k akan memakan >50% dan melipatgandakan VRAM logits |
| Pra-tokenisasi | Pola cl100k disederhanakan | Kontraksi Inggris dibuang; angka dipotong maksimal 3 digit agar vocab tidak terisi token seperti "1998" |
| Spasi | Menempel ke kata berikutnya | `" makan"` dan `"makan"` jadi token berbeda, batas kata jelas tanpa token spasi terpisah |
| Special token | Ditulis paling depan | Menempati ID 0-31 sesuai kunci di `chat_template.py`; diverifikasi otomatis setelah training |
| Sampel latih | 15% proporsional | Kalau hanya dari subtitle, aturan merge condong ke pola dialog dan boros saat menemui prosa |
| Pemisah dokumen | `<|eos|>` | Model belajar batas dokumen |
| Split validasi | Tingkat dokumen | Mengiris larik akhir akan membelah dokumen antara latih dan validasi — kebocoran |

## Verifikasi akhir

```
train.bin : 1.115.298.306 token | ID 2-16383 | EOS hadir sebagai pemisah
val.bin   :     5.683.512 token
TokenDataset: target digeser satu posisi = True
dekode dari offset 500 juta -> teks Indonesia utuh dan terbaca
```

## Cara menjalankan

```
.venv\Scripts\python.exe scripts\train_tokenizer.py
.venv\Scripts\python.exe scripts\pack_tokens.py
.venv\Scripts\python.exe -m pytest tests\test_tokenizer.py -q
```

## Berikutnya

Fase 5 — pretrain `wicara-56m-base`. Semua prasyarat sudah siap.
