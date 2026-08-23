# Fase 6b — Data tulis tangan: persona & permintaan terbuka

Tanggal: 2026-08-23

Lanjutan bagian 6 di `notes/HANDOFF.md`. Fase ini **belum dilatih** — datanya
sudah siap, training-nya dijalankan pemilik.

## Masalah yang dikerjakan

Sesudah SFT v2, hanya dua pola prompt yang masih gagal:

| Prompt gagal | Prompt setara yang berhasil |
|---|---|
| `"Ceritakan sesuatu"` | `"cerita dong"` → 84 char |
| `"Kamu bisa apa?"` | `"Jelaskan apa saja yang bisa kamu lakukan."` → 435 char |

Keduanya bukan batas kapasitas 56M. Model sudah terbukti bisa menjawab panjang
dan koheren; yang kurang adalah contoh SFT untuk dua bentuk itu.

## Penemuan utama: kategori `perkenalan` melawan persona

Ini yang paling penting dari seluruh fase ini, dan sebelumnya salah didiagnosis.

Model menjawab `"Namaku Kittan"` dan `"Aku Daphne"`. Dugaan sebelumnya: nama
tokoh film itu bocor dari **pretrain**, sama seperti kasus "aku tidak tahu".

**Dugaan itu salah.** Penyebabnya ada di data SFT sendiri. Kategori
`perkenalan` menambang pasangan dari subtitle dengan pola:

```
"Siapa namamu?"  ->  "Namaku X"        (X = nama tokoh film)
```

dan diberi kuota **1.200 contoh**. Itu persis pertanyaan yang seharusnya
dijawab dengan persona Wicara, tapi dengan jawaban yang salah — dan jumlahnya
9× lebih banyak daripada 138 contoh persona yang baru ditulis.

Menambah data persona tanpa mematikan kategori ini akan sia-sia: model
mengikuti mayoritas. Karena itu kuotanya disetel **0**, bukan dikecilkan —
seluruh isi kategori itu mengajarkan perilaku yang justru sedang diganti.

Polanya sendiri (`POLA_PASANGAN["perkenalan"]` di `src/data/sft.py`) sengaja
dibiarkan ada supaya keputusannya terlihat, bukan hilang diam-diam. Dikunci
uji `test_kuota_perkenalan_tetap_nol`.

## Keputusan soal data sintetis

Preferensi pemilik "data asli, bukan sintetis" ditetapkan untuk **korpus
pretraining**, di mana keaslian menentukan integritas 1,1 miliar token. Untuk
SFT sifatnya berbeda: SFT adalah tahap demonstrasi perilaku, dan dua pola di
atas **tidak ada di korpus mana pun**:

- **Persona** — tidak ada teks Indonesia yang memuat "Saya Wicara".
- **Permintaan terbuka** — dialog film selalu punya konteks adegan, jadi tidak
  pernah ada giliran yang benar-benar kosong topik.

Ditulis 400 contoh, sekitar **1% dari total data SFT**. Disetujui pemilik
sebelum ditulis.

**Istilahnya jujur saja: ini data sintetis.** Ditulis oleh asisten, bukan
manusia. Yang membedakannya dari data sintetis skala besar bukan keasliannya,
melainkan jumlahnya kecil sehingga **seluruh 400 barisnya bisa dibaca** sebelum
dipakai. Karena itu berkasnya JSONL polos, bukan gzip: terbaca di `git diff`
dan bisa disunting langsung.

Risiko yang biasa dibahas soal data sintetis (*model collapse*, gaya bahasa
menyeragam) menyangkut porsi puluhan persen di korpus pretrain. Pada 1% di
tahap SFT, yang berubah hanya perilaku pada pola yang tadinya tidak punya
contoh sama sekali.

## Isi `data/sft/tulis_tangan.jsonl`

400 contoh, tiga kategori:

| Kategori | Contoh | Median jawaban | Isi |
|---|---|---|---|
| `terbuka` | 239 | 215 char | Permintaan tanpa topik → model memilih topik lalu bercerita 2–4 kalimat |
| `persona` | 138 | 87 char | Identitas, kemampuan, hakikat, asal-usul |
| `batas` | 23 | 112 char | Kejujuran soal keterbatasan: sering salah, tidak punya internet, tidak ingat obrolan lama |

Fakta persona yang dijaga konsisten di seluruh contoh: nama **Wicara**, model
bahasa Indonesia, **56 juta parameter**, dilatih dari nol dari teks Indonesia,
tanpa akses internet, tanpa ingatan antar percakapan, sering keliru soal fakta.

Dua hal yang sengaja **tidak** dilakukan saat menulis:

1. **Jawaban `terbuka` tidak memuat klaim faktual.** Semuanya berupa cerita
   atau pengamatan sehari-hari. Kalau bentuk yang dilatih adalah "pernyataan
   fakta yang meyakinkan", model akan menirunya dan halusinasi justru
   memburuk — pelajaran yang sama dengan alasan HotpotQA dibuang di Fase 6.

2. **Register divariasikan.** Prompt baku dijawab dengan "Saya", prompt santai
   dijawab dengan "Aku". Panjang jawaban juga divariasikan mengikuti panjang
   prompt, sesuai tujuan pemilik.

## Temuan sampingan: anggaran token memotong jawaban panjang

`chat.py` default `--max-new-tokens 100`. Pada 4,26 karakter/token itu hanya
**±426 karakter**. Data wikihow saja median jawabannya **632 karakter** — jadi
jawaban panjang selama ini **terpotong di tengah kalimat** saat inference,
terlepas dari kualitas modelnya.

Default dinaikkan ke **160** (±680 karakter). Seluruh jawaban tulis tangan
dijaga di bawah 500 karakter supaya selalu selesai utuh; dikunci uji
`test_jawaban_muat_di_anggaran_token_chat`.

## Hasil build `data/sft-v3/`

40.200 contoh (39.396 train / 804 val). Seluruh 400 contoh tulis tangan lolos
— tidak ada yang kena dedup maupun saringan `layak_instruksi`.

| Sumber | Contoh | Porsi |
|---|---|---|
| aya | 22.000 | 54,7% |
| subtitle | 9.800 | 24,4% |
| cendol | 8.000 | 19,9% |
| **tulis tangan** | **400** | **1,0%** |

Turun dari 41.000 di v2 karena 1.200 contoh `perkenalan` dihapus dan 400
ditambahkan. `data/sft/` (v2) tidak disentuh — dibangun ke direktori terpisah
lewat `--out-dir`.

## Perubahan kode

| Berkas | Perubahan |
|---|---|
| `data/sft/tulis_tangan.jsonl` | **baru** — 400 contoh, JSONL polos, ikut ter-commit |
| `scripts/build_sft_data.py` | `kumpulkan_tulis_tangan()`; `perkenalan` → 0; tulis tangan dikumpulkan paling awal supaya menang dedup |
| `src/data/sft.py` | Docstring dikoreksi — klaim "tidak ada yang ditulis sintetis" sudah tidak benar |
| `scripts/chat.py` | `--max-new-tokens` 100 → 160; 4 prompt uji baru; default checkpoint v1 → v2 |
| `tests/test_sft.py` | +8 uji (170 → 178) |

Delapan uji baru menahan hal yang mudah rusak diam-diam pada berkas yang
disunting tangan: bentuk JSON, contoh kembar, saringan yang sama dengan jalur
tambang, jawaban `terbuka` yang pelan-pelan memendek, penyebutan nama selain
Wicara, dan kuota `perkenalan` yang dihidupkan lagi.

## Berikutnya

Latih v3 dan bandingkan dengan v2 lewat `chat.py --uji` (12 prompt, empat di
antaranya khusus menguji fase ini).

Yang **belum** dikerjakan dan sengaja di luar lingkup:

- **Penolakan pertanyaan pengetahuan** — `"Apa ibu kota Zimbabwe?"` masih
  dikarang. Butuh contoh berbentuk `"Apa ibu kota X?"` → `"Maaf, saya tidak
  tahu"`, dan bentuk itu tidak ada di dialog film. Catatan Fase 6 sudah
  menyebutnya sebagai perbaikan data yang layak dicoba.
- **Multi-turn** — pemilik memilih menundanya supaya v3 hanya menguji satu
  perubahan.
