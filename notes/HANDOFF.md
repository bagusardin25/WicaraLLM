# Wicara — Catatan Serah Terima

Dokumen untuk melanjutkan proyek di sesi baru tanpa kehilangan konteks.
Terakhir diperbarui: **2026-08-24**.

> Baca bagian 1, 6, dan 7 dulu. Sisanya rujukan.

---

## 1. Ringkasan singkat

**Wicara** — model bahasa Indonesia 56 juta parameter, dibangun dari nol di
laptop RTX 4050 6 GB. Arsitektur, training loop, dan pipeline data ditulis
sendiri di PyTorch.

Lokasi: `D:\Project\llmgusz` · Repo GitHub: `bagusardin25/WicaraLLM`

**Tujuan pemilik:** model yang bisa diajak percakapan biasa dalam bahasa
Indonesia, dan **menyesuaikan panjang respons dengan prompt** — sapaan
dijawab singkat, permintaan bercerita dijawab panjang dan natural.

**Status: pretrain dan SFT sudah selesai.** Sapaan dan permintaan terbuka
sudah bekerja (v3). Yang masih gagal: persona — model belum mengaku bernama
Wicara. Data v4 untuk memperbaikinya sudah siap, tinggal dilatih. Bagian 6.

---

## 2. Status per fase

| Fase | Isi | Status |
|---|---|---|
| 0 | Environment & benchmark GPU | ✅ |
| 1 | Pipeline data (unduh, bersih, dedup) | ✅ |
| 2 | Tokenizer BPE 16k + packing | ✅ |
| 3 | Arsitektur model | ✅ |
| 3b | Uji sanity | ✅ |
| 4 | Training loop | ✅ |
| 5 | Pretrain `wicara-56m-base` | ✅ val loss 3,05 |
| 6 | SFT `wicara-56m-chat` | ✅ val loss 1,45 (v2) |
| 6b | Data tulis tangan: persona & permintaan terbuka | 🟨 v3 dilatih — `terbuka` ✅, `persona` ❌ |
| 6c | v4: persona diperbesar 161 → 704 | 🟨 data siap, **belum dilatih** |
| 7 | Inference engine & CLI | 🟨 `chat.py` + `generate.py` jalan |
| 8 | Evaluasi | ⬜ belum |

**179 uji lolos** (`pytest tests/ -q`).

Catatan per fase ada di `notes/00-` sampai `notes/06b-`.

---

## 3. Angka kunci

### Model

| | Nilai |
|---|---|
| Parameter | 56.0M (45,5M non-embedding) |
| `d_model` / `n_layer` | 640 / 10 |
| Attention | GQA 10 query / 5 KV, head_dim 64 |
| `d_ffn` | 1728 · SwiGLU · RMSNorm pre-norm · RoPE · tied embeddings |
| Vocab | 16.384 (BPE byte-level, 4,26 karakter/token) |
| Konteks | 512 token |

Config lain tersedia: `7m`, `19m`, `32m`, `88m` (`src/model/config.py`).

### Korpus pretrain — 1,12 miliar token

| Sumber | Token | Porsi |
|---|---|---|
| OpenSubtitles v2024 | 484M | **43,2%** |
| FineWeb-2 `ind_Latn` | 210M | 18,7% |
| Wikipedia ID | 174M | 15,5% |
| Cendol v2 | 129M | 11,5% |
| Aya Collection | 121M | 10,8% |
| TED2020 | 3M | 0,3% |

`train.bin` 1.115.298.306 token · `val.bin` 5.683.512 token · 19,9 token per
parameter (Chinchilla-optimal ≈ 20).

### Hasil training

| | Pretrain | SFT v2 |
|---|---|---|
| Val loss | **3,0505** (ppl 21,2) | **1,4541** |
| Langkah | 17.010 | 3.766 |
| Durasi | 11,1 jam | ~20 menit |
| Throughput | 28.017 tok/s | ~60.000 tok/s |
| VRAM | 3,12 GB | ~2,2 GB |

### Data SFT v4 — 40.743 contoh (`data/sft-v4/`)

39.929 train · 814 val. Versi lama utuh: `data/sft/` (v2), `data/sft-v3/` (v3).

| Kategori | Sumber | Contoh | Median jawaban |
|---|---|---|---|
| SODA-inst (T) | aya | 12.000 | 226 char |
| Dolly-v2 (T) | aya | 10.000 | 179 char |
| wikihow | cendol | 8.000 | 632 char |
| sapaan | subtitle | 3.000 | 16 char |
| kabar | subtitle | 2.000 | 19 char |
| perpisahan | subtitle | 1.500 | 20 char |
| terima_kasih | subtitle | 1.500 | 14 char |
| maaf | subtitle | 1.200 | 14 char |
| tidak_tahu | subtitle | 600 | 15 char |
| **persona** | **tulis tangan** | **584** (292 unik ×2) | 71 char |
| **terbuka** | **tulis tangan** | **239** | 215 char |
| **batas** | **tulis tangan** | **120** (60 unik ×2) | 112 char |

Tulis tangan 943 instance (2,3%), dari 591 contoh unik. `perkenalan` sudah
dihapus sejak v3 — lihat bagian 6 dan 7.

**persona + batas = 704 melawan tidak_tahu = 600.** Perbandingan itu yang
menentukan; di v3 angkanya 161 lawan 600 dan persona kalah telak.

Perhatikan median `wikihow`: **632 karakter**. Itu lebih panjang daripada yang
muat di anggaran token inference sebelum diperbaiki; lihat bagian 7.

---

## 4. Checkpoint

```
checkpoints/
  wicara-56m-base/     best.pt (val 3,0505) · final.pt · log.jsonl · samples.txt
  wicara-56m-chat/     SFT v1 — tidak_tahu 3.000 contoh
  wicara-56m-chat-v2/  SFT v2 — tidak_tahu 600, Dolly-v2 masuk
  wicara-56m-chat-v3/  SFT v3 — tulis tangan 400  <-- TERBAIK SEJAUH INI
  wicara-56m-chat-v4/  SFT v4 — BELUM ADA, tinggal dilatih (bagian 6)
```

`chat.py` masih default ke v2 — ganti ke v3, atau v4 setelah terbukti.

`wicara-56m-chat-v2/log.jsonl` dan `samples.txt` **kosong (0 byte)** karena
mati listrik saat training baru selesai. Bobotnya **utuh dan terverifikasi**
(55.964.800 parameter, tanpa NaN). Log v3 utuh.

---

## 5. Perintah

**PENTING: jalankan lewat PowerShell, BUKAN Git Bash.** Di MSYS2 bash,
`import torch` bisa gagal dengan `OSError: [WinError 126] ... torch_python.dll
or one of its dependencies` karena PATH MSYS2 menaungi DLL sistem.

```powershell
cd D:\Project\llmgusz

# ngobrol (default sudah v2)
.venv\Scripts\python.exe scripts\chat.py
.venv\Scripts\python.exe scripts\chat.py --uji

# model base (melanjutkan teks, bukan menjawab)
.venv\Scripts\python.exe scripts\generate.py

# uji
.venv\Scripts\python.exe -m pytest tests\ -q

# LATIH v4 — data di data/sft-v4/ sudah dibangun, tinggal jalankan (~20 menit)
.venv\Scripts\python.exe scripts\train_sft.py --epochs 3 --sft-dir data/sft-v4 --out-dir checkpoints/wicara-56m-chat-v4

# lalu bandingkan v4 dengan v3
.venv\Scripts\python.exe scripts\chat.py --ckpt checkpoints/wicara-56m-chat-v4/best.pt --uji
.venv\Scripts\python.exe scripts\chat.py --ckpt checkpoints/wicara-56m-chat-v3/best.pt --uji

# bangun ulang data (hanya kalau tulis_tangan.jsonl atau kuota diubah)
.venv\Scripts\python.exe scripts\build_sft_data.py --out-dir data/sft-v4
```

Tiap versi dibangun ke direktori sendiri (`data/sft/`, `sft-v3/`, `sft-v4/`)
supaya versi lama tetap bisa direproduksi persis.

---

## 6. LANGKAH BERIKUTNYA — latih v4

### Yang diinginkan pemilik

Respons **menyesuaikan panjang dengan prompt**: sapaan dijawab singkat,
permintaan bercerita dijawab panjang dan natural.

### Yang sudah bekerja di v2

Sapaan, kabar, terima kasih, maaf, perpisahan — semuanya dijawab wajar dan
singkat. Model juga **sudah bisa menjawab panjang**:

```
"Tulislah cerita pendek tentang anak di desa."  -> 193 char, narasi koheren
"Jelaskan apa saja yang bisa kamu lakukan."     -> 435 char, terstruktur
"gimana cara masak nasi goreng"                 -> 411 char, langkah bernomor
```

Yang ketiga penting: itu **bahasa santai tanpa kapital**. Jadi register santai
BUKAN penghalang.

### Hasil v3: `terbuka` berhasil, `persona` gagal

Diuji pada 12 prompt dan setelan `chat.py` yang sama untuk kedua checkpoint:

| Prompt | v2 | v3 |
|---|---|---|
| **Ceritakan sesuatu.** | Aku tidak tahu. | Baik, satu hal kecil. Ada seorang gadis bernama Trie… (3 kalimat) ✅ |
| **bosen nih** | Hey, apa yang terjadi? | Baiklah, kita mulai. 🟨 |
| **Siapa kamu?** | Aku tidak… | Aku Kawis! ❌ |
| **Kamu bisa apa?** | Aku tak tahu. | Saya tidak tahu. ❌ |

Sisanya (sapaan, terima kasih, perpisahan, Zimbabwe, nasi goreng) setara.

**Kenapa satu berhasil dan satu tidak:**

| Kategori | Contoh | Porsi | Prior yang dilawan | Hasil |
|---|---|---|---|---|
| `terbuka` | 239 | 0,59% | lemah | ✅ |
| `persona` | 138 | 0,34% | nama tokoh dari 484 juta token subtitle | ❌ |

Ditambah `tidak_tahu` yang **4,3× lebih banyak** dari persona, dan terlihat
membajak "Kamu bisa apa?" di v2 maupun v3.

**Jangan pakai val loss untuk menilai ini.** 1,7485 lawan 1,4541 diukur pada
val set berbeda, dan val set-nya saling bocor 95%. Lihat bagian 7.

### v4 — perbesar persona (SIAP DILATIH)

Satu perubahan saja, supaya sebabnya jelas:

| | v3 | v4 |
|---|---|---|
| persona + batas (unik) | 161 | **352** |
| Pengulangan di campuran | 1× | **2×** |
| Instance | 161 | **704** |
| Lawan `tidak_tahu` (600) | kalah 1:3,7 | **menang 1,2:1** |

191 varian baru, bobot terbesar di pertanyaan identitas. `tidak_tahu`
**tidak** dimatikan — pemilik memilih mengubah satu hal saja.

```powershell
.venv\Scripts\python.exe scripts\train_sft.py --epochs 3 --sft-dir data/sft-v4 --out-dir checkpoints/wicara-56m-chat-v4
.venv\Scripts\python.exe scripts\chat.py --ckpt checkpoints/wicara-56m-chat-v4/best.pt --uji
```

Yang dilihat: `Siapa kamu?` dan `Kamu bisa apa?` harus menyebut Wicara.
`Ceritakan sesuatu.` harus tetap sebaik v3 — kalau memburuk, persona
mengambil terlalu banyak ruang.

### Belum dikerjakan, sengaja di luar lingkup

1. **Val split stabil** — sampai ini diperbaiki, tiap versi hanya bisa dinilai
   dari perilaku. Ini yang paling menghambat, lihat bagian 7.
2. **`tidak_tahu` → 0** — alasannya sama dengan `perkenalan`, tapi pemilik
   memilih menundanya.
3. **Penolakan pertanyaan pengetahuan** — `"Apa ibu kota Zimbabwe?"` masih
   dikarang. Butuh contoh berbentuk `"Apa ibu kota X?"` → `"Maaf, tidak tahu"`.
4. **Multi-turn** — datanya harus tulis tangan juga; lihat bagian 7.
5. **Jawaban `terbuka` yang bertanya balik** — sekitar 35 contoh menawarkan
   pilihan alih-alih langsung bercerita. Model menirunya lalu berhenti
   ("Baiklah, kita mulai."). Layak diganti jadi langsung bercerita.

---

## 7. Temuan penting — jangan diulang penelusurannya

### Nama karangan datang dari DUA lapisan sekaligus

Model menjawab `"Namaku Kittan"` / `"Aku Daphne"` — nama tokoh film.

Satu sumbernya kategori `perkenalan` di data SFT, yang menambang
`"Siapa namamu?"` → `"Namaku X"` dengan kuota **1.200 contoh**. Kuotanya
sekarang **0**, dikunci uji `test_kuota_perkenalan_tetap_nol`.

**Tapi itu ternyata bukan satu-satunya sumber.** Versi awal catatan ini
menyimpulkan penyebabnya murni data SFT. **Kesimpulan itu terlalu kuat, dan
v3 membuktikannya salah:** dengan `perkenalan` dihapus habis dan 138 contoh
persona ditambahkan, model tetap menjawab `"Aku Kawis!"` untuk "Siapa kamu?".

Jadi prior dari pretrain juga nyata. Menghapus kategori SFT yang bersaing
**perlu tapi tidak cukup** — persis pola yang sama dengan "aku tidak tahu"
di bawah.

**Pelajarannya:** untuk menimpa prior pretrain, contoh SFT harus cukup banyak,
bukan sekadar bebas dari saingan. 138 contoh (0,34%) tidak cukup; 239 contoh
`terbuka` (0,59%) cukup — tapi `terbuka` melawan prior yang jauh lebih lemah.

### "tidak tahu" berasal dari PRETRAIN, bukan data SFT

Model menjawab *"aku tidak tahu"* untuk ~26% prompt uji. Dua diagnosis awal
**salah** dan sudah dibuktikan salah:

| Dugaan | Hasil uji |
|---|---|
| Kategori `tidak_tahu` di data SFT terlalu besar | Dipangkas 3.000 → 600, hasilnya cuma 28% → **26%** |
| Suhu sampling terlalu rendah | 0,7 → 1,1 hanya menggeser 32% → **24%** |

**Penyebab sebenarnya:** model BASE sudah menjawab "Aku tidak tahu" untuk 3
dari 6 prompt terbuka — **sebelum SFT sama sekali**. Di korpus subtitle,
frasa itu ada di **0,44% seluruh baris** dan hadir di **14,3% blok**. Model
menyerapnya dari 484 juta token. 40 ribu contoh SFT tidak cukup menimpanya.

Data SFT sendiri bersih: hanya 592 dari 40.180 jawaban (1,5%) berupa "tidak
tahu", dan 583 di antaranya memang kategori `tidak_tahu`.

**Implikasi:** perbaikan lewat data SFT hanya bisa menambah pola BARU yang
lebih kuat, bukan menghapus pola lama. Menghapusnya butuh mengubah komposisi
korpus pretrain (subtitle 43% → ~25%) dan retrain 11 jam.

### Multi-turn: datanya tidak tersedia

Model kehilangan konteks di giliran ke-3 (giliran 1–2 baik). Semua contoh SFT
berbentuk satu giliran. Sumber multi-turn Indonesia yang otentik sudah
ditelusuri dan **semuanya buntu**:

| Sumber | Hasil |
|---|---|
| Tambang subtitle 4 giliran | ~14.000 rangkaian, sebagian besar rusak di paruh kedua |
| OpenAssistant OASST2 | **12 pesan** Indonesia dari 128.575 |
| Aya, Cendol | Seluruhnya satu giliran |

Satu-satunya jalan: tulis tangan.

### Penambangan pasangan subtitle: kedua sisi HARUS cocok pola

Versi pertama hanya mencocokkan baris pembuka, menghasilkan pasangan tak
nyambung (`"Terima kasih." → "Jangan hubungi kebun binatang."`). Baris
subtitle berurutan sering diucapkan penutur yang sama. Aturan sekarang di
`src/data/sft.py::POLA_PASANGAN` mensyaratkan **kedua sisi** cocok.

### Dedup di level yang salah lebih merusak daripada tidak dedup

Dedup per baris pada subtitle membuang 39,7% data — dan yang terbuang adalah
frasa percakapan paling umum (`terima kasih` 845×, `halo` 458×). Bahasa nyata
itu Zipfian. Dedup per **blok** hanya membuang 3,2%, dan itulah duplikasi
sungguhan.

### Val split TIDAK stabil antar rebuild — val loss antar versi tak bisa dibandingkan

`build_sft_data.py` memilih val dari 2% pertama sesudah `rng.shuffle(semua)`.
Panjang atau isi daftar berubah sedikit saja → permutasinya berubah total →
val set tersusun ulang. Akibatnya, terukur langsung:

```
val v2 yang ada di train v3 : 783 / 820  (95,5%)
val v3 yang ada di train v2 : 772 / 804  (96,0%)
bersih untuk keduanya       :  23
```

Dua akibatnya:

1. **Val loss antar versi tidak berarti.** v2 = 1,4541 dan v3 = 1,7485 diukur
   pada himpunan berbeda. Selisihnya bukan ukuran mana yang lebih baik.
2. **Evaluasi silang juga tidak sah.** Menguji v3 pada val v2 memberi 1,1898 —
   angka itu palsu, karena 95% isinya sudah dilatihkan ke v3.

**Perbaikannya:** tentukan split dari hash isi contoh, bukan dari posisi
sesudah shuffle. Dengan begitu keanggotaan val satu contoh tidak pernah
bergantung pada apa lagi yang ada di dataset. Belum dikerjakan.

Sampai itu diperbaiki, **nilai v3 dari perilaku, bukan dari loss.**

### Anggaran token inference memotong jawaban panjang

`chat.py` dulu default `--max-new-tokens 100`. Pada 4,26 karakter/token itu
cuma **±426 karakter**, sedangkan median jawaban wikihow di data SFT **632
karakter**. Jadi sebagian jawaban panjang terpotong di tengah kalimat —
bukan karena modelnya, tapi karena anggarannya.

Default sekarang **160** (±680 karakter). Kalau menilai panjang respons,
periksa dulu batas ini sebelum menyalahkan model atau datanya.

### Jebakan lain yang sudah ditangani

- **Spill VRAM di Windows tidak melempar error** — PyTorch bisa mengalokasikan
  7,1 GB di GPU 6,0 GB; Windows meluapkannya ke RAM dan throughput anjlok 4×
  diam-diam. Batas aman praktis ~5,5 GB.
- **Biaya VRAM logits besar** — di model kecil ber-vocab besar, tensor logits +
  cross-entropy memakan hampir sebanyak seluruh lapisan Transformer.
- **FlashAttention tidak ada di wheel PyTorch Windows** — `mem_efficient`
  dipakai, dan itu cukup.
- **memmap mengunci berkas di Windows** — `TokenDataset.close()` wajib
  dipanggil sebelum menimpa `.bin`.
- **`loss awal = ln(vocab)` adalah alat uji** — loss di bawah itu berarti
  kebocoran label.
- **Cakupan sampling** — pengambilan acak dengan pengembalian hanya mencakup
  63% korpus. `TokenDataset` sekarang memakai iterasi epoch berurutan yang
  diacak (cakupan 100%).

---

## 8. Struktur kode

```
src/
  model/      rmsnorm · rope · attention (GQA+KV-cache) · ffn · transformer · config
  data/       sources · readers · clean · dedup · sft · sft_dataset
  train/      trainer · data (memmap) · lr_schedule · checkpoint · synthetic · config
  tokenizer/  bpe · chat_template
  infer/      generate
scripts/      check_env · download_corpus · verify_corpus · clean_corpus ·
              demo_cleaning · train_tokenizer · pack_tokens · build_sft_data ·
              bench_model · train · train_sft · generate · chat
tests/        178 uji
notes/        catatan per fase + glosarium istilah LLM (PDF) + HANDOFF.md
data/sft/     tulis_tangan.jsonl (400 contoh, IKUT TER-COMMIT) ·
              train/val.jsonl.gz (v2, diabaikan git) · stats.json
```

`data/sft/tulis_tangan.jsonl` sengaja JSONL polos, bukan `.jsonl.gz` — selain
supaya terbaca di `git diff`, `.gitignore` memang mengabaikan `*.jsonl.gz`
sehingga versi gzip tidak akan pernah ter-commit.

**Chat template** (`src/tokenizer/chat_template.py`) — 32 special token di ID
0–31, dikunci dan **tidak boleh diubah** (ID-nya tertanam di `.bin` dan
checkpoint):

```
<|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|eos|>
```

**Loss masking** (`src/data/sft_dataset.py`) — loss hanya pada token jawaban
asisten. Verifikasi lewat `periksa_masking()` yang mendekode ulang posisi
yang dilatih. Sudah dikunci 21 uji.

---

## 9. Preferensi pemilik

- Semua dokumentasi, komentar kode, dan pesan CLI dalam **bahasa Indonesia**
- **Data asli, bukan sintetis** — berlaku untuk **korpus pretraining**.
  Untuk SFT, pemilik menyetujui 400 contoh tulis tangan (2026-08-23) setelah
  dijelaskan bahwa (a) sumber aslinya tidak hilang, (b) porsinya ±1%, dan
  (c) berkasnya bisa dibaca seluruhnya sebelum dipakai. Tanyakan lagi kalau
  mau menambah lebih banyak — persetujuannya untuk jumlah segitu.
- Pemilik menjalankan sendiri training panjang lewat terminal; jangan
  dijalankan sebagai proses background yang terikat sesi
- Konfirmasi dulu sebelum memulai training panjang
- Repo bersifat open source (Apache-2.0); PDF tech report pihak ketiga di
  `tech_report_llm_openweight/` **tidak boleh** ikut ter-commit
