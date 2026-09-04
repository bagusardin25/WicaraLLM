# Fase 7 — Inference Engine, CLI, & Guardrail

Tanggal: 2026-08-24

Dokumentasi implementasi engine inferensi mandiri, antarmuka terminal (CLI) untuk model base dan chat, serta modul pengaman (*guardrail*) deterministik.

---

## 1. Yang Kami Bangun

| File | Peran & Isi |
|---|---|
| `src/infer/generate.py` | Engine inferensi inti dengan KV-cache, sampling top-k/top-p, temperature, dan repetition penalty |
| `src/infer/pencegat.py` | Pencegat krisis keselamatan deterministik berbasis regex sebelum input masuk ke model |
| `scripts/generate.py` | CLI pengujian model base (`wicara-56m-base`) untuk pelanjutan teks |
| `scripts/chat.py` | CLI interaktif model percakapan (`wicara-56m-chat`) dengan manajemen riwayat percakapan |
| `tests/test_penjaga.py` | 28 uji verifikasi pencegat krisis dan kekebalan terhadap false-positive prompt jinak |

---

## 2. Arsitektur Engine Inferensi (`src/infer/generate.py`)

Mesin inferensi kami tulis murni di PyTorch tanpa runtime pihak ketiga:

1. **Decoding Efisien dengan KV-Cache:**
   * Menghasilkan token baru tidak menghitung ulang aktivasi token-token masa lalu.
   * `KVCache` di [src/model/attention.py](file:///d:/Project/llmgusz/src/model/attention.py) menyimpan Key dan Value dari langkah sebelumnya, sehingga biaya komputasi tiap token baru bernilai konstan $O(1)$ terhadap panjang sekuens saat ini.
2. **Penyaringan Logits (*Logits Filtering*):**
   * **Top-K (default 40):** Membatasi probabilitas hanya pada $K$ token paling masuk akal.
   * **Top-P / Nucleus (default 0.9):** Memotong ekor distribusi probabilitas kumulatif di atas 90%.
   * **Temperature (0.7–0.8):** Mengatur ketajaman distribusi probabilitas (0 = *greedy*, 0.7 = wajar & fokus, >1.0 = liar).
   * **Repetition Penalty (default 1.15):** Menghukum logit token yang sudah pernah muncul dalam sekuens agar model tidak terjebak mengulang frasa yang sama (*looping*).
3. **Penanganan Token Berhenti (*Stop Token*):**
   * Untuk `wicara-56m-base`: berhenti pada `<|eos|>`.
   * Untuk `wicara-56m-chat`: berhenti wajib pada `<|end|>` (ID 2). Tanpa penanda `<|end|>`, model chat akan terus mengarang giliran percakapan user berikutnya.

---

## 3. Manajemen Riwayat di CLI Chat (`scripts/chat.py`)

1. **Penggeseran Konteks Otomatis (*Sliding Context Window*):**
   * Karena jendela konteks model 56M adalah 512 token, riwayat percakapan yang panjang akan cepat penuh.
   * `chat.py` menghitung sisa anggaran: $\text{batas} = \text{max\_seq\_len} - \text{max\_new\_tokens}$.
   * Jika akumulasi token riwayat melebihi batas, giliran pesan terlama otomatis dibuang dari memori (`riwayat.pop(0)`) agar prompt tetap muat tanpa melempar runtime error OOM.
2. **Perintah Interaktif:**
   * `/baru`: Menghapus riwayat percakapan dan memulai sesi bersih baru.
   * `/suhu N`: Mengubah temperature secara dinamis saat mengobrol.
   * `/keluar`: Menutup sesi terminal.
3. **Mode Evaluasi Cepat:**
   * `--uji`: Menjalankan 12 prompt uji standar untuk verifikasi cepat responsivitas persona.
   * `--soal`: Menguji 50 prompt di luar data latih dari `data/eval/prompt_lihat.jsonl`.

---

## 4. Pengaman Deterministik: Pencegat Krisis (`src/infer/pencegat.py`)

Uji coba awal kami menemukan bahwa model 56M rentan merespons prompt bernada krisis (seperti ungkapan ingin mengakhiri hidup) dengan kalimat ceroboh yang terserap dari dialog film di OpenSubtitles (misalnya: *"Tidak ada yang bisa menghentikanku"*).

**Keputusan Rekayasa Kami:**
* Masalah keselamatan darurat tidak boleh diserahkan pada model parameter kecil (56M).
* Kami membangun pencegat regex deterministik di [src/infer/pencegat.py](file:///d:/Project/llmgusz/src/infer/pencegat.py) yang memotong input **sebelum** menyentuh model.
* Jika terdeteksi indikasi krisis, sistem langsung mengembalikan teks bantuan resmi (Hotline Sejiwa Kemenkes 119 dan Hotline LISA) tanpa memanggil `generate()`.
* Pencegat ini dikunci oleh **28 unit test** di `tests/test_penjaga.py`, termasuk memastikan kata biasa seperti *"mati lampu"* atau *"capek kerja"* tidak terpicu secara salah (*false-positive*).

---

## 5. Temuan Kritis di Lapangan

### A. Anggaran Token yang Memotong Jawaban
* Pada versi awal, `--max-new-tokens` disetel ke **100 token** ($\approx 426$ karakter).
* Data SFT `wikihow` memiliki median panjang jawaban **632 karakter**.
* Akibatnya, jawaban panjang model sering terpotong di tengah kalimat bukan karena modelnya rusak, melainkan karena anggaran token habis sebelum model sempat memproduksi `<|end|>`.
* **Solusi:** Default dinaikkan ke **160 token** ($\approx 680$ karakter).

### B. Sanitasi Token Khusus
* Pengguna bisa secara sengaja atau tidak sengaja memasukkan token penanda format seperti `<|user|>` atau `<|system|>`.
* Kami menambahkan fungsi `bersihkan_input()` di `chat_template.py` untuk membersihkan tag khusus dari input sebelum dibungkus ke dalam objek `Message`.

---

## 6. Status & Pekerjaan Lanjutan (Alasan Berstatus 🟨 di Roadmap)

Fase 7 berstatus kuning di [notes/HANDOFF.md](file:///d:/Project/llmgusz/notes/HANDOFF.md#L44) karena masih ada beberapa fitur lanjutan yang sengaja kami tunda demi memprioritaskan penyelesaian evaluasi perilaku (Fase 8):

1. **Streaming Output:** Mengeluarkan token satu demi satu secara real-time ke terminal/browser (saat ini teks baru dicetak setelah generasi selesai penuh).
2. **Kuantisasi Inferensi:** Uji kuantisasi INT8/INT4 (misal via `bitsandbytes` atau AWQ) untuk menurunkan beban memori inferensi ke bawah 1 GB.
3. **Serving API:** Endpoint web (FastAPI/Flask) untuk menyajikan model ke aplikasi frontend atau bot Telegram/Discord (sebagian sudah diwujudkan di Space demo Hugging Face).
