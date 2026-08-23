"""Bangun data SFT percakapan dari sumber yang sudah ada.

Modul ini menambang dari data NYATA. Ada satu sumber lagi yang TIDAK ditambang
dan tidak diurus di sini: 400 contoh tulis tangan di data/sft/tulis_tangan.jsonl
(sekitar 1% dari total), untuk dua pola yang tidak ada di korpus mana pun —
persona dan permintaan terbuka tanpa topik. Alasannya di
scripts/build_sft_data.py::kumpulkan_tulis_tangan.

Tiga sumber tambang, tiga peran berbeda:

    OpenSubtitles  -> sapaan dan basa-basi. Ditambang sebagai pasangan giliran
                      nyata dari dialog film. Ini satu-satunya sumber yang
                      punya "halo" -> "halo juga"; Aya dan Cendol seluruhnya
                      dataset instruksi dan sama sekali tidak memuat sapaan.

    Aya            -> mengikuti instruksi. HANYA subset SODA-inst dan Dolly-v2.

    Cendol         -> instruksi prosedural dari wikihow.

Kategori `perkenalan` di POLA_PASANGAN masih didefinisikan, tapi kuotanya
sudah disetel 0 di build_sft_data.py — pasangan "Siapa namamu?" -> "Namaku X"
mengajarkan model memakai nama tokoh film, dan itu bertentangan dengan
persona. Polanya dibiarkan ada supaya keputusan itu terlihat, bukan hilang
diam-diam.

Yang SENGAJA DIBUANG dan alasannya:

    HotpotQA, NQ-Open, Mintaka, Adversarial QA (531 ribu baris)
        Tanya-jawab faktual berjawaban pendek dan tegas. Model 56M tidak punya
        kapasitas menyimpan fakta-fakta itu, jadi yang ia pelajari bukan
        jawabannya melainkan KEBIASAAN menjawab yakin tanpa tahu. Justru
        memperparah halusinasi.

    Flan-CoT, PIQA, Flan-Coqa
        Soal pilihan ganda dan penalaran berantai. Bukan percakapan, dan
        Flan-Coqa keluarannya rusak (dialog banyak giliran digencet satu baris).

    indo_puisi
        Banyak keluaran berisi harfiah "nan" — nilai kosong yang bocor jadi teks.

    wikipedia_ban/gor/bjn/ace/bug
        Bahasa Bali, Gorontalo, Banjar, Aceh, Bugis. Bukan bahasa Indonesia.
"""

import re

# --- subset yang dipakai --------------------------------------------------

AYA_DIPAKAI = {"SODA-inst (T)", "Dolly-v2 (T)"}
CENDOL_DIPAKAI = {"wikihow"}

# --- batas kewajaran satu contoh percakapan -------------------------------

MIN_USER_CHARS = 3
MAX_USER_CHARS = 600
MIN_ASSIST_CHARS = 8
MAX_ASSIST_CHARS = 900


# ==========================================================================
# Penambangan sapaan dari subtitle
# ==========================================================================

# PASANGAN pola: baris pertama DAN baris kedua harus sama-sama cocok.
#
# Versi pertama aturan ini hanya mencocokkan baris pembuka, dengan asumsi
# baris berikutnya pasti tanggapannya. Asumsi itu SALAH di sebagian besar
# kasus — baris subtitle berurutan sering diucapkan penutur yang sama, atau
# terpisah pergantian adegan. Hasilnya bukan tanya-jawab:
#
#     "Halo, nak."      -> "Dia tak hanya takut, dia muak."
#     "Terima kasih."   -> "Jangan hubungi kebun binatang."
#
# Melatih model dengan pasangan seperti itu mengajarkan menjawab dengan
# kalimat tak nyambung — lebih merusak daripada tidak SFT sama sekali.
#
# Mensyaratkan KEDUA sisi cocok pola membuat hasilnya jauh lebih sedikit,
# tapi yang lolos adalah pasangan giliran (adjacency pair) yang sungguhan.
POLA_PASANGAN = {
    "sapaan": (
        re.compile(r"^(halo|hai|hei|hey|selamat (pagi|siang|sore|malam)|"
                   r"assalamualaikum)\b", re.IGNORECASE),
        re.compile(r"^(halo|hai|hei|hey|selamat (pagi|siang|sore|malam)|"
                   r"waalaikumsalam|ya[,.]|iya[,.])", re.IGNORECASE),
    ),
    "kabar": (
        re.compile(r"^(apa kabar|bagaimana kabar|gimana kabar|kau baik|"
                   r"kamu baik|kau tidak apa|kamu tidak apa|kau oke|"
                   r"kamu oke|kau baik-baik saja|kamu baik-baik saja)",
                   re.IGNORECASE),
        re.compile(r"^(baik|aku baik|saya baik|kabar baik|lumayan|"
                   r"tidak terlalu baik|tidak begitu baik|aku tidak apa|"
                   r"saya tidak apa|cukup baik|ya, aku|ya, saya)",
                   re.IGNORECASE),
    ),
    "perkenalan": (
        re.compile(r"^(siapa nama(mu|nya|kamu|anda)?|nama(mu| kamu| anda)?\?)",
                   re.IGNORECASE),
        re.compile(r"^(nama ?ku|nama saya|aku |saya |panggil (aku|saya))",
                   re.IGNORECASE),
    ),
    "terima_kasih": (
        re.compile(r"^(terima kasih|makasih)\b", re.IGNORECASE),
        re.compile(r"^(sama-sama|kembali|terima kasih kembali|"
                   r"tidak masalah|bukan masalah|dengan senang hati|"
                   r"tentu saja|santai saja|ya, tentu)", re.IGNORECASE),
    ),
    "perpisahan": (
        re.compile(r"^(sampai jumpa|selamat tinggal|sampai nanti|"
                   r"aku pergi dulu|saya pergi dulu)\b", re.IGNORECASE),
        re.compile(r"^(sampai jumpa|selamat tinggal|sampai nanti|"
                   r"hati-hati|jaga diri|dah|baiklah)\b", re.IGNORECASE),
    ),
    "maaf": (
        re.compile(r"^(maaf|maafkan (aku|saya))\b", re.IGNORECASE),
        re.compile(r"^(tidak apa|tak apa|tidak masalah|bukan salah|"
                   r"lupakan|sudahlah|tidak perlu minta maaf)",
                   re.IGNORECASE),
    ),
}

# Respons ketidaktahuan diperlakukan berbeda: ia sah sebagai jawaban untuk
# hampir semua PERTANYAAN, jadi cukup baris pertama berupa pertanyaan.
# Inilah pola yang ingin diajarkan sebagai ganti mengarang fakta.
POLA_PERTANYAAN = re.compile(
    r"^(apa|siapa|kenapa|mengapa|bagaimana|gimana|di ?mana|ke ?mana|kapan|"
    r"berapa|apakah)\b.*\?$", re.IGNORECASE)

POLA_TIDAK_TAHU = re.compile(
    r"^(aku tidak tahu|aku tak tahu|saya tidak tahu|entahlah|entah|"
    r"aku tidak yakin|saya tidak yakin|aku kurang tahu|tidak tahu|"
    r"mana ku tahu|aku juga tidak tahu)\b",
    re.IGNORECASE)

# Baris yang tidak layak jadi giliran mana pun.
POLA_BURUK = re.compile(
    r"(^\W+$|^[A-Z\s]{15,}$|www\.|http|\d{2}:\d{2}|"
    r"^(ya|tidak|oke|hmm|ah|oh|eh)[.!?]?$)", re.IGNORECASE)


def _layak(teks: str, min_c: int, max_c: int) -> bool:
    if not (min_c <= len(teks) <= max_c):
        return False
    if POLA_BURUK.search(teks):
        return False
    # Butuh minimal dua kata supaya bukan seruan tunggal.
    return len(re.findall(r"[a-zA-Z]+", teks)) >= 2


def tambang_pasangan_subtitle(blok: str) -> list[tuple[str, str, str]]:
    """Ambil pasangan (user, asisten, kategori) dari satu blok percakapan.

    Blok subtitle berisi baris berurutan dari satu adegan. Baris ke-n dan
    ke-(n+1) biasanya giliran dua penutur berbeda, jadi pasangan berurutan
    adalah perkiraan wajar untuk tanya-jawab.

    Perkiraan ini tidak selalu benar — kadang dua baris berurutan diucapkan
    penutur yang sama. Itu diterima: yang dipelajari model adalah pola
    "ada giliran, lalu ada tanggapan", dan sebagian derau tidak merusaknya.
    """
    baris = [b.strip() for b in blok.split("\n") if b.strip()]
    hasil = []

    for i in range(len(baris) - 1):
        a, b = baris[i], baris[i + 1]
        if not _layak(a, MIN_USER_CHARS, MAX_USER_CHARS):
            continue
        if not _layak(b, MIN_ASSIST_CHARS, MAX_ASSIST_CHARS):
            continue

        # Pertanyaan yang dijawab dengan ketidaktahuan. Sah untuk pertanyaan
        # apa pun, jadi cukup sisi pertama yang perlu berupa pertanyaan.
        if POLA_PERTANYAAN.match(a) and POLA_TIDAK_TAHU.match(b):
            hasil.append((a, b, "tidak_tahu"))
            continue

        # Sisanya: KEDUA sisi harus cocok pola pasangannya.
        for kategori, (pola_a, pola_b) in POLA_PASANGAN.items():
            if pola_a.match(a) and pola_b.match(b):
                hasil.append((a, b, kategori))
                break

    return hasil


# ==========================================================================
# Penyaringan Aya & Cendol
# ==========================================================================

def layak_instruksi(inp: str, out: str) -> bool:
    """Saringan untuk pasangan instruksi dari Aya dan Cendol."""
    inp, out = (inp or "").strip(), (out or "").strip()
    if not inp or not out:
        return False
    if not (MIN_USER_CHARS <= len(inp) <= MAX_USER_CHARS):
        return False
    if not (MIN_ASSIST_CHARS <= len(out) <= MAX_ASSIST_CHARS):
        return False

    # Nilai kosong yang bocor jadi teks — banyak ditemukan di indo_puisi.
    if out.lower() in {"nan", "none", "null", "-"}:
        return False

    # Sisa format pilihan ganda yang lolos dari penyaringan subset.
    if re.search(r"\b(pilihan|opsi)\s*:", inp, re.IGNORECASE):
        return False
    if re.match(r"^\s*\(?[A-D][\).]\s", out):
        return False

    return True
