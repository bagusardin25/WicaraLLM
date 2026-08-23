"""Tambang data SFT dari korpus yang sudah ada.

Tiga sumber:
    OpenSubtitles  sapaan. Satu-satunya yang punya "halo" -> "halo juga";
                   Aya dan Cendol murni instruksi.
    Aya            SODA-inst dan Dolly-v2 saja.
    Cendol         wikihow.

Contoh tulis tangan (persona, permintaan terbuka) tidak diurus di sini;
lihat scripts/build_sft_data.py::kumpulkan_tulis_tangan.

POLA_PASANGAN["perkenalan"] masih didefinisikan, kuotanya 0 di
build_sft_data.py. Polanya dibiarkan supaya keputusan itu terlihat.

Yang dibuang:
    HotpotQA, NQ-Open, Mintaka, Adversarial QA (531 ribu baris)
        Tanya-jawab faktual pendek. 56M tidak menyimpan faktanya; yang
        tertinggal kebiasaan menjawab yakin.
    Flan-CoT, PIQA, Flan-Coqa
        Pilihan ganda / CoT. Flan-Coqa keluarannya rusak.
    indo_puisi
        Keluaran harfiah "nan".
    wikipedia_ban/gor/bjn/ace/bug
        Bukan bahasa Indonesia.
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

# Kedua sisi harus cocok. Versi pertama cuma mencocokkan pembuka —
# baris subtitle berurutan sering penutur yang sama:
#
#     "Halo, nak."    -> "Dia tak hanya takut, dia muak."
#     "Terima kasih." -> "Jangan hubungi kebun binatang."
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

# Ketidaktahuan: cukup sisi pertama berupa pertanyaan. "aku tidak tahu"
# sah hampir di semua pertanyaan, beda dari POLA_PASANGAN.
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
    """Pasangan (user, asisten, kategori) dari satu blok subtitle.

    Baris berurutan dianggap giliran dua penutur. Tidak selalu benar;
    POLA_PASANGAN menahan yang tidak nyambung.
    """
    baris = [b.strip() for b in blok.split("\n") if b.strip()]
    hasil = []

    for i in range(len(baris) - 1):
        a, b = baris[i], baris[i + 1]
        if not _layak(a, MIN_USER_CHARS, MAX_USER_CHARS):
            continue
        if not _layak(b, MIN_ASSIST_CHARS, MAX_ASSIST_CHARS):
            continue

        if POLA_PERTANYAAN.match(a) and POLA_TIDAK_TAHU.match(b):
            hasil.append((a, b, "tidak_tahu"))
            continue

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
