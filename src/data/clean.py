"""Pembersihan teks korpus bahasa Indonesia.

Filosofinya: BUANG lebih agresif daripada yang terasa nyaman.

Model 32M kapasitasnya sangat terbatas. Setiap pola sampah yang lolos ke data
latih akan ikut dipelajari dan memakan kapasitas yang seharusnya dipakai untuk
bahasa. Membuang 30% korpus yang meragukan jauh lebih murah daripada melatih
model selama 9 jam di atas data kotor.
"""

import re
import unicodedata

# Rasio karakter yang masuk akal untuk teks Indonesia normal.
MIN_ALPHA_RATIO = 0.65      # minimal 65% huruf/spasi
MAX_DIGIT_RATIO = 0.20      # di atas ini biasanya tabel atau daftar angka
MAX_UPPER_RATIO = 0.30      # di atas ini biasanya judul atau teriakan
MAX_PUNCT_RATIO = 0.25

MIN_DOC_CHARS = 200         # dokumen pretrain
MIN_LINE_CHARS = 5          # baris subtitle
MIN_LINE_ALPHA = 3          # minimal huruf, untuk menyaring "!!!" dan "123"

# Penanda boilerplate web yang sering lolos ke korpus crawl.
BOILERPLATE = re.compile(
    r"(baca\s+juga|lihat\s+juga|artikel\s+terkait|berita\s+terkait|"
    r"copyright|hak\s+cipta|all\s+rights\s+reserved|"
    r"klik\s+di\s+sini|selengkapnya\s*[:»>]|"
    r"bagikan\s+ke|share\s+this|posted\s+by|diposting\s+oleh|"
    r"advertisement|iklan\s+scroll|scroll\s+untuk|"
    r"tags?\s*:|kategori\s*:|label\s*:)",
    re.IGNORECASE,
)

# Sisa markup dan artefak crawl.
MARKUP = re.compile(r"<[^>]{1,200}>|\[\[[^\]]{1,200}\]\]|\{\{[^}]{1,200}\}\}")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Artefak khas subtitle.
SUB_TAG = re.compile(r"<[/]?[iub]>|\{\\[^}]*\}", re.IGNORECASE)
SUB_KREDIT = re.compile(
    r"(subtitle|subtitles|diterjemahkan|penerjemah|translated|resync|"
    r"alih\s*bahasa|terjemahan|synced|corrected|opensubtitles|subscene|"
    r"idfl|lebahganteng|pein\s*akatsuki|sub\s*by|by\s+\w+\s*$)",
    re.IGNORECASE,
)
SUB_SPEAKER = re.compile(r"^\s*[-–—]\s*")
SUB_KURUNG = re.compile(r"[\(\[][^)\]]{0,60}[\)\]]")  # (mendesah), [MUSIK]

WHITESPACE = re.compile(r"[ \t ]+")
NEWLINES = re.compile(r"\n{3,}")
REPEAT_CHAR = re.compile(r"(.)\1{7,}")  # aaaaaaaaaa, ..........


def normalize(text: str) -> str:
    """Normalisasi Unicode dan spasi.

    NFKC menyatukan bentuk yang terlihat sama tapi kode-nya beda — misalnya
    huruf lebar (ｈａｌｏ) jadi huruf biasa. Tanpa ini, tokenizer akan
    memboroskan slot vocab untuk varian penulisan yang sama.
    """
    text = unicodedata.normalize("NFKC", text)
    # Zero-width space diganti SPASI, bukan dihapus. Menghapusnya
    # merekatkan kata yang terpisah ("halo<zwsp>apa" -> "haloapa"),
    # dan kata gabungan palsu itu akan jadi token sendiri di tokenizer.
    text = text.replace("​", " ").replace("﻿", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Samakan tanda kutip tipografis agar tidak jadi token terpisah.
    for a, b in (("“", '"'), ("”", '"'),
                 ("‘", "'"), ("’", "'"),
                 ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    text = REPEAT_CHAR.sub(r"\1\1\1", text)
    text = WHITESPACE.sub(" ", text)
    text = NEWLINES.sub("\n\n", text)
    return text.strip()


def char_stats(text: str) -> dict[str, float]:
    if not text:
        return {"alpha": 0.0, "digit": 0.0, "upper": 0.0, "punct": 0.0}
    n = len(text)
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    digit = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    punct = sum(not (c.isalnum() or c.isspace()) for c in text)
    return {"alpha": alpha / n, "digit": digit / n,
            "upper": upper / n, "punct": punct / n}


def looks_indonesian(text: str) -> bool:
    """Deteksi bahasa berbasis kata fungsi.

    Kata fungsi (yang, dan, di, dari, untuk...) muncul di hampir semua kalimat
    Indonesia dan hampir tidak pernah di bahasa lain. Jauh lebih cepat daripada
    model deteksi bahasa, dan untuk memisahkan Indonesia dari Inggris di korpus
    campuran, akurasinya sudah memadai.

    Bahasa Melayu akan ikut lolos. Itu disengaja — keduanya sangat berdekatan
    dan teks Melayu tetap berguna untuk model ini.
    """
    KATA = {
        "yang", "dan", "di", "dari", "untuk", "dengan", "ini", "itu", "pada",
        "tidak", "ada", "adalah", "akan", "saya", "kamu", "dia", "kita",
        "mereka", "sudah", "bisa", "juga", "atau", "karena", "tapi", "ke",
        "aku", "nya", "apa", "kalau", "saja", "lebih", "seperti", "oleh",
    }
    kata = re.findall(r"[a-zA-Z]+", text.lower())
    if len(kata) < 8:
        return False
    cocok = sum(1 for w in kata if w in KATA)
    return cocok / len(kata) >= 0.06


class Alasan:
    """Kenapa sebuah teks ditolak. Dihitung per-alasan supaya kalau ada
    filter yang terlalu galak, langsung kelihatan dari statistiknya."""

    OK = "ok"
    KOSONG = "kosong"
    TERLALU_PENDEK = "terlalu_pendek"
    HURUF_SEDIKIT = "huruf_terlalu_sedikit"
    ANGKA_BANYAK = "angka_terlalu_banyak"
    KAPITAL_BANYAK = "kapital_terlalu_banyak"
    SIMBOL_BANYAK = "simbol_terlalu_banyak"
    BOILERPLATE = "boilerplate"
    BUKAN_INDONESIA = "bukan_indonesia"
    KREDIT_SUBTITLE = "kredit_subtitle"
    KURANG_HURUF = "huruf_terlalu_sedikit_di_baris"


def clean_document_verbose(
    text: str, *, min_chars: int = MIN_DOC_CHARS
) -> tuple[str | None, str]:
    """Bersihkan satu dokumen prosa, kembalikan (hasil, alasan).

    hasil None berarti dokumen dibuang; alasan menjelaskan filter mana
    yang menolaknya.
    """
    if not text:
        return None, Alasan.KOSONG

    text = normalize(text)
    text = MARKUP.sub(" ", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()

    if len(text) < min_chars:
        return None, Alasan.TERLALU_PENDEK

    s = char_stats(text)
    if s["alpha"] < MIN_ALPHA_RATIO:
        return None, Alasan.HURUF_SEDIKIT
    if s["digit"] > MAX_DIGIT_RATIO:
        return None, Alasan.ANGKA_BANYAK
    if s["upper"] > MAX_UPPER_RATIO:
        return None, Alasan.KAPITAL_BANYAK
    if s["punct"] > MAX_PUNCT_RATIO:
        return None, Alasan.SIMBOL_BANYAK

    baris = [ln for ln in text.split("\n") if ln.strip()]
    if baris:
        n_boiler = sum(1 for ln in baris if BOILERPLATE.search(ln))
        if n_boiler / len(baris) > 0.30:
            return None, Alasan.BOILERPLATE

    if not looks_indonesian(text):
        return None, Alasan.BUKAN_INDONESIA

    return text, Alasan.OK


def clean_document(text: str, *, min_chars: int = MIN_DOC_CHARS) -> str | None:
    """Bersihkan satu dokumen prosa. None berarti dokumen dibuang."""
    return clean_document_verbose(text, min_chars=min_chars)[0]


def clean_subtitle_line_verbose(line: str) -> tuple[str | None, str]:
    """Bersihkan satu baris subtitle, kembalikan (hasil, alasan).

    Subtitle butuh penanganan berbeda dari prosa: barisnya sangat pendek,
    penuh artefak format, dan sering berisi kredit penerjemah yang sama
    persis di ribuan berkas. Kredit itulah sumber duplikasi terbesar di
    korpus subtitle, dan kalau lolos, model akan hafal nama penerjemah.
    """
    if not line:
        return None, Alasan.KOSONG

    line = normalize(line)
    line = SUB_TAG.sub("", line)
    line = SUB_KURUNG.sub("", line)   # (mendesah), [MUSIK]
    line = SUB_SPEAKER.sub("", line)  # penanda ganti pembicara
    line = URL.sub("", line)
    line = WHITESPACE.sub(" ", line).strip()

    if len(line) < MIN_LINE_CHARS:
        return None, Alasan.TERLALU_PENDEK
    if SUB_KREDIT.search(line):
        return None, Alasan.KREDIT_SUBTITLE

    s = char_stats(line)
    if s["alpha"] < 0.60:
        return None, Alasan.HURUF_SEDIKIT
    if s["digit"] > 0.25:
        return None, Alasan.ANGKA_BANYAK
    if s["upper"] > 0.50:
        return None, Alasan.KAPITAL_BANYAK

    # Dulu di sini ada syarat minimal 3 kata. Itu membuang 27% baris
    # subtitle -- termasuk "Ada apa?" dan "Menjauh darinya!", yang justru
    # PERSIS ragam percakapan yang ingin dipelajari model. Yang tersisa
    # hanya syarat ada cukup huruf, untuk menyaring "!!!" dan "123".
    if sum(c.isalpha() for c in line) < MIN_LINE_ALPHA:
        return None, Alasan.KURANG_HURUF

    return line, Alasan.OK


def clean_subtitle_line(line: str) -> str | None:
    return clean_subtitle_line_verbose(line)[0]


def group_subtitle_lines(
    lines: list[str], target_chars: int = 1200
) -> list[str]:
    """Gabung baris subtitle menjadi blok percakapan.

    Satu baris subtitle terlalu pendek untuk jadi contoh training — model
    tidak akan pernah melihat konteks lebih dari satu kalimat, dan tidak
    belajar bahwa percakapan itu bergiliran.

    Menggabungkan beberapa baris berurutan menjadi satu blok membuat model
    melihat alur tanya-jawab. Inilah yang membuat korpus subtitle berharga
    untuk model percakapan.
    """
    blok, buf, n = [], [], 0
    for ln in lines:
        buf.append(ln)
        n += len(ln) + 1
        if n >= target_chars:
            blok.append("\n".join(buf))
            buf, n = [], 0
    if len(buf) >= 4:
        blok.append("\n".join(buf))
    return blok
