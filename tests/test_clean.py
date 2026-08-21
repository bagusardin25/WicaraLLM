"""Uji pembersihan dan deduplikasi — Fase 1.

Beberapa uji di sini mengunci pelajaran yang didapat dari menjalankan
peragaan di data nyata (scripts/demo_cleaning.py). Aturan yang pertama
kali saya tulis ternyata salah di tiga tempat, dan uji ini mencegahnya
kembali.
"""

import pytest

from src.data.clean import (
    Alasan,
    char_stats,
    clean_document,
    clean_document_verbose,
    clean_subtitle_line,
    clean_subtitle_line_verbose,
    group_subtitle_lines,
    looks_indonesian,
    normalize,
)
from src.data.dedup import BloomDedup, ExactDedup, normalize_for_hash


# ============================ normalisasi ================================

def test_huruf_lebar_jadi_biasa():
    assert normalize("Ｈａｌｏ") == "Halo"


def test_kutip_tipografis_diseragamkan():
    hasil = normalize("Dia bilang “halo” dan ‘pergi’")
    assert '"halo"' in hasil and "'pergi'" in hasil


def test_spasi_berlebih_diringkas():
    assert normalize("halo     apa      kabar") == "halo apa kabar"


def test_karakter_berulang_dipotong():
    assert normalize("haloooooooooooo") == "halooo"


def test_zero_width_space_jadi_spasi_bukan_dihapus():
    """Bug nyata yang tertangkap peragaan.

    Menghapus zero-width space merekatkan kata: "halo<zwsp>apa" jadi
    "haloapa". Kata gabungan palsu itu akan jadi token tersendiri di
    tokenizer dan tidak pernah muncul di teks normal.
    """
    hasil = normalize("halo​apa​kabar")
    assert hasil == "halo apa kabar"
    assert "haloapa" not in hasil


# ============================ deteksi bahasa =============================

def test_teks_indonesia_dikenali():
    assert looks_indonesian(
        "Saya sedang belajar membuat model bahasa dari awal dengan "
        "menggunakan data yang sudah dikumpulkan dari berbagai sumber."
    )


def test_teks_inggris_ditolak():
    assert not looks_indonesian(
        "I am currently learning how to build a language model from "
        "scratch using data collected from many different sources."
    )


def test_teks_terlalu_pendek_ditolak_untuk_deteksi():
    assert not looks_indonesian("halo apa kabar")


# ============================== dokumen ==================================

TEKS_BAIK = (
    "Bahasa Indonesia adalah bahasa resmi Republik Indonesia yang digunakan "
    "oleh lebih dari dua ratus juta penutur. Bahasa ini berkembang dari "
    "bahasa Melayu dan telah mengalami banyak perubahan sejak kemerdekaan. "
    "Saat ini bahasa Indonesia juga dipelajari di berbagai negara lain."
)


def test_dokumen_bagus_lolos():
    hasil, alasan = clean_document_verbose(TEKS_BAIK)
    assert alasan == Alasan.OK
    assert hasil is not None and len(hasil) > 200


def test_dokumen_pendek_ditolak():
    assert clean_document("Halo dunia.") is None


def test_dokumen_penuh_angka_ditolak():
    """Yang penting dokumennya dibuang; pintu penolakannya boleh mana saja.

    Teks "1 2 3 4 5" terjaring lebih dulu oleh rasio huruf (spasi dihitung
    huruf, digit tidak) sebelum sampai ke pemeriksaan angka. Menguji alasan
    yang terlalu spesifik membuat uji rapuh terhadap urutan filter.
    """
    hasil, alasan = clean_document_verbose("1 2 3 4 5 " * 60)
    assert hasil is None
    assert alasan in (Alasan.ANGKA_BANYAK, Alasan.HURUF_SEDIKIT,
                      Alasan.BUKAN_INDONESIA)


def test_url_dan_email_dibuang():
    teks = TEKS_BAIK + " Kunjungi https://contoh.com atau surel a@b.com"
    hasil = clean_document(teks)
    assert hasil is not None
    assert "https" not in hasil and "@b.com" not in hasil


def test_markup_html_dibuang():
    teks = "<p>" + TEKS_BAIK + "</p><div class='x'>lagi</div>"
    hasil = clean_document(teks)
    assert hasil is not None and "<" not in hasil


def test_alasan_penolakan_dilaporkan():
    """Statistik per-alasan yang membuat filter terlalu galak ketahuan."""
    _, alasan = clean_document_verbose("")
    assert alasan == Alasan.KOSONG
    _, alasan = clean_document_verbose("pendek")
    assert alasan == Alasan.TERLALU_PENDEK


# ============================== subtitle =================================

def test_penanda_pembicara_dibuang():
    assert clean_subtitle_line("-Apa yang terjadi?") == "Apa yang terjadi?"


def test_tag_format_dibuang():
    assert clean_subtitle_line("<i>Dia pergi sekarang</i>") == "Dia pergi sekarang"


def test_keterangan_kurung_dibuang():
    hasil = clean_subtitle_line("(mendesah) Aku tidak tahu lagi")
    assert hasil is not None and "mendesah" not in hasil


def test_kredit_penerjemah_dibuang():
    """Sumber duplikasi terbesar di korpus subtitle.

    Kredit identik muncul di ribuan berkas. Kalau lolos, model akan hafal
    nama penerjemah alih-alih belajar bahasa.
    """
    for kredit in (
        "Synced and corrected by albanda",
        "Diterjemahkan oleh Lebah Ganteng",
        "Subtitle by IDFL Subs Crew",
        "Alih bahasa: Pein Akatsuki",
    ):
        _, alasan = clean_subtitle_line_verbose(kredit)
        assert alasan == Alasan.KREDIT_SUBTITLE, f"lolos: {kredit!r}"


@pytest.mark.parametrize("baris", [
    "Ada apa?",
    "Menjauh darinya!",
    "Aku tak tahu.",
    "Ya, tentu saja.",
    "Kau baik-baik saja?",
])
def test_dialog_pendek_dipertahankan(baris):
    """Bug nyata yang tertangkap peragaan.

    Ambang panjang yang pertama saya tulis (12 karakter, minimal 3 kata)
    membuang 27% baris subtitle — termasuk contoh-contoh di atas, yang
    justru PERSIS ragam percakapan yang ingin dipelajari model. Untuk
    model percakapan, ucapan pendek adalah targetnya, bukan sampahnya.
    """
    assert clean_subtitle_line(baris) is not None, f"terbuang: {baris!r}"


@pytest.mark.parametrize("sampah", ["!!!", "123", "...", "?!?!"])
def test_sampah_pendek_tetap_dibuang(sampah):
    assert clean_subtitle_line(sampah) is None


# =========================== pengelompokan ===============================

def test_baris_digabung_jadi_blok():
    baris = [f"Ini kalimat percakapan nomor {i} yang cukup panjang."
             for i in range(60)]
    blok = group_subtitle_lines(baris, target_chars=400)
    assert len(blok) > 1
    assert all("\n" in b for b in blok)


def test_blok_menjaga_urutan_percakapan():
    """Urutan harus utuh — di situlah alur tanya-jawabnya."""
    baris = ["Apa kabar?", "Baik, terima kasih.", "Kamu sendiri?"]
    blok = group_subtitle_lines(baris * 20, target_chars=200)
    isi = blok[0].split("\n")
    assert isi[0] == "Apa kabar?"
    assert isi[1] == "Baik, terima kasih."


# ============================ deduplikasi ================================

def test_normalisasi_hash_mengabaikan_beda_sepele():
    assert normalize_for_hash("Halo, apa kabar?") == normalize_for_hash(
        "halo apa kabar"
    )
    assert normalize_for_hash("Aku  TIDAK   tahu!!") == normalize_for_hash(
        "aku tidak tahu"
    )


def test_exact_dedup_menangkap_kembar():
    d = ExactDedup()
    assert not d.is_duplicate("Halo apa kabar")
    assert d.is_duplicate("halo, APA kabar!")
    assert not d.is_duplicate("Selamat pagi")
    assert d.stats["unik"] == 2


def test_bloom_tidak_pernah_false_negative():
    """Sifat yang menentukan: Bloom filter boleh keliru bilang 'sudah ada',
    tapi TIDAK BOLEH keliru bilang 'belum pernah'. Kalau sampai terjadi,
    duplikat asli akan lolos."""
    b = BloomDedup(expected_items=10_000, fp_rate=0.01)
    teks = [f"kalimat percakapan nomor {i}" for i in range(2000)]
    for t in teks:
        b.is_duplicate(t)
    for t in teks:
        assert b.is_duplicate(t), f"false negative pada {t!r}"


def test_bloom_ram_tetap_walau_item_bertambah():
    kecil = BloomDedup(expected_items=1_000_000, fp_rate=0.01)
    besar = BloomDedup(expected_items=1_000_000, fp_rate=0.01)
    for i in range(5000):
        besar.is_duplicate(f"teks {i}")
    assert kecil.bits.nbytes == besar.bits.nbytes


def test_bloom_false_positive_dalam_batas():
    b = BloomDedup(expected_items=50_000, fp_rate=0.01)
    for i in range(20_000):
        b.is_duplicate(f"item latih {i}")

    salah = sum(1 for i in range(5_000) if b.is_duplicate(f"item baru {i}"))
    assert salah / 5_000 < 0.05, f"false positive {salah / 5_000:.1%} terlalu tinggi"


def test_dedup_blok_jauh_lebih_lunak_daripada_baris():
    """Pelajaran terpenting dari peragaan.

    Dedup per BARIS membuang frasa umum ("terima kasih", "baiklah") yang
    memang WAJAR sering muncul — bahasa nyata itu Zipfian. Dedup per BLOK
    hanya menangkap duplikasi sungguhan.
    """
    percakapan = ["Halo!", "Apa kabar?", "Baik, terima kasih.", "Sama-sama."]
    baris = percakapan * 50  # 50 percakapan berbeda memakai frasa yang sama

    per_baris = ExactDedup()
    dibuang_baris = sum(1 for b in baris if per_baris.is_duplicate(b))

    blok = group_subtitle_lines(baris, target_chars=80)
    per_blok = ExactDedup()
    dibuang_blok = sum(1 for b in blok if per_blok.is_duplicate(b))

    assert dibuang_baris / len(baris) > 0.9, "dedup baris membuang hampir semua"
    assert dibuang_blok / max(len(blok), 1) < dibuang_baris / len(baris)


# ============================== statistik ================================

def test_char_stats_masuk_akal():
    s = char_stats("Halo apa kabar")
    assert s["alpha"] > 0.9 and s["digit"] == 0.0
    s = char_stats("12345")
    assert s["digit"] == 1.0
