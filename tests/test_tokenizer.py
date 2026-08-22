"""Uji tokenizer BPE."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.tokenizer.bpe import (
    build_tokenizer,
    build_trainer,
    compression_ratio,
    load,
    verify_special_tokens,
)
from src.tokenizer.chat_template import (
    ALL_SPECIAL_TOKENS,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKEN_IDS,
    Message,
    render,
)

TOKENIZER_NYATA = (
    Path(__file__).resolve().parent.parent
    / "data" / "tokenizer" / "wicara-bpe-16k.json"
)

KALIMAT = [
    "Halo, apa kabar hari ini?",
    "Baik, terima kasih sudah bertanya kepada saya.",
    "Saya sedang belajar membuat model bahasa dari awal.",
    "Bahasa Indonesia adalah bahasa resmi Republik Indonesia.",
    "Dia mengerjakan tugasnya dengan sangat baik dan teliti.",
    "Kami akan pergi ke pasar untuk membeli buah-buahan segar.",
    "Mereka tidak tahu ke mana perginya anak itu kemarin sore.",
    "Perkembangan teknologi mengubah cara manusia berkomunikasi.",
    "Menurut saya, keputusan itu sudah tepat dan bijaksana.",
    "Kenapa kamu tidak memberitahuku lebih awal tentang hal ini?",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    """Tokenizer kecil yang dilatih dari teks inline."""
    t = build_tokenizer()
    # Size 600 untuk menampung special tokens dan abjad.
    t.train_from_iterator(KALIMAT * 200, trainer=build_trainer(600))
    return t


# ========================= ID special token ==============================

def test_special_token_menempati_id_paling_depan(tok):
    """ID special token harus menempati posisi terdepan."""
    verify_special_tokens(tok)
    assert tok.token_to_id("<|pad|>") == PAD_ID == 0
    assert tok.token_to_id("<|bos|>") == BOS_ID == 1
    assert tok.token_to_id("<|eos|>") == EOS_ID == 2


def test_semua_32_special_token_ada(tok):
    assert len(ALL_SPECIAL_TOKENS) == 32
    for token, id_harapan in SPECIAL_TOKEN_IDS.items():
        assert tok.token_to_id(token) == id_harapan


def test_slot_cadangan_tetap_terpesan(tok):
    """Slot cadangan token harus terpesan."""
    for i in range(24):
        assert tok.token_to_id(f"<|reserved_{i}|>") is not None


# ============================ round-trip =================================

@pytest.mark.parametrize("teks", KALIMAT)
def test_encode_decode_lossless(tok, teks):
    """Byte-level BPE harus mengembalikan teks persis seperti semula."""
    assert tok.decode(tok.encode(teks).ids) == teks


@pytest.mark.parametrize("teks", [
    "Emoji 🎉 dan simbol ∑ ≈ π",
    "Huruf Arab: بسم الله",
    "Aksara Jawa: ꦲꦏ꧀ꦱꦫ",
    "Campur 中文 dan русский",
    "Karakter rusak: �� biasa muncul dari OCR",
])
def test_tidak_ada_unk_untuk_teks_asing(tok, teks):
    """Byte-level BPE seharusnya tidak memiliki token <unk>."""
    assert tok.decode(tok.encode(teks).ids) == teks


def test_spasi_awal_membedakan_token(tok):
    """" makan" dan "makan" harus ter-encode sebagai token berbeda."""
    a = tok.encode("makan").ids
    b = tok.encode(" makan").ids
    assert a != b
    assert tok.decode(b) == " makan"


# ============================== angka ====================================

def test_angka_dipotong_maksimal_tiga_digit(tok):
    """Angka harus dipotong maksimal tiga digit."""
    ids = tok.encode("1234567").ids
    assert tok.decode(ids) == "1234567"
    # 7 digit harus pecah jadi minimal 3 potong (3+3+1).
    assert len(ids) >= 3


# =========================== chat template ===============================

def test_chat_template_ter_encode_sebagai_special_token(tok):
    """Penanda peran harus ter-encode sebagai satu special token."""
    teks = render([Message("user", "halo")], add_generation_prompt=True)
    ids = tok.encode(teks).ids

    assert SPECIAL_TOKEN_IDS["<|bos|>"] in ids
    assert SPECIAL_TOKEN_IDS["<|user|>"] in ids
    assert SPECIAL_TOKEN_IDS["<|end|>"] in ids
    assert ids[-1] == SPECIAL_TOKEN_IDS["<|assistant|>"]


def test_chat_template_round_trip(tok):
    teks = render([Message("user", "apa kabar?"),
                   Message("assistant", "Baik, terima kasih!")])
    assert tok.decode(tok.encode(teks).ids, skip_special_tokens=False) == teks


# ============================== kompresi =================================

def test_rasio_kompresi_terhitung_benar(tok):
    r = compression_ratio(tok, KALIMAT)
    assert r["karakter"] == sum(len(k) for k in KALIMAT)
    assert r["token"] > 0
    assert r["char_per_token"] == pytest.approx(
        r["karakter"] / r["token"], rel=1e-9
    )
    # Uji validitas rumus rasio kompresi.
    assert r["char_per_token"] > 1.0


# =========================== batas uint16 ================================

def test_vocab_muat_di_uint16():
    """Ukuran vocab maksimal 65.535 untuk format uint16."""
    from src.tokenizer.bpe import VOCAB_SIZE
    assert VOCAB_SIZE <= 65_535


# ====================== tokenizer sungguhan (opsional) ===================

@pytest.mark.skipif(not TOKENIZER_NYATA.exists(),
                    reason="tokenizer belum dilatih (butuh korpus)")
class TestTokenizerNyata:
    """Uji terhadap tokenizer 16k sungguhan."""

    @staticmethod
    @pytest.fixture(scope="class")
    def nyata() -> Tokenizer:
        return load(TOKENIZER_NYATA)

    def test_ukuran_vocab_16384(self, nyata):
        assert nyata.get_vocab_size() == 16_384

    def test_id_special_token_benar(self, nyata):
        verify_special_tokens(nyata)

    def test_kompresi_korpus_memadai(self):
        """Rasio kompresi korpus harus berada di rentang wajar (3.5 - 5.0)."""
        import json

        meta = json.loads(
            TOKENIZER_NYATA.with_name("tokenizer_meta.json")
            .read_text(encoding="utf-8")
        )
        cpt = meta["char_per_token"]
        assert 3.5 < cpt < 5.0, f"rasio korpus {cpt:.2f} di luar rentang wajar"

    def test_kalimat_baku_lebih_efisien_daripada_rata_rata(self, nyata):
        """Rasio kompresi kalimat baku harus lebih tinggi dari rata-rata korpus."""
        import json

        meta = json.loads(
            TOKENIZER_NYATA.with_name("tokenizer_meta.json")
            .read_text(encoding="utf-8")
        )
        baku = compression_ratio(nyata, KALIMAT)["char_per_token"]
        assert baku > meta["char_per_token"]
        assert baku < 8.0, f"rasio {baku:.2f} terlalu tinggi, cek kebocoran"

    def test_kata_umum_jadi_satu_token(self, nyata):
        """Kata paling sering dalam bahasa Indonesia seharusnya tidak pecah."""
        for kata in (" yang", " dan", " tidak", " saya", " dengan", " untuk"):
            ids = nyata.encode(kata).ids
            assert len(ids) == 1, f"{kata!r} pecah jadi {len(ids)} token"

    @pytest.mark.parametrize("teks", KALIMAT)
    def test_round_trip_lossless(self, nyata, teks):
        assert nyata.decode(nyata.encode(teks).ids) == teks
