"""Uji SFT.

Fokus: loss masking. Kalau bocor ke teks user, training tetap terlihat
mulus. Ujinya mendekode posisi yang dilatih, bukan menghitung token.
"""

import ast
import gzip
import json
import re
import statistics
from pathlib import Path

import pytest
import torch

from src.data.sft_dataset import ABAIKAN, SFTDataset, encode_example, periksa_masking
from src.data.sft import layak_instruksi, tambang_pasangan_subtitle
from src.tokenizer.bpe import build_tokenizer, build_trainer
from src.tokenizer.chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    Message,
    PAD_ID,
    render,
)

KALIMAT = [
    "Halo, apa kabar hari ini?",
    "Baik, terima kasih sudah bertanya kepada saya.",
    "Saya sedang belajar membuat model bahasa dari awal.",
    "Nama saya Wicara dan saya asisten berbahasa Indonesia.",
    "Silakan bertanya apa saja yang ingin kamu ketahui.",
    "Maaf, saya tidak tahu jawabannya untuk pertanyaan itu.",
]


@pytest.fixture(scope="module")
def tok():
    t = build_tokenizer()
    t.train_from_iterator(KALIMAT * 200, trainer=build_trainer(800))
    return t


PERCAKAPAN = [
    Message("user", "Halo, apa kabar hari ini?"),
    Message("assistant", "Baik, terima kasih sudah bertanya kepada saya."),
]


# ============================== loss masking =============================

def test_hanya_teks_asisten_yang_dilatih(tok):
    """Posisi yang dilatih, didekode ulang, harus teks asisten — bukan user
    atau system prompt.
    """
    r = periksa_masking(tok, PERCAKAPAN, DEFAULT_SYSTEM_PROMPT)
    dilatih = r["dilatih"]

    assert "Baik, terima kasih sudah bertanya kepada saya." in dilatih
    assert "apa kabar hari ini" not in dilatih, "teks user bocor ke loss"
    assert "asisten AI berbahasa Indonesia" not in dilatih, "system prompt bocor"


def test_penanda_akhir_ikut_dilatih(tok):
    """<|end|> harus ikut dilatih, kalau tidak model tidak belajar berhenti."""
    r = periksa_masking(tok, PERCAKAPAN, DEFAULT_SYSTEM_PROMPT)
    assert "<|end|>" in r["dilatih"]


def test_semua_giliran_asisten_dilatih(tok):
    """Semua giliran asisten dilatih, bukan hanya yang terakhir."""
    msgs = [
        Message("user", "Halo, apa kabar hari ini?"),
        Message("assistant", "Baik, terima kasih sudah bertanya kepada saya."),
        Message("user", "Silakan bertanya apa saja yang ingin kamu ketahui."),
        Message("assistant", "Nama saya Wicara dan saya asisten berbahasa Indonesia."),
    ]
    dilatih = periksa_masking(tok, msgs, DEFAULT_SYSTEM_PROMPT)["dilatih"]
    assert "Baik, terima kasih" in dilatih
    assert "Nama saya Wicara" in dilatih
    assert "Silakan bertanya" not in dilatih, "giliran user kedua bocor"


def test_sebagian_besar_token_diabaikan(tok):
    """Porsi yang dilatih minoritas: sebagian besar percakapan adalah prompt."""
    r = periksa_masking(tok, PERCAKAPAN, DEFAULT_SYSTEM_PROMPT)
    assert 0.1 < r["rasio"] < 0.7, f"rasio {r['rasio']:.1%} mencurigakan"


def test_label_digeser_satu_posisi(tok):
    """labels[t] == token[t+1], bukan token[t]."""
    teks = render(PERCAKAPAN, system_prompt=DEFAULT_SYSTEM_PROMPT)
    ids = tok.encode(teks).ids
    input_ids, labels = encode_example(tok, PERCAKAPAN, 10_000,
                                       DEFAULT_SYSTEM_PROMPT)

    assert input_ids == ids[:-1]
    for t, lab in enumerate(labels):
        if lab != ABAIKAN:
            assert lab == ids[t + 1], f"label di posisi {t} tidak digeser"


def test_tanpa_system_prompt_juga_benar(tok):
    dilatih = periksa_masking(tok, PERCAKAPAN, None)["dilatih"]
    assert "Baik, terima kasih" in dilatih
    assert "apa kabar hari ini" not in dilatih


def test_contoh_terlalu_panjang_ditolak(tok):
    panjang = [Message("user", "Halo " * 500),
               Message("assistant", "Baik " * 500)]
    assert encode_example(tok, panjang, max_len=64) is None


# ============================== dataset ==================================

@pytest.fixture
def data_sft(tmp_path):
    path = tmp_path / "sft.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": KALIMAT[i % len(KALIMAT)]},
                    {"role": "assistant",
                     "content": KALIMAT[(i + 1) % len(KALIMAT)]},
                ],
                "sumber": "uji", "kategori": "uji",
            }, ensure_ascii=False) + "\n")
    return path


def test_dataset_memuat_dan_membatch(data_sft, tok):
    ds = SFTDataset(data_sft, tok, max_len=256)
    assert len(ds) > 0
    x, y = ds.get_batch(4, "cpu")
    assert x.shape == y.shape
    assert x.shape[0] == 4


def test_padding_diabaikan_di_loss(data_sft, tok):
    """Posisi pad wajib -100."""
    ds = SFTDataset(data_sft, tok, max_len=256)
    x, y = ds.get_batch(8, "cpu")
    pad = x == PAD_ID
    assert (y[pad] == ABAIKAN).all(), "ada posisi pad yang ikut dihitung loss"


def test_setiap_batch_punya_token_untuk_dilatih(data_sft, tok):
    """Batch tanpa label valid -> loss NaN."""
    ds = SFTDataset(data_sft, tok, max_len=256)
    for _ in range(5):
        _, y = ds.get_batch(4, "cpu")
        assert (y != ABAIKAN).any()


def test_epoch_mencakup_semua_contoh(data_sft, tok):
    ds = SFTDataset(data_sft, tok, max_len=256, seed=1)
    terlihat = set()
    for _ in range(len(ds)):
        terlihat.update(ds._ambil_indeks(1))
    assert len(terlihat) == len(ds)


def test_posisi_baca_bisa_dilanjutkan(data_sft, tok):
    ds = SFTDataset(data_sft, tok, max_len=256, seed=1)
    ds._ambil_indeks(10)
    state = ds.state_dict()
    lanjut = ds._ambil_indeks(5)

    ds2 = SFTDataset(data_sft, tok, max_len=256, seed=1)
    ds2.load_state_dict(state)
    assert ds2._ambil_indeks(5) == lanjut


def test_hitungan_token_dilatih_masuk_akal(data_sft, tok):
    ds = SFTDataset(data_sft, tok, max_len=256)
    assert 0 < ds.n_token_dilatih < sum(len(lab) for _, lab in ds.contoh)


# ==================== penyaringan & penambangan data =====================

def test_pasangan_subtitle_butuh_kedua_sisi_cocok():
    """Kedua sisi harus cocok. Versi pertama cuma mencocokkan pembuka."""
    blok = "Terima kasih.\nJangan hubungi kebun binatang."
    assert tambang_pasangan_subtitle(blok) == []

    blok_baik = "Terima kasih.\nSama-sama, senang bisa membantu."
    hasil = tambang_pasangan_subtitle(blok_baik)
    assert len(hasil) == 1 and hasil[0][2] == "terima_kasih"


@pytest.mark.parametrize("user,asisten,kategori", [
    ("Halo, selamat pagi.", "Halo juga, apa kabar?", "sapaan"),
    ("Kau baik-baik saja?", "Aku baik, terima kasih.", "kabar"),
    ("Siapa namamu?", "Namaku Anitha, senang berkenalan.", "perkenalan"),
    ("Maaf, aku terlambat.", "Tidak apa-apa, santai saja.", "maaf"),
    ("Di mana dia sekarang?", "Aku tidak tahu, sungguh.", "tidak_tahu"),
])
def test_kategori_pasangan_dikenali(user, asisten, kategori):
    hasil = tambang_pasangan_subtitle(f"{user}\n{asisten}")
    assert hasil, f"tidak tertambang: {user!r} -> {asisten!r}"
    assert hasil[0][2] == kategori


def test_nilai_kosong_yang_bocor_ditolak():
    """Cendol: keluaran harfiah "nan" lolos ke teks."""
    assert not layak_instruksi("Silakan ciptakan puisi yang indah", "nan")
    assert not layak_instruksi("Tulis sesuatu", "")


def test_pilihan_ganda_ditolak():
    """Pilihan ganda tidak lolos saringan."""
    assert not layak_instruksi(
        "Manakah yang benar? Pilihan: - A: ya - B: tidak", "A")


# Data tulis tangan. Disunting tangan, jadi bisa rusak tanpa ada yang
# menahan — JSON salah, kembar, atau jawaban yang pelan-pelan memendek.

REPO = Path(__file__).resolve().parent.parent
TULIS_TANGAN = REPO / "data" / "sft" / "tulis_tangan.jsonl"

# Nama diri orang pertama. "namanya" tidak ikut — di kategori terbuka
# muncul sah ("kucing yang dipanggil namanya").
POLA_NAMA_DIRI = re.compile(r"\bnama(ku|\s+(saya|aku))\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def tulis_tangan():
    if not TULIS_TANGAN.exists():
        pytest.skip(f"{TULIS_TANGAN.name} tidak ada")
    baris = TULIS_TANGAN.read_text(encoding="utf-8").splitlines()
    return [json.loads(b) for b in baris if b.strip()]


def _konstanta_skrip(nama: str):
    """Baca konstanta build_sft_data.py lewat AST.

    Jangan diimpor: skrip itu punya efek samping (stdout.reconfigure, pyarrow).
    """
    sumber = (REPO / "scripts" / "build_sft_data.py").read_text(encoding="utf-8")
    for simpul in ast.parse(sumber).body:
        if isinstance(simpul, ast.Assign):
            for target in simpul.targets:
                if isinstance(target, ast.Name) and target.id == nama:
                    return ast.literal_eval(simpul.value)
    raise AssertionError(f"{nama} tidak ada di build_sft_data.py")


def test_tulis_tangan_bentuknya_benar(tulis_tangan):
    assert len(tulis_tangan) >= 560, "contoh berkurang banyak dari 591"
    for d in tulis_tangan:
        msgs = d["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert d["sumber"] == "tulis_tangan"
        assert d["kategori"] in {"persona", "batas", "terbuka"}


def test_tulis_tangan_lolos_saringan_yang_sama(tulis_tangan):
    """Contoh tangan harus lolos saringan yang sama dengan jalur tambang."""
    for d in tulis_tangan:
        u, a = d["messages"][0]["content"], d["messages"][1]["content"]
        assert layak_instruksi(u, a), f"ditolak saringan: {u!r}"


def test_tulis_tangan_tidak_ada_kembar(tulis_tangan):
    """Prompt kembar menggeser bobot kategori tanpa terlihat di hitungan."""
    prompt = [d["messages"][0]["content"].lower() for d in tulis_tangan]
    kembar = {p for p in prompt if prompt.count(p) > 1}
    assert not kembar, f"prompt kembar: {sorted(kembar)[:5]}"


def test_persona_tidak_pernah_memakai_nama_lain(tulis_tangan):
    """Jawaban persona/batas yang menyebut nama diri harus Wicara."""
    for d in tulis_tangan:
        if d["kategori"] not in {"persona", "batas"}:
            continue
        a = d["messages"][1]["content"]
        if POLA_NAMA_DIRI.search(a):
            assert "Wicara" in a, f"mengaku nama tapi bukan Wicara: {a!r}"


def test_persona_benar_benar_mengajarkan_nama(tulis_tangan):
    n = sum(1 for d in tulis_tangan if "Wicara" in d["messages"][1]["content"])
    assert n >= 30, f"cuma {n} jawaban menyebut Wicara, terlalu sedikit"


def test_jawaban_terbuka_memang_panjang(tulis_tangan):
    """Median jawaban `terbuka` harus tetap panjang. Kalau turun, kategori
    ini sudah kehilangan tujuannya.
    """
    p = [len(d["messages"][1]["content"])
         for d in tulis_tangan if d["kategori"] == "terbuka"]
    assert statistics.median(p) >= 180, f"median terbuka cuma {statistics.median(p)}"
    assert min(p) >= 40


def test_jawaban_muat_di_anggaran_token_chat(tulis_tangan):
    """chat.py default 160 token (~680 char). Lebih panjang dari ini terpotong."""
    for d in tulis_tangan:
        a = d["messages"][1]["content"]
        assert len(a) <= 500, f"terlalu panjang ({len(a)}): {a[:60]}"


def test_kuota_perkenalan_tetap_nol():
    """perkenalan harus tetap 0. Menghidupkannya membatalkan data persona."""
    assert _konstanta_skrip("BATAS_KATEGORI")["perkenalan"] == 0


def test_persona_mengalahkan_tidak_tahu(tulis_tangan):
    """Pelajaran v3: 161 persona kalah dari 600 tidak_tahu, dan model
    menjawab "Saya tidak tahu" untuk "Kamu bisa apa?". Kalau perbandingan
    itu berbalik lagi, persona tenggelam lagi.
    """
    ulang_kat = _konstanta_skrip("ULANG_KATEGORI")
    n = sum(1 for d in tulis_tangan if d["kategori"] in ulang_kat)
    n *= _konstanta_skrip("ULANG_PERSONA")
    tidak_tahu = _konstanta_skrip("BATAS_KATEGORI")["tidak_tahu"]
    assert n > tidak_tahu, f"persona {n} <= tidak_tahu {tidak_tahu}"
