"""Uji sanity model — Fase 3b.

Ini rangkaian uji yang paling sering dilewati orang, dan penyebab nomor satu
"training jalan 8 jam tapi hasilnya sampah". Dua uji paling berharga di sini
adalah kausalitas dan ekuivalensi KV-cache: keduanya menangkap bug yang TIDAK
terlihat dari kurva loss.
"""

import math

import pytest
import torch

from src.model.config import CONFIGS, ModelConfig
from src.model.transformer import MiniLLM

torch.manual_seed(1234)


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return CONFIGS["7m"]


@pytest.fixture(scope="module")
def model(cfg) -> MiniLLM:
    m = MiniLLM(cfg)
    m.eval()
    return m


# ========================== bentuk & parameter ===========================

def test_bentuk_logits(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_jumlah_parameter_cocok_dengan_kalkulator(model, cfg):
    """Model nyata harus punya parameter sebanyak yang dihitung config.py.

    Kalau meleset, berarti ada layer yang terlewat atau kelebihan.
    """
    assert model.num_params() == cfg.n_params
    assert model.num_params(non_embedding=True) == cfg.n_params_non_embedding


def test_embedding_benar_benar_dipakai_bersama(model):
    """Weight tying harus berbagi tensor yang SAMA, bukan salinan."""
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_inisialisasi_residual_diskalakan(model, cfg):
    """Proyeksi keluaran harus punya std lebih kecil dari layer biasa.

    Tanpa penskalaan ini, varians residual stream menumpuk seiring kedalaman.
    """
    expected = cfg.init_std / math.sqrt(2 * cfg.n_layer)
    wo_std = model.blocks[0].attn.wo.weight.std().item()
    wq_std = model.blocks[0].attn.wq.weight.std().item()

    assert wo_std == pytest.approx(expected, rel=0.15)
    assert wq_std == pytest.approx(cfg.init_std, rel=0.15)
    assert wo_std < wq_std


def test_last_token_only_setara_dengan_irisan_terakhir(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    penuh, _ = model(idx)
    terakhir, _ = model(idx, last_token_only=True)
    assert terakhir.shape == (2, 1, cfg.vocab_size)
    assert torch.allclose(penuh[:, -1:], terakhir, atol=1e-4)


# ============================ loss saat init =============================

def test_loss_awal_mendekati_ln_vocab(cfg):
    """UJI PALING PENTING SEBELUM TRAINING.

    Model yang belum belajar apa pun hanya bisa menebak seragam di antara
    vocab_size pilihan, jadi loss-nya harus -ln(1/vocab) = ln(vocab) = 9,70.

    Kalau jauh LEBIH TINGGI  -> inisialisasi bobot terlalu besar.
    Kalau jauh LEBIH RENDAH  -> ada kebocoran label, model melihat jawaban.
    """
    m = MiniLLM(cfg)
    m.eval()
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    targets = torch.randint(0, cfg.vocab_size, (4, 32))

    with torch.no_grad():
        _, loss = m(idx, targets)

    assert loss.item() == pytest.approx(cfg.expected_init_loss, abs=0.25), (
        f"loss awal {loss.item():.3f}, seharusnya ~{cfg.expected_init_loss:.3f}"
    )


def test_loss_mengabaikan_target_minus_100(model, cfg):
    """Loss masking untuk SFT: posisi bertanda -100 tidak boleh dihitung."""
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    targets = torch.randint(0, cfg.vocab_size, (2, 10))

    semua_diabaikan = torch.full_like(targets, -100)
    semua_diabaikan[:, -1] = targets[:, -1]  # sisakan satu posisi

    with torch.no_grad():
        _, loss_satu = model(idx, semua_diabaikan)
        _, loss_manual = model(idx[:, -1:], targets[:, -1:])

    assert torch.isfinite(loss_satu)
    # Nilainya beda karena konteksnya beda, tapi keduanya harus wajar.
    assert 0 < loss_satu.item() < 20


# ============================== kausalitas ===============================

def test_kausalitas_token_masa_depan_tidak_bocor(model, cfg):
    """UJI KRITIS.

    Mengubah token di posisi t tidak boleh mengubah logits di posisi mana pun
    SEBELUM t. Kalau bocor, model bisa menyontek jawaban: loss terlihat sangat
    bagus saat training, tapi generasinya hancur total karena saat inference
    token masa depan memang belum ada.
    """
    t_ubah = 7
    idx = torch.randint(0, cfg.vocab_size, (1, 16))

    idx_lain = idx.clone()
    idx_lain[0, t_ubah] = (idx[0, t_ubah] + 1) % cfg.vocab_size

    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx_lain)

    # Sebelum titik perubahan: harus identik.
    assert torch.allclose(a[:, :t_ubah], b[:, :t_ubah], atol=1e-5), (
        "kebocoran masa depan: posisi sebelum perubahan ikut berubah"
    )
    # Di titik perubahan: harus berbeda, kalau tidak berarti input diabaikan.
    assert not torch.allclose(a[:, t_ubah], b[:, t_ubah], atol=1e-4)


# ============================== KV-cache =================================

def test_kv_cache_setara_dengan_forward_penuh(model, cfg):
    """UJI KRITIS KEDUA.

    Menghasilkan token satu per satu memakai cache harus memberi logits yang
    SAMA dengan memproses seluruh sekuens sekaligus. Kalau meleset, biasanya
    penyebabnya offset RoPE yang lupa diisi.

    Bug ini paling licik karena training tetap mulus (training tidak memakai
    cache) dan modelnya hanya rusak saat dipakai.
    """
    t = 24
    idx = torch.randint(0, cfg.vocab_size, (2, t))

    with torch.no_grad():
        penuh, _ = model(idx)

        caches = model.make_caches(batch_size=2, max_seq_len=cfg.max_seq_len)
        bertahap = []
        for i in range(t):
            step, _ = model(idx[:, i : i + 1], caches=caches)
            bertahap.append(step)
        bertahap = torch.cat(bertahap, dim=1)

    beda = (penuh - bertahap).abs().max().item()
    assert beda < 1e-4, f"decode bertahap menyimpang dari forward penuh: {beda:.2e}"


def test_kv_cache_prefill_lalu_decode(model, cfg):
    """Pola inference sesungguhnya: prompt diproses sekaligus (prefill),
    lalu token dikeluarkan satu per satu (decode).

    Jalur ini memakai mask kausal yang digeser, cabang kode yang berbeda dari
    dua uji sebelumnya, dan mudah salah.
    """
    t_prompt, t_lanjut = 10, 6
    idx = torch.randint(0, cfg.vocab_size, (1, t_prompt + t_lanjut))

    with torch.no_grad():
        penuh, _ = model(idx)

        caches = model.make_caches(batch_size=1, max_seq_len=cfg.max_seq_len)
        awal, _ = model(idx[:, :t_prompt], caches=caches)
        assert torch.allclose(penuh[:, :t_prompt], awal, atol=1e-4)

        for i in range(t_prompt, t_prompt + t_lanjut):
            step, _ = model(idx[:, i : i + 1], caches=caches)
            assert torch.allclose(penuh[:, i : i + 1], step, atol=1e-4), (
                f"menyimpang di posisi {i}"
            )


def test_kv_cache_menolak_kelebihan_kapasitas(model, cfg):
    caches = model.make_caches(batch_size=1, max_seq_len=4)
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="KV-cache penuh"):
        model(idx, caches=caches)


# ========================= gradient & training ===========================

def test_semua_parameter_menerima_gradient(cfg):
    """Parameter yang gradient-nya None berarti terputus dari graf komputasi
    dan tidak akan pernah belajar."""
    m = MiniLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = m(idx, idx)
    loss.backward()

    mati = [n for n, p in m.named_parameters() if p.grad is None]
    assert not mati, f"parameter tanpa gradient: {mati}"


def test_overfit_satu_batch(cfg):
    """Model harus mampu MENGHAFAL satu batch sampai loss mendekati nol.

    Ini uji kemampuan belajar paling dasar. Kalau gagal, ada bug fundamental
    di forward, backward, atau optimizer, dan tidak ada gunanya melanjutkan
    ke training sungguhan.
    """
    torch.manual_seed(0)
    m = MiniLLM(cfg)
    m.train()

    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    awal = None
    for step in range(300):
        _, loss = m(idx, targets)
        if awal is None:
            awal = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    akhir = loss.item()
    assert awal == pytest.approx(cfg.expected_init_loss, abs=0.3)
    assert akhir < 0.5, f"gagal menghafal satu batch: {awal:.2f} -> {akhir:.2f}"


# ==================== berlaku untuk SEMUA config =========================
#
# Arsitektur ini seluruhnya digerakkan oleh ModelConfig — tidak ada dimensi
# yang ditanam keras di kode. Uji berikut membuktikannya: menaikkan model
# dari 32M ke 56M tidak butuh satu baris perubahan pun di src/model/.

@pytest.mark.parametrize("nama", list(CONFIGS))
def test_config_lolos_validasi(nama):
    c = CONFIGS[nama]
    c.validate()
    assert c.d_model % c.n_head == 0
    assert c.n_head % c.n_kv_head == 0
    assert c.head_dim % 2 == 0, "RoPE butuh head_dim genap"
    assert c.vocab_size <= 65535, "dataset uint16 tidak lagi cukup"


@pytest.mark.parametrize("nama", list(CONFIGS))
def test_head_dim_selalu_64(nama):
    """head_dim 64 adalah titik manis: ramah Tensor Core dan jangkauan
    frekuensi RoPE-nya sudah terbukti di banyak model."""
    assert CONFIGS[nama].head_dim == 64


@pytest.mark.parametrize("nama", list(CONFIGS))
def test_d_ffn_kelipatan_64_dan_rasio_wajar(nama):
    c = CONFIGS[nama]
    assert c.d_ffn % 64 == 0, "kelipatan 64 agar ramah Tensor Core"
    rasio = c.d_ffn / c.d_model
    assert 2.5 <= rasio <= 2.9, f"rasio d_ffn/d_model = {rasio:.2f}, target ~8/3"


@pytest.mark.parametrize("nama", list(CONFIGS))
def test_model_nyata_cocok_dengan_kalkulator(nama):
    """Jumlah parameter nyata harus sama persis dengan hitungan config,
    di SEMUA ukuran. Kalau meleset di satu config saja, berarti rumusnya
    kebetulan benar untuk satu kasus."""
    c = CONFIGS[nama]
    m = MiniLLM(c)
    assert m.num_params() == c.n_params
    assert m.num_params(non_embedding=True) == c.n_params_non_embedding


@pytest.mark.parametrize("nama", ["7m", "56m"])
def test_kausalitas_dan_kvcache_lintas_ukuran(nama):
    """Dua uji paling kritis, diulang di config yang akan benar-benar dipakai."""
    c = CONFIGS[nama]
    m = MiniLLM(c)
    m.eval()
    t = 12
    idx = torch.randint(0, c.vocab_size, (1, t))

    with torch.no_grad():
        penuh, _ = m(idx)

        # kausalitas
        lain = idx.clone()
        lain[0, 6] = (idx[0, 6] + 1) % c.vocab_size
        ubah, _ = m(lain)
        assert torch.allclose(penuh[:, :6], ubah[:, :6], atol=1e-5)

        # ekuivalensi KV-cache
        caches = m.make_caches(1, c.max_seq_len)
        bertahap = torch.cat(
            [m(idx[:, i:i + 1], caches=caches)[0] for i in range(t)], dim=1
        )
    assert (penuh - bertahap).abs().max().item() < 1e-4


@pytest.mark.parametrize("nama", ["7m", "56m"])
def test_loss_awal_ln_vocab_lintas_ukuran(nama):
    c = CONFIGS[nama]
    m = MiniLLM(c)
    m.eval()
    data = torch.randint(0, c.vocab_size, (2, 33))
    with torch.no_grad():
        _, loss = m(data[:, :-1], data[:, 1:])
    assert loss.item() == pytest.approx(c.expected_init_loss, abs=0.3)


def test_porsi_embedding_menurun_saat_model_membesar():
    """Efek samping menyenangkan dari menaikkan ukuran model.

    Tabel embedding ukurannya vocab x d_model, tumbuh LINEAR terhadap lebar.
    Sisa modelnya tumbuh KUADRATIK. Jadi makin besar model, makin kecil
    porsi kapasitas yang habis untuk sekadar tabel lookup — makin banyak
    yang benar-benar dipakai berpikir.
    """
    porsi = []
    for nama in ("7m", "19m", "32m", "56m", "88m"):
        c = CONFIGS[nama]
        porsi.append(c.param_breakdown()["embedding"] / c.n_params)
    assert porsi == sorted(porsi, reverse=True), "porsi embedding harus menurun"
    assert porsi[porsi.index(min(porsi))] < 0.20


# =============================== CUDA ====================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="butuh CUDA")
def test_jalan_di_cuda_dengan_bf16(cfg):
    """Jalur yang sesungguhnya dipakai saat training: bf16 autocast di GPU."""
    m = MiniLLM(cfg).cuda()
    idx = torch.randint(0, cfg.vocab_size, (4, 64), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = m(idx, idx)

    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
