"""Uji training loop — Fase 4."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.model.config import CONFIGS
from src.model.transformer import MiniLLM
from src.train.checkpoint import find_latest, load_checkpoint, save_checkpoint
from src.train.config import TrainConfig
from src.train.data import TokenDataset, write_tokens
from src.train.lr_schedule import get_lr
from src.train.synthetic import generate_bigram_corpus
from src.train.trainer import Trainer

torch.manual_seed(0)

CFG = CONFIGS["7m"]


# ============================ jadwal LR ==================================

SCHED = dict(lr=1e-3, min_lr=1e-4, warmup_steps=100, total_steps=1000)


def test_warmup_naik_linear_dari_nyaris_nol():
    assert get_lr(0, **SCHED) == pytest.approx(1e-3 / 100)
    assert get_lr(49, **SCHED) == pytest.approx(1e-3 * 0.50)
    assert get_lr(99, **SCHED) == pytest.approx(1e-3)


def test_puncak_tepat_di_akhir_warmup():
    puncak = max(get_lr(s, **SCHED) for s in range(1000))
    assert puncak == pytest.approx(1e-3)
    assert get_lr(99, **SCHED) == pytest.approx(puncak)


def test_cosine_menurun_monoton_setelah_warmup():
    nilai = [get_lr(s, **SCHED) for s in range(100, 1000)]
    assert all(a >= b for a, b in zip(nilai, nilai[1:]))


def test_berakhir_di_min_lr_dan_bertahan():
    assert get_lr(1000, **SCHED) == pytest.approx(1e-4)
    assert get_lr(5000, **SCHED) == pytest.approx(1e-4)  # tahan, tidak minus


def test_titik_tengah_cosine_di_separuh_jalan():
    """Di tengah jadwal, cosine tepat di titik separuh antara lr dan min_lr."""
    tengah = get_lr(100 + (1000 - 100) // 2, **SCHED)
    assert tengah == pytest.approx((1e-3 + 1e-4) / 2, rel=0.01)


# ============================== data =====================================

def test_target_digeser_tepat_satu_posisi(tmp_path):
    """y[t] harus sama dengan x[t+1]. Inti next-token prediction."""
    tokens = np.arange(5000, dtype=np.int64) % 1000
    path = tmp_path / "t.bin"
    write_tokens(path, tokens)

    ds = TokenDataset(path, seq_len=16)
    x, y = ds.get_batch(4, "cpu")

    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "target tidak digeser satu posisi"


def test_menolak_file_terlalu_pendek(tmp_path):
    path = tmp_path / "kecil.bin"
    write_tokens(path, np.arange(10, dtype=np.int64))
    with pytest.raises(ValueError, match="terlalu pendek"):
        TokenDataset(path, seq_len=512)


def test_menolak_token_melebihi_uint16(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        write_tokens(tmp_path / "x.bin", np.array([70000], dtype=np.int64))


def test_pesan_jelas_kalau_file_tidak_ada(tmp_path):
    with pytest.raises(FileNotFoundError, match="Fase 1"):
        TokenDataset(tmp_path / "hilang.bin", seq_len=16)


def test_close_melepaskan_kunci_file_windows(tmp_path):
    """Setelah close(), file token harus bisa ditimpa.

    Di Windows, file yang sedang ter-memory-map TIDAK BISA ditimpa atau
    dihapus. Tanpa close(), pipeline data Fase 1 yang menulis ulang
    train.bin akan gagal dengan OSError selama masih ada TokenDataset hidup.
    Bug ini tidak muncul di Linux, jadi mudah lolos kalau tidak diuji di sini.
    """
    path = tmp_path / "kunci.bin"
    write_tokens(path, np.arange(5000, dtype=np.int64) % 1000)

    ds = TokenDataset(path, seq_len=16)
    with pytest.raises(OSError):
        write_tokens(path, np.arange(5000, dtype=np.int64) % 500)

    ds.close()
    write_tokens(path, np.arange(5000, dtype=np.int64) % 500)  # sekarang boleh


def test_dataset_bisa_dipakai_sebagai_context_manager(tmp_path):
    path = tmp_path / "ctx.bin"
    write_tokens(path, np.arange(5000, dtype=np.int64) % 1000)

    with TokenDataset(path, seq_len=16) as ds:
        assert ds.n_tokens == 5000
    write_tokens(path, np.arange(5000, dtype=np.int64) % 500)  # tidak terkunci


# ====================== gradient accumulation ============================

def test_gradient_accumulation_setara_dengan_batch_besar():
    """KLAIM INTI FASE 4.

    Seluruh strategi VRAM proyek ini bergantung pada satu janji: menumpuk
    gradient dari 16 batch kecil memberi hasil yang SAMA dengan satu batch
    16x lebih besar. Kalau tidak setara, batch efektif 128 cuma ilusi.

    Kuncinya ada pada pembagian dengan jumlah akumulasi. Tanpa itu gradient
    jadi JUMLAH, bukan RATA-RATA, dan efektif 4x lebih besar.
    """
    torch.manual_seed(7)
    m = MiniLLM(CFG)
    data = torch.randint(0, CFG.vocab_size, (8, 33))
    x, y = data[:, :-1], data[:, 1:]

    # A. satu batch berisi 8 sekuens
    _, loss = m(x, y)
    loss.backward()
    besar = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)

    # B. 4 micro-batch berisi 2 sekuens, masing-masing dibagi 4
    n_micro = 4
    for i in range(n_micro):
        xs, ys = x[i * 2 : (i + 1) * 2], y[i * 2 : (i + 1) * 2]
        _, l = m(xs, ys)
        (l / n_micro).backward()
    akum = [p.grad.clone() for p in m.parameters()]

    beda = max((a - b).abs().max().item() for a, b in zip(besar, akum))
    assert beda < 1e-5, f"gradient tidak setara, selisih maksimum {beda:.2e}"


# ============================ checkpoint =================================

def _mini_trainer(tmp_path, **overrides) -> Trainer:
    # Tulis sekali saja. Trainer kedua di tmp_path yang sama harus memakai
    # ulang file yang ada -- menimpanya akan gagal selama trainer pertama
    # masih memegang memory map.
    if not (tmp_path / "tr.bin").exists():
        tokens = generate_bigram_corpus(60_000, n_active=256, branching=4, seed=1)
        write_tokens(tmp_path / "tr.bin", tokens[:50_000])
        write_tokens(tmp_path / "va.bin", tokens[50_000:])

    cfg = TrainConfig(
        train_bin=str(tmp_path / "tr.bin"),
        val_bin=str(tmp_path / "va.bin"),
        batch_size=4,
        grad_accum_steps=1,
        total_tokens=200_000,
        eval_interval=10_000,
        log_interval=10_000,
        sample_interval=10_000,
        checkpoint_interval_minutes=999,
        out_dir=str(tmp_path / "ckpt"),
        device="cpu",
        **overrides,
    )
    return Trainer(CFG, cfg)


def test_checkpoint_memulihkan_bobot_dan_optimizer(tmp_path):
    """Optimizer state WAJIB ikut tersimpan.

    Kalau hanya bobot yang dipulihkan, momentum AdamW kembali nol dan loss
    melonjak saat dilanjutkan — training seolah mundur ribuan langkah.
    """
    t = _mini_trainer(tmp_path)
    t.train(max_steps=5, verbose=False)
    path = t.save("uji.pt")

    bobot_asli = t.model.tok_emb.weight.clone()
    exp_avg_asli = t.optimizer.state_dict()["state"][0]["exp_avg"].clone()

    t2 = _mini_trainer(tmp_path)
    assert not torch.allclose(t2.model.tok_emb.weight, bobot_asli)

    meta = load_checkpoint(path, model=t2.model, optimizer=t2.optimizer,
                           device="cpu")

    assert torch.allclose(t2.model.tok_emb.weight, bobot_asli)
    assert torch.allclose(
        t2.optimizer.state_dict()["state"][0]["exp_avg"], exp_avg_asli
    )
    assert meta["step"] == 5
    assert meta["tokens_seen"] == t.tokens_seen


def test_resume_melanjutkan_dari_langkah_yang_benar(tmp_path):
    t = _mini_trainer(tmp_path)
    t.train(max_steps=6, verbose=False)
    t.save("step_6.pt")

    t2 = _mini_trainer(tmp_path)
    assert t2.resume() is True
    assert t2.step == 6
    assert t2.tokens_seen == t.tokens_seen

    hasil = t2.train(max_steps=9, verbose=False)
    assert hasil["step"] == 9


def test_resume_tanpa_checkpoint_mengembalikan_false(tmp_path):
    t = _mini_trainer(tmp_path)
    assert t.resume() is False
    assert t.step == 0


def test_penulisan_checkpoint_atomik(tmp_path):
    """Tidak boleh ada file .tmp tersisa setelah penyimpanan sukses."""
    t = _mini_trainer(tmp_path)
    t.save("a.pt")
    assert not list(t.out_dir.glob("*.tmp"))
    assert (t.out_dir / "a.pt").exists()


# ========================= weight decay grouping =========================

def test_norm_dan_embedding_dikecualikan_dari_weight_decay(tmp_path):
    t = _mini_trainer(tmp_path)
    grup_decay, grup_no = t.optimizer.param_groups

    assert grup_decay["weight_decay"] == 0.1
    assert grup_no["weight_decay"] == 0.0

    # Embedding (tied dengan lm_head) harus ada di grup tanpa decay.
    ptr_emb = t.model.tok_emb.weight.data_ptr()
    ptr_no_decay = {p.data_ptr() for p in grup_no["params"]}
    assert ptr_emb in ptr_no_decay

    # Semua gain RMSNorm 1D juga.
    for p in t.model.parameters():
        if p.dim() == 1:
            assert p.data_ptr() in ptr_no_decay


# ============================ end-to-end =================================

def test_belajar_sampai_mendekati_entropi_teoretis(tmp_path):
    """UJI TERPENTING FASE 4.

    Korpus sintetis punya entropi yang diketahui persis: ln(4) = 1,386.
    Model yang belajar benar harus bergerak dari ln(vocab)=9,70 turun MENUJU
    angka itu, dan berhenti di sekitarnya.

    Dua arah kegagalan yang sama pentingnya:
      * Loss macet di ~9,70   -> loop tidak belajar sama sekali
      * Loss jauh di bawah H  -> model menghafal urutan spesifik, atau
                                 data validasi bocor ke data latih
    """
    entropi = math.log(4)

    t = _mini_trainer(tmp_path, lr=3e-3)
    hasil = t.train(max_steps=400, verbose=False)

    val = hasil["val_loss"]
    assert val < 4.0, (
        f"loss {val:.3f} tidak turun cukup jauh dari 9,70 — loop tidak belajar"
    )
    assert val > entropi - 0.30, (
        f"loss {val:.3f} di bawah entropi teoretis {entropi:.3f} — "
        "model menghafal atau ada kebocoran data"
    )


def test_loss_awal_sesuai_ln_vocab_sebelum_belajar(tmp_path):
    t = _mini_trainer(tmp_path)
    awal = t.evaluate(n_batches=5)
    assert awal == pytest.approx(CFG.expected_init_loss, abs=0.3)


def test_evaluate_deterministik(tmp_path):
    """Dua panggilan berturut-turut harus memberi angka identik.

    Tanpa generator berbenih tetap, val loss bergoyang tiap evaluasi dan
    perubahan nyata tidak bisa dibedakan dari derau sampling.
    """
    t = _mini_trainer(tmp_path)
    assert t.evaluate(n_batches=4) == pytest.approx(t.evaluate(n_batches=4))


def test_konfigurasi_langkah_konsisten():
    cfg = TrainConfig(batch_size=8, grad_accum_steps=16, total_tokens=1_300_000_000)
    seq = 512
    assert cfg.tokens_per_step(seq) == 8 * 512 * 16  # 65.536
    assert cfg.total_steps(seq) == 1_300_000_000 // 65_536
    assert cfg.warmup_steps(seq) == int(0.02 * cfg.total_steps(seq))
    assert cfg.min_lr == pytest.approx(cfg.lr * 0.1)
