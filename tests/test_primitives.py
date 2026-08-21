"""Uji RMSNorm dan RoPE — dua primitif paling bawah."""

import torch

from src.model.rmsnorm import RMSNorm
from src.model.rope import apply_rope, build_rope_cache

torch.manual_seed(0)


# ============================== RMSNorm ==================================

def test_rmsnorm_menormalkan_ke_rms_satu():
    """Dengan gain=1, keluaran harus punya RMS ~1 di dimensi terakhir."""
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 37.0  # sengaja skalanya kacau
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_tahan_skala_input():
    """Input dikali 1000 harus menghasilkan keluaran yang sama.

    Inilah gunanya normalisasi: melindungi layer berikutnya dari ledakan skala.
    """
    norm = RMSNorm(32)
    x = torch.randn(2, 8, 32)
    assert torch.allclose(norm(x), norm(x * 1000.0), atol=1e-4)


def test_rmsnorm_mempertahankan_dtype_bf16():
    """Hitung internal fp32, tapi keluaran harus kembali ke dtype semula."""
    norm = RMSNorm(32)
    out = norm(torch.randn(2, 8, 32, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_rmsnorm_gain_berpengaruh():
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    base = norm(x)
    with torch.no_grad():
        norm.weight.mul_(2.0)
    assert torch.allclose(norm(x), base * 2.0, atol=1e-5)


# ================================ RoPE ===================================

def test_rope_mempertahankan_panjang_vektor():
    """Rotasi hanya memutar arah, tidak boleh mengubah panjang vektor."""
    cos, sin = build_rope_cache(seq_len=128, head_dim=64)
    x = torch.randn(2, 4, 128, 64)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_rope_posisi_nol_tidak_berubah():
    """Di posisi 0 sudut rotasinya nol, jadi vektor harus utuh."""
    cos, sin = build_rope_cache(seq_len=8, head_dim=32)
    x = torch.randn(1, 1, 8, 32)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out[0, 0, 0], x[0, 0, 0], atol=1e-5)


def test_rope_hanya_bergantung_pada_selisih_posisi():
    """UJI INTI RoPE.

    Ini sifat yang membuat RoPE bermakna. Ambil satu query dan satu key,
    tempatkan di posisi (m, n), lalu geser keduanya sejauh d ke (m+d, n+d).
    Hasil dot product-nya harus SAMA PERSIS, karena selisihnya tidak berubah.

    Kalau uji ini lolos, model belajar "3 token sebelum saya" dan bukan
    "token ke-847" — itulah alasan RoPE bisa menangani teks lebih panjang
    daripada yang pernah dilihatnya saat training.
    """
    head_dim = 64
    cos, sin = build_rope_cache(seq_len=256, head_dim=head_dim)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def skor(pos_q: int, pos_k: int) -> torch.Tensor:
        qr = apply_rope(q, cos, sin, offset=pos_q)
        kr = apply_rope(k, cos, sin, offset=pos_k)
        return (qr * kr).sum()

    # Selisih tetap 5, digeser ke mana-mana.
    acuan = skor(10, 5)
    for geser in (0, 1, 20, 100, 200):
        assert torch.allclose(skor(10 + geser, 5 + geser), acuan, atol=1e-4), (
            f"skor berubah saat digeser {geser} — RoPE tidak relatif"
        )

    # Kontrol negatif: selisih BERBEDA harus memberi skor berbeda,
    # kalau tidak, berarti RoPE tidak melakukan apa-apa.
    assert not torch.allclose(skor(10, 4), acuan, atol=1e-3)


def test_rope_offset_setara_dengan_mengiris_sekuens_panjang():
    """Verifikasi offset untuk KV-cache.

    Memproses token di posisi 50 satu per satu (offset=50) harus identik
    dengan memproses sekuens penuh lalu mengambil posisi ke-50. Kalau tidak
    sama, decode inkremental akan menyimpang dari forward penuh.
    """
    cos, sin = build_rope_cache(seq_len=64, head_dim=32)
    x_penuh = torch.randn(1, 2, 64, 32)

    penuh = apply_rope(x_penuh, cos, sin)
    sepotong = apply_rope(x_penuh[:, :, 50:51], cos, sin, offset=50)

    assert torch.allclose(penuh[:, :, 50:51], sepotong, atol=1e-5)


def test_rope_frekuensi_menurun_sepanjang_dimensi():
    """Pasangan dimensi awal berputar cepat, yang akhir lambat.

    Ini yang memberi model 'jarum detik' dan 'jarum jam' sekaligus:
    kepekaan terhadap jarak dekat sekaligus jarak jauh.
    """
    cos, _ = build_rope_cache(seq_len=2, head_dim=64)
    # Di posisi 1, cos(sudut) makin mendekati 1 seiring sudut mengecil.
    sudut_pertama = cos[1, 0]
    sudut_terakhir = cos[1, 31]
    assert sudut_terakhir > sudut_pertama
    assert sudut_terakhir > 0.999  # nyaris tidak berputar sama sekali
