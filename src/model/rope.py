"""RoPE — Rotary Position Embedding.

Masalah yang dipecahkan: attention pada dasarnya buta urutan. Kalau tidak ada
informasi posisi, "anjing menggigit orang" dan "orang menggigit anjing" terlihat
identik bagi model, karena attention hanya menghitung kecocokan antar token
tanpa peduli siapa duluan.

Ide RoPE: PUTAR vektor query dan key sesuai posisinya. Vektor di posisi 5
diputar 5 satuan sudut, di posisi 12 diputar 12 satuan sudut.

Kenapa memutar berhasil: hasil perkalian titik antara dua vektor yang diputar
hanya bergantung pada SELISIH sudutnya. Jadi attention otomatis melihat jarak
relatif ("3 token sebelum saya"), bukan posisi absolut ("token ke-847"). Ini
yang membuat RoPE bisa diekstrapolasi ke teks lebih panjang dari saat training.

Tiap pasangan dimensi diputar dengan kecepatan berbeda: pasangan awal berputar
cepat (peka jarak dekat), pasangan akhir berputar lambat (peka jarak jauh).
Mirip jarum detik dan jarum jam pada satu jam yang sama.
"""

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hitung cos & sin sekali di awal, lalu dipakai ulang tiap forward pass.

    Ini murni fungsi dari posisi — tidak ada parameter yang dilatih di RoPE.

    Returns:
        cos, sin — keduanya (seq_len, head_dim), disimpan fp32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim harus genap untuk RoPE, dapat {head_dim}")

    # Kecepatan putar tiap pasangan dimensi: 1/theta^(2i/d).
    # Pasangan pertama berputar penuh, yang terakhir nyaris diam.
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim/2)

    # Digandakan supaya panjangnya cocok dengan head_dim penuh. Ini mengikuti
    # konvensi Llama/HuggingFace: dimensi dipasangkan (i, i + head_dim/2),
    # bukan (i, i+1) seperti di paper aslinya. Hasil matematisnya setara.
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pasangkan tiap dimensi i dengan i + head_dim/2, lalu putar 90 derajat.

    Rotasi 2D standar adalah (x, y) -> (-y, x). Fungsi ini melakukan itu
    untuk semua pasangan sekaligus.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Terapkan rotasi posisi ke tensor query atau key.

    Args:
        x: (batch, n_head, seq_len, head_dim)
        cos, sin: dari build_rope_cache
        offset: posisi awal. WAJIB diisi saat memakai KV-cache — token ke-51
            harus tahu bahwa dirinya di posisi 50, bukan posisi 0. Lupa
            mengisi offset adalah bug KV-cache paling umum, dan gejalanya
            halus: model terlihat normal tapi mengulang-ulang atau kehilangan
            konteks setelah beberapa token.

    Rumusnya: x_rotated = x * cos + rotate_half(x) * sin
    Ini persis rotasi 2D, dikerjakan paralel untuk semua pasangan dimensi.
    """
    seq_len = x.shape[-2]
    c = cos[offset : offset + seq_len].to(x.dtype)
    s = sin[offset : offset + seq_len].to(x.dtype)
    # (seq_len, head_dim) -> (1, 1, seq_len, head_dim) supaya broadcast
    c = c.unsqueeze(0).unsqueeze(0)
    s = s.unsqueeze(0).unsqueeze(0)
    return x * c + rotate_half(x) * s
