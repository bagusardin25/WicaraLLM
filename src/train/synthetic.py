"""Korpus sintetis dengan entropi yang diketahui persis.

Dipakai untuk memverifikasi training loop TANPA perlu data sungguhan.
Lihat scripts/make_synthetic_data.py untuk penjelasan lengkap dan CLI-nya.
"""

import numpy as np


def generate_bigram_corpus(
    n_tokens: int,
    n_active: int = 2048,
    branching: int = 8,
    seed: int = 42,
    n_chains: int = 2048,
) -> np.ndarray:
    """Rantai bigram dengan entropi kondisional tepat ln(branching).

    Tiap token punya tepat `branching` kemungkinan lanjutan yang seragam,
    jadi model yang belajar sempurna akan berhenti di loss = ln(branching).

    Dibuat sebagai banyak rantai independen yang dijalankan paralel lewat
    numpy, karena loop Python sepanjang puluhan juta token terlalu lambat.
    """
    rng = np.random.default_rng(seed)
    n_chains = min(n_chains, max(1, n_tokens // 2))

    # Tabel aturan: inilah "tata bahasa" yang harus ditemukan model.
    successors = rng.integers(0, n_active, size=(n_active, branching), dtype=np.int64)

    panjang = max(2, n_tokens // n_chains)
    keluaran = np.empty((n_chains, panjang), dtype=np.int64)
    keluaran[:, 0] = rng.integers(0, n_active, size=n_chains)

    for i in range(1, panjang):
        pilihan = rng.integers(0, branching, size=n_chains)
        keluaran[:, i] = successors[keluaran[:, i - 1], pilihan]

    return keluaran.reshape(-1)
