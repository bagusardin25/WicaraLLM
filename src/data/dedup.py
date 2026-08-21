"""Deduplikasi.

Langkah dengan rasio manfaat-per-usaha tertinggi di seluruh pipeline data.
Data kembar membuat model MENGHAFAL alih-alih belajar, dan membuat val loss
berbohong (potongan yang sama muncul di latih dan validasi).

Dua alat, untuk dua skala berbeda:

    ExactDedup  — set hash biasa. Tepat, tapi RAM tumbuh linear.
                  Dipakai di level DOKUMEN (jutaan item).

    BloomDedup  — bit array berukuran tetap. RAM konstan, tapi ada
                  false positive kecil. Dipakai di level BARIS subtitle
                  (ratusan juta item), di mana set biasa akan makan
                  berpuluh GB.

Untuk deduplikasi, false positive berarti membuang baris unik yang dikira
kembar. Itu kerugian yang bisa diterima: korpus kita berlebih, dan membuang
sedikit yang bersih jauh lebih murah daripada menyimpan yang kotor.
"""

import hashlib
import re

import numpy as np

SPASI = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_hash(text: str) -> str:
    """Bentuk kanonik untuk pembandingan.

    Huruf besar-kecil, tanda baca, dan spasi berlebih dibuang, supaya
    "Halo, apa kabar?" dan "halo apa kabar" dianggap kembar. Tanpa ini,
    perbedaan sepele membuat duplikat lolos.
    """
    text = text.lower()
    text = NON_ALNUM.sub(" ", text)
    return SPASI.sub(" ", text).strip()


def hash64(text: str) -> int:
    """Hash 64-bit. blake2b dipilih karena cepat dan bisa diminta pendek."""
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )


class ExactDedup:
    """Deduplikasi tepat berbasis set hash 64-bit."""

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.n_total = 0
        self.n_dup = 0

    def is_duplicate(self, text: str) -> bool:
        self.n_total += 1
        h = hash64(normalize_for_hash(text))
        if h in self.seen:
            self.n_dup += 1
            return True
        self.seen.add(h)
        return False

    @property
    def stats(self) -> dict:
        return {
            "total": self.n_total,
            "duplikat": self.n_dup,
            "unik": self.n_total - self.n_dup,
            "rasio_duplikat": self.n_dup / max(self.n_total, 1),
        }


class BloomDedup:
    """Bloom filter berbasis numpy — RAM tetap, untuk item sangat banyak.

    Bloom filter menyimpan keanggotaan dalam bit array. Tiap item menyalakan
    k bit pada posisi berbeda. Kalau semua k bit sudah menyala, item DIANGGAP
    pernah dilihat.

    Sifatnya: tidak pernah bilang "belum pernah" untuk item yang sudah ada
    (tidak ada false negative), tapi bisa keliru bilang "sudah ada" untuk
    item baru (false positive), dengan peluang yang bisa dihitung.

    Untuk 150 juta baris subtitle dengan ~1% false positive dibutuhkan
    sekitar 180 MB. Set Python untuk jumlah yang sama butuh belasan GB.
    """

    def __init__(self, expected_items: int = 150_000_000, fp_rate: float = 0.01):
        # Ukuran optimal: m = -n*ln(p) / (ln2)^2 ;  k = (m/n)*ln2
        m = int(-expected_items * np.log(fp_rate) / (np.log(2) ** 2))
        self.n_bits = max(1 << 20, 1 << int(np.ceil(np.log2(m))))
        self.k = max(1, min(6, int(round((self.n_bits / expected_items) * np.log(2)))))
        self.bits = np.zeros(self.n_bits // 8, dtype=np.uint8)
        self.mask = self.n_bits - 1
        self.n_total = 0
        self.n_dup = 0

    def _positions(self, text: str) -> list[int]:
        d = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:16], "big") | 1  # ganjil, agar menyebar
        # Kirsch-Mitzenmacher: k hash cukup diturunkan dari dua hash.
        return [(h1 + i * h2) & self.mask for i in range(self.k)]

    def is_duplicate(self, text: str) -> bool:
        self.n_total += 1
        pos = self._positions(text)
        idx = np.array([p >> 3 for p in pos], dtype=np.int64)
        bit = np.array([1 << (p & 7) for p in pos], dtype=np.uint8)

        if bool(np.all((self.bits[idx] & bit) != 0)):
            self.n_dup += 1
            return True

        for i, b in zip(idx, bit):
            self.bits[i] |= b
        return False

    @property
    def stats(self) -> dict:
        return {
            "total": self.n_total,
            "duplikat": self.n_dup,
            "unik": self.n_total - self.n_dup,
            "rasio_duplikat": self.n_dup / max(self.n_total, 1),
            "bit_mb": round(self.bits.nbytes / 1024**2, 2),
            "k_hash": self.k,
        }
