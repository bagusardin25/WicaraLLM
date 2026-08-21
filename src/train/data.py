"""Pembaca dataset token — memmap uint16, tanpa DataLoader.

Kenapa tidak memakai torch DataLoader:
    Di Windows, worker DataLoader memakai spawn (bukan fork), sehingga tiap
    worker mengimpor ulang seluruh modul dan biaya start-nya mahal. Untuk
    dataset yang cuma satu larik angka datar, mesin sebesar itu tidak perlu.

Pendekatan ala nanoGPT: seluruh korpus adalah SATU file biner berisi ID token
uint16 berurutan, dibaca lewat np.memmap. Sistem operasi yang mengurus caching
halaman, jadi file 800 MB tidak pernah dimuat penuh ke RAM.

Batch diambil dari posisi ACAK di dalam file, bukan berurutan. Untuk pretrain
ini justru lebih baik: tiap batch jadi campuran dokumen dari berbagai bagian
korpus, sehingga gradient-nya tidak bias ke satu topik.
"""

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"File token tidak ada: {self.path}\n"
                "Jalankan scripts/make_synthetic_data.py (untuk uji) atau "
                "pipeline Fase 1 (untuk data sungguhan)."
            )

        self.seq_len = seq_len
        # mode='r' = read-only, tidak menyalin apa pun ke RAM.
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

        # Butuh seq_len + 1 token: yang terakhir jadi target token sebelumnya.
        self.max_start = len(self.data) - seq_len - 1
        if self.max_start < 1:
            raise ValueError(
                f"{self.path.name} hanya {len(self.data):,} token, "
                f"terlalu pendek untuk seq_len={seq_len}."
            )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_tokens(self) -> int:
        return len(self.data)

    def close(self) -> None:
        """Lepaskan memory map.

        WAJIB di Windows: selama sebuah file masih ter-map, sistem operasi
        MELARANG file itu ditimpa atau dihapus. Tanpa ini, pipeline data yang
        menulis ulang train.bin akan gagal dengan OSError yang membingungkan
        selama masih ada TokenDataset hidup yang menunjuk ke sana.

        Di Linux hal ini tidak terjadi, jadi bug-nya mudah lolos kalau
        pengembangannya di Linux.
        """
        mm = getattr(self.data, "_mmap", None)
        if mm is not None:
            mm.close()
        self.data = None

    def __enter__(self) -> "TokenDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ambil satu batch acak.

        Returns:
            x: (batch, seq_len) input
            y: (batch, seq_len) target — SAMA dengan x tapi digeser satu posisi.

        Pergeseran satu posisi inilah inti next-token prediction: saat model
        melihat x[t], ia harus menebak y[t] yang sebenarnya adalah x[t+1].
        Lupa menggeser membuat loss awal turun di bawah ln(vocab) dan modelnya
        tidak belajar apa pun yang berguna.
        """
        ix = torch.randint(
            self.max_start, (batch_size,), generator=generator, device="cpu"
        )

        # astype(np.int64) wajib: PyTorch tidak punya tipe uint16, dan
        # F.cross_entropy menuntut target bertipe long.
        x = torch.stack(
            [
                torch.from_numpy(
                    self.data[i : i + self.seq_len].astype(np.int64)
                )
                for i in ix.tolist()
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64)
                )
                for i in ix.tolist()
            ]
        )

        if str(device).startswith("cuda"):
            # pin_memory + non_blocking membuat transfer CPU->GPU bisa
            # tumpang tindih dengan komputasi, bukan menunggu bergantian.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y


def write_tokens(path: str | Path, tokens: np.ndarray) -> None:
    """Tulis larik token ke file biner uint16.

    uint16 menampung nilai 0..65535, cukup untuk vocab 16.384 dan separuh
    lebih hemat dibanding uint32.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tokens.max() > 65535:
        raise ValueError(
            f"Token ID {tokens.max()} melebihi kapasitas uint16. "
            "Kecilkan vocab atau ganti ke uint32."
        )

    tokens.astype(np.uint16).tofile(path)
