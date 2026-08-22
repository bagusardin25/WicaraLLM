"""Pembaca dataset token dengan np.memmap, tanpa DataLoader."""

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Pembaca dataset token dengan pengambilan batch mode epoch (default) atau acak (evaluasi)."""

    def __init__(self, path: str | Path, seq_len: int, seed: int = 1337):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"File token tidak ada: {self.path}\n"
                "Jalankan scripts/make_synthetic_data.py (untuk uji) atau "
                "pipeline Fase 1 (untuk data sungguhan)."
            )

        self.seq_len = seq_len
        # Read-only memmap.
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

        # Butuh seq_len + 1 token: yang terakhir jadi target token sebelumnya.
        self.max_start = len(self.data) - seq_len - 1
        if self.max_start < 1:
            raise ValueError(
                f"{self.path.name} hanya {len(self.data):,} token, "
                f"terlalu pendek untuk seq_len={seq_len}."
            )

        # -- keadaan mode epoch --
        self.n_chunks = (len(self.data) - 1) // seq_len
        self._seed = seed
        self._epoch = 0
        self._pos = 0
        self._order: np.ndarray | None = None

    # -- mode epoch -------------------------------------------------------

    def _acak_ulang(self) -> None:
        """Acak urutan potongan untuk satu putaran baru."""
        rng = np.random.default_rng(self._seed + self._epoch)
        self._order = rng.permutation(self.n_chunks)
        self._pos = 0

    def _ambil_start(self, n: int) -> np.ndarray:
        """Posisi awal untuk n potongan berikutnya, lanjut ke epoch baru
        kalau yang sekarang habis."""
        if self._order is None:
            self._acak_ulang()

        keluar = []
        sisa = n
        while sisa > 0:
            tersedia = self.n_chunks - self._pos
            if tersedia == 0:
                self._epoch += 1
                self._acak_ulang()
                tersedia = self.n_chunks
            ambil = min(sisa, tersedia)
            keluar.append(self._order[self._pos : self._pos + ambil])
            self._pos += ambil
            sisa -= ambil

        return np.concatenate(keluar) * self.seq_len

    @property
    def epoch(self) -> float:
        """Berapa putaran korpus yang sudah dilewati (pecahan)."""
        return self._epoch + self._pos / max(self.n_chunks, 1)

    def state_dict(self) -> dict:
        """Posisi baca, supaya resume tidak mengulang potongan yang sama."""
        return {"epoch": self._epoch, "pos": self._pos, "seed": self._seed}

    def load_state_dict(self, state: dict) -> None:
        self._seed = state.get("seed", self._seed)
        self._epoch = state.get("epoch", 0)
        self._acak_ulang()
        self._pos = min(state.get("pos", 0), self.n_chunks)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_tokens(self) -> int:
        return len(self.data)

    def close(self) -> None:
        """Lepaskan memory map untuk menghindari OSError saat penimpaan file di Windows."""
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
        """Ambil satu batch.

        Args:
            batch_size: Ukuran batch.
            device: Perangkat komputasi.
            generator: Generator angka acak (jika none, pakai urutan epoch).

        Returns:
            x: (batch, seq_len) input
            y: (batch, seq_len) target (digeser satu posisi)
        """
        if generator is not None:
            ix = torch.randint(
                self.max_start, (batch_size,), generator=generator, device="cpu"
            )
        else:
            ix = torch.from_numpy(self._ambil_start(batch_size))

        # Konversi ke np.int64 sebelum ke PyTorch.
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
            # Transfer asinkron ke GPU.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y


def write_tokens(path: str | Path, tokens: np.ndarray) -> None:
    """Tulis larik token ke file biner uint16."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tokens.max() > 65535:
        raise ValueError(
            f"Token ID {tokens.max()} melebihi kapasitas uint16. "
            "Kecilkan vocab atau ganti ke uint32."
        )

    tokens.astype(np.uint16).tofile(path)
