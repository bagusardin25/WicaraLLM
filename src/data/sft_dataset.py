"""Dataset SFT: ubah percakapan jadi (input_ids, labels) dengan loss masking.

INTI FASE 6 ADA DI SINI, dan ini juga titik paling rawan.

Loss HANYA dihitung pada token jawaban asisten. Token pertanyaan user diberi
label -100 supaya diabaikan `F.cross_entropy(ignore_index=-100)`.

Kalau masking-nya salah dan loss ikut dihitung di teks user, model belajar
MENIRU cara user bertanya alih-alih menjawabnya. Yang menakutkan: training
tetap terlihat mulus, loss tetap turun, angkanya tetap wajar. Kesalahannya
baru ketahuan saat model dipakai — dan saat itu 1 jam training sudah terbuang.

Karena itu modul ini menyediakan `periksa_masking()`: mendekode ulang posisi
yang loss-nya dihitung, dan membandingkannya dengan teks asisten yang
seharusnya. Bukti, bukan asumsi.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from src.tokenizer.chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    Message,
    PAD_ID,
    assistant_spans,
    render,
)

ABAIKAN = -100  # nilai yang diabaikan cross_entropy


def encode_example(
    tokenizer,
    messages: list[Message],
    max_len: int = 512,
    system_prompt: str | None = None,
) -> tuple[list[int], list[int]] | None:
    """Ubah satu percakapan jadi (input_ids, labels).

    Returns None kalau hasilnya terlalu panjang atau tidak ada token asisten
    yang tersisa untuk dilatih.

    Pergeseran satu posisi:
        input_ids = token[:-1]
        labels    = token[1:], dengan -100 di posisi non-asisten

    Model memprediksi token t+1 saat melihat token t, jadi label di posisi t
    adalah token t+1. Loss dipertahankan hanya kalau token t+1 itu milik
    jawaban asisten.
    """
    teks = render(messages, system_prompt=system_prompt)
    spans = assistant_spans(messages, system_prompt=system_prompt)

    enc = tokenizer.encode(teks)
    ids, offsets = enc.ids, enc.offsets

    if len(ids) < 2 or len(ids) > max_len + 1:
        return None

    # Tandai token yang termasuk jawaban asisten.
    #
    # Titik TENGAH offset dipakai, bukan awalnya. Token byte-level bisa
    # mencakup spasi sebelum kata, sehingga awalnya jatuh tepat di batas span
    # dan menghasilkan penilaian yang meleset satu token.
    milik_asisten = np.zeros(len(ids), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        tengah = (a + b) / 2
        for s, e in spans:
            if s <= tengah < e:
                milik_asisten[i] = True
                break

    input_ids = ids[:-1]
    labels = [
        ids[i + 1] if milik_asisten[i + 1] else ABAIKAN
        for i in range(len(ids) - 1)
    ]

    if not any(v != ABAIKAN for v in labels):
        return None

    return input_ids, labels


def periksa_masking(tokenizer, messages: list[Message],
                    system_prompt: str | None = None) -> dict:
    """Bukti bahwa masking benar: dekode ulang posisi yang dilatih.

    Mengembalikan teks yang benar-benar dihitung loss-nya. Kalau di dalamnya
    ada potongan pertanyaan user, berarti masking bocor.
    """
    hasil = encode_example(tokenizer, messages, max_len=10_000,
                           system_prompt=system_prompt)
    if hasil is None:
        return {"dilatih": "", "n_dilatih": 0, "n_total": 0}

    input_ids, labels = hasil
    dilatih = [v for v in labels if v != ABAIKAN]
    return {
        "dilatih": tokenizer.decode(dilatih, skip_special_tokens=False),
        "n_dilatih": len(dilatih),
        "n_total": len(labels),
        "rasio": len(dilatih) / max(len(labels), 1),
    }


class SFTDataset:
    """Percakapan SFT yang sudah ter-encode, siap dibatch.

    Berbeda dari TokenDataset (pretrain) yang membaca larik datar: di sini
    tiap contoh berdiri sendiri dan panjangnya berbeda-beda, jadi perlu
    padding per batch.
    """

    def __init__(self, path: str | Path, tokenizer, max_len: int = 512,
                 system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
                 seed: int = 1337):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Data SFT tidak ada: {self.path}\n"
                "Jalankan scripts/build_sft_data.py dulu."
            )

        self.max_len = max_len
        self.contoh: list[tuple[list[int], list[int]]] = []
        self.n_dibuang = 0

        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for baris in f:
                d = json.loads(baris)
                msgs = [Message(m["role"], m["content"]) for m in d["messages"]]
                out = encode_example(tokenizer, msgs, max_len, system_prompt)
                if out is None:
                    self.n_dibuang += 1
                    continue
                self.contoh.append(out)

        if not self.contoh:
            raise ValueError(f"Tidak ada contoh yang layak di {self.path}")

        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(self.contoh))
        self._pos = 0
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.contoh)

    @property
    def epoch(self) -> float:
        return self._epoch + self._pos / max(len(self.contoh), 1)

    @property
    def n_token_dilatih(self) -> int:
        """Total token yang benar-benar dihitung loss-nya."""
        return sum(sum(1 for v in lab if v != ABAIKAN) for _, lab in self.contoh)

    def _ambil_indeks(self, n: int) -> list[int]:
        keluar = []
        while len(keluar) < n:
            if self._pos >= len(self._order):
                self._epoch += 1
                self._order = self._rng.permutation(len(self.contoh))
                self._pos = 0
            ambil = min(n - len(keluar), len(self._order) - self._pos)
            keluar.extend(self._order[self._pos : self._pos + ambil].tolist())
            self._pos += ambil
        return keluar

    def get_batch(
        self, batch_size: int, device, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Satu batch, di-padding ke contoh terpanjang di dalam batch itu.

        Padding hanya sampai contoh terpanjang di batch (bukan max_len),
        supaya tidak membuang komputasi untuk token pad yang toh diabaikan.

        Args:
            generator: kalau diberikan, indeks diambil acak memakai generator
                ini (dipakai evaluasi). Karena pemanggil membuat generator
                berbenih tetap di tiap evaluasi, batch validasinya identik
                setiap kali dan angkanya bisa dibandingkan lintas langkah.
                Kalau None, contoh diambil berurutan dari putaran epoch yang
                urutannya sudah diacak — menjamin setiap contoh terlihat.
        """
        if generator is not None:
            idx = torch.randint(len(self.contoh), (batch_size,),
                                generator=generator).tolist()
        else:
            idx = self._ambil_indeks(batch_size)
        batch = [self.contoh[i] for i in idx]
        panjang = max(len(x) for x, _ in batch)

        x = torch.full((batch_size, panjang), PAD_ID, dtype=torch.long)
        y = torch.full((batch_size, panjang), ABAIKAN, dtype=torch.long)

        for i, (ids, labels) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)

        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "pos": self._pos}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state.get("epoch", 0)
        self._pos = min(state.get("pos", 0), len(self._order))

    def close(self) -> None:
        """Ada demi kesamaan antarmuka dengan TokenDataset; contoh SFT
        seluruhnya di RAM, tidak ada memory map yang perlu dilepas."""
