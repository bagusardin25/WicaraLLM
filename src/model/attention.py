"""Grouped-Query Attention (GQA) dengan RoPE dan KV-cache.

Attention adalah mekanisme yang membuat tiap token bisa menoleh ke token lain.
Analogi perpustakaan:
    Query  = apa yang sedang saya cari
    Key    = label yang saya pasang di diri saya
    Value  = isi yang saya berikan kalau ternyata dipilih

Tiap query dicocokkan ke semua key, lalu value yang cocok diambil dan
dijumlahkan secara berbobot.

Kenapa GQA, bukan MHA biasa:
    Saat inference, K dan V dari semua token sebelumnya harus disimpan
    (KV-cache). Dengan MHA, tiap head punya K/V sendiri sehingga cache-nya
    besar. GQA membuat beberapa query head BERBAGI satu set K/V.

    Model ini: 8 query head berbagi 4 set K/V (tiap K/V dipakai 2 query head).
    KV-cache jadi separuh, kualitas nyaris tidak turun.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.rope import apply_rope


class KVCache:
    """Penyimpan K dan V untuk SATU layer selama generasi teks.

    Tanpa cache, menghasilkan token ke-100 berarti menghitung ulang K dan V
    untuk 99 token sebelumnya, padahal nilainya tidak pernah berubah. Cache
    mengubah biaya generasi dari kuadratik menjadi linear.

    Memori dialokasikan sekali di awal (bukan concat berulang) supaya tidak
    ada realokasi tiap token.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (batch_size, n_kv_head, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # berapa posisi yang sudah terisi

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sisipkan K/V token baru, kembalikan SELURUH isi cache sejauh ini."""
        t = k_new.shape[2]
        if self.length + t > self.k.shape[2]:
            raise ValueError(
                f"KV-cache penuh: {self.length} + {t} > {self.k.shape[2]}. "
                "Naikkan max_seq_len saat membuat cache."
            )
        self.k[:, :, self.length : self.length + t] = k_new
        self.v[:, :, self.length : self.length + t] = v_new
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self) -> None:
        self.length = 0


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Gandakan tiap KV head agar jumlahnya cocok dengan query head.

    (B, n_kv_head, T, head_dim) -> (B, n_kv_head * n_rep, T, head_dim)

    Ini hanya penyesuaian bentuk supaya bisa dihitung sekaligus. Penghematan
    GQA yang sesungguhnya ada di ukuran KV-cache dan jumlah parameter Wk/Wv,
    bukan di sini.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.dropout = cfg.dropout

        # Tanpa bias, praktik standar sejak Llama: sedikit lebih stabil dan
        # menghemat parameter tanpa kehilangan kualitas.
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        # Posisi awal token-token ini. Saat memakai cache, token baru TIDAK
        # berada di posisi 0, dia melanjutkan dari isi cache. Salah di sini
        # adalah bug KV-cache paling umum, dan gejalanya halus.
        offset = cache.length if cache is not None else 0

        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE diterapkan ke Q dan K saja, TIDAK ke V.
        # Posisi hanya perlu memengaruhi pencocokan (siapa melihat siapa),
        # bukan isi informasi yang dibawa.
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            k, v = cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        kv_len = k.shape[2]
        drop = self.dropout if self.training else 0.0

        if t == kv_len:
            # Forward penuh tanpa cache: mask kausal segitiga biasa.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=drop
            )
        elif t == 1:
            # Decode satu token: seluruh isi cache ada di masa lalu, jadi
            # tidak ada yang perlu ditutup.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:
            # Prefill di atas cache yang sudah terisi. is_causal=True SALAH
            # di sini karena matriksnya tidak lagi persegi; yang dibutuhkan
            # adalah segitiga yang digeser sejauh isi cache.
            mask = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(
                diagonal=kv_len - t
            )
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=drop
            )

        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.wo(out)
