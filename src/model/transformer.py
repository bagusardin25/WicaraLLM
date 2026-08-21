"""Blok Transformer dan model mini-LLM utuh.

Struktur satu blok (gaya Llama, pre-norm):

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Dua hal yang layak diperhatikan:

1. PENJUMLAHAN, bukan penimpaan. Jalur `x` yang mengalir lurus dari input ke
   output disebut residual stream. Tiap blok hanya MENAMBAHKAN hasil olahannya
   ke jalur itu. Ibarat papan tulis yang terus dicoret-tambahi, bukan dihapus
   lalu ditulis ulang. Inilah yang membuat model dalam tetap bisa dilatih:
   gradient punya jalan pintas mengalir sampai ke lapisan paling bawah.

2. PRE-norm, bukan post-norm. Normalisasi dipasang SEBELUM sub-layer, sehingga
   jalur residual tetap bersih tanpa normalisasi di tengahnya. Post-norm
   (seperti Transformer asli 2017) butuh warmup yang jauh lebih hati-hati dan
   sering gagal pada model dalam.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import Attention, KVCache
from src.model.config import ModelConfig
from src.model.ffn import SwiGLU
from src.model.rmsnorm import RMSNorm
from src.model.rope import build_rope_cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin, cache)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Satu tabel bobot dipakai dua arah: token -> vektor (input) dan
            # vektor -> skor (output). Masuk akal karena keduanya memetakan
            # hubungan yang sama, hanya arahnya terbalik.
            # Menghemat 8,4 juta parameter pada config utama (26% dari model).
            self.lm_head.weight = self.tok_emb.weight

        # RoPE tidak punya parameter yang dilatih, jadi disimpan sebagai buffer.
        # persistent=False artinya tidak ikut masuk file checkpoint (bisa
        # dihitung ulang kapan saja), sehingga checkpoint tetap ramping.
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._scale_residual_init()

    # -- inisialisasi -----------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _scale_residual_init(self) -> None:
        """Kecilkan bobot proyeksi yang menulis ke residual stream.

        Tiap blok menambahkan hasilnya ke jalur residual. Dengan 8 blok yang
        semuanya berinisialisasi sama, varians jalur itu menumpuk dan makin
        membesar seiring kedalaman, membuat training awal tidak stabil.

        Solusi standar (GPT-2 dan seterusnya): skalakan bobot proyeksi KELUARAN
        tiap sub-layer dengan 1/sqrt(2 * n_layer). Angka 2 karena tiap blok
        menulis dua kali ke residual: sekali dari attention, sekali dari FFN.
        """
        std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("attn.wo.weight") or name.endswith("ffn.w_down.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    # -- info -------------------------------------------------------------

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    # -- KV-cache ---------------------------------------------------------

    def make_caches(
        self, batch_size: int, max_seq_len: int | None = None
    ) -> list[KVCache]:
        """Buat satu KVCache per layer untuk dipakai saat generate."""
        max_seq_len = max_seq_len or self.cfg.max_seq_len
        device = self.tok_emb.weight.device
        dtype = self.tok_emb.weight.dtype
        return [
            KVCache(
                batch_size,
                max_seq_len,
                self.cfg.n_kv_head,
                self.cfg.head_dim,
                device,
                dtype,
            )
            for _ in range(self.cfg.n_layer)
        ]

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        last_token_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: (batch, seq_len) berisi ID token.
            targets: (batch, seq_len) token yang seharusnya diprediksi.
                Pakai -100 untuk posisi yang loss-nya diabaikan (dipakai saat
                SFT, agar loss hanya dihitung pada jawaban asisten).
            caches: KV-cache per layer, hanya saat generate.
            last_token_only: saat generate, hanya posisi terakhir yang perlu
                logits-nya. Menghindari perkalian matriks 16.384 kolom untuk
                posisi yang hasilnya toh dibuang.

        Returns:
            (logits, loss). loss None kalau targets tidak diberikan.
        """
        b, t = idx.shape
        offset = caches[0].length if caches is not None else 0

        if offset + t > self.cfg.max_seq_len:
            raise ValueError(
                f"Panjang {offset + t} melebihi max_seq_len "
                f"({self.cfg.max_seq_len})."
            )

        x = self.tok_emb(idx)

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                self.rope_cos,
                self.rope_sin,
                caches[i] if caches is not None else None,
            )

        x = self.norm_out(x)

        if last_token_only:
            x = x[:, -1:]

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
