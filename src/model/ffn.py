"""SwiGLU feed-forward network.

Attention mencampur informasi ANTAR token. FFN mengolah tiap token
SENDIRI-SENDIRI, tanpa melihat token lain. Riset interpretabilitas menunjukkan
di sinilah sebagian besar pengetahuan model tersimpan: FFN berperan mirip
memori key-value.

Bentuknya melebar lalu menyempit: 512 -> 1408 -> 512. Model berpikir di ruang
yang lebih luas sebentar, lalu merangkum hasilnya kembali ke jalur utama.

SwiGLU vs MLP biasa:
    MLP biasa : down(gelu(up(x)))               -- 2 matriks
    SwiGLU    : down(silu(gate(x)) * up(x))     -- 3 matriks

Jalur gate bertindak sebagai kran: ia menentukan seberapa banyak tiap dimensi
dari jalur up yang boleh lewat. Karena memakai 3 matriks (bukan 2), d_ffn
dipilih sekitar (8/3) x d_model supaya total parameternya tetap setara dengan
MLP biasa yang ber-d_ffn 4x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_down = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(z) = z * sigmoid(z), versi mulus dari ReLU. Nilai negatif
        # tidak dimatikan total, sehingga gradient tetap mengalir di sana.
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
