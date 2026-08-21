"""RMSNorm — normalisasi yang dipakai Llama, Qwen, DeepSeek, dan hampir semua
model modern, menggantikan LayerNorm.

Bedanya dengan LayerNorm:
    LayerNorm : x -> (x - mean) / std * gain + bias
    RMSNorm   : x -> x / rms(x) * gain            (tanpa mean, tanpa bias)

RMSNorm membuang langkah pengurangan mean dan suku bias. Ternyata pemusatan
mean itu tidak penting untuk Transformer — yang penting hanya menjaga SKALA
vektor tetap wajar. Hasilnya lebih sedikit operasi, lebih sedikit parameter,
dan kualitas setara.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Satu gain per dimensi. Mulai dari 1.0 = awalnya tidak mengubah apa-apa.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dihitung di fp32 walaupun modelnya bf16.
        #
        # Alasannya: x.pow(2) menguadratkan nilai. Aktivasi sebesar 300 jadi
        # 90.000, dan bf16 hanya punya ~3 digit presisi — hasilnya kasar dan
        # error-nya menyebar ke seluruh layer. Normalisasi adalah titik paling
        # sensitif di Transformer, jadi presisi penuh di sini murah tapi penting.
        dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)

        # Gain WAJIB ikut di-cast ke dtype input. Parameter selalu disimpan
        # fp32, dan bf16 * fp32 otomatis dipromosikan kembali ke fp32 oleh
        # PyTorch — rantai dtype model jadi putus tanpa error apa pun.
        return (x_fp32 * rms).to(dtype) * self.weight.to(dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"
