"""Konfigurasi model + kalkulator anggaran parameter dan VRAM.

File ini sengaja tidak meng-import torch, supaya bisa dipakai untuk merencanakan
ukuran model sebelum environment GPU siap.

Rumus parameter (gaya Llama: tanpa bias, tied embedding, SwiGLU, GQA):

    embedding      = vocab * d_model                       (dipakai 2x lewat tying)
    per layer:
      attn         = d*d (Wq) + 2 * d*(n_kv*head_dim) (Wk,Wv) + d*d (Wo)
      ffn (SwiGLU) = 3 * d * d_ffn                     (gate, up, down)
      norm         = 2 * d
    final norm     = d
"""

from dataclasses import dataclass
from math import log


# Biaya memori optimizer AdamW + AMP bf16, dalam byte per parameter:
#   master weights fp32 (4) + gradients fp32 (4) + Adam m (4) + Adam v (4)
BYTES_PER_PARAM_ADAMW = 16

# Byte aktivasi per token, per layer, per satuan d_model.
#
# Angka ini DIKALIBRASI DARI PENGUKURAN NYATA (scripts/bench_model.py di
# RTX 4050, config main), bukan hasil hitungan teoretis. Perkiraan teoretis
# awal sebesar 34 meleset ~2x terlalu rendah, karena melupakan bahwa tensor
# di dalam FFN berukuran d_ffn (2,75x lebih lebar dari d_model) dan bahwa
# autocast menyisakan sejumlah salinan fp32.
ACTIVATION_BYTES_PER_TOKEN_PER_LAYER = 55

# Byte per token per entri vocab untuk logits + perhitungan cross-entropy.
#
# Sering terlupakan, padahal di model kecil ini biayanya BESAR: vocab (16.384)
# 32x lebih lebar dari d_model (512), jadi satu tensor logits saja bisa
# menyaingi seluruh aktivasi 8 layer. Isinya: logits bf16 (2 byte), salinan
# fp32 untuk cross_entropy (4 byte), dan gradient-nya (dominan sisanya).
OUTPUT_HEAD_BYTES_PER_TOKEN_PER_VOCAB = 12

# CUDA context + workspace cuBLAS/cuDNN di Windows.
CUDA_OVERHEAD_BYTES = 500 * 1024**2


@dataclass
class ModelConfig:
    """Hyperparameter arsitektur. Lihat configs/*.yaml untuk nilai konkretnya."""

    name: str = "wicara-56m"

    # Dimensi inti
    vocab_size: int = 16384
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4  # GQA: n_head harus habis dibagi n_kv_head
    d_ffn: int = 1408  # ~(8/3)*d_model, dibulatkan ke kelipatan 64
    max_seq_len: int = 512

    # RoPE
    rope_theta: float = 10000.0

    # Regularisasi
    dropout: float = 0.0  # 0.0 untuk pretrain (data cukup banyak), naikkan saat SFT

    # Numerik
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    # -- validasi ---------------------------------------------------------

    def validate(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) harus habis dibagi n_head ({self.n_head})"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) harus habis dibagi n_kv_head "
                f"({self.n_kv_head}) untuk GQA"
            )
        if self.vocab_size > 65535:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) > 65535 — dataset uint16 memmap "
                "tidak lagi cukup, ubah ke uint32 di src/data/pack.py"
            )

    # -- properti turunan -------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_rep(self) -> int:
        """Berapa kali tiap KV head diulang untuk melayani query head-nya."""
        return self.n_head // self.n_kv_head

    @property
    def expected_init_loss(self) -> float:
        """Loss yang seharusnya muncul saat inisialisasi: ln(vocab_size).

        Model yang belum belajar apa-apa harus menebak seragam di antara
        vocab_size pilihan. Kalau loss awal jauh dari angka ini, ada bug
        di inisialisasi atau di embedding.
        """
        return log(self.vocab_size)

    # -- hitung parameter -------------------------------------------------

    def param_breakdown(self) -> dict[str, int]:
        """Rincian jumlah parameter per komponen."""
        d, kv_dim = self.d_model, self.n_kv_head * self.head_dim

        embedding = self.vocab_size * d
        attn_per_layer = d * d + 2 * (d * kv_dim) + d * d
        ffn_per_layer = 3 * d * self.d_ffn
        norm_per_layer = 2 * d

        layers = self.n_layer * (attn_per_layer + ffn_per_layer + norm_per_layer)
        final_norm = d
        # Tanpa tying, output head menambah satu salinan penuh tabel embedding.
        lm_head = 0 if self.tie_embeddings else embedding

        return {
            "embedding": embedding,
            "attention": self.n_layer * attn_per_layer,
            "ffn": self.n_layer * ffn_per_layer,
            "norms": self.n_layer * norm_per_layer + final_norm,
            "lm_head": lm_head,
            "total": embedding + layers + final_norm + lm_head,
        }

    @property
    def n_params(self) -> int:
        return self.param_breakdown()["total"]

    @property
    def n_params_non_embedding(self) -> int:
        """Parameter yang benar-benar 'berpikir' — di luar tabel lookup.

        Ini angka yang lebih jujur untuk membandingkan kapasitas antar model
        dengan ukuran vocab berbeda.
        """
        b = self.param_breakdown()
        return b["total"] - b["embedding"] - b["lm_head"]

    # -- estimasi memori --------------------------------------------------

    def optimizer_state_bytes(self) -> int:
        """VRAM untuk weights + gradients + state AdamW (tidak tergantung batch)."""
        return self.n_params * BYTES_PER_PARAM_ADAMW

    def activation_bytes(self, batch_size: int, seq_len: int | None = None) -> int:
        """Perkiraan VRAM aktivasi. Tumbuh linear terhadap batch dan seq_len.

        Dua sumbangan: badan Transformer (per layer) dan kepala keluaran
        (logits + loss). Yang kedua tidak bergantung jumlah layer sama sekali,
        tapi di model kecil ber-vocab besar porsinya bisa hampir separuh.
        """
        seq_len = seq_len or self.max_seq_len
        tokens = batch_size * seq_len

        body = (
            ACTIVATION_BYTES_PER_TOKEN_PER_LAYER * tokens * self.d_model * self.n_layer
        )
        head = OUTPUT_HEAD_BYTES_PER_TOKEN_PER_VOCAB * tokens * self.vocab_size
        return body + head

    def activation_breakdown(self, batch_size: int, seq_len: int | None = None
                             ) -> dict[str, int]:
        seq_len = seq_len or self.max_seq_len
        tokens = batch_size * seq_len
        return {
            "body": ACTIVATION_BYTES_PER_TOKEN_PER_LAYER
            * tokens
            * self.d_model
            * self.n_layer,
            "output_head": OUTPUT_HEAD_BYTES_PER_TOKEN_PER_VOCAB
            * tokens
            * self.vocab_size,
        }

    def max_batch_size(self, vram_budget_bytes: int, seq_len: int | None = None
                       ) -> int:
        """Batch terbesar yang muat dalam anggaran VRAM tertentu.

        Dipakai untuk memilih batch TANPA harus coba-coba sampai OOM.
        """
        seq_len = seq_len or self.max_seq_len
        tetap = self.optimizer_state_bytes() + CUDA_OVERHEAD_BYTES
        tersisa = vram_budget_bytes - tetap
        if tersisa <= 0:
            return 0
        per_batch = self.activation_bytes(1, seq_len)
        return max(0, int(tersisa // per_batch))

    def estimated_vram_bytes(self, batch_size: int, seq_len: int | None = None) -> int:
        return (
            self.optimizer_state_bytes()
            + self.activation_bytes(batch_size, seq_len)
            + CUDA_OVERHEAD_BYTES
        )

    def kv_cache_bytes(self, batch_size: int, seq_len: int, dtype_bytes: int = 2) -> int:
        """VRAM KV-cache saat inference. Inilah yang dihemat GQA."""
        kv_dim = self.n_kv_head * self.head_dim
        return 2 * self.n_layer * batch_size * seq_len * kv_dim * dtype_bytes

    # -- estimasi compute -------------------------------------------------

    def training_flops(self, n_tokens: int) -> float:
        """Perkiraan FLOPs training dengan aturan baku 6*N*D.

        Faktor 6 = 2 (forward) + 4 (backward), per parameter per token.
        """
        return 6.0 * self.n_params_non_embedding * n_tokens

    def chinchilla_tokens(self) -> int:
        """Jumlah token compute-optimal menurut Chinchilla (~20 token/parameter)."""
        return 20 * self.n_params

    # -- laporan ----------------------------------------------------------

    def summary(self, batch_size: int = 16) -> str:
        b = self.param_breakdown()
        total = b["total"]
        mb = 1024**2
        gb = 1024**3

        lines = [
            f"=== {self.name} ===",
            f"  d_model={self.d_model}  n_layer={self.n_layer}  "
            f"n_head={self.n_head}  n_kv_head={self.n_kv_head}  "
            f"d_ffn={self.d_ffn}  head_dim={self.head_dim}",
            f"  vocab={self.vocab_size:,}  seq_len={self.max_seq_len}  "
            f"tied={self.tie_embeddings}",
            "",
            "  Parameter:",
        ]
        for key in ("embedding", "attention", "ffn", "norms", "lm_head"):
            if b[key]:
                lines.append(
                    f"    {key:<12} {b[key]:>12,}  ({100 * b[key] / total:5.1f}%)"
                )
        lines += [
            f"    {'TOTAL':<12} {total:>12,}  ({total / 1e6:.1f}M)",
            f"    non-embedding {self.n_params_non_embedding:>10,}  "
            f"({self.n_params_non_embedding / 1e6:.1f}M)",
            "",
            "  VRAM (training):",
            f"    optimizer state   {self.optimizer_state_bytes() / mb:8.0f} MB",
            f"    aktivasi badan    "
            f"{self.activation_breakdown(batch_size)['body'] / mb:8.0f} MB"
            f"   (B={batch_size})",
            f"    aktivasi logits   "
            f"{self.activation_breakdown(batch_size)['output_head'] / mb:8.0f} MB"
            f"   (vocab {self.vocab_size:,})",
            f"    CUDA overhead     {CUDA_OVERHEAD_BYTES / mb:8.0f} MB",
            f"    TOTAL             "
            f"{self.estimated_vram_bytes(batch_size) / gb:8.2f} GB",
            "",
            "  Training:",
            f"    loss saat init    {self.expected_init_loss:.4f}  (= ln(vocab))",
            f"    Chinchilla        {self.chinchilla_tokens() / 1e6:.0f}M token",
            f"    KV-cache (1x{self.max_seq_len})  "
            f"{self.kv_cache_bytes(1, self.max_seq_len) / mb:.1f} MB",
        ]
        return "\n".join(lines)


# --- Preset dari plan (§5) -----------------------------------------------

CONFIGS: dict[str, ModelConfig] = {
    # A — Debug: cukup kecil untuk iterasi dalam hitungan detik.
    "7m": ModelConfig(
        name="wicara-7m",
        d_model=256,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        d_ffn=704,
        max_seq_len=256,
    ),
    # B — Kecil: run ~3 jam.
    "19m": ModelConfig(
        name="wicara-19m",
        d_model=384,
        n_layer=8,
        n_head=6,
        n_kv_head=2,
        d_ffn=1024,
        max_seq_len=512,
    ),
    # C — UTAMA: target run semalam di RTX 4050 6GB.
    "32m": ModelConfig(
        name="wicara-32m",
        d_model=512,
        n_layer=8,
        n_head=8,
        n_kv_head=4,
        d_ffn=1408,
        max_seq_len=512,
    ),
    # C2 — Menengah: ditambahkan setelah Fase 1 ternyata menghasilkan ~3
    # miliar token mentah (6,6x target awal). Dengan data sebanyak itu,
    # config "main" 32M jadi under-parameterized menurut Chinchilla —
    # pasangan optimal untuk ~1,1B token adalah sekitar 55M parameter.
    "56m": ModelConfig(
        name="wicara-56m",
        d_model=640,
        n_layer=10,
        n_head=10,
        n_kv_head=5,
        d_ffn=1728,  # ~(8/3)*640, dibulatkan ke kelipatan 64
        max_seq_len=512,
    ),
    # D — Stretch: hanya jika bersedia run multi-hari.
    "88m": ModelConfig(
        name="wicara-88m",
        d_model=768,
        n_layer=12,
        n_head=12,
        n_kv_head=4,
        d_ffn=2048,
        max_seq_len=512,
    ),
}


if __name__ == "__main__":
    batches = {"7m": 64, "19m": 32, "32m": 16, "56m": 8, "88m": 8}
    for key, cfg in CONFIGS.items():
        print(cfg.summary(batch_size=batches[key]))
        print()
