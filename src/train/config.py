"""Konfigurasi training — terpisah dari konfigurasi arsitektur.

Pemisahan ini disengaja: satu arsitektur bisa dilatih berkali-kali dengan
learning rate, data, atau jadwal berbeda. Mencampurnya membuat checkpoint
sulit dibandingkan.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Learning rate acuan, dikalibrasi untuk d_model=512 (config "32m").
LR_ACUAN = 1e-3
D_MODEL_ACUAN = 512


def scaled_lr(d_model: int, base_lr: float = LR_ACUAN) -> float:
    """Learning rate yang diskalakan terhadap lebar model.

    Model yang lebih lebar butuh LR lebih KECIL. Alasannya: tiap neuron di
    layer berikutnya menjumlahkan d_model masukan, jadi makin lebar model,
    makin besar pula perubahan keluaran untuk langkah bobot yang sama.
    Memakai LR yang sama di model lebih lebar membuat langkahnya efektif
    terlalu besar, dan training jadi tidak stabil di awal.

    Penskalaan 1/d_model mengikuti maximal-update parametrization (muP).
    Sejalan juga dengan praktik empiris: GPT-2 124M (d=768) memakai 6e-4,
    sementara model kecil biasa memakai 1e-3 ke atas.

        d_model=512  (32M) -> 1.0e-3
        d_model=640  (56M) -> 8.0e-4
        d_model=768  (88M) -> 6.7e-4
    """
    return base_lr * (D_MODEL_ACUAN / d_model)


@dataclass
class TrainConfig:
    # -- data ------------------------------------------------------------
    train_bin: str = "data/tokens/train.bin"
    val_bin: str = "data/tokens/val.bin"

    # -- arsitektur ------------------------------------------------------
    model_config: str = "56m"  # kunci di src/model/config.py CONFIGS

    # -- ukuran batch ----------------------------------------------------
    # batch_size ditetapkan 8 dari hasil ukur Fase 3: throughput setara
    # batch 16 tapi VRAM separuh. Batch efektif dinaikkan lewat akumulasi.
    batch_size: int = 8
    grad_accum_steps: int = 16  # batch efektif = 8 * 16 = 128 sekuens

    # -- optimizer -------------------------------------------------------
    # 1e-3 wajar untuk model ~32M. Model kecil menoleransi LR lebih tinggi
    # daripada model besar (GPT-2 124M memakai 6e-4).
    lr: float = 1e-3
    min_lr_ratio: float = 0.1  # LR akhir = lr * rasio ini
    warmup_ratio: float = 0.02  # 2% langkah pertama untuk pemanasan
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # 0.95, bukan 0.999 — standar untuk LLM
    grad_clip: float = 1.0

    # -- panjang training ------------------------------------------------
    total_tokens: int = 1_300_000_000

    # -- fase anneal -----------------------------------------------------
    # 10% langkah terakhir: LR meluruh ke minimum. Kalau tersedia, data
    # berkualitas tinggi juga digeser masuk di sini (lihat plan tahap 2).
    anneal_ratio: float = 0.10

    # -- evaluasi & logging ----------------------------------------------
    eval_interval: int = 250
    eval_batches: int = 40
    log_interval: int = 10
    sample_interval: int = 500
    sample_prompts: list[str] = field(default_factory=list)

    # -- checkpoint ------------------------------------------------------
    out_dir: str = "checkpoints/run1"
    checkpoint_interval_minutes: float = 30.0
    keep_last_n: int = 3

    # -- lain-lain -------------------------------------------------------
    seed: int = 1337
    compile_model: bool = False  # torch.compile sering bermasalah di Windows
    device: str = "cuda"

    # -- turunan ---------------------------------------------------------

    def tokens_per_step(self, seq_len: int) -> int:
        """Token yang diproses per satu update parameter."""
        return self.batch_size * seq_len * self.grad_accum_steps

    def total_steps(self, seq_len: int) -> int:
        return max(1, self.total_tokens // self.tokens_per_step(seq_len))

    def warmup_steps(self, seq_len: int) -> int:
        return max(1, int(self.warmup_ratio * self.total_steps(seq_len)))

    @property
    def min_lr(self) -> float:
        return self.lr * self.min_lr_ratio

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)

    def to_dict(self) -> dict:
        return asdict(self)
