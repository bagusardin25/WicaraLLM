"""Simpan dan lanjutkan training."""

import json
import time
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    model_config: dict,
    train_config: dict,
    data_state: dict | None = None,
) -> Path:
    """Tulis checkpoint secara atomik."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "train_config": train_config,
        "data_state": data_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "saved_at": time.time(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
    restore_rng: bool = True,
) -> dict:
    """Muat checkpoint dan kembalikan metadata-nya."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    # weights_only=False karena butuh payload dict config dan RNG
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if restore_rng:
        # Restore RNG supaya urutan batch tidak berulang.
        torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in ckpt["cuda_rng"]]
            )

    return {
        "step": ckpt["step"],
        "tokens_seen": ckpt["tokens_seen"],
        "best_val_loss": ckpt["best_val_loss"],
        "model_config": ckpt["model_config"],
        "train_config": ckpt["train_config"],
        "data_state": ckpt.get("data_state"),
    }


def rotate_checkpoints(out_dir: str | Path, keep_last_n: int) -> None:
    """Hapus checkpoint lama, sisakan n terbaru (kecuali best.pt dan final.pt)."""
    out_dir = Path(out_dir)
    berkala = sorted(
        out_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in berkala[:-keep_last_n] if keep_last_n > 0 else berkala:
        old.unlink(missing_ok=True)


def find_latest(out_dir: str | Path) -> Path | None:
    """Checkpoint berkala terbaru, untuk resume otomatis."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None
    berkala = sorted(
        out_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return berkala[-1] if berkala else None


def append_jsonl(path: str | Path, record: dict) -> None:
    """Tambah satu baris log JSON ke file log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
