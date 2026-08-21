"""Jadwal learning rate: warmup linear lalu peluruhan cosine.

Bentuk kurvanya:

    LR
     |      .-'''-.
     |    .'       '-.
     |  .'             '--.
     |.'                    '---....____
     +--------------------------------------> langkah
      warmup      cosine decay        min_lr
      (2%)

Kenapa WARMUP:
    Di langkah-langkah pertama, parameter masih acak dan gradient-nya sangat
    liar. Langkah besar di titik ini bisa melempar bobot ke wilayah yang
    tidak bisa dipulihkan — training terlihat "meledak" lalu mandek di loss
    tinggi selamanya. Warmup menaikkan LR pelan-pelan sampai gradient
    menenangkan diri.

Kenapa PELURUHAN COSINE:
    Di fase awal perlu langkah besar untuk menjelajah. Mendekati akhir,
    langkah besar justru membuat model melompati minimum yang bagus. Cosine
    menurunkannya secara mulus; turunannya nol di kedua ujung, jadi tidak ada
    perubahan mendadak yang mengguncang optimizer.

Kenapa berhenti di min_lr, bukan nol:
    LR nol berarti model berhenti belajar sama sekali di langkah terakhir.
    Menyisakan ~10% membuat fase akhir tetap produktif.
"""

import math


def get_lr(
    step: int,
    *,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Learning rate untuk satu langkah.

    Args:
        step: nomor langkah, mulai dari 0.
    """
    # 1. Warmup linear. step+1 supaya langkah 0 tidak dapat LR nol persis.
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps

    # 2. Lewat akhir jadwal (misal training diperpanjang): tahan di minimum.
    if step >= total_steps:
        return min_lr

    # 3. Peluruhan cosine dari lr ke min_lr.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 -> 0.0
    return min_lr + coeff * (lr - min_lr)


def describe_schedule(
    *, lr: float, min_lr: float, warmup_steps: int, total_steps: int, n: int = 8
) -> str:
    """Cetak beberapa titik dari kurva — berguna untuk memeriksa jadwal
    sebelum menjalankan training panjang."""
    lines = [f"  {'langkah':>10}{'LR':>12}{'fase':>16}"]
    lines.append(f"  {'-' * 38}")
    titik = sorted({0, warmup_steps - 1, warmup_steps} |
                   {int(total_steps * i / (n - 1)) for i in range(n)})
    for s in titik:
        if s > total_steps:
            continue
        cur = get_lr(s, lr=lr, min_lr=min_lr, warmup_steps=warmup_steps,
                     total_steps=total_steps)
        fase = "warmup" if s < warmup_steps else "cosine decay"
        if s >= total_steps:
            fase = "selesai"
        lines.append(f"  {s:>10,}{cur:>12.2e}{fase:>16}")
    return "\n".join(lines)
