"""Fase 0 — verifikasi environment dan pengukuran TFLOPS efektif RTX 4050.

Jalankan lewat interpreter venv, BUKAN `python` dari PATH:

    .venv\\Scripts\\python.exe scripts\\check_env.py

Output skrip ini yang mengkalibrasi ulang estimasi waktu training di plan §6.
Jangan percaya angka teoretis vendor — ukur mesin sendiri.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.config import CONFIGS  # noqa: E402

GB = 1024**3
MB = 1024**2


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check_python() -> bool:
    """Cegah kesalahan paling mahal: menjalankan proyek ini di MSYS2 Python 3.14."""
    section("1. Python interpreter")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")

    ok = True
    if sys.version_info[:2] != (3, 11):
        print(f"  [!] Diharapkan Python 3.11, dapat {sys.version_info[0]}."
              f"{sys.version_info[1]}")
        ok = False
    if "msys" in sys.executable.lower() or "ucrt64" in sys.executable.lower():
        print("  [X] Ini MSYS2 Python — wheel PyTorch CUDA Windows tidak kompatibel.")
        ok = False
    if sys.prefix == sys.base_prefix:
        print("  [!] Tidak berjalan di dalam venv.")
        ok = False

    print(f"  -> {'OK' if ok else 'BERMASALAH'}")
    return ok


def check_torch():
    section("2. PyTorch + CUDA")
    try:
        import torch
    except ImportError:
        print("  [X] torch belum terinstall.")
        return None

    print(f"  torch          : {torch.__version__}")
    print(f"  CUDA tersedia  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  [X] CUDA tidak terdeteksi — training akan berjalan di CPU (mustahil).")
        return None

    print(f"  CUDA build     : {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU            : {props.name}")
    print(f"  Compute cap    : sm_{props.major}{props.minor}")
    print(f"  SM count       : {props.multi_processor_count}")
    print(f"  VRAM total     : {total / GB:.2f} GB")
    print(f"  VRAM bebas     : {free / GB:.2f} GB   <-- angka ini yang jadi anggaran")
    print(f"  VRAM terpakai  : {(total - free) / GB:.2f} GB (desktop Windows + app lain)")
    print(f"  bf16 didukung  : {torch.cuda.is_bf16_supported()}")

    if free < 3.5 * GB:
        print("  [!] VRAM bebas < 3.5 GB. Tutup browser/aplikasi lain sebelum training.")

    return torch, free


def check_sdpa(torch) -> None:
    """Cek kernel attention hemat memori.

    Yang penting bukan nama kernelnya, melainkan apakah ADA kernel yang
    menghindari materialisasi matriks seq_len x seq_len. Itu yang membuat
    GPU 6 GB sanggup. Flash dan mem_efficient sama-sama memenuhi syarat itu.
    """
    import warnings

    section("3. Backend attention (SDPA)")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    available = {}
    for backend, label in [
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "mem_efficient"),
        (SDPBackend.MATH, "math (fallback)"),
    ]:
        try:
            # Memaksa satu backend membuat PyTorch memperingatkan soal backend
            # lain yang dinonaktifkan sementara — itu derau, bukan masalah.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, q, q, is_causal=True)
            available[label] = True
            print(f"  {label:<18} tersedia")
        except Exception:
            available[label] = False
            print(f"  {label:<18} TIDAK tersedia")

    if not available["flash"]:
        print("\n  Catatan: wheel PyTorch untuk Windows memang tidak dikompilasi")
        print("  dengan FlashAttention. Ini normal dan BUKAN masalah — mem_efficient")
        print("  juga tidak pernah membentuk matriks seq_len x seq_len di memori,")
        print("  jadi anggaran VRAM di plan tetap berlaku. Bedanya hanya kecepatan")
        print("  (perkiraan 10-20% lebih lambat), dan di seq_len 512 selisihnya kecil.")

    if not available["mem_efficient"] and not available["flash"]:
        print("\n  [X] Hanya kernel 'math' yang tersedia — attention akan memakai")
        print("  memori kuadratik terhadap seq_len. Turunkan batch secara signifikan.")


def benchmark_matmul(torch) -> float:
    """Ukur throughput bf16 matmul nyata. Mengembalikan TFLOPS puncak terukur."""
    section("4. Benchmark matmul bf16 (TFLOPS efektif)")
    torch.set_float32_matmul_precision("high")

    peak = 0.0
    print(f"  {'ukuran':<14}{'TFLOPS':>10}")
    print(f"  {'-' * 24}")
    for n in (1024, 2048, 4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

        for _ in range(5):  # warmup: biarkan clock naik & kernel ter-autotune
            a @ b
        torch.cuda.synchronize()

        iters = 20
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tflops = (2 * n**3 * iters) / elapsed / 1e12
        peak = max(peak, tflops)
        print(f"  {n}x{n:<9}{tflops:>10.1f}")
        del a, b
        torch.cuda.empty_cache()

    print(f"\n  Puncak terukur: {peak:.1f} TFLOPS")
    print("  Catatan: model kecil tidak akan mencapai angka ini — matmul-nya jauh")
    print("  lebih kecil dan GPU kurang terisi. Harapkan ~25-40% dari puncak.")
    return peak


def estimate_training_time(peak_tflops: float) -> None:
    section("5. Estimasi waktu training (kalibrasi ulang plan §6)")
    cfg = CONFIGS["56m"]
    n_tokens = 1_300_000_000
    flops = cfg.training_flops(n_tokens)

    print(f"  Model    : {cfg.name} ({cfg.n_params / 1e6:.1f}M parameter)")
    print(f"  Token    : {n_tokens / 1e9:.1f}B")
    print(f"  FLOPs    : {flops:.2e}")
    print()
    print(f"  {'MFU':<8}{'TFLOPS efektif':>16}{'tok/detik':>14}{'waktu':>12}")
    print(f"  {'-' * 50}")
    for mfu in (0.15, 0.25, 0.35, 0.45):
        effective = peak_tflops * mfu
        hours = flops / (effective * 1e12) / 3600
        tok_s = n_tokens / (hours * 3600)
        print(f"  {mfu:<8.0%}{effective:>16.1f}{tok_s:>14,.0f}{hours:>10.1f} jam")

    print("\n  Skenario realistis untuk model kecil di laptop Windows: MFU 25-35%.")
    print("  Angka final ditetapkan dari throughput nyata 200 step di Fase 5.")


def check_vram_budget(torch, free_bytes: int) -> None:
    section("6. Anggaran VRAM vs config")
    print(f"  VRAM bebas saat ini: {free_bytes / GB:.2f} GB\n")
    print(f"  {'config':<14}{'param':>9}{'B':>5}{'perkiraan':>12}{'verdict':>12}")
    print(f"  {'-' * 52}")
    for key, batch in (("7m", 64), ("19m", 32), ("32m", 16), ("56m", 8), ("88m", 8)):
        cfg = CONFIGS[key]
        need = cfg.estimated_vram_bytes(batch)
        margin = free_bytes - need
        verdict = "OK" if margin > 0.8 * GB else ("KETAT" if margin > 0 else "OOM")
        print(f"  {key:<14}{cfg.n_params / 1e6:>7.1f}M{batch:>5}"
              f"{need / GB:>10.2f} GB{verdict:>12}")


def main() -> int:
    print("Fase 0 — verifikasi environment mini-LLM\n")

    py_ok = check_python()
    result = check_torch()
    if result is None:
        print("\n[X] Environment belum siap. Selesaikan instalasi PyTorch dulu.")
        return 1

    torch, free = result
    check_sdpa(torch)
    peak = benchmark_matmul(torch)
    estimate_training_time(peak)
    check_vram_budget(torch, free)

    section("Ringkasan")
    print(f"  Python OK        : {py_ok}")
    print(f"  CUDA siap        : True")
    print(f"  Puncak bf16      : {peak:.1f} TFLOPS")
    print(f"  VRAM bebas       : {free / GB:.2f} GB")
    print("\n  Fase 0 selesai. Lanjut ke Fase 1 (pipeline data).")
    return 0 if py_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
