"""Verifikasi model nyata di GPU dan kalibrasi ulang estimasi training.

Menjawab tiga pertanyaan yang selama ini masih berupa perkiraan:
  1. Apakah jumlah parameter nyata sama dengan hitungan di config.py?
  2. Apakah VRAM yang benar-benar terpakai sesuai prediksi?
  3. Berapa throughput sesungguhnya, dan berarti berapa lama pretrain nanti?

Jalankan:  .venv\\Scripts\\python.exe scripts\\bench_model.py
"""

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.config import CONFIGS  # noqa: E402
from src.model.transformer import MiniLLM  # noqa: E402

GB = 1024**3
MB = 1024**2
TARGET_TOKENS = 1_300_000_000


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def bench_batch(cfg, model, opt, batch_size: int, steps: int = 12):
    """Ukur VRAM puncak dan throughput untuk satu batch size.

    Mengembalikan (tok_per_s, peak_gb) atau None kalau OOM.
    """
    seq = cfg.max_seq_len
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Data latih sungguhan: target adalah input yang DIGESER satu posisi.
    # Memakai targets=idx (tanpa geser) membuat loss awal palsu, karena
    # residual stream + tied embedding sudah condong memprediksi token
    # dirinya sendiri.
    data = torch.randint(0, cfg.vocab_size, (batch_size, seq + 1), device="cuda")
    idx, targets = data[:, :-1].contiguous(), data[:, 1:].contiguous()

    try:
        # Beberapa langkah pemanasan: alokasi awal dan autotune kernel
        # tidak boleh ikut terhitung.
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(idx, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(idx, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    tok_per_s = (batch_size * seq * steps) / elapsed
    peak = torch.cuda.max_memory_allocated() / GB
    del data, idx, targets
    return tok_per_s, peak


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="56m", choices=list(CONFIGS))
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[X] CUDA tidak tersedia.")
        return 1

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    cfg = CONFIGS[args.model]
    free_before, total = torch.cuda.mem_get_info()

    section("1. Membangun model")
    model = MiniLLM(cfg).cuda()
    actual = model.num_params()
    predicted = cfg.n_params

    print(f"  Config          : {cfg.name}")
    print(f"  Parameter nyata : {actual:,}")
    print(f"  Prediksi config : {predicted:,}")
    print(f"  Cocok           : {'YA' if actual == predicted else 'TIDAK'}")
    print(f"  Non-embedding   : {model.num_params(non_embedding=True):,}")
    print(f"  Bobot saja      : {actual * 4 / MB:.0f} MB (fp32)")

    section("2. Loss saat inisialisasi")
    data = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len + 1), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(data[:, :-1], data[:, 1:])
    print(f"  Terukur   : {loss.item():.4f}")
    print(f"  Seharusnya: {cfg.expected_init_loss:.4f}  (= ln {cfg.vocab_size})")
    ok_loss = abs(loss.item() - cfg.expected_init_loss) < 0.25
    print(f"  Verdict   : {'OK' if ok_loss else 'BERMASALAH'}")
    if not ok_loss and loss.item() < cfg.expected_init_loss:
        print("  Petunjuk  : loss di bawah ln(vocab) berarti model sudah 'tahu'")
        print("              jawaban. Cek apakah target sudah digeser satu")
        print("              posisi, dan apakah mask kausal bekerja.")
    del data

    section("3. VRAM & throughput per batch size")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    print(f"  VRAM bebas sebelum mulai: {free_before / GB:.2f} GB\n")
    print(f"  {'batch':>6}{'tok/detik':>13}{'VRAM puncak':>14}"
          f"{'prediksi':>11}{'selisih':>10}")
    print(f"  {'-' * 54}")

    # Di atas ambang ini, Windows diam-diam meluapkan VRAM ke RAM sistem
    # (shared memory) alih-alih melempar OOM. Tidak ada error, tapi kecepatan
    # anjlok berkali-kali lipat. Ini jebakan khas Windows/WDDM.
    ambang_spill = 0.92 * total / GB

    hasil = []
    for bs in (8, 16, 24, 32, 48):
        out = bench_batch(cfg, model, opt, bs)
        if out is None:
            print(f"  {bs:>6}{'OOM':>13}")
            break
        tok_s, peak = out
        pred = cfg.estimated_vram_bytes(bs) / GB
        spill = peak > ambang_spill
        tanda = "  <-- SPILL ke RAM" if spill else ""
        print(f"  {bs:>6}{tok_s:>13,.0f}{peak:>12.2f} GB"
              f"{pred:>9.2f} GB{peak - pred:>+9.2f}{tanda}")
        if spill:
            # Berhenti di sini. Melanjutkan ke batch lebih besar akan
            # meluapkan makin banyak memori ke RAM sistem, dan sisa
            # efeknya membuat pengukuran BERIKUTNYA ikut melambat --
            # benchmark jadi meracuni dirinya sendiri.
            break
        hasil.append((bs, tok_s, peak))

    if not hasil:
        print("\n  [X] Tidak ada batch size yang muat.")
        return 1

    print(f"\n  Catatan: peak di atas {ambang_spill:.2f} GB berarti sebagian data")
    print("  dipindah ke RAM sistem. PyTorch tidak melempar error, hanya jadi")
    print("  jauh lebih lambat -- batch seperti itu dibuang dari pertimbangan.")

    section("4. Estimasi pretrain (kalibrasi nyata)")
    best_bs, best_tok, best_peak = max(hasil, key=lambda r: r[1])
    flops_per_tok = 6 * cfg.n_params_non_embedding
    achieved_tflops = best_tok * flops_per_tok / 1e12
    hours = TARGET_TOKENS / best_tok / 3600

    print(f"  Batch terbaik     : {best_bs}")
    print(f"  Throughput        : {best_tok:,.0f} token/detik")
    print(f"  VRAM puncak       : {best_peak:.2f} GB dari {total / GB:.2f} GB")
    print(f"  TFLOPS tercapai   : {achieved_tflops:.1f}")
    print(f"  MFU (vs 25.2 peak): {achieved_tflops / 25.2:.1%}")
    print()
    print(f"  Target {TARGET_TOKENS / 1e9:.1f}B token -> "
          f"{hours:.1f} jam ({hours / 24:.1f} hari)")
    print(f"  Chinchilla {cfg.chinchilla_tokens() / 1e6:.0f}M token -> "
          f"{cfg.chinchilla_tokens() / best_tok / 3600:.1f} jam")

    section("Ringkasan")
    print(f"  Parameter cocok   : {'YA' if actual == predicted else 'TIDAK'}")
    print(f"  Loss awal benar   : {'YA' if ok_loss else 'TIDAK'}")
    print(f"  Batch disarankan  : {best_bs}")
    print(f"  Perkiraan pretrain: {hours:.1f} jam")
    print("\n  Fase 3 selesai. Model siap dipakai Fase 4 (training loop).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
