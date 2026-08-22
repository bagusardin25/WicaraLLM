"""Ubah korpus bersih menjadi train.bin / val.bin berupa larik datar uint16.

Batas dokumen ditandai dengan token <|eos|>. Pemisahan validasi dilakukan
pada tingkat dokumen untuk mencegah data leakage.

    .venv\\Scripts\\python.exe scripts\\pack_tokens.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tokenizer.bpe import iter_jsonl_gz_text, load  # noqa: E402
from src.tokenizer.chat_template import EOS_ID  # noqa: E402

CLEAN = REPO_ROOT / "data" / "clean"
TOKENS = REPO_ROOT / "data" / "tokens"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"

SUMBER = ["opensubtitles", "fineweb2", "wikipedia", "aya", "cendol", "ted"]

BATCH_DOK = 2000       # dokumen per panggilan encode_batch (paralel di Rust)
FLUSH_TOKEN = 8_000_000  # tulis ke disk tiap ~16 MB


class PenulisBin:
    """Penulis berkas biner uint16 dengan buffer untuk menghemat memori."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("wb")
        self.buf: list[int] = []
        self.n_token = 0

    def tulis(self, ids: list[int]) -> None:
        self.buf.extend(ids)
        # Dihitung saat diterima agar akurat untuk pelaporan per-sumber.
        self.n_token += len(ids)
        if len(self.buf) >= FLUSH_TOKEN:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        np.asarray(self.buf, dtype=np.uint16).tofile(self.f)
        self.buf.clear()

    def close(self) -> int:
        self._flush()
        self.f.close()
        return self.n_token


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    p.add_argument("--val-ratio", type=float, default=0.005,
                   help="porsi dokumen yang disisihkan untuk validasi")
    p.add_argument("--budget", type=int, default=1_300_000_000,
                   help="hentikan setelah sekian token train tercapai")
    p.add_argument("--out-dir", default=str(TOKENS))
    args = p.parse_args()

    tok = load(args.tokenizer)
    vocab = tok.get_vocab_size()
    if vocab > 65_535:
        print(f"vocab {vocab:,} melebihi kapasitas uint16.")
        return 1

    out = Path(args.out_dir)
    train = PenulisBin(out / "train.bin")
    val = PenulisBin(out / "val.bin")

    # Pisahkan latih dan validasi di tingkat dokumen.
    tiap_val = max(2, round(1 / args.val_ratio))

    print("=" * 74)
    print("  PACKING TOKEN")
    print("=" * 74)
    print(f"  Tokenizer  : {Path(args.tokenizer).name} (vocab {vocab:,})")
    print(f"  EOS antar dokumen : ID {EOS_ID}")
    print(f"  Validasi   : 1 dari {tiap_val} dokumen ({args.val_ratio:.1%})")
    print(f"  Anggaran   : {args.budget / 1e6:,.0f} juta token train")
    print()

    rekap = []
    t_mulai = time.time()
    n_dok_global = 0
    berhenti = False

    for nama in SUMBER:
        path = CLEAN / f"{nama}.jsonl.gz"
        if not path.exists():
            print(f"  [{nama}] tidak ada, dilewati")
            continue

        t0 = time.time()
        awal_train, awal_val = train.n_token, val.n_token
        n_dok = n_char = 0
        batch: list[str] = []
        batch_ke_val: list[bool] = []

        def proses(batch, ke_val):
            for enc, v in zip(tok.encode_batch(batch), ke_val):
                ids = enc.ids + [EOS_ID]
                (val if v else train).tulis(ids)

        for teks in iter_jsonl_gz_text(path):
            batch.append(teks)
            batch_ke_val.append(n_dok_global % tiap_val == 0)
            n_char += len(teks)
            n_dok += 1
            n_dok_global += 1

            if len(batch) >= BATCH_DOK:
                proses(batch, batch_ke_val)
                batch, batch_ke_val = [], []

                if train.n_token >= args.budget:
                    berhenti = True
                    break
                if n_dok % 200_000 == 0:
                    laju = n_dok / max(time.time() - t0, 1e-9)
                    print(f"      {n_dok:>9,} dok  "
                          f"{(train.n_token + val.n_token) / 1e6:>7.0f}M token  "
                          f"{laju:>6.0f} dok/s", flush=True)

        if batch:
            proses(batch, batch_ke_val)

        tr = train.n_token - awal_train
        va = val.n_token - awal_val
        # Rasio aktual untuk sumber ini.
        cpt = n_char / max(tr + va, 1)
        print(f"  [{nama}] {n_dok:,} dok -> {(tr + va) / 1e6:.0f}M token "
              f"({cpt:.2f} char/token, {time.time() - t0:.0f}s)")

        rekap.append({"sumber": nama, "dokumen": n_dok, "karakter": n_char,
                      "token_train": tr, "token_val": va,
                      "char_per_token": round(cpt, 4)})
        if berhenti:
            print(f"      anggaran {args.budget / 1e6:.0f}M token tercapai, "
                  f"sumber berikutnya dilewati")
            break

    n_train = train.close()
    n_val = val.close()

    total_char = sum(r["karakter"] for r in rekap)
    cpt_global = total_char / max(n_train + n_val, 1)

    meta = {
        "tokenizer": Path(args.tokenizer).name,
        "vocab_size": vocab,
        "eos_id": EOS_ID,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "val_ratio": args.val_ratio,
        "char_per_token": round(cpt_global, 4),
        "per_sumber": rekap,
        "detik": round(time.time() - t_mulai, 1),
    }
    (out / "pack_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 74)
    print("  RINGKASAN")
    print("=" * 74)
    print(f"  {'sumber':<16}{'dokumen':>11}{'token':>11}{'char/token':>12}")
    print("  " + "-" * 50)
    for r in rekap:
        print(f"  {r['sumber']:<16}{r['dokumen']:>11,}"
              f"{(r['token_train'] + r['token_val']) / 1e6:>10.0f}M"
              f"{r['char_per_token']:>12.2f}")
    print("  " + "-" * 50)
    print(f"\n  train.bin : {n_train:>13,} token  "
          f"({(out / 'train.bin').stat().st_size / 1024**3:.2f} GB)")
    print(f"  val.bin   : {n_val:>13,} token  "
          f"({(out / 'val.bin').stat().st_size / 1024**2:.0f} MB)")
    print(f"  gabungan  : {cpt_global:.2f} karakter per token")
    print(f"  waktu     : {(time.time() - t_mulai) / 60:.1f} menit")
    print(f"\n  Siap untuk Fase 5 (pretrain).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
