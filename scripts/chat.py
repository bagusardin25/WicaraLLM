"""Ngobrol dengan wicara-56m-chat.

    .venv\\Scripts\\python.exe scripts\\chat.py
    .venv\\Scripts\\python.exe scripts\\chat.py --uji

Perintah saat mengobrol:
    /baru    mulai percakapan dari awal (lupakan riwayat)
    /suhu N  ubah temperature
    /keluar  selesai
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.infer.generate import generate  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.transformer import MiniLLM  # noqa: E402
from src.tokenizer.bpe import load as load_tokenizer  # noqa: E402
from src.tokenizer.chat_template import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    END_TURN_ID,
    Message,
    render,
)

CKPT = REPO_ROOT / "checkpoints" / "wicara-56m-chat-v2" / "best.pt"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"

# Prompt uji yang menyentuh tiap kategori data SFT, plus satu pertanyaan
# faktual untuk melihat apakah model mengarang atau mengaku tidak tahu.
PROMPT_UJI = [
    "Halo!",
    "Apa kabar?",
    "Siapa namamu?",
    "Terima kasih banyak!",
    "Maaf, saya terlambat.",
    "Sampai jumpa!",
    "Apa ibu kota Zimbabwe?",
    "Jelaskan cara membuat nasi goreng.",
    # Empat di bawah menguji data tulis tangan (Fase 6b). Sebelum data itu
    # ada, "Siapa kamu?" dijawab dengan nama tokoh film dan "Ceritakan
    # sesuatu" dijawab satu kalimat pendek atau "aku tidak tahu".
    "Siapa kamu?",
    "Kamu bisa apa?",
    "Ceritakan sesuatu.",
    "bosen nih",
]


def muat(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def jawab(model, tok, riwayat, sysprompt, opsi) -> str:
    """Satu giliran jawaban, dengan seluruh riwayat sebagai konteks."""
    teks = render(riwayat, add_generation_prompt=True, system_prompt=sysprompt)
    ids = tok.encode(teks).ids

    # Konteks 512 token. Kalau riwayat melebihi, giliran terlama dibuang —
    # model kecil ini memang tidak dirancang untuk percakapan panjang.
    batas = model.cfg.max_seq_len - opsi["max_new_tokens"]
    while len(ids) > batas and len(riwayat) > 1:
        riwayat.pop(0)
        teks = render(riwayat, add_generation_prompt=True,
                      system_prompt=sysprompt)
        ids = tok.encode(teks).ids

    return generate(model, tok, prompt_ids=ids,
                    stop_ids=(END_TURN_ID,), **opsi).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    # 100 token hanya cukup untuk ~426 karakter (4,26 char/token), dan itu
    # memotong jawaban panjang di tengah kalimat. Data wikihow saja median
    # jawabannya 632 karakter. 160 memberi ruang ~680 karakter.
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--uji", action="store_true",
                   help="jalankan daftar prompt uji lalu keluar")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint chat tidak ada: {ckpt_path}")
        print("Jalankan SFT dulu: scripts/train_sft.py")
        return 1

    tok = load_tokenizer(args.tokenizer)
    model, cfg, ckpt = muat(ckpt_path, args.device)
    sysprompt = None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT

    opsi = dict(max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device)

    print("=" * 70)
    print(f"  {cfg.name}-chat — {model.num_params() / 1e6:.1f}M parameter")
    print(f"  checkpoint : {ckpt_path.name} (langkah {ckpt['step']:,}, "
          f"val loss {ckpt['best_val_loss']:.4f})")
    print("=" * 70)

    if args.uji:
        for prompt in PROMPT_UJI:
            hasil = jawab(model, tok, [Message("user", prompt)],
                          sysprompt, opsi)
            print(f"\n  Anda   : {prompt}")
            print(f"  Wicara : {hasil}")
        return 0

    print("\n  /baru untuk mulai ulang · /suhu N untuk ubah suhu · "
          "/keluar untuk selesai")
    print("  Catatan: model 56M — lancar berbahasa, tapi fakta sering ngawur.\n")

    riwayat: list[Message] = []
    while True:
        try:
            teks = input("  Anda   : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not teks or teks == "/keluar":
            break
        if teks == "/baru":
            riwayat = []
            print("  (percakapan direset)\n")
            continue
        if teks.startswith("/suhu "):
            try:
                opsi["temperature"] = float(teks.split()[1])
                print(f"  (suhu -> {opsi['temperature']})\n")
            except (IndexError, ValueError):
                print("  (format: /suhu 0.7)\n")
            continue

        riwayat.append(Message("user", teks))
        hasil = jawab(model, tok, riwayat, sysprompt, opsi)
        riwayat.append(Message("assistant", hasil))
        print(f"  Wicara : {hasil}\n")

    print("\n  Sampai jumpa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
