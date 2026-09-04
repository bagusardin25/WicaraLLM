"""Ngobrol dengan wicara-56m-chat.

    .venv\\Scripts\\python.exe scripts\\chat.py
    .venv\\Scripts\\python.exe scripts\\chat.py --uji
    .venv\\Scripts\\python.exe scripts\\chat.py --soal

Perintah saat mengobrol:
    /baru    mulai percakapan dari awal (lupakan riwayat)
    /suhu N  ubah temperature
    /keluar  selesai
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.infer.generate import generate  # noqa: E402
from src.infer.pencegat import cegat  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.transformer import MiniLLM  # noqa: E402
from src.tokenizer.bpe import load as load_tokenizer  # noqa: E402
from src.tokenizer.chat_template import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    END_TURN_ID,
    Message,
    bersihkan_input,
    render,
)

CKPT = REPO_ROOT / "checkpoints" / "wicara-56m-chat-v5" / "best.pt"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"
SOAL_LIHAT = REPO_ROOT / "data" / "eval" / "prompt_lihat.jsonl"

# Uji cepat sebelum rilis: satu prompt per kategori, plus satu pertanyaan
# faktual. Empat terakhir menyasar perilaku yang kita tulis sendiri.
PROMPT_UJI = [
    "Halo!",
    "Apa kabar?",
    "Siapa namamu?",
    "Terima kasih banyak!",
    "Maaf, saya terlambat.",
    "Sampai jumpa!",
    "Apa ibu kota Zimbabwe?",
    "Jelaskan cara membuat nasi goreng.",
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
    """Satu giliran jawaban, dengan seluruh riwayat sebagai konteks.

    Pencegat krisis di sini jaring pengaman untuk pemanggil mana pun, termasuk
    jalur API nanti. Loop interaktif mencegat lebih dulu supaya pertukarannya
    tidak masuk riwayat.
    """
    if riwayat and riwayat[-1].role == "user":
        tetap = cegat(riwayat[-1].content)
        if tetap:
            return tetap

    teks = render(riwayat, add_generation_prompt=True, system_prompt=sysprompt)
    ids = tok.encode(teks).ids

    # Konteks 512. Giliran terlama dibuang kalau penuh.
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
    # Anggaran keluaran. 100 token hanya ~426 char dan memotong jawaban
    # panjang di tengah kalimat; median wikihow saja 632 char.
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--uji", action="store_true",
                   help="jalankan daftar prompt uji lalu keluar")
    p.add_argument("--soal", action="store_true",
                   help="kirim 50 soal di luar data latih, cetak jawaban lengkap")
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

    if args.soal:
        if not SOAL_LIHAT.exists():
            print(f"Berkas soal tidak ada: {SOAL_LIHAT}")
            return 1
        soal = [json.loads(b) for b in
                SOAL_LIHAT.read_text(encoding="utf-8").splitlines() if b.strip()]
        print(f"\n  50 soal di luar data latih — jawaban lengkap, satu sampel")
        print(f"  {len(soal)} prompt · suhu {opsi['temperature']}\n")
        for i, s in enumerate(soal, 1):
            prompt = s["prompt"]
            hasil = jawab(model, tok, [Message("user", prompt)],
                          sysprompt, opsi)
            print(f"  [{i:02d}/{len(soal)}] {s.get('kategori', '-')}")
            print(f"  Anda   : {prompt}")
            print(f"  Wicara : {hasil}\n")
        print("  Selesai.")
        return 0

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

        # Disanitasi di sini, bukan di render(): teks yang tersimpan di riwayat
        # harus sudah bersih supaya giliran berikutnya ikut aman.
        teks = bersihkan_input(teks)

        # Pertukaran yang dicegat tidak masuk riwayat. Model tidak pernah
        # melihatnya, jadi tidak boleh diberi tahu seolah ia yang menjawab —
        # dan tanpa ini frasa dari pesan tetap terbawa ke giliran berikutnya.
        tetap = cegat(teks)
        if tetap:
            print(f"  Wicara : {tetap}\n")
            continue

        riwayat.append(Message("user", teks))
        hasil = jawab(model, tok, riwayat, sysprompt, opsi)
        riwayat.append(Message("assistant", hasil))
        print(f"  Wicara : {hasil}\n")

    print("\n  Sampai jumpa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
