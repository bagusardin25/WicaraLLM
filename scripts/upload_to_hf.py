"""Unggah model Wicara yang sudah diekspor ke Hugging Face Hub."""

import argparse
from pathlib import Path
import sys
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FOLDER = REPO_ROOT / "hf_export" / "wicara-56m-chat"
DEFAULT_REPO_ID = "num1notsvn/wicara-56m-chat"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload folder model ke Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Repo ID di HF (username/repo_name)")
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER), help="Folder lokal yang berisi file model")
    parser.add_argument("--private", action="store_true", help="Set repo sebagai private (default: public)")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder tidak ditemukan: {folder_path}")
        return 1

    api = HfApi()

    print(f"[*] Memeriksa / membuat repositori: {args.repo_id} (private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"[*] Mengunggah isi folder {folder_path} ke https://huggingface.co/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\n[SUKSES] Model Wicara berhasil diunggah ke Hugging Face Hub!")
    print(f"URL: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
