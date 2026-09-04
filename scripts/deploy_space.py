"""Deploy demo Gradio ke Hugging Face Spaces."""

import argparse
from pathlib import Path
import sys
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_FOLDER = REPO_ROOT / "space"
DEFAULT_SPACE_ID = "num1notsvn/wicara-56m-chat-demo"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy app Gradio ke Hugging Face Spaces")
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID, help="Space ID (username/space_name)")
    parser.add_argument("--folder", default=str(SPACE_FOLDER), help="Folder berisi app.py, requirements.txt, dan README.md")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder Space tidak ditemukan: {folder_path}")
        return 1

    api = HfApi()
    print(f"[*] Membuat / memeriksa Space: {args.space_id} (SDK: gradio)...")
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    print(f"[*] Mengunggah file aplikasi dari {folder_path} ke Space...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=args.space_id,
        repo_type="space",
    )

    print(f"\n[SUKSES] Demo berhasil dideploy ke Hugging Face Spaces!")
    print(f"URL: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
