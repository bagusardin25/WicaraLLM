"""Uji coba memuat model hasil ekspor menggunakan Hugging Face Transformers."""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "hf_export" / "wicara-56m-chat"


def main() -> None:
    print(f"[*] Memuat tokenizer dari: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"[*] Memuat model dari: {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.bfloat16)
    print(f"[OK] Berhasil dimuat sebagai: {model.__class__.__name__}")

    messages = [
        {"role": "user", "content": "Halo! Siapa namamu?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n[*] Prompt yang dirender:\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
        eos_token_id=[tokenizer.eos_token_id, 2],
    )

    jawaban = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[*] Respon Wicara:\n{jawaban.strip()}\n")
    print("[SUKSES] Model Hugging Face terverifikasi dan siap diunggah!")


if __name__ == "__main__":
    main()
