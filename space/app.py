import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from pathlib import Path
import threading
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATH = REPO_ROOT / "hf_export" / "wicara-56m-chat"
MODEL_ID = str(LOCAL_PATH) if LOCAL_PATH.exists() else "num1notsvn/wicara-56m-chat"

print(f"Loading model: {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
model.eval()
print(f"Model loaded successfully on {device} ({dtype})!", flush=True)


def _to_text(val) -> str:
    """Ekstrak teks murni dari input pesan atau riwayat Gradio 5/6."""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for part in val:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return " ".join(parts)
    if isinstance(val, dict):
        return str(val.get("text", val.get("content", "")))
    return str(val)


def predict(
    message,
    history: list,
    system_prompt: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
):
    user_text = _to_text(message).strip()
    if not user_text:
        yield ""
        return

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = _to_text(item.get("content", ""))
            if content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u, b = item
            u_text = _to_text(u)
            b_text = _to_text(b)
            if u_text:
                messages.append({"role": "user", "content": u_text})
            if b_text:
                messages.append({"role": "assistant", "content": b_text})

    messages.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        do_sample=True,
        eos_token_id=[tokenizer.eos_token_id, 2],
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text


demo = gr.ChatInterface(
    fn=predict,
    title="🇮🇩 Wicara 56M Chat",
    description=(
        "**Wicara** (56M parameters) is an efficient Indonesian language model trained from scratch "
        "on a single RTX 4050 6 GB laptop. Architecture: LLaMA-style (RMSNorm, RoPE, SwiGLU, GQA)."
    ),
    additional_inputs=[
        gr.Textbox(
            value="Kamu adalah asisten AI berbahasa Indonesia yang ramah.",
            label="System Prompt",
            lines=2,
        ),
        gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="Repetition Penalty"),
        gr.Slider(32, 256, value=160, step=16, label="Max New Tokens"),
    ],
    examples=[
        ["Halo! Siapa namamu?"],
        ["Apa kabar hari ini?"],
        ["Jelaskan cara membuat nasi goreng sederhana."],
        ["Ceritakan sesuatu yang menarik tentang Indonesia."],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    app, local_url, share_url = demo.launch(share=True)
    print(f"\n==========================================", flush=True)
    print(f"LOCAL URL : {local_url}", flush=True)
    print(f"PUBLIC URL: {share_url}", flush=True)
    print(f"==========================================\n", flush=True)
    if share_url:
        with open(REPO_ROOT / "space" / "public_url.txt", "w", encoding="utf-8") as f:
            f.write(share_url.strip())
