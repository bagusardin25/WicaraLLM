"""Konversi checkpoint Wicara ke format standar Hugging Face (LlamaForCausalLM).

Skrip ini membaca checkpoint PyTorch Wicara (MiniLLM), memetakan nama tensor
ke format LLaMA standar, lalu menyimpannya dalam format safetensors lengkap
dengan config.json, tokenizer, dan generation_config.

Contoh pemakaian:
    .venv\\Scripts\\python.exe scripts\\export_to_hf.py
    .venv\\Scripts\\python.exe scripts\\export_to_hf.py --out-dir hf_export/wicara-56m-chat
    .venv\\Scripts\\python.exe scripts\\export_to_hf.py --push-to-hub num1notsvn/wicara-56m-chat
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer.chat_template import (  # noqa: E402
    ALL_SPECIAL_TOKENS,
    BOS_ID,
    DEFAULT_SYSTEM_PROMPT,
    END_TURN_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKEN_IDS,
    UNK_ID,
)

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "wicara-56m-chat-v5" / "best.pt"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "wicara-bpe-16k.json"
DEFAULT_OUT_DIR = REPO_ROOT / "hf_export" / "wicara-56m-chat"


def remap_state_dict_to_llama(state_dict: dict[str, torch.Tensor], n_layer: int) -> dict[str, torch.Tensor]:
    """Petakan nama-nama parameter Wicara (MiniLLM) ke standar LlamaForCausalLM."""
    hf_dict: dict[str, torch.Tensor] = {}

    # Token embeddings
    if "tok_emb.weight" in state_dict:
        hf_dict["model.embed_tokens.weight"] = state_dict["tok_emb.weight"]

    # Final norm
    if "norm_out.weight" in state_dict:
        hf_dict["model.norm.weight"] = state_dict["norm_out.weight"]

    # LM Head (jika tied, pastikan tetap ada atau tied ke embed_tokens)
    if "lm_head.weight" in state_dict:
        hf_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    elif "tok_emb.weight" in state_dict:
        hf_dict["lm_head.weight"] = state_dict["tok_emb.weight"]

    # Layer blocks
    for i in range(n_layer):
        prefix_wicara = f"blocks.{i}"
        prefix_llama = f"model.layers.{i}"

        # Attention Norm
        norm_attn_key = f"{prefix_wicara}.norm_attn.weight"
        if norm_attn_key in state_dict:
            hf_dict[f"{prefix_llama}.input_layernorm.weight"] = state_dict[norm_attn_key]

        # Attention Projections: wq, wk, wv, wo -> q_proj, k_proj, v_proj, o_proj
        wq_key = f"{prefix_wicara}.attn.wq.weight"
        wk_key = f"{prefix_wicara}.attn.wk.weight"
        wv_key = f"{prefix_wicara}.attn.wv.weight"
        wo_key = f"{prefix_wicara}.attn.wo.weight"

        if wq_key in state_dict:
            hf_dict[f"{prefix_llama}.self_attn.q_proj.weight"] = state_dict[wq_key]
        if wk_key in state_dict:
            hf_dict[f"{prefix_llama}.self_attn.k_proj.weight"] = state_dict[wk_key]
        if wv_key in state_dict:
            hf_dict[f"{prefix_llama}.self_attn.v_proj.weight"] = state_dict[wv_key]
        if wo_key in state_dict:
            hf_dict[f"{prefix_llama}.self_attn.o_proj.weight"] = state_dict[wo_key]

        # FFN Norm
        norm_ffn_key = f"{prefix_wicara}.norm_ffn.weight"
        if norm_ffn_key in state_dict:
            hf_dict[f"{prefix_llama}.post_attention_layernorm.weight"] = state_dict[norm_ffn_key]

        # SwiGLU Projections: w_gate, w_up, w_down -> gate_proj, up_proj, down_proj
        w_gate_key = f"{prefix_wicara}.ffn.w_gate.weight"
        w_up_key = f"{prefix_wicara}.ffn.w_up.weight"
        w_down_key = f"{prefix_wicara}.ffn.w_down.weight"

        if w_gate_key in state_dict:
            hf_dict[f"{prefix_llama}.mlp.gate_proj.weight"] = state_dict[w_gate_key]
        if w_up_key in state_dict:
            hf_dict[f"{prefix_llama}.mlp.up_proj.weight"] = state_dict[w_up_key]
        if w_down_key in state_dict:
            hf_dict[f"{prefix_llama}.mlp.down_proj.weight"] = state_dict[w_down_key]

    return hf_dict


def build_llama_config(model_cfg: dict[str, Any], dtype_str: str) -> dict[str, Any]:
    """Bangun konfigurasi LlamaConfig untuk config.json."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": BOS_ID,
        "eos_token_id": [EOS_ID, END_TURN_ID],
        "pad_token_id": PAD_ID,
        "hidden_act": "silu",
        "hidden_size": model_cfg.get("d_model", 640),
        "initializer_range": model_cfg.get("init_std", 0.02),
        "intermediate_size": model_cfg.get("d_ffn", 1728),
        "max_position_embeddings": model_cfg.get("max_seq_len", 512),
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": model_cfg.get("n_head", 10),
        "num_hidden_layers": model_cfg.get("n_layer", 10),
        "num_key_value_heads": model_cfg.get("n_kv_head", 5),
        "pretraining_tp": 1,
        "rms_norm_eps": model_cfg.get("norm_eps", 1e-5),
        "rope_scaling": None,
        "rope_theta": float(model_cfg.get("rope_theta", 10000.0)),
        "tie_word_embeddings": bool(model_cfg.get("tie_embeddings", True)),
        "torch_dtype": dtype_str,
        "transformers_version": "4.40.0",
        "use_cache": True,
        "vocab_size": model_cfg.get("vocab_size", 16384),
    }


def build_generation_config() -> dict[str, Any]:
    """Default generation hyperparameters."""
    return {
        "_from_model_config": True,
        "bos_token_id": BOS_ID,
        "eos_token_id": [EOS_ID, END_TURN_ID],
        "pad_token_id": PAD_ID,
        "do_sample": True,
        "max_new_tokens": 160,
        "repetition_penalty": 1.15,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "transformers_version": "4.40.0",
    }


def export_tokenizer_bundle(tokenizer_src: Path, out_dir: Path) -> None:
    """Simpan seluruh file tokenizer yang dibutuhkan oleh transformers."""
    # 1. tokenizer.json (fast tokenizer)
    shutil.copyfile(tokenizer_src, out_dir / "tokenizer.json")

    # 2. Jinja chat template yang cocok dengan render() Wicara
    jinja_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] != 'system' %}"
        "{{ '<|system|>" + DEFAULT_SYSTEM_PROMPT + "<|end|>' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>' + message['content'] + '<|end|>' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>' + message['content'] + '<|end|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>' + message['content'] + '<|end|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|assistant|>' }}"
        "{% endif %}"
    )

    # 3. tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|bos|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|end|>",
        "model_max_length": 512,
        "pad_token": "<|pad|>",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|unk|>",
        "chat_template": jinja_template,
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    # 4. special_tokens_map.json
    added_tokens = [tok for tok in ALL_SPECIAL_TOKENS if tok not in ["<|pad|>", "<|bos|>", "<|unk|>", "<|end|>"]]
    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "additional_special_tokens": added_tokens,
    }
    with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)


def write_model_card(out_dir: Path, repo_id: str | None = None) -> None:
    """Tulis Model Card README.md dengan frontmatter Hugging Face."""
    display_name = repo_id if repo_id else "Wicara 56M Chat"
    card_content = f"""---
language:
- id
license: apache-2.0
tags:
- wicara
- pytorch
- causal-lm
- text-generation
- conversational
- indonesian
pipeline_tag: text-generation
inference: false
---

# {display_name}

**Wicara** (*Weight-efficient Indonesian Conversational Architecture, Research Artifact*) adalah model bahasa Indonesia 56 juta parameter yang dilatih dari nol (*from scratch*) dengan efisiensi komputasi tinggi.

- **Bahasa**: Indonesia
- **Parameter**: 56,0 Juta (45,5 Juta non-embedding)
- **Arsitektur**: Gaya LLaMA (Pre-norm RMSNorm, RoPE, SwiGLU, Grouped-Query Attention, Tied Embeddings)
- **Tipe**: Chat / Instruct (SFT v5)
- **Lisensi**: Apache-2.0

## Arsitektur

| Komponen | Spesifikasi |
|---|---|
| Parameter | **56.027.520** |
| `d_model` / `n_layer` | 640 / 10 |
| Attention | 10 query / 5 KV (GQA 2:1) |
| `head_dim` / `d_ffn` | 64 / 1728 |
| Context Length | 512 token |
| Vocab Size | 16.384 BPE byte-level |

## Cara Penggunaan

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id or display_name}"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

messages = [
    {{"role": "system", "content": "Kamu adalah asisten AI berbahasa Indonesia yang ramah."}},
    {{"role": "user", "content": "Halo! Siapa namamu?"}},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=160,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    do_sample=True,
    eos_token_id=[tokenizer.eos_token_id, 2],
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```
"""
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model Wicara ke format Hugging Face")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Path ke checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER), help="Path ke wicara-bpe-16k.json")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Direktori output")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Tipe data bobot")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Repo ID di Hugging Face (misal: username/wicara-56m-chat)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    tokenizer_path = Path(args.tokenizer)
    out_dir = Path(args.out_dir)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return 1
    if not tokenizer_path.exists():
        print(f"[ERROR] Tokenizer tidak ditemukan: {tokenizer_path}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Memuat checkpoint dari: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_config_raw = raw_ckpt.get("model_config", {})
    state_dict = raw_ckpt.get("model", {})

    n_layer = model_config_raw.get("n_layer", 10)
    print(f"[*] Memetakan layer ({n_layer} blok) ke format LLaMA...")
    hf_state_dict = remap_state_dict_to_llama(state_dict, n_layer)

    # Konversi dtype
    target_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[*] Mengubah bobot tensor ke {args.dtype}...")
    for k, v in hf_state_dict.items():
        hf_state_dict[k] = v.to(target_dtype).contiguous()

    # Simpan safetensors
    weights_path = out_dir / "model.safetensors"
    print(f"[*] Menyimpan safetensors ke: {weights_path}")
    save_file(hf_state_dict, weights_path)

    # Simpan config.json
    print("[*] Menyimpan config.json...")
    llama_cfg = build_llama_config(model_config_raw, args.dtype)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(llama_cfg, f, indent=2)

    # Simpan generation_config.json
    print("[*] Menyimpan generation_config.json...")
    gen_cfg = build_generation_config()
    with open(out_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    # Simpan bundle tokenizer
    print("[*] Mengemas bundle tokenizer...")
    export_tokenizer_bundle(tokenizer_path, out_dir)

    # Buat Model Card README.md
    print("[*] Membuat Model Card README.md...")
    write_model_card(out_dir, repo_id=args.push_to_hub)

    print(f"\n[SUKSES] Seluruh file Hugging Face berhasil diekspor ke: {out_dir}")

    # Push to Hub jika diminta
    if args.push_to_hub:
        print(f"\n[*] Mengunggah ke Hugging Face Hub: {args.push_to_hub}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.push_to_hub, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=args.push_to_hub,
                repo_type="model",
            )
            print(f"[SUKSES] Model berhasil diunggah! Kunjungi: https://huggingface.co/{args.push_to_hub}")
        except Exception as e:
            print(f"[GAGAL UPLOAD] Terjadi kesalahan saat upload: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
