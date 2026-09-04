[🇮🇩 Bahasa Indonesia](README.md) | **🇬🇧 English**

<img src="assets/wicarallm-logomark.png" alt="Logo Wicara" width="150">

# Wicara

**W**eight-efficient **I**ndonesian **C**onversational **A**rchitecture, **R**esearch **A**rtifact

A 56-million-parameter Indonesian language model. Trained from scratch on a
single RTX 4050 6 GB laptop.

*Wicara* (Sanskrit *vicāra*): speech, discourse. That is what it does.

Data, tokenizer, architecture, training, inference — we wrote it ourselves
in PyTorch. No off-the-shelf model, no API.

![tests](https://github.com/bagusardin25/WicaraLLM/actions/workflows/tests.yml/badge.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/num1notsvn/wicara-56m-chat)
[![GGUF Quantized](https://img.shields.io/badge/%F0%9F%A4%97%20GGUF-Quantized-green)](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF)
[![Ollama](https://img.shields.io/badge/Ollama-hf.co%2Fnum1notsvn%2Fwicara--56m--chat--GGUF-black)](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-orange)](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

```bash
# Option 1: Run directly via Ollama (~70 MB, no manual download/clone)
ollama run hf.co/num1notsvn/wicara-56m-chat-GGUF:Q8_0

# Option 2: Run locally with PyTorch
.venv\Scripts\python.exe scripts\chat.py
```

| You | Wicara |
|---|---|
| Halo! | Halo, Tn. |
| Apa kabar? | Baik, baik. |
| Siapa namamu? | Aku tidak punya nama. |
| Terima kasih banyak! | Tidak masalah, tidak masalah. |
| Sampai jumpa! | Sampai jumpa! |

`wicara-56m-chat` answers one turn and stops.
`wicara-56m-base` continues text — that is pretrain, before SFT.

Pretraining done. SFT done. Behavioral evaluation done (Phase 8). **217 tests.**

---

## Model

```
wicara-{size}-{type}
```

`56m` / `350m` / `1b` — parameter count. `base` is pretrain, `chat` has SFT.
First release: **`wicara-56m-base`**, then **`wicara-56m-chat`**. A generation
number (`wicara-2-…`) only appears if the architecture changes in a
fundamental way.

Llama-style: pre-norm RMSNorm, RoPE, GQA, SwiGLU, no bias, tied embeddings.

| | `wicara-56m` |
|---|---|
| Parameters | **56.0 million** (45.5 million non-embedding) |
| `d_model` / `n_layer` | 640 / 10 |
| Attention | 10 query / 5 KV (GQA 2:1) |
| `head_dim` / `d_ffn` | 64 / 1728 |
| Vocab | 16,384 byte-level BPE (4.26 characters/token) |
| Context | 512 tokens |
| Precision | bf16 autocast |

56 million because that is what fits in ~11 hours on this GPU. 1.1 billion
tokens puts it on the Chinchilla point (~20 tokens/parameter). `7m` and
`19m` are for fast iteration only.

### Pretrain — `wicara-56m-base`

| | Value |
|---|---|
| Val loss | **3.0505** (perplexity 21.2) |
| Train loss | 2.937 — gap to val 0.12 |
| Steps | 17,010 (1 epoch, 100% coverage) |
| Throughput | 28,017 tokens/sec |
| VRAM | 3.12 GB of 6.00 GB |
| Duration | 11.1 hours |

At step 17,000, prompt `"Halo, apa kabar?"` with no SFT at all:

```
Halo, ini aku, ada yang bisa kubantu?
Aku segera ke sana.
Halo, ini aku Anna.
```

Absorbed from phone conversations in the subtitle corpus. Greetings trigger
*aku/kau*; formal questions trigger *Saya/Anda*. Facts are unreliable:
`"Indonesia adalah negara"` continues as `"...yang paling maju di dunia."`

Details: [`notes/05-fase-5-pretrain.md`](notes/05-fase-5-pretrain.md).

### SFT — `wicara-56m-chat`

3,619 steps, ~20 minutes, val loss **1.37**. SFT loss is not comparable to
pretrain: only assistant-answer tokens are counted.

39,400 examples, mined from Aya, Cendol, and subtitles, plus 591 handwritten
examples to anchor persona ("Wicara") and curb hallucinations.

Details: [`notes/06-fase-6-sft.md`](notes/06-fase-6-sft.md) and [`notes/06b-fase-6b-tulis-tangan.md`](notes/06b-fase-6b-tulis-tangan.md).

---

## Corpus

**1.12 billion tokens** from 3,985,535 documents. Real text, no synthetic.

| Source | Role | Tokens | Share | Passed filter | char/token |
|---|---|---|---|---|---|
| [OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles) | Dialogue | 484M | 43.2% | 93.9% | 3.98 |
| [FineWeb-2 `ind_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Informal writing | 210M | 18.7% | 99.4% | 4.58 |
| [Wikipedia Indonesia](https://huggingface.co/datasets/wikimedia/wikipedia) | Full sentences | 174M | 15.5% | 83.1% | 4.16 |
| [Cendol v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | Instructions | 129M | 11.5% | 73.3% | 4.46 |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Q&A | 121M | 10.8% | 79.7% | 4.78 |
| [TED2020](https://opus.nlpl.eu/TED2020/id&id/v1/TED2020) | Formal speech | 3M | 0.3% | 98.7% | 4.92 |

`train.bin` 1,115,298,306 tokens, `val.bin` 5,683,512. Split at document
level — slicing the array would split documents across train and val.

Subtitles take the largest share because the focus is conversation, but they
are not allowed to dominate: a model fed only movies talks in fragments.
Wikipedia and FineWeb hold that back. CulturaX and OSCAR are gated;
FineWeb-2 is what we used.

The 16,384 BPE tokenizer was trained on the clean corpus. 4.26
characters/token — English tokenizers sit around 2.2 on Indonesian.
`" halo"`, `" hai"`, `" iya"`, `" tidak"` are each one token. 32 special
tokens at IDs 0–31 (8 live, 24 reserved) so later experiments do not need
an embedding resize.

Downloads, licenses, citations, SHA256: [`data/raw/SOURCES.md`](data/raw/SOURCES.md).
Filter stats: `data/clean/stats.json`.

---

## Repo

```
src/
  model/       rmsnorm · rope · attention (GQA+KV-cache) · ffn · transformer
  data/        sources · readers · clean · dedup · sft · sft_dataset
  train/       trainer · data (memmap) · lr_schedule · checkpoint
  infer/       generate · pencegat
  tokenizer/   bpe · chat_template
scripts/       check_env · download_corpus · verify_corpus · clean_corpus ·
               train_tokenizer · pack_tokens · train · train_sft ·
               build_sft_data · eval_perilaku · banding_eval ·
               export_to_hf · upload_to_hf · deploy_space · generate · chat
tests/         217 tests
notes/         notes per phase (01 to 08) + HANDOFF.md
space/         Gradio web demo app for Hugging Face Spaces
```

---

## Running

### Online demo (no local GPU required)
- **Hugging Face Space:** [num1notsvn/wicara-56m-chat-demo](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
- **Google Colab:** [Wicara_56M_Chat_Demo.ipynb](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

### Local machine
Python **3.11** from python.org (not MSYS2). NVIDIA GPU. PowerShell, not
Git Bash.

Chat, if a chat checkpoint is already there:

```powershell
.venv\Scripts\python.exe scripts\chat.py
.venv\Scripts\python.exe scripts\chat.py --uji
```

From scratch:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
.venv\Scripts\python.exe scripts\download_corpus.py
.venv\Scripts\python.exe scripts\verify_corpus.py
.venv\Scripts\python.exe scripts\clean_corpus.py
.venv\Scripts\python.exe scripts\train_tokenizer.py
.venv\Scripts\python.exe scripts\pack_tokens.py
.venv\Scripts\python.exe -m pytest tests\ -q
```

Also:

```powershell
.venv\Scripts\python.exe src\model\config.py
.venv\Scripts\python.exe scripts\bench_model.py --model 56m
.venv\Scripts\python.exe scripts\train.py --model 56m
.venv\Scripts\python.exe scripts\train_sft.py
```

---

## What we hit on 6 GB

Full notes in [`notes/`](notes/). The ones that keep biting:

- **Logits eat nearly as much VRAM as the whole Transformer.** Early
  estimates were off 2× from ignoring that (768 MB vs 880 MB at batch 8).
- **Windows does not throw OOM.** Batch 32 allocates 7.1 GB on a 6.0 GB GPU
  — spill to RAM, throughput drops 4×. Safe ceiling ~5.5 GB.
- **No FlashAttention in Windows PyTorch wheels.** `mem_efficient` is enough.
- **memmap locks files on Windows.** Call `TokenDataset.close()` before
  overwriting a `.bin`. Does not happen on Linux.
- **Line-level subtitle dedup drops 39.7%** — including `terima kasih` 845×
  and `halo` 458×, which are Zipfian. Block-level dedup is 3.2%.
- **Loss below ln(vocab) at the first steps = label leakage.** The number
  looks good, so the curve will not catch it.

---

## Limits

`wicara-56m` is fluent. It does not reason, does not do math, and cannot
be trusted on facts. Its hallucinations sound confident. That is 56 million
parameters, not a bug. Distance to a 7B model is about 125× parameters.

---

## License

Three things, three rules.

**Code** (`src/`, `scripts/`, `tests/`) — [Apache-2.0](LICENSE). Includes
a patent grant.

**Corpus** is not in the repo. Only the download recipe. Each source has
its own license; the list is in [`data/raw/SOURCES.md`](data/raw/SOURCES.md).

| Source | License |
|---|---|
| Wikipedia Indonesia | CC-BY-SA-3.0 |
| FineWeb-2 | ODC-BY-1.0 |
| Aya Collection | Apache-2.0 |
| Cendol v2 | Apache-2.0 |
| TED2020 | CC-BY-NC-ND-4.0 (non-commercial) |
| OpenSubtitles | see opus.nlpl.eu |

**Model weights:**
- **Chat Model (PyTorch / Safetensors):** [`num1notsvn/wicara-56m-chat`](https://huggingface.co/num1notsvn/wicara-56m-chat)
- **GGUF Format (Ollama & LM Studio):** [`num1notsvn/wicara-56m-chat-GGUF`](https://huggingface.co/num1notsvn/wicara-56m-chat-GGUF) (`Q8_0` ~70 MB & `F16` ~133 MB)
- **Interactive Web Demo:** [Hugging Face Space](https://huggingface.co/spaces/num1notsvn/wicara-56m-chat-demo)
- **Google Colab Notebook:** [Open In Colab](https://colab.research.google.com/github/bagusardin25/WicaraLLM/blob/main/Wicara_56M_Chat_Demo.ipynb)

Base model weights (`wicara-56m-base`) have not been released publicly yet. When published:
OpenSubtitles is ~43% of the corpus, and the legal status of weights as a
derivative work is unresolved anywhere; TED2020 is 0.3% and non-commercial
— for commercial use that source has to come out of `KOMPOSISI` in
`scripts/clean_corpus.py`, then retrain.

For research and learning, neither is a problem.

## Citation

```bibtex
@misc{ardin2026wicara,
  title={Wicara: A Weight-efficient Indonesian Conversational Architecture Built from Scratch},
  author={Bagus Ardin Prayoga},
  year={2026},
  url={https://github.com/bagusardin25/WicaraLLM}
}
```

Code needs Apache-2.0 attribution. Cite the corpus at the original
sources — full list in [`data/raw/SOURCES.md`](data/raw/SOURCES.md).
