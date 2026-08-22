# Wicara

[🇮🇩 Bahasa Indonesia](README.md) | **🇬🇧 English**

**W**eight-efficient **I**ndonesian **C**onversational **A**rchitecture, **R**esearch **A**rtifact

An Indonesian language model built from scratch — data, architecture, training, to inference — on **a single laptop with a 6 GB RTX 4050**.

*Wicara* (from Sanskrit *vicāra*) means speech or discourse.

The goal is to learn how LLMs work completely by building one ourselves, rather than using an existing model. The functional target is intentionally simple: responding to basic Indonesian conversations (`" halo"`, `" apa kabar?"`).

> *Building an Indonesian language model from scratch on a single 6 GB laptop
> GPU. Every component — RMSNorm, RoPE, GQA, SwiGLU, KV-cache, training loop —
> written by hand in PyTorch. Documentation and code comments are in Indonesian.*

![tests](https://github.com/bagusardin25/WicaraLLM/actions/workflows/tests.yml/badge.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

---

## Status

| Phase | Content | Status |
|---|---|---|
| 0 | Environment & GPU benchmark | ✅ Done |
| 1 | Data pipeline (download, clean, dedup) | ✅ Done |
| 2 | 16k BPE tokenizer + packing | ✅ Done |
| 3 | Model architecture | ✅ Done |
| 3b | Sanity check | ✅ Done |
| 4 | Training loop | ✅ Done |
| 5 | Pretrain | ⬜ Ready to run — see [guide](notes/04-panduan-pretrain.md) |
| 6 | Conversational SFT | ⬜ |
| 7 | Inference engine & CLI | ⬜ |
| 8 | Evaluation | ⬜ |

**149 tests passed.** The model, trainer, checkpoints, corpus, and tokenizer are ready. `train.bin` contains **1,115,298,306 tokens** — pretraining is ready to run.

---

## Naming Convention

```
wicara-{size}-{type}
```

| Component | Rule | Examples |
|---|---|---|
| `size` | parameter count; `m` below 1 billion, `b` above | `56m`, `350m`, `1b` |
| `type` | training stage | `base`, `chat` |

| Type | Meaning |
|---|---|
| `base` | Pretrained only — fluent in language, cannot handle Q&A yet |
| `chat` | Has undergone SFT with a chat template |

First release: **`wicara-56m-base`** then **`wicara-56m-chat`**.

If the architecture fundamentally changes later, a generation number will be inserted after the name — `wicara-2-350m-base` — following the Llama pattern. As long as it is the first generation, the number is omitted.

In the code, the config is selected via a size-based key: `7m`, `19m`, `32m`, `56m`, `88m`.

---

## Specifications

Llama-style architecture: pre-norm RMSNorm, RoPE, GQA, SwiGLU, no bias, *tied embeddings*.

| | `wicara-56m` |
|---|---|
| Parameters | **56.0 million** (45.5 million non-embedding) |
| `d_model` / `n_layer` | 640 / 10 |
| Attention heads | 10 query / 5 key-value (GQA 2:1) |
| `head_dim` / `d_ffn` | 64 / 1728 |
| Vocab | 16,384 (custom byte-level BPE, 4.26 characters/token) |
| Context length | 512 tokens |
| Precision | bf16 autocast |
| Learning rate | 8.0e-4 (scaled by 1/`d_model`, muP) |

### Why 56M

The optimal model size is determined by the **time budget**, not VRAM capacity. All figures below are directly measured on a Laptop RTX 4050, not estimated:

| Config | Parameters | tok/sec | VRAM | MFU | 1.3B tokens |
|---|---|---|---|---|---|
| `32m` | 32.0M | 40,615 | 2.19 GB | 22.8% | 8.9 hours |
| **`56m`** | **56.0M** | **20,733** | **2.96 GB** | 22.5% | **17.4 hours** |
| `88m` | 88.1M | 14,276 | 4.10 GB | 25.7% | 25.3 hours |

For a compute budget of ~3.6×10¹⁷ FLOPs (≈17 hours on this GPU), the Chinchilla-optimal size is √(C/120) ≈ **55 million parameters**. Given the same time budget, 32M becomes overtrained (79 tokens/parameter) and 88M becomes undertrained (10 tokens/parameter). VRAM only becomes the bottleneck around 100M.

The `7m` and `19m` configs are used for rapid iteration and testing, not for release.

---

## Corpus

**1.12 billion tokens** from 3,985,535 documents. Entirely real text from public repositories — **no synthetic data**.

| Source | Role | Tokens | Share | Passed filter | char/token |
|---|---|---|---|---|---|
| [OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/id&id/v2024/OpenSubtitles) | Conversational dialogue | 484M | 43.2% | 93.9% | 3.98 |
| [FineWeb-2 `ind_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Informal written | 210M | 18.7% | 99.4% | 4.58 |
| [Wikipedia Indonesia](https://huggingface.co/datasets/wikimedia/wikipedia) | Factual, complete sentences | 174M | 15.5% | 83.1% | 4.16 |
| [Cendol v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | Indonesian instructions | 129M | 11.5% | 73.3% | 4.46 |
| [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Q&A | 121M | 10.8% | 79.7% | 4.78 |
| [TED2020](https://opus.nlpl.eu/TED2020/id&id/v1/TED2020) | Formal spoken | 3M | 0.3% | 98.7% | 4.92 |

Final result: `train.bin` **1,115,298,306 tokens**, `val.bin` 5,683,512 tokens.
Validation is split at the **document** level, not by slicing the final array — slicing would split documents between train and val, causing data leakage.

**19.9 tokens per parameter** — hitting exactly the Chinchilla-optimal point (20) for a 56M model, unplanned.

### Tokenizer

Byte-level BPE, 16,384 vocab size, trained on a **clean** corpus (not raw, to ensure no vocab slots are wasted on filtered garbage).

Compression ratio of **4.26 characters/token** — well above the planned target of 3.5-4.0. English-based tokenizers only reach ~2.2 for Indonesian text. The greetings targeted by this project are single tokens each: `" halo"`, `" hai"`, `" iya"`, `" tidak"`.

32 special tokens occupy IDs 0-31 (8 active + 24 reserved), automatically verified after training. The reserved slots allow adding new tokens later without retraining the tokenizer or resizing the embeddings.

Direct download links, licenses, citations, and SHA256 hashes for each file are in [`data/raw/SOURCES.md`](data/raw/SOURCES.md). Full cleaning statistics — including breakdowns per rejection reason — are in `data/clean/stats.json`.

Subtitles are given the largest portion to match the conversational focus, but deliberately do **not** dominate: a model fed entirely on subtitles will speak in fragments like movie dialogue. Wikipedia and FineWeb balance it out with complete sentences.

CulturaX and OSCAR — two of the most frequently recommended sources for Indonesian — turned out to be **gated** and require an account. FineWeb-2 was used instead: it's open, and its filtering and deduplication are stricter.

---

## Getting Started

Requires Python **3.11** from python.org (not MSYS2) and an NVIDIA GPU.

```bash
# 1. Environment
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Verify GPU + measure effective TFLOPS of your own machine
.venv\Scripts\python.exe scripts\check_env.py

# 3. Download corpus (~3.6 GB) then verify its integrity
.venv\Scripts\python.exe scripts\download_corpus.py
.venv\Scripts\python.exe scripts\verify_corpus.py

# 4. Clean corpus (~40 minutes, CPU only)
.venv\Scripts\python.exe scripts\clean_corpus.py

# 5. Train 16k BPE tokenizer then pack to .bin (~8 minutes)
.venv\Scripts\python.exe scripts\train_tokenizer.py
.venv\Scripts\python.exe scripts\pack_tokens.py

# 6. Test
.venv\Scripts\python.exe -m pytest tests\ -q
```

Other useful commands:

```bash
.venv\Scripts\python.exe src\model\config.py                 # parameters & VRAM per config
.venv\Scripts\python.exe scripts\bench_model.py --model 56m  # measure throughput
.venv\Scripts\python.exe scripts\demo_cleaning.py            # cleaning demonstration
```

---

## Structure

```
src/
  model/       rmsnorm · rope · attention (GQA+KV-cache) · ffn · transformer
  data/        sources · readers · clean · dedup
  train/       trainer · data (memmap, full epoch) · lr_schedule · checkpoint
  infer/       generate (KV-cache, top-k/top-p, repetition penalty)
  tokenizer/   bpe · chat_template
scripts/       check_env · download_corpus · verify_corpus · clean_corpus ·
               demo_cleaning · train_tokenizer · pack_tokens · bench_model ·
               train · generate
tests/         149 tests
notes/         notes per phase + LLM glossary (PDF)
```

---

## Technical Notes

A few findings from building this on a 6 GB GPU, full details in [`notes/`](notes/):

**Logits VRAM is massive and easily overlooked.** In small models with large vocabularies, the logits + cross-entropy tensors can consume nearly as much as the entire Transformer layers (768 MB vs 880 MB at batch 8). Early estimates were off by 2× from ignoring this.

**VRAM spills on Windows don't throw an error.** At batch 32, PyTorch allocates 7.1 GB on a 6.0 GB GPU without an `OutOfMemoryError` — Windows silently spills it to system RAM, and throughput drops 4×. The practical safe limit is ~5.5 GB, not 6.0 GB.

**FlashAttention is unavailable in Windows PyTorch wheels.** Not a problem: the `mem_efficient` kernel also avoids materializing the seq×seq matrix in memory.

**memmap locks files on Windows.** A mapped file cannot be overwritten, meaning the data pipeline fails as long as a `TokenDataset` is alive. This doesn't happen on Linux, so it's easy to miss.

**Deduplication at the wrong level does more harm than no dedup.** Line-level dedup on subtitles discarded 39.7% of the data — but what got discarded were the most common conversational phrases (`terima kasih` [thank you] 845×, `halo` [hello] 458×). Natural language is Zipfian; those phrases *should* appear frequently. Block-level dedup only discarded 3.2%, which is actual duplication.

**`initial loss = ln(vocab)` is a test tool, not just info.** A loss below that number implies label leakage — and leakage makes numbers look *good*, so it won't be caught by looking at the loss curve.

---

## Limitations

Needs to be stated upfront, as the project name doesn't promise it: `wicara-56m` **cannot reason**. A model this size can generate grammatical Indonesian, respond to greetings with the appropriate tone, and maintain coherence for one to three sentences. It will **not** be factually accurate, cannot do math, cannot perform step-by-step reasoning, and will confidently hallucinate.

That is not a bug — that's just the capacity of 56 million parameters. The distance to a 7B model is roughly 125× the parameters and 1000× the compute.

---

## License

In an LLM project, there are **three different things** with distinct licenses, and conflating them is a common mistake:

### 1. Code — Apache-2.0

The entire contents of `src/`, `scripts/`, and `tests/` are licensed under [Apache-2.0](LICENSE). Free to use, modify, and distribute, including for commercial purposes, with attribution. Apache was chosen (rather than MIT) because it includes an explicit patent grant.

### 2. Corpus — not redistributed

This repo does **not** host a single byte of corpus text. What is included are the recipes to build it: `scripts/download_corpus.py` downloads directly from the original sources, and [`data/raw/SOURCES.md`](data/raw/SOURCES.md) records the links, licenses, citations, and SHA256 hashes for each file.

The license for each source varies and must be respected individually:

| Source | License | Note |
|---|---|---|
| Wikipedia Indonesia | CC-BY-SA-3.0 | *share-alike* |
| FineWeb-2 | ODC-BY-1.0 | attribution |
| Aya Collection | Apache-2.0 | free |
| Cendol v2 | Apache-2.0 | free |
| TED2020 | CC-BY-NC-ND-4.0 | **non-commercial** |
| OpenSubtitles | see opus.nlpl.eu | most unclear status |

### 3. Model weights — none yet, requires thought before release

No weights have been released yet. When `wicara-56m-base` is published later, these two things must be clearly stated in its model card:

**OpenSubtitles contributes 40% of the corpus.** It contains movie dialogues uploaded by users to opensubtitles.org. OPUS distributes it for research purposes, and training models on it is standard practice in research. But the legal status of *model weights* as derivative works is unresolved anywhere, and this repo does not pretend to know the answer.

**TED2020 has a non-commercial license** (CC-BY-NC-ND). Its portion is only 0.3%, but if the model weights are to be used commercially, this source should be removed and the model retrained without it — corpus composition can be configured in `KOMPOSISI` inside `scripts/clean_corpus.py`.

For use as a learning and research resource — the goal of this project — both of the above are not an issue. They only matter if the direction shifts to commercial.

---

## Citation

```bibtex
@misc{ardin2026wicara,
  title={Wicara: A Weight-efficient Indonesian Conversational Architecture Built from Scratch},
  author={Bagus Ardin Prayoga},
  year={2026},
  url={https://github.com/bagusardin25/WicaraLLM}
}
```

## Attribution

If using this code, simply include the attribution required by Apache-2.0. The corpus must be cited to their respective original sources — the complete citation list is in [`data/raw/SOURCES.md`](data/raw/SOURCES.md).
