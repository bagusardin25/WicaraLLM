"""Pembangkitan teks dengan KV-cache untuk pengujian model base."""

import torch
import torch.nn.functional as F

from src.model.transformer import MiniLLM
from src.tokenizer.chat_template import BOS_ID, EOS_ID


def _saring_logits(
    logit: torch.Tensor, top_k: int = 0, top_p: float = 0.0
) -> torch.Tensor:
    """Terapkan filter top-k dan top-p pada logits sebelum sampling."""
    if top_k and top_k < logit.size(-1):
        ambang = torch.topk(logit, top_k)[0][..., -1:]
        logit = logit.masked_fill(logit < ambang, float("-inf"))

    if top_p and 0.0 < top_p < 1.0:
        urut, idx = torch.sort(logit, descending=True, dim=-1)
        kumulatif = torch.cumsum(F.softmax(urut, dim=-1), dim=-1)
        buang = kumulatif - F.softmax(urut, dim=-1) > top_p
        urut = urut.masked_fill(buang, float("-inf"))
        logit = torch.empty_like(logit).scatter_(-1, idx, urut)

    return logit


@torch.no_grad()
def generate(
    model: MiniLLM,
    tokenizer,
    prompt: str = "",
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_at_eos: bool = True,
    device: str | torch.device = "cuda",
    seed: int | None = None,
) -> str:
    """Hasilkan lanjutan teks dari sebuah prompt untuk model base."""
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = torch.device(device)

    ids = [BOS_ID] + (tokenizer.encode(prompt).ids if prompt else [])
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    caches = model.make_caches(1, model.cfg.max_seq_len)
    keluar: list[int] = []
    autocast = device.type == "cuda"

    for langkah in range(max_new_tokens):
        if caches[0].length + idx.shape[1] > model.cfg.max_seq_len:
            break  # konteks penuh

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            logits, _ = model(idx, caches=caches, last_token_only=True)

        logit = logits[0, -1].float()

        # Terapkan repetition penalty.
        if repetition_penalty != 1.0 and keluar:
            unik = torch.tensor(sorted(set(keluar)), device=device)
            nilai = logit[unik]
            logit[unik] = torch.where(
                nilai > 0, nilai / repetition_penalty, nilai * repetition_penalty
            )

        if temperature <= 0:
            nxt = int(logit.argmax())  # greedy
        else:
            logit = _saring_logits(logit / temperature, top_k, top_p)
            nxt = int(torch.multinomial(F.softmax(logit, dim=-1), 1))

        if stop_at_eos and nxt == EOS_ID:
            break

        keluar.append(nxt)
        idx = torch.tensor([[nxt]], dtype=torch.long, device=device)

    return tokenizer.decode(keluar, skip_special_tokens=False)
