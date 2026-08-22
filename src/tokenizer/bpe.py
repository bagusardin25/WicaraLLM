"""Tokenizer BPE byte-level untuk bahasa Indonesia."""

from collections.abc import Iterator
from pathlib import Path

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

from src.tokenizer.chat_template import ALL_SPECIAL_TOKENS, SPECIAL_TOKEN_IDS

VOCAB_SIZE = 16_384

# Pola pemecah awal, disederhanakan dari GPT-4 (cl100k) untuk bahasa Indonesia.
POLA_PRA_TOKENISASI = (
    r" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def build_tokenizer() -> Tokenizer:
    """Rakit tokenizer kosong yang siap dilatih."""
    tok = Tokenizer(models.BPE(unk_token=None))

    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(POLA_PRA_TOKENISASI),
            behavior="isolated",
        ),
        # add_prefix_space=False agar cocok dengan chat template.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def build_trainer(vocab_size: int = VOCAB_SIZE) -> trainers.BpeTrainer:
    """Trainer BPE dengan special token dipesan di ID paling depan."""
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(ALL_SPECIAL_TOKENS),
        # Seluruh 256 byte masuk alfabet awal.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )


def verify_special_tokens(tok: Tokenizer) -> None:
    """Pastikan ID special token sesuai dengan yang didefinisikan di chat_template."""
    salah = []
    for token, id_harapan in SPECIAL_TOKEN_IDS.items():
        id_nyata = tok.token_to_id(token)
        if id_nyata != id_harapan:
            salah.append(f"{token}: {id_nyata} (harusnya {id_harapan})")
    if salah:
        raise ValueError(
            "ID special token tidak sesuai chat_template.py:\n  "
            + "\n  ".join(salah)
        )


def compression_ratio(tok: Tokenizer, teks: list[str]) -> dict:
    """Hitung rasio kompresi (karakter per token). Target: 3.5-4.0."""
    n_char = sum(len(t) for t in teks)
    n_token = sum(len(e.ids) for e in tok.encode_batch(teks))
    return {
        "karakter": n_char,
        "token": n_token,
        "char_per_token": n_char / max(n_token, 1),
        "token_per_char": n_token / max(n_char, 1),
    }


def save(tok: Tokenizer, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return path


def load(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer tidak ada: {path}\n"
            "Jalankan scripts/train_tokenizer.py dulu."
        )
    return Tokenizer.from_file(str(path))


def iter_jsonl_gz_text(path: Path, setiap_ke: int = 1) -> Iterator[str]:
    """Baca teks dari jsonl.gz dengan sampling tiap dokumen ke-n."""
    import gzip
    import json

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, baris in enumerate(f):
            if i % setiap_ke:
                continue
            try:
                teks = json.loads(baris).get("text")
            except json.JSONDecodeError:
                continue
            if teks:
                yield teks
