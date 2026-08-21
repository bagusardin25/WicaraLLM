"""Special token dan chat template.

Modul ini murni Python (tanpa torch/tokenizers) supaya bisa di-import dari mana
saja, termasuk saat melatih tokenizer.

Kenapa special token didefinisikan SEBELUM tokenizer dilatih:
    Ukuran tabel embedding = vocab_size * d_model. Kalau token baru ditambahkan
    setelah pretrain, tabel embedding harus di-resize dan baris barunya mulai
    dari acak — sementara sisa model sudah matang. Hasilnya tidak stabil.
    Jauh lebih mudah memesan slot-nya sejak awal, termasuk slot cadangan yang
    belum terpakai, supaya eksperimen berikutnya tidak perlu resize.
"""

from dataclasses import dataclass

# --- Token khusus, ID 0..N ------------------------------------------------
# Urutan ini dikunci: ID-nya tertanam di dataset .bin dan di checkpoint.
# JANGAN diubah setelah tokenizer dilatih dan data di-pack.

PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
UNK = "<|unk|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end|>"

SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, UNK, SYSTEM, USER, ASSISTANT, END_TURN]

# Slot cadangan: memberi ruang untuk eksperimen di masa depan (tool-calling,
# penanda dokumen, penanda bahasa) tanpa perlu melatih ulang tokenizer.
N_RESERVED = 24
RESERVED_TOKENS: list[str] = [f"<|reserved_{i}|>" for i in range(N_RESERVED)]

ALL_SPECIAL_TOKENS: list[str] = SPECIAL_TOKENS + RESERVED_TOKENS

# ID ditetapkan berdasarkan posisi — tokenizer harus menambahkannya lebih dulu.
SPECIAL_TOKEN_IDS: dict[str, int] = {
    tok: idx for idx, tok in enumerate(ALL_SPECIAL_TOKENS)
}

PAD_ID = SPECIAL_TOKEN_IDS[PAD]
BOS_ID = SPECIAL_TOKEN_IDS[BOS]
EOS_ID = SPECIAL_TOKEN_IDS[EOS]
UNK_ID = SPECIAL_TOKEN_IDS[UNK]

DEFAULT_SYSTEM_PROMPT = "Kamu adalah asisten AI berbahasa Indonesia yang ramah."


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


_ROLE_TOKEN = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def render(
    messages: list[Message],
    add_generation_prompt: bool = False,
    system_prompt: str | None = None,
) -> str:
    """Ubah daftar pesan menjadi satu string siap-tokenisasi.

    Format:
        <|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|eos|>

    Kenapa ada <|end|> DAN <|eos|>: <|end|> menutup satu giliran (model berhenti
    di sini saat generate), sedangkan <|eos|> menutup seluruh percakapan sebagai
    satu dokumen training. Tanpa pemisahan ini, model tidak punya sinyal jelas
    kapan harus berhenti bicara dan malah lanjut mengarang giliran user.

    Args:
        add_generation_prompt: kalau True, akhiri dengan `<|assistant|>` tanpa isi
            — dipakai saat inference supaya model langsung melanjutkan sebagai
            asisten.
    """
    parts = [BOS]

    if system_prompt is not None:
        parts.append(f"{SYSTEM}{system_prompt}{END_TURN}")

    for msg in messages:
        token = _ROLE_TOKEN.get(msg.role)
        if token is None:
            raise ValueError(f"Role tidak dikenal: {msg.role!r}")
        parts.append(f"{token}{msg.content}{END_TURN}")

    if add_generation_prompt:
        parts.append(ASSISTANT)
    else:
        parts.append(EOS)

    return "".join(parts)


def assistant_spans(messages: list[Message], system_prompt: str | None = None
                    ) -> list[tuple[int, int]]:
    """Rentang karakter (start, end) dari isi tiap giliran asisten.

    Dipakai Fase 6 untuk loss masking: loss HANYA dihitung pada token asisten,
    tidak pada prompt user. Ini bug paling umum di SFT — kalau loss ikut dihitung
    di teks user, model belajar meniru user dan bukan menjawabnya.

    Rentang mencakup <|end|> penutup, supaya model belajar KAPAN harus berhenti.
    """
    spans: list[tuple[int, int]] = []
    pos = len(BOS)

    if system_prompt is not None:
        pos += len(SYSTEM) + len(system_prompt) + len(END_TURN)

    for msg in messages:
        token = _ROLE_TOKEN[msg.role]
        content_start = pos + len(token)
        content_end = content_start + len(msg.content) + len(END_TURN)
        if msg.role == "assistant":
            spans.append((content_start, content_end))
        pos = content_end

    return spans


if __name__ == "__main__":
    convo = [
        Message("user", "halo"),
        Message("assistant", "Halo! Ada yang bisa saya bantu?"),
        Message("user", "apa kabar?"),
        Message("assistant", "Baik, terima kasih sudah bertanya!"),
    ]

    print(f"Total special token : {len(ALL_SPECIAL_TOKENS)} "
          f"({len(SPECIAL_TOKENS)} aktif + {N_RESERVED} cadangan)")
    print(f"ID token khusus     : {dict(list(SPECIAL_TOKEN_IDS.items())[:8])}")

    text = render(convo, system_prompt=DEFAULT_SYSTEM_PROMPT)
    print(f"\n--- training (dengan system prompt) ---\n{text}")

    print("\n--- rentang yang dihitung loss-nya (SFT) ---")
    for start, end in assistant_spans(convo, DEFAULT_SYSTEM_PROMPT):
        print(f"  [{start:4d}:{end:4d}] {text[start:end]!r}")

    print("\n--- inference (siap dilanjutkan model) ---")
    print(render([Message("user", "halo")], add_generation_prompt=True))
