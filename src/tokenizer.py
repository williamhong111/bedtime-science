"""
tokenizer.py — Tokenizer wrapper.

Plan: train a Byte-Pair Encoding (BPE) tokenizer on the corpus, or reuse a
pretrained one. Exposes encode / decode and special-token ids.
"""
from tokenizers import Tokenizer  # noqa: F401  (used when implemented)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


class StoryTokenizer:
    def __init__(self, tokenizer_path: str | None = None):
        # TODO: load a trained tokenizer from tokenizer_path, or train one.
        self.tokenizer = None
        raise NotImplementedError("TODO: implement tokenizer loading/training")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        raise NotImplementedError
