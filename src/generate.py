"""
generate.py — Inference: turn an abstract into a story.

Implements greedy / top-k / temperature decoding with the trained model.
"""
import torch


@torch.no_grad()
def generate(model, tokenizer, abstract: str, max_len: int = 200,
             temperature: float = 1.0, top_k: int = 0, device: str = "cpu") -> str:
    """TODO: autoregressive decoding loop.

    1. encode the abstract -> memory
    2. start decoder with <bos>
    3. repeatedly predict next token, sample with temperature/top_k, append
    4. stop at <eos> or max_len
    5. decode token ids back to text
    """
    raise NotImplementedError
