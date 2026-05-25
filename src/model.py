"""
model.py — A sequence-to-sequence Transformer (encoder-decoder), from scratch.

This is the core of the project: instead of importing a black-box model, we build
the Transformer ourselves so every component can be explained.

Architecture (Vaswani et al., 2017, "Attention Is All You Need"):

    abstract tokens ──> [Encoder] ──> memory
                                         │
    story tokens (shifted) ──> [Decoder w/ cross-attention to memory] ──> next-token logits

Components implemented here:
  - Token + positional embeddings
  - Multi-head attention (self & cross)
  - Position-wise feed-forward
  - Residual connections + layer normalization
  - Causal mask (decoder) + padding mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Positional encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Adds sinusoidal position information to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# 2. Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with multiple heads.

    Used three ways:
      - encoder self-attention   (q=k=v=encoder states, no causal mask)
      - decoder self-attention   (q=k=v=decoder states, causal mask)
      - decoder cross-attention  (q=decoder states, k=v=encoder memory)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Stored after each forward pass so the Explainable-AI module can read it.
        self.last_attn = None

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        def split_heads(x):
            # (B, T, d_model) -> (B, n_heads, T, d_head)
            return x.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()  # save for visualization

        out = torch.matmul(attn, v)                       # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# 3. Position-wise feed-forward
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. Encoder & Decoder layers
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# 5. Full model
# ---------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d_model, vocab_size)

    # --- masks ---
    def make_src_mask(self, src):
        # (B, 1, 1, S) — 1 where token is real, 0 where padding
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)         # (B,1,1,T)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T,T)
        return pad_mask & causal

    # --- forward ---
    def encode(self, src, src_mask):
        x = self.dropout(self.pos(self.embed(src)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.dropout(self.pos(self.embed(tgt)))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        dec = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.out(dec)  # (B, T, vocab_size)


if __name__ == "__main__":
    # tiny smoke test
    model = Seq2SeqTransformer(vocab_size=1000)
    src = torch.randint(1, 1000, (2, 20))
    tgt = torch.randint(1, 1000, (2, 15))
    logits = model(src, tgt)
    print("logits shape:", logits.shape)  # expected (2, 15, 1000)
