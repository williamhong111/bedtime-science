import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.last_attn = None

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.wq(q).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
        x = self.n3(x + self.drop(self.ff(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def tgt_mask(self, tgt):
        B, T = tgt.shape
        pad = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        return pad & causal

    def encode(self, src, mask):
        x = self.drop(self.pos(self.embed(src)))
        for layer in self.encoder:
            x = layer(x, mask)
        return x

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.drop(self.pos(self.embed(tgt)))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt):
        sm = self.src_mask(src)
        tm = self.tgt_mask(tgt)
        memory = self.encode(src, sm)
        dec = self.decode(tgt, memory, tm, sm)
        return self.out(dec)


if __name__ == "__main__":
    m = Seq2SeqTransformer(vocab_size=1000)
    src = torch.randint(1, 1000, (2, 20))
    tgt = torch.randint(1, 1000, (2, 15))
    print(m(src, tgt).shape)
