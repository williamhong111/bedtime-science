import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.generate import load_model, get_device
from src.dataset import PAD_ID, BOS_ID, EOS_ID


@torch.no_grad()
def generate_with_attention(model, tok, abstract, device, max_new=80, temperature=0.9, top_k=40):
    src_ids = tok.encode(abstract).ids[:512]
    src = torch.tensor([src_ids], device=device)
    src_mask = model.src_mask(src)
    memory = model.encode(src, src_mask)

    out = torch.tensor([[BOS_ID]], device=device)
    attn_rows = []      # one row per generated token (averaged over heads, last layer)
    gen_tokens = []     # the token id we picked at each step

    for _ in range(max_new):
        tgt_mask = model.tgt_mask(out)
        dec = model.decode(out, memory, tgt_mask, src_mask)

        # last decoder layer's cross-attention: shape (1, n_heads, T_dec, T_src)
        cross = model.decoder[-1].cross_attn.last_attn
        last_row = cross[0, :, -1, :].mean(dim=0)        # avg over heads -> (T_src,)
        attn_rows.append(last_row.cpu().numpy())

        logits = model.out(dec[:, -1, :]) / max(temperature, 1e-6)
        if top_k > 0:
            vals, idx = logits.topk(top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            choice = idx.gather(-1, torch.multinomial(probs, 1))
        else:
            probs = F.softmax(logits, dim=-1)
            choice = torch.multinomial(probs, 1)

        cid = choice.item()
        gen_tokens.append(cid)
        out = torch.cat([out, choice], dim=1)
        if cid == EOS_ID:
            break

    # decode tokens (one at a time) so we keep them as labels
    src_labels = [tok.decode([i]).strip() or "·" for i in src_ids]
    tgt_labels = [tok.decode([i]).strip() or "·" for i in gen_tokens]

    return np.stack(attn_rows), src_labels, tgt_labels


def plot(attn, src_labels, tgt_labels, out_path, src_window=80):
    # the source side can be hundreds of tokens long; keep first src_window for legibility
    n_src = min(src_window, attn.shape[1])
    attn = attn[:, :n_src]
    src_labels = src_labels[:n_src]

    fig_w = max(10, n_src * 0.18)
    fig_h = max(6, len(tgt_labels) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_src))
    ax.set_xticklabels(src_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_yticklabels(tgt_labels, fontsize=8)
    ax.set_xlabel("source (abstract) tokens")
    ax.set_ylabel("generated (story) tokens")
    ax.set_title("Cross-attention (last decoder layer, averaged over heads)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--abstract", default=(
        "We report the discovery of a new exoplanet orbiting a Sun-like star, "
        "detected via the transit method using TESS photometry, with an orbital "
        "period of 3.5 days and a mass consistent with a hot Jupiter."
    ))
    p.add_argument("--out", default="notebooks/attention.png")
    p.add_argument("--max_new", type=int, default=60)
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer, device)
    attn, src_labels, tgt_labels = generate_with_attention(
        model, tok, args.abstract, device, max_new=args.max_new
    )
    print("attention shape:", attn.shape)
    print("generated:", " ".join(tgt_labels))
    plot(attn, src_labels, tgt_labels, args.out)
