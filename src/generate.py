import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model import Seq2SeqTransformer
from src.dataset import PAD_ID, BOS_ID, EOS_ID


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(ckpt_path, tokenizer_path, device):
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    ckpt = torch.load(ckpt_path, map_location=device)
    # default architecture has to match what we trained with
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=256, n_heads=8, n_layers=4, d_ff=1024,
        max_len=512, pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, abstract, device, max_new=400, temperature=1.0, top_k=40):
    src_ids = tok.encode(abstract).ids[:512]
    src = torch.tensor([src_ids], device=device)
    src_mask = model.src_mask(src)
    memory = model.encode(src, src_mask)

    out = torch.tensor([[BOS_ID]], device=device)
    for _ in range(max_new):
        tgt_mask = model.tgt_mask(out)
        dec = model.decode(out, memory, tgt_mask, src_mask)
        logits = model.out(dec[:, -1, :]) / max(temperature, 1e-6)

        if top_k > 0:
            vals, idx = logits.topk(top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            choice = idx.gather(-1, torch.multinomial(probs, 1))
        else:
            probs = F.softmax(logits, dim=-1)
            choice = torch.multinomial(probs, 1)

        out = torch.cat([out, choice], dim=1)
        if choice.item() == EOS_ID:
            break

    ids = out[0].tolist()
    if ids[0] == BOS_ID:
        ids = ids[1:]
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    return tok.decode(ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--abstract", required=True)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer, device)
    story = generate(model, tok, args.abstract, device,
                     temperature=args.temperature, top_k=args.top_k)
    print(story)
