"""Basic shape tests for the from-scratch Transformer."""
import torch
from src.model import Seq2SeqTransformer


def test_forward_shape():
    model = Seq2SeqTransformer(vocab_size=500, d_model=64, n_heads=4, n_layers=2, d_ff=128)
    src = torch.randint(1, 500, (3, 18))
    tgt = torch.randint(1, 500, (3, 12))
    logits = model(src, tgt)
    assert logits.shape == (3, 12, 500)


def test_attention_is_saved():
    model = Seq2SeqTransformer(vocab_size=500, d_model=64, n_heads=4, n_layers=2, d_ff=128)
    src = torch.randint(1, 500, (1, 10))
    tgt = torch.randint(1, 500, (1, 8))
    _ = model(src, tgt)
    # cross-attention weights should be captured for the Explainable-AI module
    assert model.decoder[0].cross_attn.last_attn is not None
