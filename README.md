# Bedtime Science 🌙🔬

**Translating dense research abstracts into children's bedtime stories.**

A sequence-to-sequence text generation system that takes a jargon-heavy scientific
abstract as input and rewrites it as a simple, imaginative story a five-year-old could
understand — turning *"quantum entanglement"* into *"two magic friends who always know
what the other is feeling, no matter how far apart they are."*

This is a course project for a Text Generation course. The core model is a
**Transformer (encoder–decoder) implemented from scratch in PyTorch**.

---

## Project status

| Stage | Status |
|-------|--------|
| Project scaffold | ✅ done |
| Data collection (arXiv abstracts) | ⬜ todo |
| Parallel data construction (abstract → story) | ⬜ todo |
| Tokenizer | ⬜ todo |
| From-scratch Transformer | ⬜ todo |
| Training loop | ⬜ todo |
| Evaluation (readability + fidelity) | ⬜ todo |
| Explainable AI (attention viz) | ⬜ todo |
| Streamlit GUI | ⬜ todo |

## Repository layout

```
bedtime-science/
├── src/              # model, tokenizer, training, generation code
│   ├── model.py          # from-scratch Transformer (encoder-decoder)
│   ├── tokenizer.py      # tokenizer wrapper
│   ├── dataset.py        # PyTorch Dataset / DataLoader
│   ├── train.py          # training loop
│   ├── generate.py       # inference / decoding
│   └── evaluate.py       # readability + fidelity metrics
├── scripts/          # one-off scripts (download data, build parallel corpus)
│   ├── download_arxiv.py
│   └── build_parallel_data.py
├── data/
│   ├── raw/              # downloaded arXiv abstracts (gitignored)
│   └── processed/        # parallel (abstract, story) pairs (gitignored)
├── app/              # Streamlit GUI
│   └── streamlit_app.py
├── notebooks/        # exploration & attention-visualization notebooks
├── tests/            # unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Approach

1. **Data** — Download arXiv abstracts (the "complex" side). Build the "simple" side as a
   parallel corpus by rewriting abstracts into child-friendly stories. The LLM (if used) only
   builds *training data*; the delivered translation model is trained by us.
2. **Model** — Encoder–decoder Transformer written from scratch: embeddings, multi-head
   self-attention, cross-attention, causal masking, feed-forward, residuals, layer norm.
3. **Evaluation** — Flesch–Kincaid readability drop + semantic-fidelity checks.
4. **Extra criteria** — Attention visualization (Explainable AI) + Streamlit GUI.

## License

MIT (see LICENSE).
