# Companion code — *Large Language Models from the Ground Up*

Runnable notebooks for the book **[Large Language Models from the Ground
Up](https://books.wazeem.com/llm)** by **Waseem Khan**.

Every notebook opens in **Google Colab** — free, in your browser, nothing
to install. Click a badge, then choose **Copy to Drive** to keep your work.
(Complete beginners: Chapter 15 of the book walks the two-minute workflow.)

## Notebooks

- **Ch 16 — Roadmap & reading Python** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/00_roadmap.ipynb)
- **Ch 17 — Math foundations** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/01_math_foundations.ipynb)
- **Ch 18 — Tokenization** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/02_tokenization.ipynb)
- **Ch 19–20 — Your first language model (bigram)** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/03_bigram_model.ipynb)
- **Ch 21–23 — Autograd from scratch (micrograd)** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/04_micrograd.ipynb)
- **Ch 24 — Self-attention** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/05_attention.ipynb)
- **Ch 25 — The transformer block** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/06_transformer_block.ipynb)
- **Ch 26 — PyTorch & autograd** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/07_pytorch_autograd.ipynb)
- **Ch 27 — Assembling your GPT** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/08_build_gpt.ipynb)
- **Ch 28 — Training your GPT** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/09_train_gpt.ipynb)
- **Ch 30 — Sampling & beyond** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/10_sampling_and_beyond.ipynb)
- **Ch 31 — Train your own BPE tokenizer** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/bpe_tokenizer.ipynb)
- **Ch 37 — Fine-tune a model with LoRA/QLoRA (needs T4 GPU)** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/finetune_lora.ipynb)
- **Ch 44 — RAG from scratch** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/rag_from_scratch.ipynb)
- **Ch 45 — Build a tool-using agent (needs local Ollama)** &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wazeemlabs/llm-book-code/blob/main/notebooks/agent_loop.ipynb)

The notebooks that need data (`data/input.txt`) or the model file (`gpt.py`)
download them automatically in their first cell when run on Colab, so each
opens and runs on its own.

## Run locally instead

```bash
git clone https://github.com/wazeemlabs/llm-book-code
cd llm-book-code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab            # then open the notebooks/ folder
```

Use Python 3.10–3.12 (PyTorch lags the newest release). Appendix A of the
book has the full setup and troubleshooting guide.

## What's here, and what isn't

This repository holds only the **code** — it is deliberately public so the
book's Colab links work for every reader. The book's text is a separate,
paid product and is **not** published here.

Found a bug in a notebook? Please [open an
issue](https://github.com/wazeemlabs/llm-book-code/issues).

## License

Code: MIT (see `LICENSE`). The Shakespeare text in `data/input.txt` is in
the public domain.
