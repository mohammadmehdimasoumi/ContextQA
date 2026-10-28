# ContextQA — Transformer-Based Extractive Question Answering

ContextQA takes a passage of text and a question, then **extracts the answer span from that passage**
with a pretrained Transformer encoder. It also recognizes questions that the passage cannot answer
(with the SQuAD 2.0 model). It is an extractive QA system — not a generative chatbot and not a RAG app.

```
Context:  "Python was created by Guido van Rossum and was first released in 1991."
Question: "Who created Python?"
Answer:   "Guido van Rossum"
```

## Problem statement and motivation

Extractive QA is a compact way to study how Transformer encoders are used for token-level
prediction: no text generation, only a choice of *where* the answer starts and ends. This project is
a learning-oriented portfolio piece that implements the full inference and evaluation pipeline
around pretrained models (tokenization, sliding windows, offset mapping, span decoding, SQuAD 2.0
metrics, latency/size benchmarking, a small web UI) instead of calling a black-box `pipeline`.

## Model architecture and inference workflow

Both models are Transformer **encoders** with a QA head: a linear layer maps each token's final hidden
state to two numbers, a *start logit* and an *end logit*. The answer is the span whose start/end tokens
score highest.

| Model | Layers | Params (approx.) | Training data | Can say "no answer"? |
|---|---|---|---|---|
| `deepset/bert-base-cased-squad2` | 12 | ~110M | SQuAD 2.0 | Yes |
| `distilbert/distilbert-base-cased-distilled-squad` | 6 | ~65M | SQuAD 1.1 (distilled from BERT) | **No** — always returns a span |

DistilBERT keeps most of BERT's accuracy with fewer layers (knowledge distillation), trading a little
quality for lower latency and memory. Exact numbers for *your* machine come from the evaluation script.

Pipeline (`src/contextqa/`):

1. **Validate** (`preprocessing.validate_inputs`): type, emptiness, length limits.
2. **Tokenize** (`preprocessing.tokenize_pair`): `[CLS] question [SEP] context [SEP]`, with the context cut
   into overlapping windows (`max_length=384`, `stride=128`) using fast-tokenizer offsets; question and
   special tokens get no offset so they can never be selected.
3. **Forward** (`inference.QAEngine`): windows are processed in small batches; `token_type_ids` are passed
   only if the tokenizer produces them (BERT yes, DistilBERT no).
4. **Decode** (`inference.decode_answer`): top-k start × top-k end candidates, valid spans only
   (inside context, end ≥ start, ≤ 30 tokens), score = start_logit + end_logit. Duplicate spans from
   overlapping windows are merged. For SQuAD 2.0 models, "no answer" wins when
   `null_score − best_span_score > threshold`, where the null score is the `[CLS]` start+end logit
   (minimum over windows).
5. **Map back**: character offsets point into the *original* context, so `context[start:end] == answer`.

**Confidence.** The reported value is a softmax over the model's own candidate spans (plus "no answer"
when supported). It is a *relative model score*, **not a calibrated probability** of being correct.

## Technology stack

Python 3.11+, PyTorch, Hugging Face Transformers + Datasets, NumPy, Gradio, pytest.
No LangChain, vector DBs, RAG or external LLM APIs. (`evaluate` is listed as an optional cross-check;
the metrics are implemented locally following the official SQuAD v2 script.)

## Installation (Windows, PowerShell)

```powershell
git clone <your-repo-url> ContextQA
cd ContextQA
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch first. For an NVIDIA GPU pick the CUDA build matching your driver at pytorch.org, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU-only alternative: pip install torch

pip install -r requirements.txt
```

Models and datasets are downloaded once and cached by Hugging Face (`%USERPROFILE%\.cache\huggingface`).

## Usage

```powershell
python -m pytest                                   # unit tests (no model download)
python app.py                                      # Gradio app at http://127.0.0.1:7860
python scripts/evaluate_models.py --num-examples 200 --seed 42   # BERT vs DistilBERT on SQuAD 2.0
```

Optional real-model smoke tests (download the models): `set RUN_MODEL_TESTS=1` then
`python -m pytest tests/test_integration.py`.

Python usage:

```python
from contextqa.inference import QAEngine
r = QAEngine("bert").answer("Who created Python?", "Python was created by Guido van Rossum in 1991.")
print(r.answer, r.has_answer, r.confidence)
```

## Evaluation methodology

- **Data:** SQuAD 2.0 validation split (`rajpurkar/squad_v2`), a seeded random subset of configurable size,
  identical for both models.
- **Metrics:** Exact Match and token-level F1 after the official normalization (lowercase, strip punctuation
  and articles, collapse whitespace), maximum over all gold answers. For unanswerable questions the gold
  answer is `""`: predicting nothing scores 1, anything else 0. "Raw" (un-normalized) matching would be stricter
  and score lower. Results are reported overall, on answerable questions (HasAns) and on unanswerable ones (NoAns).
- **Latency:** end-to-end per example (tokenize + forward + decode) after 3 warm-up calls; mean, median, p95.
- **Size:** parameter count and fp32 weight bytes.

### Results (template — generate by running the script)

> No benchmark results are included. Run `scripts/evaluate_models.py` and paste `results/results.md` here.

| Model | N | EM | F1 | HasAns F1 | NoAns F1 | Mean ms | p95 ms | Params (M) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| deepset/bert-base-cased-squad2 | _run it_ | | | | | | | | |
| distilbert/distilbert-base-cased-distilled-squad | _run it_ | | | | | | | | |

Hardware used: _fill in (GPU/CPU, PyTorch version) — also saved in `results/results.json`._

## BERT vs DistilBERT

DistilBERT is expected to be faster and smaller. Compare them on **HasAns F1**: it was trained on SQuAD 1.1 and cannot
abstain, so its NoAns score is near zero and its overall EM/F1 is penalized by design. Interpret overall numbers
only together with the HasAns/NoAns split.

## Known limitations

- Extractive only: the answer must appear in the context; no reasoning across passages, no generation.
- Contexts over 50,000 characters are rejected; spans longer than 30 tokens are never returned.
- The model score is uncalibrated; "no answer" depends on a threshold (`InferenceConfig.no_answer_threshold`).
- DistilBERT cannot abstain.
- Small evaluation subsets are not statistically representative of the full validation set.
- English only; models are cased.

## Hardware and reproducibility notes

Developed for a laptop with 6 GB VRAM / 16 GB RAM: small batch size (8 windows), CPU fallback is automatic.
Evaluation subsets are chosen with a fixed seed, so the same `--seed` and `--num-examples` select the same
examples. GPU kernels and library versions can still cause small numeric differences between machines;
record your versions with the results.

## Project structure

```
app.py                       Gradio entry point
src/contextqa/config.py      models + hyperparameters
src/contextqa/preprocessing.py   validation, windowed tokenization
src/contextqa/inference.py   QAEngine, span decoding
src/contextqa/evaluation.py  EM/F1, latency, size, tables
src/contextqa/ui.py          UI formatting/error handling (Gradio-free)
src/contextqa/utils.py       exceptions, logging, softmax
scripts/evaluate_models.py   benchmark CLI
tests/                       pytest suite (mock-based; integration tests opt-in)
examples/sample_questions.json
```

## Future improvements

Fine-tuning on SQuAD or a domain dataset, threshold tuning on a dev split, probability calibration,
multi-passage retrieval (RAG) as a separate project, half-precision / ONNX inference, more models.
