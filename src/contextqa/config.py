"""Centralized configuration for ContextQA.

Lives inside the package (instead of the repo root) so that it can be imported
the same way from the app, scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a supported QA model."""

    key: str
    hf_id: str
    display_name: str
    supports_no_answer: bool  # trained on SQuAD 2.0 (has "no answer" supervision)?
    description: str


MODELS: dict[str, ModelSpec] = {
    "bert": ModelSpec(
        key="bert",
        hf_id="deepset/bert-base-cased-squad2",
        display_name="BERT base (cased, SQuAD 2.0)",
        supports_no_answer=True,
        description="12-layer encoder, ~110M params, trained with unanswerable questions.",
    ),
    "distilbert": ModelSpec(
        key="distilbert",
        hf_id="distilbert/distilbert-base-cased-distilled-squad",
        display_name="DistilBERT base (cased, distilled on SQuAD 1.1)",
        supports_no_answer=False,  # SQuAD 1.1: always has an answer during training
        description="6-layer encoder, ~65M params, distilled from BERT; no 'no-answer' training.",
    ),
}
DEFAULT_MODEL_KEY = "bert"


@dataclass(frozen=True)
class InferenceConfig:
    """Hyperparameters of the inference pipeline."""

    max_length: int = 384          # tokens per window (question + context + special tokens)
    stride: int = 128              # overlapping tokens between consecutive windows
    max_answer_tokens: int = 30    # longest allowed answer span, in tokens
    n_best: int = 20               # top-k start/end logits considered per window
    batch_size: int = 8            # windows per forward pass (keeps 6 GB VRAM safe)
    no_answer_threshold: float = 0.0  # SQuAD 2.0 null-score-diff threshold (0 = model default)
    max_question_chars: int = 1_000
    max_context_chars: int = 50_000


DEFAULT_INFERENCE = InferenceConfig()

# UI only: answers whose relative score is below this are flagged as "low confidence".
LOW_CONFIDENCE_THRESHOLD = 0.30
SAMPLES_PATH = PROJECT_ROOT / "examples" / "sample_questions.json"
