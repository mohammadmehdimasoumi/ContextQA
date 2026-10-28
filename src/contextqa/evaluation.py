"""SQuAD 2.0 evaluation: Exact Match, token-level F1, latency and model size.

The metric code follows the official SQuAD v2.0 evaluation script
(normalize_answer / compute_exact / compute_f1), so numbers are comparable with it.
Everything except `load_squad_v2_examples` and `model_size_mb` is torch/datasets-free.
"""

from __future__ import annotations

import collections
import logging
import random
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .utils import InvalidInputError

logger = logging.getLogger(__name__)

Example = dict[str, Any]  # keys: id, question, context, answers (list[str]; empty = unanswerable)


# ----------------------------------------------------------------------------- metrics
def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles (a/an/the), collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split() if text else []


def compute_exact(gold: str, pred: str) -> int:
    """1 if the normalized strings are identical, else 0."""
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    """Token-level F1 on bag-of-words overlap of normalized strings.

    If either side is empty (no-answer), F1 is 1 only when both are empty.
    """
    gold_toks, pred_toks = _tokens(gold), _tokens(pred)
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_toks), overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_prediction(gold_answers: Sequence[str], pred: str) -> tuple[int, float]:
    """(EM, F1) of one prediction: best match over all gold answers.

    An unanswerable question has gold_answers == [] and is treated as gold [""].
    """
    golds = list(gold_answers) or [""]
    return (
        max(compute_exact(g, pred) for g in golds),
        max(compute_f1(g, pred) for g in golds),
    )


def aggregate_scores(
    em: Sequence[int], f1: Sequence[float], has_answer: Sequence[bool]
) -> dict[str, float]:
    """Overall, HasAns and NoAns EM/F1 in percent (NoAns = unanswerable questions)."""
    em_a, f1_a, ha = np.array(em, float), np.array(f1, float), np.array(has_answer, bool)

    def _pct(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean() * 100) if mask.any() else float("nan")

    allm = np.ones_like(ha)
    return {
        "EM": _pct(em_a, allm), "F1": _pct(f1_a, allm),
        "HasAns_EM": _pct(em_a, ha), "HasAns_F1": _pct(f1_a, ha),
        "NoAns_EM": _pct(em_a, ~ha), "NoAns_F1": _pct(f1_a, ~ha),
    }


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Mean / median / p95 latency in milliseconds."""
    a = np.asarray(latencies_ms, float)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95))}


# ----------------------------------------------------------------------------- data
def select_subset(examples: Sequence[Example], n: int | None, seed: int) -> list[Example]:
    """Reproducible random subset (same seed + same n => same examples for every model)."""
    if n is None or n >= len(examples):
        return list(examples)
    idx = sorted(random.Random(seed).sample(range(len(examples)), n))
    return [examples[i] for i in idx]


def load_squad_v2_examples(n: int | None, seed: int) -> list[Example]:
    """Load the SQuAD 2.0 validation split and return a seeded subset (cached by `datasets`)."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    examples = [
        {"id": r["id"], "question": r["question"], "context": r["context"],
         "answers": list(r["answers"]["text"])}
        for r in ds
    ]
    return select_subset(examples, n, seed)


# ----------------------------------------------------------------------------- running
class _Engine(Protocol):
    def answer(self, question: str, context: str) -> Any: ...


@dataclass
class ModelReport:
    model_key: str
    hf_id: str
    num_examples: int
    scores: dict[str, float]
    latency: dict[str, float]
    size_mb: float
    num_params_m: float
    skipped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def model_size_mb(model: Any) -> tuple[float, float]:
    """(approx. in-memory/storage size in MB, parameters in millions) from the weights."""
    params = list(model.parameters())
    nbytes = sum(p.numel() * p.element_size() for p in params)
    return nbytes / 1e6, sum(p.numel() for p in params) / 1e6


def evaluate_engine(
    engine: _Engine, examples: Sequence[Example], warmup: int = 3
) -> tuple[dict[str, float], dict[str, float], int]:
    """Run `engine` on all examples. Returns (scores, latency stats, #skipped)."""
    for ex in examples[:warmup]:  # first calls include CUDA/kernels warm-up; exclude them
        engine.answer(ex["question"], ex["context"])
    em, f1, has_ans, lat, skipped = [], [], [], [], 0
    for i, ex in enumerate(examples):
        try:
            t0 = time.perf_counter()
            result = engine.answer(ex["question"], ex["context"])
            lat.append(getattr(result, "latency_ms", None) or (time.perf_counter() - t0) * 1000)
            pred = result.answer
        except InvalidInputError as exc:
            logger.warning("Skipping %s: %s", ex.get("id", i), exc)
            skipped, pred = skipped + 1, ""
        e, f = score_prediction(ex["answers"], pred)
        em.append(e); f1.append(f); has_ans.append(bool(ex["answers"]))
    return aggregate_scores(em, f1, has_ans), latency_stats(lat or [0.0]), skipped


def format_summary_table(reports: Sequence[ModelReport]) -> str:
    """Markdown table: quality, latency, size."""
    head = ("| Model | N | EM | F1 | HasAns F1 | NoAns F1 | Mean ms | p95 ms | Params (M) | Size (MB) |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = [
        f"| {r.hf_id} | {r.num_examples} | {r.scores['EM']:.1f} | {r.scores['F1']:.1f} | "
        f"{r.scores['HasAns_F1']:.1f} | {r.scores['NoAns_F1']:.1f} | {r.latency['mean_ms']:.0f} | "
        f"{r.latency['p95_ms']:.0f} | {r.num_params_m:.0f} | {r.size_mb:.0f} |"
        for r in reports
    ]
    return head + "\n".join(rows)
