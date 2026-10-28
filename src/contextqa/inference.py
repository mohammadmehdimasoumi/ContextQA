"""Extractive QA inference: model wrapper + torch-free span decoding.

Pipeline:  validate -> tokenize (sliding windows) -> model forward -> decode spans.
The decoding step (`decode_answer`) only needs NumPy arrays, so it can be unit-tested
with hand-made logits and no model download.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import DEFAULT_INFERENCE, DEFAULT_MODEL_KEY, MODELS, InferenceConfig, ModelSpec
from .preprocessing import Offset, TokenizedWindows, tokenize_pair, validate_inputs
from .utils import ModelLoadError, UnknownModelError, resolve_device, softmax

logger = logging.getLogger(__name__)

SCORE_DISCLAIMER = (
    "The confidence value is a softmax over the model's own candidate spans "
    "(plus 'no answer' when supported). It is a relative score, NOT a calibrated "
    "probability that the answer is correct."
)


@dataclass(frozen=True)
class WindowOutput:
    """Raw model output for one window."""

    start_logits: np.ndarray
    end_logits: np.ndarray
    offsets: Sequence[Offset]


@dataclass(frozen=True)
class AnswerResult:
    """Consistent output structure of every prediction."""

    model_key: str
    answer: str            # "" when has_answer is False
    has_answer: bool
    start_char: int | None  # offsets into the ORIGINAL context; context[start:end] == answer
    end_char: int | None
    confidence: float      # relative softmax weight of the returned decision (see SCORE_DISCLAIMER)
    span_score: float | None  # raw start_logit + end_logit of the best span
    null_score: float | None  # raw [CLS] start+end logit (None if model has no no-answer head)
    num_windows: int
    latency_ms: float = 0.0


def get_model_spec(model_key: str) -> ModelSpec:
    """Look up a registered model; raise UnknownModelError with the valid keys otherwise."""
    try:
        return MODELS[model_key]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model '{model_key}'. Available models: {', '.join(MODELS)}."
        ) from None


def find_best_spans(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: Sequence[Offset],
    n_best: int,
    max_answer_tokens: int,
) -> list[tuple[float, int, int]]:
    """Return candidate spans (score, start_char, end_char) for ONE window.

    Considers the `n_best` highest start logits x `n_best` highest end logits and keeps
    only valid pairs: both tokens inside the context, end >= start, length <= limit.
    score = start_logit + end_logit (log-space product of the two softmax probabilities,
    up to a constant), which is how the original BERT SQuAD code ranks spans.
    """
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)
    k = min(n_best, len(start_logits))
    start_idx = np.argsort(start_logits)[::-1][:k]
    end_idx = np.argsort(end_logits)[::-1][:k]

    spans: list[tuple[float, int, int]] = []
    for s in start_idx:
        for e in end_idx:
            if s >= len(offsets) or e >= len(offsets):
                continue
            if offsets[s] is None or offsets[e] is None:
                continue
            if e < s or (e - s + 1) > max_answer_tokens:
                continue
            spans.append(
                (float(start_logits[s] + end_logits[e]), offsets[s][0], offsets[e][1])
            )
    return spans


def decode_answer(
    context: str,
    windows: Sequence[WindowOutput],
    model_key: str,
    supports_no_answer: bool,
    config: InferenceConfig = DEFAULT_INFERENCE,
) -> AnswerResult:
    """Combine all windows into one final answer (or a 'no answer' decision).

    SQuAD 2.0 convention: each window has a null score = start_logit[CLS] + end_logit[CLS].
    We use the MINIMUM null score over windows (the question is unanswerable only if
    every window agrees) and predict 'no answer' when
        null_score - best_span_score > no_answer_threshold.
    Models trained only on SQuAD 1.1 never saw unanswerable questions, so for them the
    best span is always returned.
    """
    # Best score per distinct (start, end): overlapping windows may produce the same span.
    best: dict[tuple[int, int], float] = {}
    for w in windows:
        for score, s, e in find_best_spans(
            w.start_logits, w.end_logits, w.offsets, config.n_best, config.max_answer_tokens
        ):
            if score > best.get((s, e), -np.inf):
                best[(s, e)] = score
    ranked = sorted(((sc, s, e) for (s, e), sc in best.items()), reverse=True)[: config.n_best]

    null_score = (
        min(float(w.start_logits[0] + w.end_logits[0]) for w in windows)
        if supports_no_answer and windows
        else None
    )

    def _no_answer(conf: float) -> AnswerResult:
        return AnswerResult(model_key, "", False, None, None, conf,
                            ranked[0][0] if ranked else None, null_score, len(windows))

    if not ranked:
        return _no_answer(1.0 if null_score is not None else 0.0)

    best_score, start, end = ranked[0]
    if null_score is not None:
        probs = softmax([null_score] + [r[0] for r in ranked])  # index 0 = "no answer"
        if null_score - best_score > config.no_answer_threshold:
            return _no_answer(float(probs[0]))
        confidence = float(probs[1])
    else:
        confidence = float(softmax([r[0] for r in ranked])[0])

    return AnswerResult(
        model_key=model_key,
        answer=context[start:end],
        has_answer=True,
        start_char=start,
        end_char=end,
        confidence=confidence,
        span_score=best_score,
        null_score=null_score,
        num_windows=len(windows),
    )


class QAEngine:
    """Loads one pretrained QA model and answers (question, context) pairs."""

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        device: str | None = None,
        config: InferenceConfig = DEFAULT_INFERENCE,
    ) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.spec = get_model_spec(model_key)
        self.config = config
        self.device = resolve_device(device)
        logger.info("Loading %s on %s", self.spec.hf_id, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.spec.hf_id)
        except Exception as exc:  # network, bad id, corrupt cache ...
            raise ModelLoadError(f"Could not load '{self.spec.hf_id}': {exc}") from exc
        if not self.tokenizer.is_fast:
            raise ModelLoadError(f"'{self.spec.hf_id}' has no fast tokenizer (offsets needed).")
        self.model.to(self.device).eval()
        self._torch: Any = torch

    def _forward(self, windows: TokenizedWindows) -> list[WindowOutput]:
        """Run the model over windows in small batches; return per-window logits."""
        torch = self._torch
        keys = [k for k in self.tokenizer.model_input_names if k in windows.model_inputs]
        outputs: list[WindowOutput] = []
        bs = self.config.batch_size
        for i in range(0, windows.num_windows, bs):
            batch = {
                k: torch.tensor(windows.model_inputs[k][i : i + bs], device=self.device)
                for k in keys
            }
            with torch.inference_mode():
                out = self.model(**batch)
            start = out.start_logits.float().cpu().numpy()
            end = out.end_logits.float().cpu().numpy()
            for j in range(start.shape[0]):
                outputs.append(WindowOutput(start[j], end[j], windows.context_offsets[i + j]))
        return outputs

    def answer(self, question: str, context: str) -> AnswerResult:
        """Answer `question` using only `context`."""
        question, context = validate_inputs(question, context, self.config)
        t0 = time.perf_counter()
        windows = tokenize_pair(self.tokenizer, question, context, self.config)
        outputs = self._forward(windows)
        result = decode_answer(
            context, outputs, self.spec.key, self.spec.supports_no_answer, self.config
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return AnswerResult(**{**result.__dict__, "latency_ms": latency_ms})


_ENGINES: dict[str, QAEngine] = {}


def get_engine(model_key: str, device: str | None = None) -> QAEngine:
    """Return a cached engine so each model is loaded only once per process."""
    get_model_spec(model_key)
    if model_key not in _ENGINES:
        _ENGINES[model_key] = QAEngine(model_key, device)
    return _ENGINES[model_key]
