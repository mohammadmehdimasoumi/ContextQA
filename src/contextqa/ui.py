"""UI-facing logic, kept free of Gradio so it can be unit-tested without it."""

from __future__ import annotations

import json
import logging
from typing import Callable

from .config import LOW_CONFIDENCE_THRESHOLD, MODELS, SAMPLES_PATH
from .inference import SCORE_DISCLAIMER, AnswerResult, QAEngine, get_engine
from .utils import ContextQAError

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "No reliable answer found in the provided context."


def load_samples() -> list[list[str]]:
    """Flatten examples/sample_questions.json into [context, question] rows."""
    try:
        data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s; no samples will be shown.", SAMPLES_PATH)
        return []
    return [[item["context"], q] for item in data for q in item["questions"]]


def format_result(result: AnswerResult) -> tuple[str, str, str]:
    """Return (answer text, score text, status text) for display."""
    spec = MODELS[result.model_key]
    if result.has_answer:
        answer = result.answer
        status = "Answer found."
        if result.confidence < LOW_CONFIDENCE_THRESHOLD:
            status = "Answer found, but the model score is low: treat it with caution."
    else:
        answer, status = NO_ANSWER_TEXT, "The model judged the question unanswerable from this context."
    if not spec.supports_no_answer:
        status += " (This model was not trained to abstain, so it always returns a span.)"
    score = (
        f"Model score: {result.confidence:.1%}  |  windows: {result.num_windows}  |  "
        f"{result.latency_ms:.0f} ms\n{SCORE_DISCLAIMER}"
    )
    return answer, score, status


def answer_question(
    model_key: str,
    context: str,
    question: str,
    engine_getter: Callable[[str], QAEngine] = get_engine,
) -> tuple[str, str, str]:
    """Run QA and never raise: errors become readable status messages."""
    try:
        result = engine_getter(model_key).answer(question, context)
        return format_result(result)
    except ContextQAError as exc:  # invalid input, unknown/unloadable model
        return "", "", f"Error: {exc.args[0] if exc.args else exc}"
    except Exception:  # unexpected: log details, show a generic message
        logger.exception("Unexpected failure during inference")
        return "", "", "Unexpected error during inference. See the console log for details."
