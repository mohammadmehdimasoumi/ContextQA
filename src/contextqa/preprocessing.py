"""Input validation and question-context tokenization (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import DEFAULT_INFERENCE, InferenceConfig
from .utils import InvalidInputError

Offset = tuple[int, int] | None  # (char_start, char_end) in the context; None = not a context token

_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


@dataclass
class TokenizedWindows:
    """Tokenized question+context, split into overlapping windows.

    model_inputs:    key -> list (one entry per window) of token-id lists, ready for torch.tensor.
    context_offsets: per window, per token: char span in the ORIGINAL context, or None for
                     question tokens, special tokens ([CLS]/[SEP]) and padding.
    """

    model_inputs: dict[str, list[list[int]]]
    context_offsets: list[list[Offset]]

    @property
    def num_windows(self) -> int:
        return len(self.context_offsets)


def validate_inputs(
    question: Any, context: Any, config: InferenceConfig = DEFAULT_INFERENCE
) -> tuple[str, str]:
    """Validate and normalize the question and context.

    Returns the question stripped of surrounding whitespace and the context unchanged
    (the context is NOT modified, so character offsets stay valid for the caller).
    Raises InvalidInputError with a user-readable message otherwise.
    """
    if not isinstance(question, str) or not isinstance(context, str):
        raise InvalidInputError("Question and context must both be text (str).")
    question = question.strip()
    if not question:
        raise InvalidInputError("The question is empty. Please enter a question.")
    if not context.strip():
        raise InvalidInputError("The context is empty. Please provide a passage of text.")
    if len(question) > config.max_question_chars:
        raise InvalidInputError(
            f"The question is too long ({len(question)} characters; "
            f"maximum is {config.max_question_chars})."
        )
    if len(context) > config.max_context_chars:
        raise InvalidInputError(
            f"The context is too long ({len(context)} characters; "
            f"maximum is {config.max_context_chars}). Please shorten it or split it."
        )
    return question, context


def mask_non_context_offsets(
    offsets: Sequence[Sequence[int]], sequence_ids: Sequence[int | None]
) -> list[Offset]:
    """Keep offsets only for context tokens (sequence id 1); set all others to None.

    Fast tokenizers give every token an offset. Question tokens would otherwise point at
    the *question* string, so they must be masked or a predicted span could land there.
    """
    return [
        (int(o[0]), int(o[1])) if sid == 1 else None
        for o, sid in zip(offsets, sequence_ids)
    ]


def tokenize_pair(
    tokenizer: Any,
    question: str,
    context: str,
    config: InferenceConfig = DEFAULT_INFERENCE,
) -> TokenizedWindows:
    """Tokenize a (question, context) pair with a sliding window over the context.

    Only the context is truncated ("only_second"); overflowing tokens become additional
    windows that overlap by `config.stride` tokens, so an answer cut at one window's edge
    is fully visible in the next.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise InvalidInputError(
            "A fast tokenizer is required (offset mappings are not available otherwise)."
        )
    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=config.max_length,
        stride=config.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )
    n = len(enc["input_ids"])
    offsets = [
        mask_non_context_offsets(enc["offset_mapping"][i], enc.sequence_ids(i)) for i in range(n)
    ]
    inputs = {k: [list(row) for row in enc[k]] for k in _MODEL_INPUT_KEYS if k in enc}
    return TokenizedWindows(model_inputs=inputs, context_offsets=offsets)
