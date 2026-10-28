import pytest

from contextqa.config import InferenceConfig
from contextqa.preprocessing import mask_non_context_offsets, validate_inputs
from contextqa.utils import InvalidInputError


@pytest.mark.parametrize(
    "q, c",
    [("", "text"), ("   ", "text"), ("Who?", ""), ("Who?", "   "), (None, "text"), ("Who?", 5)],
)
def test_invalid_inputs_rejected(q, c):
    with pytest.raises(InvalidInputError):
        validate_inputs(q, c)


def test_too_long_inputs_rejected_with_clear_message():
    cfg = InferenceConfig(max_context_chars=10, max_question_chars=5)
    with pytest.raises(InvalidInputError, match="context is too long"):
        validate_inputs("Who?", "x" * 11, cfg)
    with pytest.raises(InvalidInputError, match="question is too long"):
        validate_inputs("x" * 6, "ok", cfg)


def test_valid_input_strips_question_but_not_context():
    q, c = validate_inputs("  Who?  ", " ctx ")
    assert q == "Who?" and c == " ctx "


def test_mask_keeps_only_context_tokens():
    offsets = [(0, 0), (0, 3), (0, 0), (5, 9), (10, 12), (0, 0)]
    seq_ids = [None, 0, None, 1, 1, None]
    assert mask_non_context_offsets(offsets, seq_ids) == [None, None, None, (5, 9), (10, 12), None]
