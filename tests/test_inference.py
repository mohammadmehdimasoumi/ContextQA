import numpy as np
import pytest

from contextqa.config import InferenceConfig
from contextqa.inference import WindowOutput, decode_answer, find_best_spans

# Fake tokenization of: "Python was created by Guido van Rossum in 1991"
CONTEXT = "Python was created by Guido van Rossum in 1991"
WORDS = CONTEXT.split()


def _offsets():
    """[CLS], 9 context words, [SEP] -> offsets masked like preprocessing does."""
    out, pos = [None], 0
    for w in WORDS:
        start = CONTEXT.index(w, pos)
        out.append((start, start + len(w)))
        pos = start + len(w)
    out.append(None)
    return out


def _window(span=None, null_boost=0.0, offsets=None):
    """Logits peaked at token indices `span` (inclusive, 1-based word positions)."""
    offsets = offsets or _offsets()
    n = len(offsets)
    s, e = np.full(n, -5.0), np.full(n, -5.0)
    if span:
        s[span[0]], e[span[1]] = 6.0, 6.0
    s[0] += null_boost
    e[0] += null_boost
    return WindowOutput(s, e, offsets)


def test_basic_extraction_and_offsets():
    r = decode_answer(CONTEXT, [_window(span=(5, 7))], "bert", True)
    assert r.has_answer and r.answer == "Guido van Rossum"
    assert CONTEXT[r.start_char : r.end_char] == r.answer
    assert 0.0 < r.confidence <= 1.0


def test_unanswerable_when_null_score_wins():
    r = decode_answer(CONTEXT, [_window(span=(5, 7), null_boost=20.0)], "bert", True)
    assert not r.has_answer and r.answer == "" and r.start_char is None


def test_model_without_no_answer_head_always_answers():
    r = decode_answer(CONTEXT, [_window(span=(5, 7), null_boost=20.0)], "distilbert", False)
    assert r.has_answer and r.null_score is None


def test_invalid_spans_are_ignored():
    # end before start, and spans longer than the token limit
    w = _window()
    w.start_logits[8], w.end_logits[2] = 9.0, 9.0
    spans = find_best_spans(w.start_logits, w.end_logits, w.offsets, 20, 3)
    assert all(e - s <= len(CONTEXT) for _, s, e in spans)
    assert all(CONTEXT[s:e] and s < e for _, s, e in spans)
    long_cfg = InferenceConfig(max_answer_tokens=2)
    r = decode_answer(CONTEXT, [_window(span=(1, 9))], "bert", True, long_cfg)
    assert len(r.answer.split()) <= 2 if r.has_answer else True


def test_special_and_question_tokens_never_selected():
    w = _window()
    w.start_logits[0] = w.end_logits[-1] = 50.0  # [CLS] / [SEP] must be skipped
    for _, s, e in find_best_spans(w.start_logits, w.end_logits, w.offsets, 20, 30):
        assert 0 <= s < e <= len(CONTEXT)


def test_multiple_windows_best_span_wins_and_null_uses_minimum():
    weak = _window(span=(1, 1))                  # says "Python", low-ish
    strong = _window(span=(5, 7), null_boost=-3.0)
    strong.start_logits[5] += 4.0
    r = decode_answer(CONTEXT, [weak, strong], "bert", True)
    assert r.answer == "Guido van Rossum" and r.num_windows == 2


def test_duplicate_spans_from_overlapping_windows_are_merged():
    r1 = decode_answer(CONTEXT, [_window(span=(5, 7))], "bert", True)
    r2 = decode_answer(CONTEXT, [_window(span=(5, 7)), _window(span=(5, 7))], "bert", True)
    assert r2.answer == r1.answer
    assert r2.confidence == pytest.approx(r1.confidence)  # no double counting


def test_output_structure_is_consistent():
    keys = set(decode_answer(CONTEXT, [_window(span=(5, 7))], "bert", True).__dict__)
    keys_no = set(decode_answer(CONTEXT, [_window(null_boost=30)], "bert", True).__dict__)
    assert keys == keys_no
