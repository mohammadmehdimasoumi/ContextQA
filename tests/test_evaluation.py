import pytest

from contextqa.evaluation import (
    aggregate_scores, compute_exact, compute_f1, evaluate_engine, normalize_answer,
    score_prediction, select_subset,
)
from contextqa.utils import InvalidInputError


def test_normalize_removes_case_punctuation_articles():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert normalize_answer("A cat, an owl") == "cat owl"


def test_exact_match_is_normalized():
    assert compute_exact("the Guido van Rossum", "guido van rossum.") == 1
    assert compute_exact("Guido", "Rossum") == 0


def test_f1_partial_overlap():
    # gold 3 tokens, pred 2 tokens, 2 shared -> P=1, R=2/3 -> F1=0.8
    assert compute_f1("guido van rossum", "guido van") == pytest.approx(0.8)
    assert compute_f1("abc", "xyz") == 0.0


def test_f1_handles_repeated_tokens_with_counts():
    # NOTE: avoid "a"/"an"/"the" here, they are removed as articles by normalization.
    # overlap = min counts = 2 ("y" twice); P = 2/3, R = 2/3 -> F1 = 2/3
    assert compute_f1("x y y", "y y y") == pytest.approx(2 / 3)


def test_no_answer_scoring():
    assert score_prediction([], "") == (1, 1.0)
    assert score_prediction([], "something") == (0, 0.0)
    assert score_prediction(["Paris"], "") == (0, 0.0)


def test_best_over_multiple_gold_answers():
    em, f1 = score_prediction(["Paris", "city of Paris"], "Paris")
    assert em == 1 and f1 == 1.0


def test_aggregate_splits_has_and_no_answer():
    s = aggregate_scores([1, 0, 1], [1.0, 0.5, 1.0], [True, True, False])
    assert s["EM"] == pytest.approx(200 / 3)
    assert s["HasAns_F1"] == pytest.approx(75.0)
    assert s["NoAns_EM"] == 100.0


def test_subset_is_reproducible_and_seed_dependent():
    data = [{"id": i} for i in range(100)]
    a, b = select_subset(data, 10, 42), select_subset(data, 10, 42)
    assert a == b and len(a) == 10
    assert a != select_subset(data, 10, 7)
    assert select_subset(data, None, 1) == data


class _FakeResult:
    def __init__(self, answer): self.answer, self.latency_ms = answer, 5.0


class _FakeEngine:
    def answer(self, q, c):
        if q == "too long":
            raise InvalidInputError("too long")
        return _FakeResult("paris" if "France" in q else "")


def test_evaluate_engine_end_to_end_with_fake_engine():
    ex = [
        {"question": "Capital of France?", "context": "c", "answers": ["Paris"]},
        {"question": "Who is X?", "context": "c", "answers": []},
        {"question": "too long", "context": "c", "answers": ["y"]},
    ]
    scores, lat, skipped = evaluate_engine(_FakeEngine(), ex, warmup=1)
    assert skipped == 1 and lat["mean_ms"] == 5.0
    assert scores["HasAns_EM"] == 50.0 and scores["NoAns_EM"] == 100.0
