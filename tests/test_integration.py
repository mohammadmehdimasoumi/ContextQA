"""Real-model smoke tests. Skipped by default (they download ~600 MB).

Run with:  set RUN_MODEL_TESTS=1   (Windows cmd)   then   python -m pytest tests/test_integration.py
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run"
)

CTX = "Python was created by Guido van Rossum and was first released in 1991."


@pytest.mark.parametrize("key", ["bert", "distilbert"])
def test_real_model_extracts_answer(key):
    from contextqa.inference import QAEngine
    r = QAEngine(key).answer("Who created Python?", CTX)
    assert r.has_answer and "Guido" in r.answer
    assert CTX[r.start_char:r.end_char] == r.answer


def test_bert_abstains_on_unrelated_question():
    from contextqa.inference import QAEngine
    assert not QAEngine("bert").answer("What is the capital of France?", CTX).has_answer
