import numpy as np
import pytest

from contextqa.inference import get_model_spec
from contextqa.utils import UnknownModelError, softmax


def test_softmax_sums_to_one_and_is_stable():
    p = softmax([1000.0, 1001.0, 999.0])
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 1


def test_model_selection_known_and_unknown():
    assert get_model_spec("bert").supports_no_answer is True
    assert get_model_spec("distilbert").supports_no_answer is False
    with pytest.raises(UnknownModelError, match="Available models"):
        get_model_spec("gpt")
