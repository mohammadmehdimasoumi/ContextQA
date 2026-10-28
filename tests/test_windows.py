"""Sliding-window handling tested with a tiny fake *fast* tokenizer (no downloads)."""
import re

import pytest

from contextqa.config import InferenceConfig
from contextqa.inference import WindowOutput, decode_answer
from contextqa.preprocessing import tokenize_pair
from contextqa.utils import InvalidInputError
import numpy as np


class FakeEnc(dict):
    def __init__(self, data, seq_ids): super().__init__(data); self._s = seq_ids
    def sequence_ids(self, i): return self._s[i]


class FakeFastTokenizer:
    """Word-level tokenizer: [CLS] q [SEP] ctx-window [SEP], windows overlap by `stride`."""
    is_fast = True
    def __call__(self, q, c, truncation, max_length, stride, **_):
        q_toks = q.split()
        cap = max_length - len(q_toks) - 3
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", c)]
        wins, i = [], 0
        while True:
            wins.append(words[i : i + cap])
            if i + cap >= len(words): break
            i += cap - stride
        ids, offs, seqs, mask = [], [], [], []
        width = max(len(q_toks) + 3 + len(w) for w in wins)
        for w in wins:
            n = len(q_toks) + 3 + len(w)
            seq = [None] + [0] * len(q_toks) + [None] + [1] * len(w) + [None] + [None] * (width - n)
            off = [(0, 0)] * (2 + len(q_toks)) + w + [(0, 0)] * (1 + width - n)
            ids.append([1] * width); offs.append(off); seqs.append(seq)
            mask.append([1] * n + [0] * (width - n))
        return FakeEnc({"input_ids": ids, "attention_mask": mask, "offset_mapping": offs}, seqs)


CFG = InferenceConfig(max_length=12, stride=3)


def test_long_context_produces_overlapping_windows_with_masked_offsets():
    ctx = " ".join(f"w{i}" for i in range(30))
    w = tokenize_pair(FakeFastTokenizer(), "who is it", ctx, CFG)
    assert w.num_windows > 1
    for offs in w.context_offsets:
        assert offs[0] is None and offs[1] is None  # [CLS] and question tokens masked
    for row in w.model_inputs["input_ids"]:
        assert len(row) == len(w.model_inputs["input_ids"][0])  # padded equally


def test_answer_in_later_window_maps_to_original_text():
    ctx = " ".join(f"w{i}" for i in range(30)) + " Guido van Rossum"
    w = tokenize_pair(FakeFastTokenizer(), "who is it", ctx, CFG)
    outs = []
    for offs in w.context_offsets:
        s, e = np.full(len(offs), -5.0), np.full(len(offs), -5.0)
        for t, o in enumerate(offs):
            if o and ctx[o[0]:o[1]] == "Guido": s[t] = 8.0
            if o and ctx[o[0]:o[1]] == "Rossum": e[t] = 8.0
        s[0] = e[0] = 0.0
        outs.append(WindowOutput(s, e, offs))
    r = decode_answer(ctx, outs, "bert", True)
    assert r.answer == "Guido van Rossum" and ctx[r.start_char:r.end_char] == r.answer
    assert r.num_windows == w.num_windows > 1


def test_slow_tokenizer_rejected():
    class Slow: is_fast = False
    with pytest.raises(InvalidInputError, match="fast tokenizer"):
        tokenize_pair(Slow(), "q", "c")
