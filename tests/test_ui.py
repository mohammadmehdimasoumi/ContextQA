from contextqa.inference import AnswerResult
from contextqa.ui import NO_ANSWER_TEXT, answer_question, format_result, load_samples
from contextqa.utils import InvalidInputError, ModelLoadError, UnknownModelError


def _res(key="bert", has=True, conf=0.9):
    return AnswerResult(key, "Guido" if has else "", has, 0 if has else None,
                        5 if has else None, conf, 1.0, 0.0, 1, 12.0)


class _Engine:
    def __init__(self, res): self.res = res
    def answer(self, q, c): return self.res


def test_format_found_and_disclaimer():
    answer, score, status = format_result(_res())
    assert answer == "Guido" and "not a calibrated" in score.lower() and "found" in status.lower()


def test_format_no_answer():
    answer, _, status = format_result(_res(has=False))
    assert answer == NO_ANSWER_TEXT and "unanswerable" in status


def test_low_confidence_flagged_and_distilbert_note():
    _, _, status = format_result(_res("distilbert", conf=0.1))
    assert "caution" in status and "not trained to abstain" in status


def test_model_selection_is_passed_to_engine_getter():
    seen = []
    def getter(key):
        seen.append(key); return _Engine(_res(key))
    answer_question("distilbert", "ctx", "q?", getter)
    assert seen == ["distilbert"]


def test_errors_become_messages_not_exceptions():
    def raiser(exc):
        def getter(_): raise exc
        return getter
    for exc in (InvalidInputError("The question is empty."), UnknownModelError("bad"),
                ModelLoadError("offline"), RuntimeError("boom")):
        answer, score, status = answer_question("bert", "c", "q", raiser(exc))
        assert answer == "" and score == "" and status.startswith(("Error", "Unexpected"))
    assert "empty" in answer_question("bert", "c", "q", raiser(InvalidInputError("The question is empty.")))[2]


def test_samples_load_as_context_question_rows():
    rows = load_samples()
    assert rows and all(len(r) == 2 and r[0] and r[1] for r in rows)
