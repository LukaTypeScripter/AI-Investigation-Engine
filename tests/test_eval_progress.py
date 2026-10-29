import pytest

from investigation_engine.evaluation.answer_eval import error_result
from investigation_engine.evaluation.cli import print_progress
from investigation_engine.evaluation.questions import EvalQuestion

QUESTION = EvalQuestion(
    id="inc-16",
    text="Which service was modified by change CHG-1041?",
    category="simple",
    gold_answers=("payment-gateway",),
    gold_document_ids=("changes.csv#CHG-1041",),
    expected_status="answered",
    origin="incidents",
)


def test_the_progress_line_says_why_a_question_failed(capsys: pytest.CaptureFixture[str]) -> None:
    print_progress(error_result(QUESTION, "m1: HTTP 429: slow down"))

    assert "inc-16: wrong (error): m1: HTTP 429: slow down" in capsys.readouterr().err


def test_a_long_failure_reason_is_shortened_in_the_progress_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_progress(error_result(QUESTION, "x" * 500))

    line = capsys.readouterr().err
    assert "x" * 200 + "..." in line
    assert "x" * 201 not in line
