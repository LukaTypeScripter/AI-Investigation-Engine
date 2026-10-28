import pytest

from investigation_engine.evaluation.failure_analysis import (
    RetrievalFailure,
    find_failures,
    render_failures,
)
from investigation_engine.evaluation.questions import EvalQuestion, QuestionCategory


def make_question(
    question_id: str, gold: tuple[str, ...], category: QuestionCategory = "multi_step"
) -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        text=f"Text of {question_id}?",
        category=category,
        gold_answers=("x",),
        gold_document_ids=gold,
        expected_status="answered",
        origin="test",
    )


def make_report(
    results: list[tuple[str, list[str]]], config: str = "hybrid_rrf"
) -> dict[str, object]:
    return {
        "config": config,
        "questions": [
            {"question_id": question_id, "category": "multi_step", "ranked_documents": ranking}
            for question_id, ranking in results
        ],
    }


def test_a_question_whose_gold_documents_are_all_retrieved_is_not_a_failure() -> None:
    report = make_report([("q1", ["a", "b", "c"])])

    assert find_failures(report, [make_question("q1", ("a", "b"))], k=5) == []


def test_a_question_missing_a_gold_document_in_the_top_k_is_a_failure() -> None:
    report = make_report([("q1", ["a", "x", "y", "z", "w", "b"])])

    failures = find_failures(report, [make_question("q1", ("a", "b"))], k=5)

    assert failures == [
        RetrievalFailure(
            question_id="q1",
            category="multi_step",
            text="Text of q1?",
            gold=("a", "b"),
            missing=("b",),
            retrieved=("a", "x", "y", "z", "w"),
        )
    ]


def test_a_larger_k_can_turn_a_failure_into_a_success() -> None:
    report = make_report([("q1", ["a", "x", "y", "z", "w", "b"])])
    questions = [make_question("q1", ("a", "b"))]

    assert len(find_failures(report, questions, k=5)) == 1
    assert find_failures(report, questions, k=6) == []


def test_failures_keep_the_order_of_the_report() -> None:
    report = make_report([("q2", ["x"]), ("q1", ["x"])])
    questions = [make_question("q1", ("a",)), make_question("q2", ("a",))]

    assert [failure.question_id for failure in find_failures(report, questions, k=5)] == [
        "q2",
        "q1",
    ]


def test_a_result_without_a_matching_question_is_reported() -> None:
    report = make_report([("ghost", ["a"])])

    with pytest.raises(KeyError, match="ghost"):
        find_failures(report, [make_question("q1", ("a",))], k=5)


def test_rendered_failures_show_the_question_the_missing_documents_and_what_was_found() -> None:
    failure = RetrievalFailure("q1", "multi_step", "Why?", ("a", "b"), ("b",), ("a", "x"))

    rendered = render_failures([failure])

    assert "### q1 (multi_step)" in rendered
    assert "Why?" in rendered
    assert "Missing: b" in rendered
    assert "Retrieved: a, x" in rendered


def test_rendering_no_failures_says_so() -> None:
    assert render_failures([]) == "No retrieval failures."
