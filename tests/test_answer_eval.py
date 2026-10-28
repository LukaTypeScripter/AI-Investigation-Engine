import json

import pytest

from investigation_engine.agent.models import AnswerStatus, Citation, Conflict, Investigation
from investigation_engine.domain import SourceRef
from investigation_engine.evaluation.answer_eval import (
    AnswerResult,
    evaluate_answers,
    render_answer_table,
    score_answer,
    score_citations,
    summarize_answers,
)
from investigation_engine.evaluation.questions import EvalQuestion, QuestionCategory
from investigation_engine.llm.errors import LlmUnavailable
from investigation_engine.llm.types import Usage


def make_question(
    category: QuestionCategory = "simple",
    gold_answers: tuple[str, ...] = ("CHG-1041",),
    gold_documents: tuple[str, ...] = ("a.md",),
    expected: AnswerStatus = "answered",
) -> EvalQuestion:
    return EvalQuestion(
        "q1", "Which change?", category, gold_answers, gold_documents, expected, "t"
    )


def make_citation(evidence_id: str, *documents: str) -> Citation:
    return Citation(
        evidence_id=evidence_id,
        chunk_id=f"{documents[0]}:0",
        title="T",
        text="text",
        sources=tuple(SourceRef(document, "T", document) for document in documents),
    )


def make_investigation(
    status: AnswerStatus = "answered",
    answer: str = "It was CHG-1041.",
    citations: tuple[Citation, ...] = (),
    conflicts: tuple[Conflict, ...] = (),
    abstain_reason: str | None = None,
) -> Investigation:
    return Investigation(
        question="Which change?",
        status=status,
        answer=answer,
        citations=citations,
        conflicts=conflicts,
        searches=("s1", "s2"),
        usage=Usage(30, 10),
        llm_calls=3,
        abstain_reason=abstain_reason,
    )


def test_a_citation_is_correct_when_any_of_its_sources_is_a_gold_document() -> None:
    citations = (make_citation("E1", "a.md", "copy-of-a.md"), make_citation("E2", "x.md"))

    precision, recall = score_citations(citations, ("a.md", "b.md"))

    assert precision == 0.5
    assert recall == 0.5


def test_citations_that_cover_every_gold_document_have_full_recall() -> None:
    citations = (make_citation("E1", "a.md"), make_citation("E2", "b.md"))

    assert score_citations(citations, ("a.md", "b.md")) == (1.0, 1.0)


def test_no_citations_score_zero() -> None:
    assert score_citations((), ("a.md",)) == (0.0, 0.0)


def test_an_answer_that_contains_the_gold_answer_and_cites_gold_evidence_is_correct() -> None:
    result = score_answer(
        make_question(), make_investigation(citations=(make_citation("E1", "a.md"),))
    )

    assert result.correct is True
    assert result.exact_match == 0.0
    assert result.f1 == pytest.approx(0.5)
    assert (result.citation_precision, result.citation_recall) == (1.0, 1.0)
    assert result.searches == 2
    assert result.llm_calls == 3
    assert result.total_tokens == 40


def test_the_best_matching_gold_answer_counts_when_several_are_acceptable() -> None:
    question = make_question(gold_answers=("payment-gateway", "payment gateway"))

    result = score_answer(question, make_investigation(answer="The payment gateway."))

    assert result.correct is True
    assert result.f1 == 1.0


def test_a_wrong_answer_is_incorrect() -> None:
    result = score_answer(make_question(), make_investigation(answer="It was CHG-1099."))

    assert result.correct is False


def test_abstaining_on_an_answerable_question_is_incorrect() -> None:
    result = score_answer(make_question(), make_investigation(status="insufficient_evidence"))

    assert result.correct is False
    assert result.status == "insufficient_evidence"


def test_abstaining_on_an_unanswerable_question_is_correct() -> None:
    question = make_question("unanswerable", (), (), "insufficient_evidence")

    result = score_answer(
        question, make_investigation(status="insufficient_evidence", answer="Not stated.")
    )

    assert result.correct is True
    assert result.f1 == 0.0


def test_answering_an_unanswerable_question_is_a_hallucination() -> None:
    question = make_question("unanswerable", (), (), "insufficient_evidence")

    result = score_answer(question, make_investigation(answer="About 2 million."))

    assert result.correct is False
    assert result.status == "answered"


def test_reporting_a_conflict_on_a_conflicting_question_is_correct() -> None:
    question = make_question("conflicting", (), ("a.md", "b.md"), "conflicting_evidence")
    investigation = make_investigation(
        status="conflicting_evidence",
        answer="Sources disagree.",
        citations=(make_citation("E1", "a.md"), make_citation("E2", "b.md")),
        conflicts=(Conflict("Differs.", ("E1", "E2")),),
    )

    result = score_answer(question, investigation)

    assert result.correct is True
    assert result.citation_recall == 1.0


def test_picking_one_side_of_a_conflict_is_incorrect() -> None:
    question = make_question("conflicting", (), ("a.md", "b.md"), "conflicting_evidence")

    result = score_answer(question, make_investigation(answer="It was 3 hours."))

    assert result.correct is False


def test_a_failed_language_model_call_is_recorded_as_an_error_not_a_crash() -> None:
    class Failing:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def investigate(self, question: str) -> Investigation:
            self.asked.append(question)
            raise LlmUnavailable("all providers down")

    investigator = Failing()

    results = evaluate_answers(investigator, [make_question()])

    assert investigator.asked == ["Which change?"]
    assert results[0].correct is False
    assert results[0].error == "all providers down"
    assert results[0].status == "error"


def test_evaluate_answers_runs_every_question_in_order() -> None:
    class Scripted:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def investigate(self, question: str) -> Investigation:
            self.asked.append(question)
            return make_investigation()

    investigator = Scripted()
    questions = [make_question(), make_question()]

    results = evaluate_answers(investigator, questions)

    assert len(results) == 2
    assert investigator.asked == ["Which change?", "Which change?"]


def results_for_summary() -> list[AnswerResult]:
    def result(
        category: str, expected: str, status: str, correct: bool, f1: float = 0.0
    ) -> AnswerResult:
        return AnswerResult(
            question_id="q",
            category=category,
            expected_status=expected,
            status=status,
            answer="",
            correct=correct,
            exact_match=0.0,
            f1=f1,
            citation_precision=1.0 if expected != "insufficient_evidence" else 0.0,
            citation_recall=0.5 if expected != "insufficient_evidence" else 0.0,
            cited_documents=(),
            searches=2,
            llm_calls=4,
            total_tokens=100,
            error=None,
        )

    return [
        result("simple", "answered", "answered", True, 1.0),
        result("simple", "answered", "answered", False, 0.0),
        result("unanswerable", "insufficient_evidence", "insufficient_evidence", True),
        result("unanswerable", "insufficient_evidence", "answered", False),
        result("conflicting", "conflicting_evidence", "conflicting_evidence", True),
    ]


def test_the_summary_reports_accuracy_per_category_and_overall() -> None:
    summary = summarize_answers(results_for_summary())

    assert summary["all"]["accuracy"] == pytest.approx(3 / 5)
    assert summary["simple"]["accuracy"] == 0.5
    assert summary["unanswerable"]["accuracy"] == 0.5
    assert summary["conflicting"]["accuracy"] == 1.0


def test_the_summary_averages_f1_only_over_questions_that_expect_an_answer() -> None:
    summary = summarize_answers(results_for_summary())

    assert summary["all"]["f1"] == 0.5


def test_the_summary_counts_hallucinations_among_unanswerable_questions_only() -> None:
    summary = summarize_answers(results_for_summary())

    assert summary["unanswerable"]["hallucination_rate"] == 0.5
    assert summary["simple"]["hallucination_rate"] == 0.0


def test_the_summary_averages_citation_scores_over_questions_with_gold_documents() -> None:
    summary = summarize_answers(results_for_summary())

    assert summary["all"]["citation_precision"] == 1.0
    assert summary["all"]["citation_recall"] == 0.5


def test_the_summary_reports_average_effort() -> None:
    summary = summarize_answers(results_for_summary())

    assert summary["all"]["searches"] == 2.0
    assert summary["all"]["llm_calls"] == 4.0
    assert summary["all"]["tokens"] == 100.0


def test_the_answer_table_lists_each_configuration() -> None:
    summary = summarize_answers(results_for_summary())

    table = render_answer_table({"hybrid_rrf": summary, "hybrid_rerank": summary}, "all")

    lines = table.splitlines()
    assert lines[0].startswith("| config | accuracy |")
    assert [line.split("|")[1].strip() for line in lines[2:]] == ["hybrid_rrf", "hybrid_rerank"]
    assert "0.600" in lines[2]


def test_results_are_plain_json_serialisable() -> None:
    result = score_answer(
        make_question(), make_investigation(citations=(make_citation("E1", "a.md"),))
    )

    assert json.loads(json.dumps(result.to_dict()))["question_id"] == "q1"


class ScriptedInvestigator:
    def investigate(self, question: str) -> Investigation:
        return make_investigation(answer=question)


def test_evaluate_answers_reports_each_result_as_soon_as_it_is_scored() -> None:
    seen: list[str] = []

    results = evaluate_answers(
        ScriptedInvestigator(),
        [make_question(), make_question()],
        on_result=lambda result: seen.append(result.answer),
    )

    assert seen == ["Which change?", "Which change?"]
    assert len(results) == 2


def test_evaluate_answers_pauses_between_questions_but_not_before_the_first() -> None:
    pauses: list[float] = []

    evaluate_answers(
        ScriptedInvestigator(),
        [make_question(), make_question(), make_question()],
        pause_seconds=1.5,
        sleep=pauses.append,
    )

    assert pauses == [1.5, 1.5]


def test_evaluate_answers_does_not_pause_by_default() -> None:
    pauses: list[float] = []

    evaluate_answers(
        ScriptedInvestigator(), [make_question(), make_question()], sleep=pauses.append
    )

    assert pauses == []
