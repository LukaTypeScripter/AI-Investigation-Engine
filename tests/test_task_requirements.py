import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from investigation_engine.evaluation.questions import CATEGORIES, EvalQuestion, read_questions

REPOSITORY = Path(__file__).resolve().parent.parent
QUESTION_FILES = (
    REPOSITORY / "eval" / "hotpot_questions.jsonl",
    REPOSITORY / "eval" / "incident_questions.jsonl",
)
RETRIEVAL_REPORT = REPOSITORY / "reports" / "retrieval-after-fix.json"
CHUNKING_REPORT = REPOSITORY / "reports" / "chunking.json"

MINIMUM_QUESTIONS = 40
MINIMUM_DOCUMENTS = 50_000
MINIMUM_CONFIGURATIONS = 2
MINIMUM_CHUNKERS = 2
REQUIRED_METRICS = ("recall@5", "recall@10", "full_recall@5", "mrr")


def read_reports(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["reports"]


@pytest.fixture(scope="module")
def questions() -> list[EvalQuestion]:
    return [question for path in QUESTION_FILES for question in read_questions(path)]


@pytest.fixture(scope="module")
def retrieval_reports() -> list[dict[str, Any]]:
    return read_reports(RETRIEVAL_REPORT)


def test_the_evaluation_set_has_at_least_forty_questions_with_unique_ids(
    questions: list[EvalQuestion],
) -> None:
    assert len(questions) >= MINIMUM_QUESTIONS
    assert len({question.id for question in questions}) == len(questions)


def test_the_evaluation_set_contains_every_kind_of_question_the_task_names(
    questions: list[EvalQuestion],
) -> None:
    assert set(Counter(question.category for question in questions)) == set(CATEGORIES)


def test_multi_document_questions_name_at_least_two_gold_documents(
    questions: list[EvalQuestion],
) -> None:
    multi_document = [question for question in questions if question.category == "multi_document"]

    assert multi_document
    assert all(len(question.gold_document_ids) >= 2 for question in multi_document)


def test_the_retrieval_evaluation_ran_on_at_least_fifty_thousand_documents(
    retrieval_reports: list[dict[str, Any]],
) -> None:
    assert min(report["documents"] for report in retrieval_reports) >= MINIMUM_DOCUMENTS


def test_the_retrieval_evaluation_compares_at_least_two_configurations(
    retrieval_reports: list[dict[str, Any]],
) -> None:
    assert len({report["config"] for report in retrieval_reports}) >= MINIMUM_CONFIGURATIONS


def test_the_retrieval_evaluation_reports_recall_at_k_and_mrr_for_every_configuration(
    retrieval_reports: list[dict[str, Any]],
) -> None:
    for report in retrieval_reports:
        overall = report["metrics"]["all"]
        assert all(0.0 <= overall[name] <= 1.0 for name in REQUIRED_METRICS)


def test_the_chunking_evaluation_compares_at_least_two_chunkers() -> None:
    chunkers = {report["chunker"] for report in read_reports(CHUNKING_REPORT)}

    assert len(chunkers) >= MINIMUM_CHUNKERS
