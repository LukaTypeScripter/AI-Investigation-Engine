from collections import Counter
from pathlib import Path

import pytest

from investigation_engine.domain import Document
from investigation_engine.evaluation.questions import EvalQuestion, read_questions
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.parsers import parse_path
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.storage.memory import InMemoryDocumentStore

REPOSITORY = Path(__file__).resolve().parent.parent
SAMPLES = REPOSITORY / "samples" / "incidents"
QUESTIONS = REPOSITORY / "eval" / "incident_questions.jsonl"


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    parsed: list[Document] = []
    for path in sorted(SAMPLES.iterdir()):
        parsed.extend(parse_path(path))
    return parsed


@pytest.fixture(scope="module")
def questions() -> list[EvalQuestion]:
    return read_questions(QUESTIONS)


def test_the_samples_cover_every_supported_document_kind(documents: list[Document]) -> None:
    formats = {document.metadata["format"] for document in documents}

    assert {"md", "txt", "html", "json", "csv"} <= formats


def test_the_samples_contain_enough_documents_with_unique_ids(documents: list[Document]) -> None:
    ids = [document.id for document in documents]

    assert len(ids) >= 25
    assert len(set(ids)) == len(ids)


def test_every_gold_document_exists_in_the_samples(
    documents: list[Document], questions: list[EvalQuestion]
) -> None:
    known = {document.id for document in documents}
    missing = {
        (question.id, document_id)
        for question in questions
        for document_id in question.gold_document_ids
        if document_id not in known
    }

    assert missing == set()


def test_the_questions_cover_every_category_in_sufficient_numbers(
    questions: list[EvalQuestion],
) -> None:
    counts = Counter(question.category for question in questions)

    assert counts["simple"] >= 6
    assert counts["multi_step"] >= 6
    assert counts["multi_document"] >= 4
    assert counts["unanswerable"] >= 5
    assert counts["conflicting"] >= 5


def test_question_ids_are_unique(questions: list[EvalQuestion]) -> None:
    assert len({question.id for question in questions}) == len(questions)


def test_expected_statuses_match_the_categories(questions: list[EvalQuestion]) -> None:
    expected = {
        "unanswerable": "insufficient_evidence",
        "conflicting": "conflicting_evidence",
    }

    for question in questions:
        assert question.expected_status == expected.get(question.category, "answered")


def test_answerable_questions_have_gold_answers_and_unanswerable_ones_have_no_gold_documents(
    questions: list[EvalQuestion],
) -> None:
    for question in questions:
        if question.category == "unanswerable":
            assert question.gold_document_ids == ()
        else:
            assert question.gold_document_ids
        if question.expected_status == "answered":
            assert question.gold_answers


def test_ingesting_the_samples_detects_the_duplicated_incident_report(
    documents: list[Document],
) -> None:
    service = IngestionService(InMemoryDocumentStore(), SentenceChunker(max_words=120))

    report = service.ingest(documents)

    assert report.duplicate_chunks >= 1
    assert sorted(report.added) == sorted(document.id for document in documents)


def test_the_contradiction_between_postmortem_and_status_page_is_present_in_the_data(
    documents: list[Document],
) -> None:
    by_id = {document.id: document.text for document in documents}

    assert "3 hours 10 minutes" in by_id["postmortem-INC-2207.html"]
    assert "47 minutes" in by_id["status-page-export.html"]
