import pytest

from investigation_engine.agent.evidence import EvidenceBoard
from investigation_engine.agent.tools import (
    SEARCH_TOOL,
    SUBMIT_TOOL,
    ToolArgumentError,
    parse_search_query,
    parse_submission,
)
from investigation_engine.domain import Chunk, SourceRef
from investigation_engine.retrieval.hybrid import Hit


class NoSources:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def sources_of(self, chunk_id: str) -> list[SourceRef]:
        self.requested.append(chunk_id)
        return []


def board_with_three_items() -> EvidenceBoard:
    board = EvidenceBoard(NoSources())
    hits = [
        Hit(Chunk(f"d{n}:0", f"d{n}", 0, f"text {n}", f"Title {n}", f"d{n}.txt"), 1.0, 1, 1)
        for n in range(1, 4)
    ]
    board.add(hits)
    return board


def test_the_tools_are_named_search_and_submit_answer() -> None:
    assert (SEARCH_TOOL.name, SUBMIT_TOOL.name) == ("search", "submit_answer")


def test_parse_search_query_returns_the_trimmed_query() -> None:
    assert parse_search_query({"query": "  payments march  "}) == "payments march"


@pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "   "}, {"query": 5}])
def test_parse_search_query_rejects_missing_or_blank_queries(arguments: dict[str, object]) -> None:
    with pytest.raises(ToolArgumentError, match="query"):
        parse_search_query(arguments)


def test_an_answered_submission_with_known_citations_is_accepted() -> None:
    submission = parse_submission(
        {"status": "answered", "answer": "Because of X.", "citations": ["E1", "E3"]},
        board_with_three_items(),
    )

    assert submission.status == "answered"
    assert submission.answer == "Because of X."
    assert submission.evidence_ids == ("E1", "E3")
    assert submission.conflicts == ()


def test_citation_ids_are_normalised_and_deduplicated() -> None:
    submission = parse_submission(
        {"status": "answered", "answer": "A", "citations": ["e1", "[E1]", "E2"]},
        board_with_three_items(),
    )

    assert submission.evidence_ids == ("E1", "E2")


def test_citing_evidence_that_was_never_retrieved_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="E9"):
        parse_submission(
            {"status": "answered", "answer": "A", "citations": ["E1", "E9"]},
            board_with_three_items(),
        )


def test_an_answered_submission_without_citations_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="cite"):
        parse_submission(
            {"status": "answered", "answer": "A", "citations": []}, board_with_three_items()
        )


def test_an_unknown_status_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="status"):
        parse_submission(
            {"status": "maybe", "answer": "A", "citations": ["E1"]}, board_with_three_items()
        )


@pytest.mark.parametrize("answer", ["", "   ", None, 3])
def test_a_blank_answer_is_rejected(answer: object) -> None:
    with pytest.raises(ToolArgumentError, match="answer"):
        parse_submission(
            {"status": "answered", "answer": answer, "citations": ["E1"]}, board_with_three_items()
        )


def test_citations_must_be_a_list_of_strings() -> None:
    with pytest.raises(ToolArgumentError, match="citations"):
        parse_submission(
            {"status": "answered", "answer": "A", "citations": "E1"}, board_with_three_items()
        )


def test_insufficient_evidence_needs_no_citations() -> None:
    submission = parse_submission(
        {"status": "insufficient_evidence", "answer": "The documents do not say."},
        board_with_three_items(),
    )

    assert submission.status == "insufficient_evidence"
    assert submission.evidence_ids == ()


def test_conflicting_evidence_keeps_each_conflict_and_cites_its_evidence() -> None:
    submission = parse_submission(
        {
            "status": "conflicting_evidence",
            "answer": "The sources disagree.",
            "conflicts": [{"description": "Start date differs.", "evidence": ["E1", "E2"]}],
        },
        board_with_three_items(),
    )

    assert submission.status == "conflicting_evidence"
    assert [(c.description, c.evidence_ids) for c in submission.conflicts] == [
        ("Start date differs.", ("E1", "E2"))
    ]
    assert submission.evidence_ids == ("E1", "E2")


def test_conflicting_evidence_without_a_conflict_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="conflict"):
        parse_submission(
            {"status": "conflicting_evidence", "answer": "They disagree.", "citations": ["E1"]},
            board_with_three_items(),
        )


def test_a_conflict_needs_at_least_two_pieces_of_evidence() -> None:
    with pytest.raises(ToolArgumentError, match="two"):
        parse_submission(
            {
                "status": "conflicting_evidence",
                "answer": "They disagree.",
                "conflicts": [{"description": "Differs.", "evidence": ["E1"]}],
            },
            board_with_three_items(),
        )


def test_a_conflict_must_be_an_object_with_a_description() -> None:
    with pytest.raises(ToolArgumentError, match="conflict"):
        parse_submission(
            {"status": "conflicting_evidence", "answer": "A", "conflicts": ["E1 vs E2"]},
            board_with_three_items(),
        )
