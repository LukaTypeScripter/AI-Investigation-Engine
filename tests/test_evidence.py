import pytest

from investigation_engine.agent.evidence import EvidenceBoard, normalize_evidence_id
from investigation_engine.domain import Chunk, SourceRef
from investigation_engine.retrieval.hybrid import Hit


class StubSources:
    def __init__(self, sources: dict[str, list[SourceRef]] | None = None) -> None:
        self._sources = sources or {}

    def sources_of(self, chunk_id: str) -> list[SourceRef]:
        return self._sources.get(chunk_id, [])


def make_hit(chunk_id: str, text: str = "Some text.", title: str = "Title") -> Hit:
    document_id = chunk_id.split(":")[0]
    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        index=0,
        text=text,
        title=title,
        source=f"{document_id}.txt",
    )
    return Hit(chunk=chunk, score=1.0, vector_rank=1, keyword_rank=1)


def make_board(sources: StubSources | None = None, max_chars: int = 1200) -> EvidenceBoard:
    return EvidenceBoard(sources or StubSources(), max_chars)


def test_new_chunks_are_numbered_in_the_order_they_are_added() -> None:
    items = make_board().add([make_hit("a:0"), make_hit("b:0")])

    assert [item.evidence_id for item in items] == ["E1", "E2"]


def test_a_chunk_seen_again_keeps_its_evidence_id() -> None:
    board = make_board()
    board.add([make_hit("a:0")])

    items = board.add([make_hit("b:0"), make_hit("a:0")])

    assert [item.evidence_id for item in items] == ["E2", "E1"]


def test_items_can_be_looked_up_by_evidence_id() -> None:
    board = make_board()
    board.add([make_hit("a:0")])

    assert board.get("E1") is not None
    assert board.get("E9") is None


def test_unknown_lists_ids_that_are_not_on_the_board_after_normalising() -> None:
    board = make_board()
    board.add([make_hit("a:0")])

    assert board.unknown(["e1", "E7", "[E8]"]) == ["E7", "E8"]


@pytest.mark.parametrize(("raw", "expected"), [("[e1]", "E1"), (" E2. ", "E2"), ("(e3),", "E3")])
def test_normalize_evidence_id_strips_brackets_and_punctuation(raw: str, expected: str) -> None:
    assert normalize_evidence_id(raw) == expected


def test_sources_come_from_the_lookup() -> None:
    sources = StubSources(
        {"a:0": [SourceRef("a", "Alpha", "a.txt"), SourceRef("b", "Beta", "b.txt")]}
    )
    board = make_board(sources)

    item = board.add([make_hit("a:0")])[0]

    assert item.sources == (SourceRef("a", "Alpha", "a.txt"), SourceRef("b", "Beta", "b.txt"))


def test_sources_fall_back_to_the_chunk_itself_when_the_lookup_knows_nothing() -> None:
    item = make_board().add([make_hit("a:0", title="Alpha")])[0]

    assert item.sources == (SourceRef("a", "Alpha", "a.txt"),)


def test_render_lists_each_item_with_its_id_title_sources_and_text() -> None:
    board = make_board()
    items = board.add(
        [make_hit("a:0", "Alpha text.", "Alpha"), make_hit("b:0", "Beta text.", "Beta")]
    )

    rendered = board.render("what is alpha", items)

    assert rendered.splitlines()[0] == "Search results for: what is alpha"
    assert '[E1] "Alpha" (sources: a.txt)\nAlpha text.' in rendered
    assert '[E2] "Beta" (sources: b.txt)\nBeta text.' in rendered


def test_render_trims_long_text() -> None:
    board = make_board(max_chars=10)
    items = board.add([make_hit("a:0", "x" * 50)])

    assert "x" * 10 + "..." in board.render("q", items)
    assert "x" * 11 not in board.render("q", items)


def test_render_with_no_items_says_nothing_was_found() -> None:
    assert make_board().render("missing topic", []) == "No results found for: missing topic"


def test_render_names_untitled_chunks() -> None:
    board = make_board()
    items = board.add([make_hit("a:0", title="")])

    assert '[E1] "untitled"' in board.render("q", items)


def test_citation_carries_the_chunk_text_and_every_source() -> None:
    sources = StubSources(
        {"a:0": [SourceRef("a", "Alpha", "a.txt"), SourceRef("b", "Beta", "b.txt")]}
    )
    board = make_board(sources)
    board.add([make_hit("a:0", "Alpha text.", "Alpha")])

    citation = board.citation("E1")

    assert (citation.evidence_id, citation.chunk_id, citation.title) == ("E1", "a:0", "Alpha")
    assert citation.text == "Alpha text."
    assert [ref.document_id for ref in citation.sources] == ["a", "b"]


def test_citation_for_an_unknown_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        make_board().citation("E1")
