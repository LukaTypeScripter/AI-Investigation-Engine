from investigation_engine.domain import Chunk
from investigation_engine.ingestion.dedup import fingerprint, split_duplicates


def make_chunk(chunk_id: str, text: str) -> Chunk:
    document_id = chunk_id.split(":")[0]
    return Chunk(id=chunk_id, document_id=document_id, index=0, text=text, title="", source="")


def test_fingerprint_ignores_case_and_whitespace() -> None:
    assert fingerprint("Hello   World\n") == fingerprint("hello world")


def test_fingerprint_differs_for_different_text() -> None:
    assert fingerprint("hello world") != fingerprint("hello there")


def test_split_duplicates_keeps_the_first_occurrence() -> None:
    chunks = [
        make_chunk("a:0", "Same text"),
        make_chunk("b:0", "same   TEXT"),
        make_chunk("c:0", "Other"),
    ]

    result = split_duplicates(chunks)

    assert [chunk.id for chunk in result.unique] == ["a:0", "c:0"]
    assert result.duplicate_of == {"b:0": "a:0"}


def test_split_duplicates_points_every_copy_at_the_first_one() -> None:
    chunks = [make_chunk(f"d{n}:0", "Repeated") for n in range(3)]

    result = split_duplicates(chunks)

    assert [chunk.id for chunk in result.unique] == ["d0:0"]
    assert result.duplicate_of == {"d1:0": "d0:0", "d2:0": "d0:0"}


def test_split_duplicates_recognises_chunks_ingested_earlier() -> None:
    known = {fingerprint("Existing"): "old:3"}

    result = split_duplicates([make_chunk("new:0", "existing")], known=known)

    assert result.unique == []
    assert result.duplicate_of == {"new:0": "old:3"}


def test_split_duplicates_does_not_modify_the_known_mapping() -> None:
    known = {fingerprint("Existing"): "old:3"}

    split_duplicates([make_chunk("new:0", "Fresh")], known=known)

    assert known == {fingerprint("Existing"): "old:3"}


def test_split_duplicates_without_repeats_keeps_everything() -> None:
    chunks = [make_chunk("a:0", "one"), make_chunk("b:0", "two")]

    result = split_duplicates(chunks)

    assert result.unique == chunks
    assert result.duplicate_of == {}
