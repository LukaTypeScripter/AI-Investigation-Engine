import pytest

from investigation_engine.domain import Chunk, Document
from investigation_engine.ingestion.chunking import FixedWindowChunker, SentenceChunker


def make_document(text: str) -> Document:
    return Document(id="doc", text=text, title="Title", source="source.txt")


def texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


def test_fixed_window_overlapping_windows_share_boundary_words() -> None:
    document = make_document("w0 w1 w2 w3 w4 w5 w6 w7 w8 w9")

    chunks = FixedWindowChunker(size=4, overlap=1).chunk(document)

    assert texts(chunks) == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_fixed_window_text_shorter_than_window_gives_one_chunk() -> None:
    document = make_document("one two three")

    chunks = FixedWindowChunker(size=10, overlap=2).chunk(document)

    assert texts(chunks) == ["one two three"]


def test_fixed_window_exact_multiple_has_no_trailing_empty_chunk() -> None:
    document = make_document("a b c d e f g h")

    chunks = FixedWindowChunker(size=4, overlap=0).chunk(document)

    assert texts(chunks) == ["a b c d", "e f g h"]


def test_fixed_window_last_window_may_be_shorter() -> None:
    document = make_document("a b c d e")

    chunks = FixedWindowChunker(size=4, overlap=0).chunk(document)

    assert texts(chunks) == ["a b c d", "e"]


def test_fixed_window_blank_text_gives_no_chunks() -> None:
    assert FixedWindowChunker(size=4, overlap=1).chunk(make_document("  \n ")) == []


def test_fixed_window_chunks_carry_document_identity() -> None:
    document = make_document("a b c d e")

    chunks = FixedWindowChunker(size=4, overlap=0).chunk(document)

    assert [chunk.id for chunk in chunks] == ["doc:0", "doc:1"]
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert {chunk.document_id for chunk in chunks} == {"doc"}
    assert {chunk.title for chunk in chunks} == {"Title"}
    assert {chunk.source for chunk in chunks} == {"source.txt"}


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_fixed_window_invalid_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="window"):
        FixedWindowChunker(size=size, overlap=overlap)


def test_sentence_chunker_packs_whole_sentences_up_to_the_limit() -> None:
    document = make_document("One two. Three four. Five six.")

    chunks = SentenceChunker(max_words=4).chunk(document)

    assert texts(chunks) == ["One two. Three four.", "Five six."]


def test_sentence_chunker_keeps_everything_together_when_it_fits() -> None:
    document = make_document("One two. Three four.")

    chunks = SentenceChunker(max_words=10).chunk(document)

    assert texts(chunks) == ["One two. Three four."]


def test_sentence_chunker_splits_a_sentence_longer_than_the_limit() -> None:
    document = make_document("a b c d e f g h i j.")

    chunks = SentenceChunker(max_words=4).chunk(document)

    assert texts(chunks) == ["a b c d", "e f g h", "i j."]


def test_sentence_chunker_flushes_pending_sentences_before_a_long_one() -> None:
    document = make_document("Short one. a b c d e f. Tail.")

    chunks = SentenceChunker(max_words=4).chunk(document)

    assert texts(chunks) == ["Short one.", "a b c d", "e f.", "Tail."]


def test_sentence_chunker_collapses_newlines_and_repeated_spaces() -> None:
    document = make_document("First line.\nSecond   line.")

    chunks = SentenceChunker(max_words=10).chunk(document)

    assert texts(chunks) == ["First line. Second line."]


def test_sentence_chunker_blank_text_gives_no_chunks() -> None:
    assert SentenceChunker(max_words=4).chunk(make_document("   ")) == []


def test_sentence_chunker_chunks_carry_document_identity() -> None:
    document = make_document("One two. Three four. Five six.")

    chunks = SentenceChunker(max_words=4).chunk(document)

    assert [chunk.id for chunk in chunks] == ["doc:0", "doc:1"]
    assert {chunk.source for chunk in chunks} == {"source.txt"}


def test_sentence_chunker_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_words"):
        SentenceChunker(max_words=0)
