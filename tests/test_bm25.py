import math

import pytest

from investigation_engine.retrieval.bm25 import Bm25Index, tokenize


def build_index(documents: dict[str, str]) -> Bm25Index:
    index = Bm25Index()
    for item_id, text in documents.items():
        index.add(item_id, text)
    return index


def ranked_ids(index: Bm25Index, query: str, limit: int = 10) -> list[str]:
    return [item_id for item_id, _ in index.search(query, limit)]


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World! It's 2024.") == ["hello", "world", "it", "s", "2024"]


def test_tokenize_handles_georgian_text() -> None:
    assert tokenize("გამარჯობა მსოფლიო") == ["გამარჯობა", "მსოფლიო"]


def test_search_single_document_matches_the_hand_computed_score() -> None:
    index = build_index({"d1": "apple"})

    assert index.search("apple", 5) == [("d1", pytest.approx(math.log(4 / 3)))]


def test_search_ranks_the_document_with_the_rarer_term_first() -> None:
    index = build_index({"d1": "common rare", "d2": "common filler", "d3": "common other"})

    assert ranked_ids(index, "rare common")[0] == "d1"


def test_search_prefers_higher_term_frequency_at_equal_length() -> None:
    index = build_index({"d1": "apple apple filler", "d2": "apple filler filler"})

    assert ranked_ids(index, "apple") == ["d1", "d2"]


def test_search_prefers_the_shorter_document_at_equal_term_frequency() -> None:
    index = build_index({"long": "apple " + "filler " * 20, "short": "apple"})

    assert ranked_ids(index, "apple") == ["short", "long"]


def test_search_ignores_repeated_query_terms() -> None:
    index = build_index({"d1": "apple banana", "d2": "banana banana"})

    assert index.search("apple apple", 5) == index.search("apple", 5)


def test_search_unknown_terms_return_nothing() -> None:
    index = build_index({"d1": "apple"})

    assert index.search("zebra", 5) == []


def test_search_empty_index_returns_nothing() -> None:
    assert Bm25Index().search("apple", 5) == []


def test_search_respects_the_limit() -> None:
    index = build_index({"d1": "apple", "d2": "apple pie", "d3": "apple tart"})

    assert len(index.search("apple", 2)) == 2


def test_search_breaks_score_ties_by_id() -> None:
    index = build_index({"d2": "apple", "d1": "apple"})

    assert ranked_ids(index, "apple") == ["d1", "d2"]


def test_remove_drops_the_document_from_results_and_size() -> None:
    index = build_index({"d1": "apple", "d2": "apple pie"})

    index.remove("d1")

    assert ranked_ids(index, "apple") == ["d2"]
    assert len(index) == 1


def test_remove_restores_the_statistics_of_the_remaining_documents() -> None:
    index = build_index({"d1": "apple banana", "d2": "apple " + "filler " * 30})
    index.remove("d2")
    fresh = build_index({"d1": "apple banana"})

    assert index.search("apple", 5) == fresh.search("apple", 5)


def test_remove_unknown_id_is_ignored() -> None:
    index = build_index({"d1": "apple"})

    index.remove("missing")

    assert len(index) == 1


def test_add_existing_id_replaces_its_content() -> None:
    index = Bm25Index()
    index.add("d1", "apple")
    index.add("d1", "banana")

    assert index.search("apple", 5) == []
    assert ranked_ids(index, "banana") == ["d1"]
    assert len(index) == 1


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.5, -0.1), (1.5, 1.1)])
def test_invalid_parameters_are_rejected(k1: float, b: float) -> None:
    with pytest.raises(ValueError, match="BM25"):
        Bm25Index(k1=k1, b=b)
