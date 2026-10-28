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


VOCABULARY = [
    "the",
    "of",
    "was",
    "gateway",
    "payment",
    "ledger",
    "refund",
    "outage",
    "march",
    "deploy",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "theta",
    "kappa",
    "sigma",
    "omega",
]


def synthetic_corpus(size: int) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for number in range(size):
        length = 4 + number % 13
        words = [
            VOCABULARY[(number * 7 + position * 13 + (number * position) % 5) % 4]
            if position % 3 == 0
            else VOCABULARY[(number * 3 + position * 11 + number // 7) % len(VOCABULARY)]
            for position in range(length)
        ]
        corpus[f"d{number:03d}"] = " ".join(words)
    return corpus


def reference_search(
    corpus: dict[str, str], query: str, limit: int, k1: float = 1.5, b: float = 0.75
) -> list[tuple[str, float]]:
    tokens = {item_id: tokenize(text) for item_id, text in corpus.items()}
    average = sum(len(words) for words in tokens.values()) / len(tokens)
    scores: dict[str, float] = {}
    for term in set(tokenize(query)):
        containing = [item_id for item_id, words in tokens.items() if term in words]
        for item_id in containing:
            idf = math.log(1 + (len(tokens) - len(containing) + 0.5) / (len(containing) + 0.5))
            frequency = tokens[item_id].count(term)
            norm = 1 - b + b * len(tokens[item_id]) / average
            scores[item_id] = scores.get(item_id, 0.0) + idf * (
                frequency * (k1 + 1) / (frequency + k1 * norm)
            )
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


@pytest.mark.parametrize(
    "query",
    [
        "the of was",
        "gateway payment",
        "the gateway deploy march alpha",
        "zeta omega the",
        "payment outage refund ledger gateway",
        "unknownterm gateway",
        "the",
    ],
)
@pytest.mark.parametrize("limit", [1, 3, 10, 500])
def test_search_matches_a_plain_bm25_computation_on_a_larger_corpus(query: str, limit: int) -> None:
    corpus = synthetic_corpus(300)
    index = build_index(corpus)

    found = index.search(query, limit)
    expected = reference_search(corpus, query, limit)

    assert [item_id for item_id, _ in found] == [item_id for item_id, _ in expected]
    assert [score for _, score in found] == pytest.approx([score for _, score in expected])
