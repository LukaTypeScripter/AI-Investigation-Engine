import pytest

from investigation_engine.evaluation.metrics import (
    CitationScores,
    citation_scores,
    contains_answer,
    exact_match,
    full_recall_at_k,
    mean,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)


def test_normalize_answer_lowercases_and_drops_punctuation_and_articles() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


@pytest.mark.parametrize(
    ("typographic", "plain"),
    [
        ("ALRT‑5531", "ALRT-5531"),
        ("payment‑gateway", "payment-gateway"),
        ("1980–1987", "1980-1987"),
        ("a — b", "a - b"),
        ("Orion Pay’s", "Orion Pay's"),
        ("a “quoted” word", 'a "quoted" word'),
    ],
)
def test_normalize_answer_treats_typographic_punctuation_like_plain_ascii(
    typographic: str, plain: str
) -> None:
    assert normalize_answer(typographic) == normalize_answer(plain)


def test_contains_answer_finds_an_identifier_written_with_a_non_breaking_hyphen() -> None:
    assert contains_answer("The first alert (ID ALRT‑5531) fired.", "ALRT-5531") is True


def test_exact_match_ignores_case_articles_and_punctuation() -> None:
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match("Eiffel Tower", "Louvre") == 0.0


def test_token_f1_rewards_partial_overlap() -> None:
    assert token_f1("Barack Obama", "Obama") == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_token_f1_is_one_for_identical_answers_and_zero_without_overlap() -> None:
    assert token_f1("New York", "new york") == 1.0
    assert token_f1("Paris", "Berlin") == 0.0


def test_token_f1_counts_repeated_tokens_only_as_often_as_the_gold_answer_has_them() -> None:
    assert token_f1("go go go", "go") == pytest.approx(2 * (1 / 3) * 1.0 / (1 / 3 + 1.0))


def test_token_f1_treats_yes_and_no_as_all_or_nothing() -> None:
    assert token_f1("yes", "yes") == 1.0
    assert token_f1("no", "yes") == 0.0


def test_token_f1_of_two_empty_answers_is_one_and_of_one_empty_answer_is_zero() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("", "Paris") == 0.0
    assert token_f1("Paris", "") == 0.0


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("It was Barack Obama.", "Obama", True),
        ("The pool size was 50 connections", "50", True),
        ("somebody", "body", False),
        ("nobody came", "no", False),
        ("in New York City", "new york", True),
        ("York New", "new york", False),
        ("anything", "", False),
    ],
)
def test_contains_answer_matches_whole_words_in_order(
    prediction: str, gold: str, expected: bool
) -> None:
    assert contains_answer(prediction, gold) is expected


def test_recall_at_k_is_the_share_of_gold_documents_in_the_top_k() -> None:
    ranked = ["a", "x", "b", "y"]

    assert recall_at_k(ranked, {"a", "b"}, 1) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 3) == 1.0
    assert recall_at_k(ranked, {"c"}, 4) == 0.0


def test_recall_at_k_without_gold_documents_is_zero() -> None:
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_full_recall_at_k_needs_every_gold_document() -> None:
    ranked = ["a", "x", "b"]

    assert full_recall_at_k(ranked, {"a", "b"}, 2) == 0.0
    assert full_recall_at_k(ranked, {"a", "b"}, 3) == 1.0


def test_reciprocal_rank_uses_the_first_gold_document() -> None:
    assert reciprocal_rank(["x", "y", "a", "b"], {"a", "b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_citation_scores_measure_precision_and_recall_against_the_gold_documents() -> None:
    scores = citation_scores(cited={"a", "x"}, gold={"a", "b"})

    assert scores == CitationScores(precision=0.5, recall=0.5)


def test_citation_scores_without_citations_are_zero() -> None:
    assert citation_scores(cited=set(), gold={"a"}) == CitationScores(precision=0.0, recall=0.0)


def test_mean_of_nothing_is_zero() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == 0.5
