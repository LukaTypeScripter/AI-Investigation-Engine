import pytest

from investigation_engine.domain import Chunk
from investigation_engine.retrieval.rerank import FeatureReranker, RerankWeights, informative_terms


def make_chunk(chunk_id: str, text: str = "unrelated words", title: str = "") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=chunk_id,
        index=0,
        text=text,
        title=title,
        source="",
    )


def ranked_ids(ranking: list[tuple[Chunk, float]]) -> list[str]:
    return [chunk.id for chunk, _ in ranking]


def test_informative_terms_drop_stopwords_punctuation_and_repeats() -> None:
    assert informative_terms("Why did the payments fail in March? Payments!") == {
        "payments",
        "fail",
        "march",
    }


def test_a_title_found_in_the_query_can_overtake_a_higher_fused_score() -> None:
    leader = make_chunk("leader", title="Other")
    named = make_chunk("named", text="x", title="Scott Derrickson")

    ranking = FeatureReranker().rerank(
        "Scott Derrickson nationality", [(leader, 1.0), (named, 0.9)]
    )

    assert ranked_ids(ranking) == ["named", "leader"]


def test_score_combines_fused_coverage_and_title_features() -> None:
    chunk = make_chunk("a", text="payments failed in march")

    ranking = FeatureReranker().rerank("payments march", [(chunk, 0.02)])

    assert ranking == [(chunk, pytest.approx(1.0 + 0.5 * 1.0 + 0.5 * 0.0))]


def test_coverage_counts_query_terms_found_in_the_text_or_the_title() -> None:
    from_text = make_chunk("text", text="payments delayed")
    from_title = make_chunk("title", text="nothing here", title="Payments delayed")
    neither = make_chunk("neither", text="nothing here")

    ranking = FeatureReranker(RerankWeights(fused=0.0, coverage=1.0, title=0.0)).rerank(
        "payments delayed", [(neither, 1.0), (from_title, 1.0), (from_text, 1.0)]
    )

    assert ranked_ids(ranking) == ["text", "title", "neither"]
    assert {chunk.id: score for chunk, score in ranking}["neither"] == 0.0


def test_a_partly_matching_title_scores_proportionally() -> None:
    chunk = make_chunk("a", title="Ed Wood")

    ranking = FeatureReranker(RerankWeights(fused=0.0, coverage=0.0, title=1.0)).rerank(
        "Scott Derrickson Ed", [(chunk, 1.0)]
    )

    assert ranking[0][1] == pytest.approx(0.5)


def test_zero_feature_weights_keep_the_fused_order() -> None:
    first, second = make_chunk("first", title="Scott"), make_chunk("second")
    weights = RerankWeights(fused=1.0, coverage=0.0, title=0.0)

    ranking = FeatureReranker(weights).rerank("Scott", [(first, 0.9), (second, 0.8)])

    assert ranked_ids(ranking) == ["first", "second"]


def test_a_query_made_only_of_stopwords_leaves_the_fused_order() -> None:
    first, second = make_chunk("first"), make_chunk("second")

    ranking = FeatureReranker().rerank("what is the", [(first, 0.9), (second, 0.8)])

    assert ranked_ids(ranking) == ["first", "second"]


def test_equal_scores_are_ordered_by_chunk_id() -> None:
    ranking = FeatureReranker().rerank("x", [(make_chunk("b"), 1.0), (make_chunk("a"), 1.0)])

    assert ranked_ids(ranking) == ["a", "b"]


def test_reranking_nothing_returns_nothing() -> None:
    assert FeatureReranker().rerank("query", []) == []


def test_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RerankWeights(fused=1.0, coverage=-0.1, title=0.0)
