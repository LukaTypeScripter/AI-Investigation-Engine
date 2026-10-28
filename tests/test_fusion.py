import pytest

from investigation_engine.retrieval.fusion import reciprocal_rank_fusion


def test_fusion_top_rank_in_two_lists_scores_two_over_sixty_one() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["a", "c"]]))

    assert fused["a"] == pytest.approx(2 / 61)


def test_fusion_item_found_by_both_rankers_beats_a_single_ranker_leader() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])

    assert [item_id for item_id, _ in fused] == ["b", "a", "c"]


def test_fusion_breaks_score_ties_by_id() -> None:
    fused = reciprocal_rank_fusion([["b"], ["a"]])

    assert [item_id for item_id, _ in fused] == ["a", "b"]


def test_fusion_without_rankings_returns_nothing() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_fusion_constant_of_zero_gives_the_top_rank_a_full_point() -> None:
    assert reciprocal_rank_fusion([["a"]], k=0) == [("a", pytest.approx(1.0))]


def test_fusion_negative_constant_is_rejected() -> None:
    with pytest.raises(ValueError, match="constant"):
        reciprocal_rank_fusion([["a"]], k=-1)
