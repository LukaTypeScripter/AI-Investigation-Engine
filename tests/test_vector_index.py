import pytest

from investigation_engine.retrieval.vector import InMemoryVectorIndex


def make_index() -> InMemoryVectorIndex:
    index = InMemoryVectorIndex(dimension=2, initial_capacity=2)
    index.add("a", [1.0, 0.0])
    index.add("b", [0.0, 1.0])
    index.add("c", [1.0, 1.0])
    return index


def ranked_ids(index: InMemoryVectorIndex, vector: list[float], limit: int = 10) -> list[str]:
    return [item_id for item_id, _ in index.search(vector, limit)]


def test_search_orders_items_by_cosine_similarity() -> None:
    assert ranked_ids(make_index(), [1.0, 0.1], 3) == ["a", "c", "b"]


def test_search_scores_are_cosine_similarities_regardless_of_vector_length() -> None:
    index = make_index()

    assert index.search([2.0, 0.0], 1) == [("a", pytest.approx(1.0))]
    assert dict(index.search([5.0, 0.0], 3))["c"] == pytest.approx(0.70710678)


def test_search_respects_the_limit() -> None:
    assert len(make_index().search([1.0, 0.0], 2)) == 2


def test_search_on_an_empty_index_returns_nothing() -> None:
    assert InMemoryVectorIndex(dimension=2).search([1.0, 0.0], 5) == []


def test_search_with_a_non_positive_limit_returns_nothing() -> None:
    assert make_index().search([1.0, 0.0], 0) == []


def test_search_with_a_zero_vector_returns_nothing() -> None:
    assert make_index().search([0.0, 0.0], 5) == []


def test_search_breaks_ties_by_id() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("b", [1.0, 0.0])
    index.add("a", [1.0, 0.0])

    assert ranked_ids(index, [1.0, 0.0]) == ["a", "b"]


def test_remove_drops_the_item_from_results_and_size() -> None:
    index = make_index()

    index.remove("a")

    assert ranked_ids(index, [1.0, 0.0]) == ["c", "b"]
    assert len(index) == 2


def test_remove_keeps_the_remaining_vectors_intact() -> None:
    index = make_index()
    index.remove("a")

    assert index.search([0.0, 1.0], 1) == [("b", pytest.approx(1.0))]
    assert dict(index.search([1.0, 1.0], 2))["c"] == pytest.approx(1.0)


def test_remove_unknown_item_is_ignored() -> None:
    index = make_index()

    index.remove("missing")

    assert len(index) == 3


def test_add_existing_id_replaces_its_vector() -> None:
    index = make_index()

    index.add("a", [0.0, 1.0])

    assert len(index) == 3
    assert dict(index.search([1.0, 0.0], 3))["a"] == pytest.approx(0.0, abs=1e-6)


def test_index_grows_beyond_its_initial_capacity() -> None:
    index = InMemoryVectorIndex(dimension=2, initial_capacity=2)
    for number in range(100):
        index.add(f"item{number}", [1.0, float(number)])

    assert len(index) == 100
    assert len(index.search([1.0, 0.0], 100)) == 100


def test_a_vector_of_the_wrong_dimension_is_rejected() -> None:
    index = InMemoryVectorIndex(dimension=2)

    with pytest.raises(ValueError, match="dimension"):
        index.add("a", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0], 1)


def test_a_zero_vector_cannot_be_indexed() -> None:
    with pytest.raises(ValueError, match="zero"):
        InMemoryVectorIndex(dimension=2).add("a", [0.0, 0.0])


@pytest.mark.parametrize(("dimension", "capacity"), [(0, 4), (2, 0)])
def test_invalid_sizes_are_rejected(dimension: int, capacity: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryVectorIndex(dimension=dimension, initial_capacity=capacity)
