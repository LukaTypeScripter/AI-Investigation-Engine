from collections.abc import Callable
from typing import Protocol

import pytest

from investigation_engine.retrieval.postgres_vector import PostgresVectorIndex
from investigation_engine.retrieval.protocols import VectorIndex
from investigation_engine.retrieval.vector import InMemoryVectorIndex
from investigation_engine.storage.database import Database


class IndexUnderTest(VectorIndex, Protocol):
    def __len__(self) -> int: ...


@pytest.fixture(params=["memory", "postgres"])
def empty_index(request: pytest.FixtureRequest) -> IndexUnderTest:
    if request.param == "memory":
        return InMemoryVectorIndex(dimension=2, initial_capacity=2)
    make_database: Callable[[int], Database] = request.getfixturevalue("make_database")
    return PostgresVectorIndex(make_database(2))


@pytest.fixture
def index(empty_index: IndexUnderTest) -> IndexUnderTest:
    empty_index.add("a", [1.0, 0.0])
    empty_index.add("b", [0.0, 1.0])
    empty_index.add("c", [1.0, 1.0])
    return empty_index


def ranked_ids(index: VectorIndex, vector: list[float], limit: int = 10) -> list[str]:
    return [item_id for item_id, _ in index.search(vector, limit)]


def test_search_orders_items_by_cosine_similarity(index: IndexUnderTest) -> None:
    assert ranked_ids(index, [1.0, 0.1], 3) == ["a", "c", "b"]


def test_search_scores_are_cosine_similarities_regardless_of_vector_length(
    index: IndexUnderTest,
) -> None:
    assert index.search([2.0, 0.0], 1) == [("a", pytest.approx(1.0))]
    assert dict(index.search([5.0, 0.0], 3))["c"] == pytest.approx(0.70710678)


def test_search_respects_the_limit(index: IndexUnderTest) -> None:
    assert len(index.search([1.0, 0.0], 2)) == 2


def test_search_on_an_empty_index_returns_nothing(empty_index: IndexUnderTest) -> None:
    assert empty_index.search([1.0, 0.0], 5) == []


def test_search_with_a_non_positive_limit_returns_nothing(index: IndexUnderTest) -> None:
    assert index.search([1.0, 0.0], 0) == []


def test_search_with_a_zero_vector_returns_nothing(index: IndexUnderTest) -> None:
    assert index.search([0.0, 0.0], 5) == []


def test_search_breaks_ties_by_id(empty_index: IndexUnderTest) -> None:
    empty_index.add("b", [1.0, 0.0])
    empty_index.add("a", [1.0, 0.0])

    assert ranked_ids(empty_index, [1.0, 0.0]) == ["a", "b"]


def test_remove_drops_the_item_from_results_and_size(index: IndexUnderTest) -> None:
    index.remove("a")

    assert ranked_ids(index, [1.0, 0.0]) == ["c", "b"]
    assert len(index) == 2


def test_remove_keeps_the_remaining_vectors_intact(index: IndexUnderTest) -> None:
    index.remove("a")

    assert index.search([0.0, 1.0], 1) == [("b", pytest.approx(1.0))]
    assert dict(index.search([1.0, 1.0], 2))["c"] == pytest.approx(1.0)


def test_remove_unknown_item_is_ignored(index: IndexUnderTest) -> None:
    index.remove("missing")

    assert len(index) == 3


def test_add_existing_id_replaces_its_vector(index: IndexUnderTest) -> None:
    index.add("a", [0.0, 1.0])

    assert len(index) == 3
    assert dict(index.search([1.0, 0.0], 3))["a"] == pytest.approx(0.0, abs=1e-6)


def test_add_many_indexes_every_item(empty_index: IndexUnderTest) -> None:
    empty_index.add_many([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])

    assert len(empty_index) == 2
    assert ranked_ids(empty_index, [0.0, 1.0]) == ["b", "a"]


def test_add_many_replaces_the_vectors_of_ids_that_already_exist(index: IndexUnderTest) -> None:
    index.add_many([("a", [0.0, 1.0]), ("d", [1.0, 0.0])])

    assert len(index) == 4
    assert dict(index.search([1.0, 0.0], 4))["a"] == pytest.approx(0.0, abs=1e-6)


def test_add_many_with_nothing_changes_nothing(index: IndexUnderTest) -> None:
    index.add_many([])

    assert len(index) == 3


def test_add_many_rejects_a_bad_vector_before_adding_any(empty_index: IndexUnderTest) -> None:
    with pytest.raises(ValueError, match="zero"):
        empty_index.add_many([("a", [1.0, 0.0]), ("b", [0.0, 0.0])])

    assert len(empty_index) == 0


def test_a_vector_of_the_wrong_dimension_is_rejected(empty_index: IndexUnderTest) -> None:
    with pytest.raises(ValueError, match="dimension"):
        empty_index.add("a", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        empty_index.search([1.0], 1)


def test_a_zero_vector_cannot_be_indexed(empty_index: IndexUnderTest) -> None:
    with pytest.raises(ValueError, match="zero"):
        empty_index.add("a", [0.0, 0.0])


def test_the_memory_index_grows_beyond_its_initial_capacity() -> None:
    index = InMemoryVectorIndex(dimension=2, initial_capacity=2)
    for number in range(100):
        index.add(f"item{number}", [1.0, float(number)])

    assert len(index) == 100
    assert len(index.search([1.0, 0.0], 100)) == 100


@pytest.mark.parametrize(("dimension", "capacity"), [(0, 4), (2, 0)])
def test_the_memory_index_rejects_invalid_sizes(dimension: int, capacity: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryVectorIndex(dimension=dimension, initial_capacity=capacity)
