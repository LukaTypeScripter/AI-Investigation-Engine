import os
from collections.abc import Iterable, Iterator

import numpy as np
import pytest

from investigation_engine.embeddings import EmbeddingError, FastEmbedEmbedder


class StubModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.passage_batches: list[list[str]] = []
        self.batch_sizes: list[int] = []
        self.queries: list[str] = []

    def passage_embed(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[np.ndarray]:
        batch = list(texts)
        self.passage_batches.append(batch)
        self.batch_sizes.append(batch_size)
        for index, _ in enumerate(batch):
            yield np.full(self.dimension, float(index), dtype=np.float32)

    def query_embed(self, query: str) -> Iterator[np.ndarray]:
        self.queries.append(query)
        yield np.full(self.dimension, 9.0, dtype=np.float32)


def make_embedder(model: StubModel, loads: list[str] | None = None) -> FastEmbedEmbedder:
    def factory(model_name: str) -> StubModel:
        if loads is not None:
            loads.append(model_name)
        return model

    return FastEmbedEmbedder(model_name="stub-model", dimension=3, model_factory=factory)


def test_embed_documents_returns_one_plain_float_vector_per_text() -> None:
    model = StubModel()

    vectors = make_embedder(model).embed_documents(["first", "second"])

    assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert type(vectors[0][0]) is float
    assert model.passage_batches == [["first", "second"]]
    assert model.batch_sizes == [64]


def test_embed_query_uses_the_query_side_of_the_model() -> None:
    model = StubModel()

    vector = make_embedder(model).embed_query("why did it fail")

    assert vector == [9.0, 9.0, 9.0]
    assert model.queries == ["why did it fail"]


def test_embedding_no_texts_does_not_load_the_model() -> None:
    loads: list[str] = []

    assert make_embedder(StubModel(), loads).embed_documents([]) == []
    assert loads == []


def test_model_is_loaded_lazily_and_only_once() -> None:
    loads: list[str] = []
    embedder = make_embedder(StubModel(), loads)

    assert loads == []
    embedder.embed_query("a")
    embedder.embed_documents(["b"])
    embedder.embed_query("c")

    assert loads == ["stub-model"]


def test_a_model_with_the_wrong_dimension_is_rejected() -> None:
    embedder = make_embedder(StubModel(dimension=2))

    with pytest.raises(EmbeddingError, match="dimension"):
        embedder.embed_query("a")


def test_the_embedder_reports_its_dimension() -> None:
    assert make_embedder(StubModel()).dimension == 3


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="downloads a model, set RUN_MODEL_TESTS=1 to run",
)
def test_real_model_places_related_text_closer_to_the_query() -> None:
    embedder = FastEmbedEmbedder()
    query = embedder.embed_query("Why did payments fail in March?")
    related, unrelated = embedder.embed_documents(
        [
            "Payment processing was delayed after a bad deploy in March.",
            "Banana bread needs ripe bananas.",
        ]
    )

    similarity_to_related = sum(a * b for a, b in zip(query, related, strict=True))
    similarity_to_unrelated = sum(a * b for a, b in zip(query, unrelated, strict=True))

    assert len(query) == embedder.dimension
    assert similarity_to_related > similarity_to_unrelated
