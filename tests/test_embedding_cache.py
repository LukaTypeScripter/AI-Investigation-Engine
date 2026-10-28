from pathlib import Path

import pytest

from fakes import HashingEmbedder
from investigation_engine.embeddings import CachedEmbedder, SqliteEmbeddingCache


def make_cached(path: Path | str = ":memory:") -> tuple[CachedEmbedder, HashingEmbedder]:
    inner = HashingEmbedder(dimension=8)
    return CachedEmbedder(inner, SqliteEmbeddingCache(path)), inner


def test_documents_are_embedded_once_and_then_served_from_the_cache() -> None:
    cached, inner = make_cached()

    first = cached.embed_documents(["alpha beta", "gamma"])
    second = cached.embed_documents(["alpha beta", "gamma"])

    assert second == first
    assert len(inner.document_batches) == 1


def test_only_missing_texts_are_sent_to_the_wrapped_embedder_and_order_is_kept() -> None:
    cached, inner = make_cached()
    cached.embed_documents(["alpha"])

    vectors = cached.embed_documents(["gamma", "alpha", "delta"])

    assert inner.document_batches[-1] == ["gamma", "delta"]
    assert vectors == [
        inner.embed_query("gamma"),
        inner.embed_query("alpha"),
        inner.embed_query("delta"),
    ]


def test_repeated_texts_inside_one_batch_are_embedded_once() -> None:
    cached, inner = make_cached()

    vectors = cached.embed_documents(["same", "same", "other"])

    assert inner.document_batches == [["same", "other"]]
    assert vectors[0] == vectors[1]


def test_cached_vectors_match_the_originals_to_float32_precision() -> None:
    cached, inner = make_cached()
    cached.embed_documents(["alpha beta"])

    assert cached.embed_documents(["alpha beta"])[0] == pytest.approx(
        inner.embed_query("alpha beta")
    )


def test_queries_are_cached_separately_from_documents() -> None:
    inner = HashingEmbedder(dimension=8)
    cached = CachedEmbedder(inner, SqliteEmbeddingCache(":memory:"))

    cached.embed_documents(["alpha"])
    cached.embed_query("alpha")
    cached.embed_query("alpha")

    assert len(inner.document_batches) == 1
    assert inner.queries == ["alpha"]


def test_the_cache_survives_being_reopened(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    first, _ = make_cached(path)
    first.embed_documents(["alpha"])
    first.close()

    second, inner = make_cached(path)
    second.embed_documents(["alpha"])

    assert inner.document_batches == []


def test_embedding_nothing_returns_nothing() -> None:
    cached, inner = make_cached()

    assert cached.embed_documents([]) == []
    assert inner.document_batches == []


def test_the_dimension_is_that_of_the_wrapped_embedder() -> None:
    assert make_cached()[0].dimension == 8
