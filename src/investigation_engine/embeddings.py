import hashlib
import sqlite3
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class TextEmbeddingModel(Protocol):
    def passage_embed(
        self, texts: Iterable[str], batch_size: int = ...
    ) -> Iterable[np.ndarray]: ...

    def query_embed(self, query: str) -> Iterable[np.ndarray]: ...


def load_fastembed_model(model_name: str) -> TextEmbeddingModel:
    from fastembed import TextEmbedding

    return cast(TextEmbeddingModel, TextEmbedding(model_name=model_name))


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_factory: Callable[[str], TextEmbeddingModel] = load_fastembed_model,
    ) -> None:
        self.dimension = dimension
        self._model_name = model_name
        self._batch_size = batch_size
        self._model_factory = model_factory
        self._model: TextEmbeddingModel | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().passage_embed(texts, batch_size=self._batch_size)
        return [self._checked(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._checked(next(iter(self._load().query_embed(text))))

    def _load(self) -> TextEmbeddingModel:
        if self._model is None:
            self._model = self._model_factory(self._model_name)
        return self._model

    def _checked(self, vector: np.ndarray) -> list[float]:
        values: list[float] = vector.tolist()
        if len(values) != self.dimension:
            raise EmbeddingError(
                f"model {self._model_name} returned {len(values)} dimensions, "
                f"expected {self.dimension}"
            )
        return values


EMBEDDING_KEY_PASSAGE = "passage"
EMBEDDING_KEY_QUERY = "query"


def text_key(kind: str, text: str) -> str:
    return f"{kind}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def to_float32(vector: Sequence[float]) -> list[float]:
    return np.asarray(vector, dtype=np.float32).tolist()


class SqliteEmbeddingCache:
    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._lock:
            for key in keys:
                row = self._connection.execute(
                    "SELECT vector FROM embeddings WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    found[key] = np.frombuffer(row[0], dtype=np.float32).tolist()
        return found

    def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        rows = [
            (key, np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in vectors.items()
        ]
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", rows
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: SqliteEmbeddingCache) -> None:
        self.dimension = inner.dimension
        self._inner = inner
        self._cache = cache

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [text_key(EMBEDDING_KEY_PASSAGE, text) for text in texts]
        known = self._cache.get_many(keys)
        missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in known}
        if missing:
            computed = self._inner.embed_documents(list(missing.values()))
            fresh = {key: to_float32(vector) for key, vector in zip(missing, computed, strict=True)}
            self._cache.put_many(fresh)
            known.update(fresh)
        return [known[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        key = text_key(EMBEDDING_KEY_QUERY, text)
        known = self._cache.get_many([key])
        if key in known:
            return known[key]
        vector = to_float32(self._inner.embed_query(text))
        self._cache.put_many({key: vector})
        return vector

    def close(self) -> None:
        self._cache.close()
