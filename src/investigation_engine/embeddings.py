from collections.abc import Callable, Iterable, Sequence
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
