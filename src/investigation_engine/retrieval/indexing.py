from collections.abc import Iterable, Sequence
from itertools import batched
from typing import Protocol

from investigation_engine.domain import Chunk
from investigation_engine.embeddings import Embedder
from investigation_engine.ingestion.service import IndexChanges
from investigation_engine.retrieval.protocols import KeywordIndex, VectorIndex

DEFAULT_EMBEDDING_BATCH = 64


class VectorGaps(Protocol):
    def chunk_ids_without_vector(self) -> list[str]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...


def passage_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


class SearchIndexer:
    def __init__(
        self,
        embedder: Embedder,
        vectors: VectorIndex,
        keywords: KeywordIndex,
        batch_size: int = DEFAULT_EMBEDDING_BATCH,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._keywords = keywords
        self._batch_size = batch_size

    def apply(self, changes: IndexChanges) -> None:
        for chunk_id in changes.chunks_removed:
            self._keywords.remove(chunk_id)
            self._vectors.remove(chunk_id)
        self.index(changes.chunks_added)

    def index(self, chunks: Iterable[Chunk]) -> None:
        for batch in batched(chunks, self._batch_size):
            texts = [passage_text(chunk) for chunk in batch]
            vectors = self._embedder.embed_documents(texts)
            self._vectors.add_many(
                [(chunk.id, vector) for chunk, vector in zip(batch, vectors, strict=True)]
            )
            for chunk, text in zip(batch, texts, strict=True):
                self._keywords.add(chunk.id, text)


def index_missing_vectors(
    source: VectorGaps,
    embedder: Embedder,
    vectors: VectorIndex,
    batch_size: int = DEFAULT_EMBEDDING_BATCH,
) -> int:
    indexed = 0
    for chunk_ids in batched(source.chunk_ids_without_vector(), batch_size):
        chunks = list(source.get_chunks(chunk_ids).values())
        if not chunks:
            continue
        embeddings = embedder.embed_documents([passage_text(chunk) for chunk in chunks])
        vectors.add_many(
            [(chunk.id, vector) for chunk, vector in zip(chunks, embeddings, strict=True)]
        )
        indexed += len(chunks)
    return indexed
