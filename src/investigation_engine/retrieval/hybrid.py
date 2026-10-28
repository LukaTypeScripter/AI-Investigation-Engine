from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.domain import Chunk
from investigation_engine.embeddings import Embedder
from investigation_engine.retrieval.fusion import DEFAULT_RANK_CONSTANT, reciprocal_rank_fusion
from investigation_engine.retrieval.protocols import KeywordSearch, VectorSearch
from investigation_engine.retrieval.rerank import Reranker

DEFAULT_CANDIDATES = 50


class ChunkLookup(Protocol):
    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None
    keyword_rank: int | None


def rank_positions(ids: Sequence[str]) -> dict[str, int]:
    return {item_id: position for position, item_id in enumerate(ids, start=1)}


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vectors: VectorSearch,
        keywords: KeywordSearch,
        chunks: ChunkLookup,
        reranker: Reranker | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._keywords = keywords
        self._chunks = chunks
        self._reranker = reranker
        self._candidates = candidates
        self._rank_constant = rank_constant

    def retrieve(self, query: str, limit: int) -> list[Hit]:
        if not query.strip():
            return []
        vector_ids = self._ids(
            self._vectors.search(self._embedder.embed_query(query), self._candidates)
        )
        keyword_ids = self._ids(self._keywords.search(query, self._candidates))
        fused = reciprocal_rank_fusion([vector_ids, keyword_ids], self._rank_constant)
        pool = fused[: self._candidates]
        chunks = self._chunks.get_chunks([chunk_id for chunk_id, _ in pool])
        scored = [(chunks[chunk_id], score) for chunk_id, score in pool if chunk_id in chunks]
        ordered = self._reranker.rerank(query, scored) if self._reranker else scored
        vector_ranks = rank_positions(vector_ids)
        keyword_ranks = rank_positions(keyword_ids)
        return [
            Hit(chunk, score, vector_ranks.get(chunk.id), keyword_ranks.get(chunk.id))
            for chunk, score in ordered[:limit]
        ]

    @staticmethod
    def _ids(results: Sequence[tuple[str, float]]) -> list[str]:
        return [item_id for item_id, _ in results]
