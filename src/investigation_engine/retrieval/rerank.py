from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.domain import Chunk
from investigation_engine.retrieval.bm25 import tokenize

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
    ]
)


def informative_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


@dataclass(frozen=True, slots=True)
class RerankWeights:
    fused: float = 1.0
    coverage: float = 0.5
    title: float = 0.5

    def __post_init__(self) -> None:
        if min(self.fused, self.coverage, self.title) < 0:
            raise ValueError("rerank weights must be non-negative")


DEFAULT_WEIGHTS = RerankWeights()


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]: ...


class FeatureReranker:
    def __init__(self, weights: RerankWeights = DEFAULT_WEIGHTS) -> None:
        self._weights = weights

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        query_terms = informative_terms(query)
        best_fused = max(score for _, score in candidates)
        rescored = [
            (chunk, self._score(chunk, score / best_fused if best_fused > 0 else 0.0, query_terms))
            for chunk, score in candidates
        ]
        return sorted(rescored, key=lambda pair: (-pair[1], pair[0].id))

    def _score(self, chunk: Chunk, fused: float, query_terms: set[str]) -> float:
        title_terms = informative_terms(chunk.title)
        chunk_terms = title_terms | informative_terms(chunk.text)
        coverage = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
        title_match = len(title_terms & query_terms) / len(title_terms) if title_terms else 0.0
        weights = self._weights
        return weights.fused * fused + weights.coverage * coverage + weights.title * title_match
