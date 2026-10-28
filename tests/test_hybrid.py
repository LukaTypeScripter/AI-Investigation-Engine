from collections.abc import Sequence

import pytest

from fakes import HashingEmbedder
from investigation_engine.domain import Chunk, Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.hybrid import HybridRetriever
from investigation_engine.retrieval.indexing import SearchIndexer
from investigation_engine.retrieval.rerank import FeatureReranker
from investigation_engine.retrieval.vector import InMemoryVectorIndex
from investigation_engine.storage.memory import InMemoryDocumentStore


def scripted_results(ranking: list[str], limit: int) -> list[tuple[str, float]]:
    return [(chunk_id, 1.0 - 0.1 * position) for position, chunk_id in enumerate(ranking[:limit])]


class StubVectorSearch:
    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking
        self.limits: list[int] = []
        self.vectors: list[list[float]] = []

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[str, float]]:
        self.limits.append(limit)
        self.vectors.append(list(vector))
        return scripted_results(self._ranking, limit)


class StubKeywordSearch:
    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking
        self.limits: list[int] = []
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.limits.append(limit)
        self.queries.append(query)
        return scripted_results(self._ranking, limit)


class StubChunks:
    def __init__(self, *chunk_ids: str) -> None:
        self._chunks = {chunk_id: make_chunk(chunk_id) for chunk_id in chunk_ids}

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        return {
            chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks
        }


class ReversingReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        self.queries.append(query)
        return [(chunk, 42.0) for chunk, _ in reversed(candidates)]


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id=chunk_id, index=0, text=chunk_id, title="", source="")


def make_retriever(
    vector_ranking: list[str],
    keyword_ranking: list[str],
    known: tuple[str, ...] = ("a", "b", "c", "d"),
    reranker: ReversingReranker | None = None,
    candidates: int = 50,
) -> tuple[HybridRetriever, StubVectorSearch, StubKeywordSearch]:
    vectors, keywords = StubVectorSearch(vector_ranking), StubKeywordSearch(keyword_ranking)
    retriever = HybridRetriever(
        embedder=HashingEmbedder(),
        vectors=vectors,
        keywords=keywords,
        chunks=StubChunks(*known),
        reranker=reranker,
        candidates=candidates,
    )
    return retriever, vectors, keywords


def hit_ids(retriever: HybridRetriever, query: str = "query", limit: int = 10) -> list[str]:
    return [hit.chunk.id for hit in retriever.retrieve(query, limit)]


def test_a_chunk_found_by_both_searches_beats_the_leader_of_one() -> None:
    retriever, _, _ = make_retriever(["a", "b"], ["c", "b"])

    assert hit_ids(retriever) == ["b", "a", "c"]


def test_hits_report_where_each_search_ranked_them() -> None:
    retriever, _, _ = make_retriever(["a", "b"], ["c", "b"])

    hits = {hit.chunk.id: hit for hit in retriever.retrieve("query", 10)}

    assert (hits["b"].vector_rank, hits["b"].keyword_rank) == (2, 2)
    assert (hits["a"].vector_rank, hits["a"].keyword_rank) == (1, None)
    assert (hits["c"].vector_rank, hits["c"].keyword_rank) == (None, 1)


def test_without_a_reranker_the_score_is_the_fused_rank_score() -> None:
    retriever, _, _ = make_retriever(["a", "b"], ["c", "b"])

    best = retriever.retrieve("query", 1)[0]

    assert best.score == pytest.approx(2 / 62)


def test_the_limit_caps_the_number_of_hits() -> None:
    retriever, _, _ = make_retriever(["a", "b"], ["c", "d"])

    assert len(hit_ids(retriever, limit=3)) == 3


def test_each_search_is_asked_for_the_candidate_pool_size() -> None:
    retriever, vectors, keywords = make_retriever(["a"], ["b"], candidates=7)

    retriever.retrieve("query", 1)

    assert (vectors.limits, keywords.limits) == ([7], [7])
    assert keywords.queries == ["query"]
    assert vectors.vectors == [HashingEmbedder().embed_query("query")]


def test_ids_the_store_no_longer_knows_are_skipped() -> None:
    retriever, _, _ = make_retriever(["ghost", "a"], ["b"], known=("a", "b"))

    assert hit_ids(retriever) == ["b", "a"]


def test_a_blank_query_returns_nothing_without_searching() -> None:
    retriever, vectors, keywords = make_retriever(["a"], ["b"])

    assert retriever.retrieve("   ", 5) == []
    assert (vectors.limits, keywords.limits) == ([], [])


def test_the_reranker_decides_the_final_order_and_scores() -> None:
    reranker = ReversingReranker()
    retriever, _, _ = make_retriever(["a", "b"], ["c", "b"], reranker=reranker)

    hits = retriever.retrieve("query", 10)

    assert [hit.chunk.id for hit in hits] == ["c", "a", "b"]
    assert {hit.score for hit in hits} == {42.0}
    assert reranker.queries == ["query"]


def build_real_retriever() -> HybridRetriever:
    embedder = HashingEmbedder()
    store = InMemoryDocumentStore()
    keywords = Bm25Index()
    vectors = InMemoryVectorIndex(dimension=embedder.dimension)
    service = IngestionService(store, SentenceChunker(max_words=30))
    texts = {
        "payments": "Payment processing was delayed after a bad deploy in March.",
        "bread": "Banana bread needs ripe bananas.",
        "disk": "The database ran out of disk space on db-2.",
    }
    documents = [
        Document(id=name, text=text, title=name, source=f"{name}.txt")
        for name, text in texts.items()
    ]
    SearchIndexer(embedder, vectors, keywords).apply(service.ingest(documents).changes)
    return HybridRetriever(embedder, vectors, keywords, store, FeatureReranker())


def test_end_to_end_the_relevant_document_comes_first() -> None:
    retriever = build_real_retriever()

    hits = retriever.retrieve("why was payment processing delayed in March", 3)

    assert hits[0].chunk.document_id == "payments"
    assert hits[0].vector_rank == 1
    assert hits[0].keyword_rank == 1
