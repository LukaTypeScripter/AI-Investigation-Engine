import hashlib
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass

from investigation_engine.agent.evidence import SourceLookup
from investigation_engine.domain import Document
from investigation_engine.embeddings import Embedder
from investigation_engine.evaluation.metrics import (
    full_recall_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from investigation_engine.evaluation.questions import EvalQuestion
from investigation_engine.ingestion.chunking import Chunker
from investigation_engine.ingestion.service import IngestionService, IngestReport
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.hybrid import Hit, HybridRetriever
from investigation_engine.retrieval.indexing import SearchIndexer
from investigation_engine.retrieval.rerank import FeatureReranker
from investigation_engine.retrieval.vector import InMemoryVectorIndex
from investigation_engine.storage.memory import InMemoryDocumentStore

DEFAULT_KS = (1, 5, 10)
CHUNKS_PER_DOCUMENT = 3
ALL_GROUP = "all"
EMBEDDING_BATCH = 64
MILLISECONDS = 1000.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    name: str
    use_vectors: bool = True
    use_keywords: bool = True
    rerank: bool = False
    candidates: int = 50

    def __post_init__(self) -> None:
        if not (self.use_vectors or self.use_keywords):
            raise ValueError("at least one of vector search and keyword search must be enabled")


DEFAULT_CONFIGS = (
    RetrievalConfig("bm25_only", use_vectors=False),
    RetrievalConfig("vector_only", use_keywords=False),
    RetrievalConfig("hybrid_rrf"),
    RetrievalConfig("hybrid_rerank", rerank=True),
)


@dataclass(frozen=True, slots=True)
class SearchStack:
    store: InMemoryDocumentStore
    keywords: Bm25Index
    vectors: InMemoryVectorIndex
    embedder: Embedder


@dataclass(frozen=True, slots=True)
class QuestionRetrieval:
    question_id: str
    category: str
    ranked_documents: tuple[str, ...]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    config: str
    chunker: str
    documents: int
    chunks: int
    duplicate_chunks: int
    mean_latency_ms: float
    metrics: dict[str, dict[str, float]]
    questions: tuple[QuestionRetrieval, ...]


def build_stack(
    documents: Iterable[Document],
    chunker: Chunker,
    embedder: Embedder,
    batch_size: int = EMBEDDING_BATCH,
) -> tuple[SearchStack, IngestReport]:
    store = InMemoryDocumentStore()
    keywords = Bm25Index()
    vectors = InMemoryVectorIndex(dimension=embedder.dimension)
    report = IngestionService(store, chunker).ingest(documents)
    SearchIndexer(embedder, vectors, keywords, batch_size).apply(report.changes)
    return SearchStack(store, keywords, vectors, embedder), report


def make_retriever(stack: SearchStack, config: RetrievalConfig) -> HybridRetriever:
    return HybridRetriever(
        embedder=stack.embedder,
        vectors=stack.vectors if config.use_vectors else None,
        keywords=stack.keywords if config.use_keywords else None,
        chunks=stack.store,
        reranker=FeatureReranker() if config.rerank else None,
        candidates=config.candidates,
    )


def document_ranking(hits: Sequence[Hit], sources: SourceLookup) -> list[str]:
    by_chunk = sources.sources_of_many([hit.chunk.id for hit in hits])
    ordered = (ref.document_id for hit in hits for ref in by_chunk.get(hit.chunk.id, []))
    return list(dict.fromkeys(ordered))


def metric_names(ks: Sequence[int]) -> list[str]:
    names = [name for k in ks for name in (f"recall@{k}", f"full_recall@{k}")]
    return [*names, "mrr"]


def evaluate_question(
    retriever: HybridRetriever,
    sources: SourceLookup,
    question: EvalQuestion,
    ks: Sequence[int],
) -> QuestionRetrieval:
    hits = retriever.retrieve(question.text, max(ks) * CHUNKS_PER_DOCUMENT)
    ranking = document_ranking(hits, sources)
    gold = question.gold_document_ids
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranking, gold, k)
        metrics[f"full_recall@{k}"] = full_recall_at_k(ranking, gold, k)
    metrics["mrr"] = reciprocal_rank(ranking, gold)
    return QuestionRetrieval(
        question_id=question.id,
        category=question.category,
        ranked_documents=tuple(ranking[: max(ks)]),
        metrics=metrics,
    )


def summarize(results: Sequence[QuestionRetrieval]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QuestionRetrieval]] = {ALL_GROUP: list(results)}
    for result in results:
        groups.setdefault(result.category, []).append(result)
    return {
        group: {
            name: mean(result.metrics[name] for result in members)
            for name in (members[0].metrics if members else {})
        }
        for group, members in groups.items()
    }


def run_configs(
    stack: SearchStack,
    ingest_report: IngestReport,
    questions: Sequence[EvalQuestion],
    configs: Iterable[RetrievalConfig],
    chunker_name: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> list[RetrievalReport]:
    scorable = [question for question in questions if question.gold_document_ids]
    reports: list[RetrievalReport] = []
    for config in configs:
        retriever = make_retriever(stack, config)
        started = time.perf_counter()
        results = tuple(
            evaluate_question(retriever, stack.store, question, ks) for question in scorable
        )
        elapsed = time.perf_counter() - started
        reports.append(
            RetrievalReport(
                config=config.name,
                chunker=chunker_name,
                documents=len(stack.store.document_ids()),
                chunks=len(stack.store.chunk_ids()),
                duplicate_chunks=ingest_report.duplicate_chunks,
                mean_latency_ms=elapsed / len(scorable) * MILLISECONDS if scorable else 0.0,
                metrics=summarize(results),
                questions=results,
            )
        )
    return reports


def limit_documents(
    documents: Iterable[Document], required_ids: Collection[str], limit: int, seed: int
) -> list[Document]:
    required: list[Document] = []
    others: list[Document] = []
    for document in documents:
        (required if document.id in required_ids else others).append(document)

    def sort_key(document: Document) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{document.id}".encode()).hexdigest()
        return digest, document.id

    room = max(limit - len(required), 0)
    chosen = required + sorted(others, key=sort_key)[:room]
    return sorted(chosen, key=lambda document: document.id)


def report_to_dict(report: RetrievalReport) -> dict[str, object]:
    return asdict(report)


def render_table(reports: Sequence[RetrievalReport], group: str, metrics: Sequence[str]) -> str:
    lines = [
        "| config | " + " | ".join(metrics) + " |",
        "| --- | " + " | ".join("---" for _ in metrics) + " |",
    ]
    for report in reports:
        values = report.metrics.get(group, {})
        cells = " | ".join(f"{values.get(metric, 0.0):.3f}" for metric in metrics)
        lines.append(f"| {report.config} | {cells} |")
    return "\n".join(lines)
