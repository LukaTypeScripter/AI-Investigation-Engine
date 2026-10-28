import threading
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from investigation_engine.agent.evidence import SourceLookup
from investigation_engine.agent.investigator import Investigator, InvestigatorConfig
from investigation_engine.agent.models import Investigation
from investigation_engine.domain import SourceRef
from investigation_engine.embeddings import Embedder
from investigation_engine.ingestion.chunking import Chunker
from investigation_engine.ingestion.parsers import parse_bytes
from investigation_engine.ingestion.service import DocumentStore, IngestionService
from investigation_engine.llm.client import ChatClient
from investigation_engine.llm.factory import ConfigurationError
from investigation_engine.retrieval.hybrid import (
    DEFAULT_CANDIDATES,
    ChunkLookup,
    Hit,
    HybridRetriever,
)
from investigation_engine.retrieval.indexing import SearchIndexer, passage_text
from investigation_engine.retrieval.locked import LockedKeywordIndex
from investigation_engine.retrieval.protocols import KeywordIndex, VectorIndex
from investigation_engine.retrieval.rerank import Reranker
from investigation_engine.storage.catalog import DEFAULT_PAGE_SIZE, DocumentCatalog, DocumentSummary

NO_LANGUAGE_MODEL_MESSAGE = (
    "no language model is configured: set GEMINI_API_KEY or GROQ_API_KEY to enable answers"
)


class NothingToIndexError(ValueError):
    pass


class EngineStore(DocumentStore, DocumentCatalog, ChunkLookup, SourceLookup, Protocol):
    pass


class EngineStats(NamedTuple):
    documents: int
    chunks: int


@dataclass(frozen=True, slots=True)
class UploadResult:
    filename: str
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    duplicate_chunks: int
    chunks_added: int
    chunks_removed: int


@dataclass(frozen=True, slots=True)
class DocumentPage:
    items: list[DocumentSummary]
    total: int


class Engine:
    def __init__(
        self,
        store: EngineStore,
        embedder: Embedder,
        vectors: VectorIndex,
        keywords: KeywordIndex,
        chunker: Chunker,
        llm: ChatClient | None = None,
        reranker: Reranker | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        investigator_config: InvestigatorConfig | None = None,
    ) -> None:
        self._store = store
        self._keywords = LockedKeywordIndex(keywords)
        self._ingestion = IngestionService(store, chunker)
        self._indexer = SearchIndexer(embedder, vectors, self._keywords)
        self._retriever = HybridRetriever(
            embedder, vectors, self._keywords, store, reranker, candidates
        )
        self._investigator = (
            Investigator(llm, self._retriever, store, investigator_config) if llm else None
        )
        self._write_lock = threading.Lock()

    @property
    def can_answer(self) -> bool:
        return self._investigator is not None

    def load_keyword_index(self) -> int:
        loaded = 0
        for chunk in self._store.iter_chunks():
            self._keywords.add(chunk.id, passage_text(chunk))
            loaded += 1
        return loaded

    def upload(self, filename: str, content: bytes) -> UploadResult:
        documents = parse_bytes(filename, content)
        if not documents:
            raise NothingToIndexError(f"{filename} contains no text to index")
        with self._write_lock:
            report = self._ingestion.ingest(documents)
            self._indexer.apply(report.changes)
        return UploadResult(
            filename=filename,
            added=report.added,
            updated=report.updated,
            unchanged=report.unchanged,
            duplicate_chunks=report.duplicate_chunks,
            chunks_added=len(report.changes.chunks_added),
            chunks_removed=len(report.changes.chunks_removed),
        )

    def delete_document(self, document_id: str) -> bool:
        with self._write_lock:
            changes = self._ingestion.delete(document_id)
            if changes is None:
                return False
            self._indexer.apply(changes)
            return True

    def documents(
        self, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, containing: str = ""
    ) -> DocumentPage:
        return DocumentPage(
            items=self._store.list_documents(limit, offset, containing),
            total=self._store.count_documents(containing),
        )

    def stats(self) -> EngineStats:
        return EngineStats(self._store.count_documents(), self._store.count_chunks())

    def search(self, query: str, limit: int) -> list[Hit]:
        return self._retriever.retrieve(query, limit)

    def sources_of(self, chunk_id: str) -> list[SourceRef]:
        return self._store.sources_of(chunk_id)

    def ask(self, question: str) -> Investigation:
        if self._investigator is None:
            raise ConfigurationError(NO_LANGUAGE_MODEL_MESSAGE)
        return self._investigator.investigate(question)
