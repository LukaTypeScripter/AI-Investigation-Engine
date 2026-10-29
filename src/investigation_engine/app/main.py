from fastapi import FastAPI

from investigation_engine.app.api import ApiLimits, create_app
from investigation_engine.app.engine import Engine
from investigation_engine.embeddings import Embedder, FastEmbedEmbedder
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.llm.cache import SqliteResponseCache
from investigation_engine.llm.client import ChatClient
from investigation_engine.llm.factory import (
    ConfigurationError,
    build_chat_client,
    build_http_client,
    configured_providers,
)
from investigation_engine.logging_setup import configure_logging
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.indexing import index_missing_vectors
from investigation_engine.retrieval.postgres_vector import PostgresVectorIndex
from investigation_engine.retrieval.rerank import FeatureReranker
from investigation_engine.settings import Settings
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import PostgresDocumentStore

MISSING_DATABASE_MESSAGE = "DATABASE_URL is not set: point it at a Postgres database with pgvector"


def build_language_model(settings: Settings) -> ChatClient | None:
    if not configured_providers(settings):
        return None
    settings.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
    return build_chat_client(
        settings, build_http_client(settings), SqliteResponseCache(settings.llm_cache_path)
    )


def build_engine(settings: Settings, embedder: Embedder | None = None) -> Engine:
    if settings.database_url is None:
        raise ConfigurationError(MISSING_DATABASE_MESSAGE)
    database = Database(settings.database_url.get_secret_value(), settings.embedding_dimension)
    database.ensure_schema()
    store = PostgresDocumentStore(database)
    vectors = PostgresVectorIndex(database)
    chosen = embedder or FastEmbedEmbedder(settings.embedding_model, settings.embedding_dimension)
    index_missing_vectors(store, chosen, vectors)
    engine = Engine(
        store=store,
        embedder=chosen,
        vectors=vectors,
        keywords=Bm25Index(),
        chunker=SentenceChunker(max_words=settings.chunk_max_words),
        llm=build_language_model(settings),
        reranker=FeatureReranker() if settings.rerank_results else None,
        candidates=settings.retrieval_candidates,
        protect_built_in=settings.protect_built_in_documents,
    )
    engine.load_keyword_index()
    return engine


def create_default_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings.log_level)
    limits = ApiLimits(settings.max_upload_bytes, settings.max_question_chars)
    return create_app(lambda: build_engine(settings), limits)
