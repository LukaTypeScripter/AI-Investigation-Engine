import argparse
import sys
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

from investigation_engine.domain import Document
from investigation_engine.embeddings import (
    CachedEmbedder,
    Embedder,
    FastEmbedEmbedder,
    SqliteEmbeddingCache,
)
from investigation_engine.evaluation.hotpot import build_documents, load_examples
from investigation_engine.evaluation.questions import read_questions
from investigation_engine.evaluation.retrieval_eval import limit_documents
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.parsers import parse_path
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.retrieval.indexing import index_missing_vectors
from investigation_engine.retrieval.postgres_vector import PostgresVectorIndex
from investigation_engine.settings import Settings
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import PostgresDocumentStore

DEFAULT_SEED = 42
DEFAULT_CACHE_PATH = Path("data/embeddings.sqlite")
DEFAULT_INGEST_BATCH = 500
MISSING_DATABASE_MESSAGE = "DATABASE_URL is not set: point it at a Postgres database with pgvector"
MISSING_SOURCE_MESSAGE = "nothing to load: pass --samples DIRECTORY and/or --hotpot FILE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="investigation-seed")
    parser.add_argument("--samples", type=Path, help="directory of files to parse and load")
    parser.add_argument("--hotpot", type=Path, help="downloaded HotpotQA validation file")
    parser.add_argument("--questions", type=Path, nargs="*", default=[])
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--batch", type=int, default=DEFAULT_INGEST_BATCH)
    return parser


def load_documents(arguments: argparse.Namespace) -> list[Document]:
    documents: list[Document] = []
    if arguments.hotpot:
        documents = build_documents(load_examples(arguments.hotpot))
        if arguments.max_documents:
            required = {
                document_id
                for path in arguments.questions
                for question in read_questions(path)
                for document_id in question.gold_document_ids
            }
            documents = limit_documents(
                documents, required, arguments.max_documents, arguments.seed
            )
    if arguments.samples:
        for path in sorted(arguments.samples.iterdir()):
            documents.extend(parse_path(path))
    return documents


def ingest_in_batches(
    service: IngestionService, documents: Sequence[Document], batch_size: int
) -> tuple[int, int]:
    changed = 0
    duplicate_chunks = 0
    for batch in batched(documents, batch_size):
        report = service.ingest(batch)
        changed += len(report.added) + len(report.updated)
        duplicate_chunks += report.duplicate_chunks
        print(f"ingested {changed} new or changed documents", file=sys.stderr)
    return changed, duplicate_chunks


def open_embedder(cache_path: Path) -> CachedEmbedder:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return CachedEmbedder(FastEmbedEmbedder(), SqliteEmbeddingCache(cache_path))


def seed(
    arguments: argparse.Namespace, embedder: Embedder, settings: Settings, database_url: str
) -> int:
    documents = load_documents(arguments)
    database = Database(database_url, settings.embedding_dimension)
    database.ensure_schema()
    store = PostgresDocumentStore(database)
    vectors = PostgresVectorIndex(database)
    service = IngestionService(store, SentenceChunker(max_words=settings.chunk_max_words))
    changed, duplicate_chunks = ingest_in_batches(service, documents, arguments.batch)
    embedded = index_missing_vectors(
        store,
        embedder,
        vectors,
        on_progress=lambda total: print(f"embedded {total} chunks", file=sys.stderr),
    )
    print(
        f"{store.count_documents()} documents and {store.count_chunks()} chunks stored "
        f"({changed} documents changed, {duplicate_chunks} duplicate chunks skipped, "
        f"{embedded} chunks embedded)"
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    chosen_settings = settings or Settings()
    if chosen_settings.database_url is None:
        print(MISSING_DATABASE_MESSAGE, file=sys.stderr)
        return 2
    database_url = chosen_settings.database_url.get_secret_value()
    if not (arguments.samples or arguments.hotpot):
        print(MISSING_SOURCE_MESSAGE, file=sys.stderr)
        return 2
    if embedder is not None:
        return seed(arguments, embedder, chosen_settings, database_url)
    cached = open_embedder(arguments.cache)
    try:
        return seed(arguments, cached, chosen_settings, database_url)
    finally:
        cached.close()


if __name__ == "__main__":
    sys.exit(main())
