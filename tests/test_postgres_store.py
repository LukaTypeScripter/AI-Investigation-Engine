from collections.abc import Callable

from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import PostgresDocumentStore


def test_chunk_ids_without_a_vector_lists_the_chunks_that_were_never_embedded(
    make_database: Callable[[int], Database],
) -> None:
    database = make_database(2)
    store = PostgresDocumentStore(database)
    IngestionService(store, SentenceChunker(max_words=3)).ingest(
        [Document(id="a", text="One two. Three four.", title="a", source="a.txt")]
    )
    database.run(
        lambda connection: connection.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES ('a:0', '[1,0]')"
        )
    )

    assert store.chunk_ids_without_vector() == ["a:1"]


def test_chunk_ids_without_a_vector_is_empty_for_an_empty_store(
    make_database: Callable[[int], Database],
) -> None:
    assert PostgresDocumentStore(make_database(2)).chunk_ids_without_vector() == []
