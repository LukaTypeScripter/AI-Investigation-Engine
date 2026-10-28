from collections.abc import Callable

from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import DocumentSummary, PostgresDocumentStore


def make_document(document_id: str, text: str, title: str = "", source: str = "") -> Document:
    return Document(
        id=document_id,
        text=text,
        title=title or document_id,
        source=source or f"{document_id}.txt",
        metadata={"format": "txt"},
    )


def make_loaded_store(
    make_database: Callable[[int], Database], *documents: Document
) -> PostgresDocumentStore:
    store = PostgresDocumentStore(make_database(2))
    IngestionService(store, SentenceChunker(max_words=3)).ingest(documents)
    return store


def test_list_documents_starts_with_the_most_recently_added(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(make_database, make_document("a", "One."), make_document("b", "Two."))

    assert [summary.id for summary in store.list_documents()] == ["b", "a"]


def test_list_documents_reports_the_number_of_chunks_each_document_contains(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "One two. Three four."), make_document("b", "Five.")
    )

    assert {summary.id: summary.chunk_count for summary in store.list_documents()} == {
        "a": 2,
        "b": 1,
    }


def test_list_documents_counts_a_shared_chunk_for_every_document_that_contains_it(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "Same text."), make_document("b", "Same text.")
    )

    assert {summary.id: summary.chunk_count for summary in store.list_documents()} == {
        "a": 1,
        "b": 1,
    }


def test_list_documents_describes_each_document_by_title_and_source(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "One.", title="Outage", source="outage.md")
    )

    assert store.list_documents() == [DocumentSummary("a", "Outage", "outage.md", 1)]


def test_list_documents_pages_through_the_results(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, *(make_document(name, f"Text {name}.") for name in ("a", "b", "c"))
    )

    assert [summary.id for summary in store.list_documents(limit=2)] == ["c", "b"]
    assert [summary.id for summary in store.list_documents(limit=2, offset=2)] == ["a"]


def test_list_documents_filters_by_text_in_the_title_or_the_source_ignoring_case(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database,
        make_document("a", "One.", title="Payment Outage"),
        make_document("b", "Two.", source="runbook-payments.md"),
        make_document("c", "Three.", title="Refunds"),
    )

    assert {summary.id for summary in store.list_documents(containing="PAYMENT")} == {"a", "b"}


def test_list_documents_treats_percent_signs_in_the_filter_literally(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database,
        make_document("a", "One.", title="100% uptime"),
        make_document("b", "Two.", title="Refunds"),
    )

    assert [summary.id for summary in store.list_documents(containing="%")] == ["a"]


def test_counts_report_how_many_documents_and_chunks_are_stored(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "One two. Three four."), make_document("b", "Five.")
    )

    assert (store.count_documents(), store.count_chunks()) == (2, 3)


def test_count_documents_can_apply_the_same_filter_as_the_listing(
    make_database: Callable[[int], Database],
) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "One.", title="Outage"), make_document("b", "Two.")
    )

    assert store.count_documents(containing="outage") == 1


def test_iter_chunks_yields_every_chunk_once(make_database: Callable[[int], Database]) -> None:
    store = make_loaded_store(
        make_database, make_document("a", "One two. Three four."), make_document("b", "Five.")
    )

    chunks = list(store.iter_chunks(batch_size=2))

    assert sorted(chunk.id for chunk in chunks) == ["a:0", "a:1", "b:0"]
    assert {chunk.id: chunk.text for chunk in chunks}["a:1"] == "Three four."


def test_chunk_ids_without_a_vector_lists_the_chunks_that_were_never_embedded(
    make_database: Callable[[int], Database],
) -> None:
    database = make_database(2)
    store = PostgresDocumentStore(database)
    IngestionService(store, SentenceChunker(max_words=3)).ingest(
        [make_document("a", "One two. Three four.")]
    )
    database.run(
        lambda connection: connection.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES ('a:0', '[1,0]')"
        )
    )

    assert store.chunk_ids_without_vector() == ["a:1"]
