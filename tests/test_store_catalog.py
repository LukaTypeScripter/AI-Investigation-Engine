from collections.abc import Callable
from typing import Protocol

import pytest

from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import DocumentStore, IngestionService
from investigation_engine.storage.catalog import DocumentCatalog, DocumentSummary
from investigation_engine.storage.database import Database
from investigation_engine.storage.memory import InMemoryDocumentStore
from investigation_engine.storage.postgres import PostgresDocumentStore


class CatalogStore(DocumentStore, DocumentCatalog, Protocol):
    pass


@pytest.fixture(params=["memory", "postgres"])
def new_store(request: pytest.FixtureRequest) -> Callable[..., CatalogStore]:
    def build(*documents: Document) -> CatalogStore:
        if request.param == "memory":
            store: CatalogStore = InMemoryDocumentStore()
        else:
            make_database: Callable[[int], Database] = request.getfixturevalue("make_database")
            store = PostgresDocumentStore(make_database(2))
        IngestionService(store, SentenceChunker(max_words=3)).ingest(documents)
        return store

    return build


def make_document(document_id: str, text: str, title: str = "", source: str = "") -> Document:
    return Document(
        id=document_id,
        text=text,
        title=title or document_id,
        source=source or f"{document_id}.txt",
        metadata={"format": "txt"},
    )


def test_list_documents_starts_with_the_most_recently_added(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "One."), make_document("b", "Two."))

    assert [summary.id for summary in store.list_documents()] == ["b", "a"]


def test_list_documents_reports_the_number_of_chunks_each_document_contains(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "One two. Three four."), make_document("b", "Five."))

    assert {summary.id: summary.chunk_count for summary in store.list_documents()} == {
        "a": 2,
        "b": 1,
    }


def test_list_documents_counts_a_shared_chunk_for_every_document_that_contains_it(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "Same text."), make_document("b", "Same text."))

    assert {summary.id: summary.chunk_count for summary in store.list_documents()} == {
        "a": 1,
        "b": 1,
    }


def test_list_documents_describes_each_document_by_title_and_source(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "One.", title="Outage", source="outage.md"))

    assert store.list_documents() == [DocumentSummary("a", "Outage", "outage.md", 1)]


def test_list_documents_pages_through_the_results(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(*(make_document(name, f"Text {name}.") for name in ("a", "b", "c")))

    assert [summary.id for summary in store.list_documents(limit=2)] == ["c", "b"]
    assert [summary.id for summary in store.list_documents(limit=2, offset=2)] == ["a"]


def test_list_documents_filters_by_text_in_the_title_or_the_source_ignoring_case(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(
        make_document("a", "One.", title="Payment Outage"),
        make_document("b", "Two.", source="runbook-payments.md"),
        make_document("c", "Three.", title="Refunds"),
    )

    assert {summary.id for summary in store.list_documents(containing="PAYMENT")} == {"a", "b"}


def test_list_documents_treats_percent_signs_in_the_filter_literally(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(
        make_document("a", "One.", title="100% uptime"),
        make_document("b", "Two.", title="Refunds"),
    )

    assert [summary.id for summary in store.list_documents(containing="%")] == ["a"]


def test_counts_report_how_many_documents_and_chunks_are_stored(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "One two. Three four."), make_document("b", "Five."))

    assert (store.count_documents(), store.count_chunks()) == (2, 3)


def test_count_documents_can_apply_the_same_filter_as_the_listing(
    new_store: Callable[..., CatalogStore],
) -> None:
    store = new_store(make_document("a", "One.", title="Outage"), make_document("b", "Two."))

    assert store.count_documents(containing="outage") == 1


def test_iter_chunks_yields_every_chunk_once(new_store: Callable[..., CatalogStore]) -> None:
    store = new_store(make_document("a", "One two. Three four."), make_document("b", "Five."))

    chunks = list(store.iter_chunks(batch_size=2))

    assert sorted(chunk.id for chunk in chunks) == ["a:0", "a:1", "b:0"]
    assert {chunk.id: chunk.text for chunk in chunks}["a:1"] == "Three four."


def test_origins_of_reports_where_each_stored_document_came_from(
    new_store: Callable[..., CatalogStore],
) -> None:
    uploaded = Document(
        id="u", text="One.", title="u", source="u.txt", metadata={"origin": "upload"}
    )
    store = new_store(uploaded, make_document("b", "Two."))

    assert store.origins_of(["u", "b", "ghost"]) == {"u": "upload", "b": ""}


def test_origins_of_nothing_is_empty(new_store: Callable[..., CatalogStore]) -> None:
    assert new_store().origins_of([]) == {}
