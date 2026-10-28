from collections.abc import Sequence
from typing import Protocol

import pytest

from investigation_engine.domain import Chunk, Document, SourceRef
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.parsers import parse_bytes
from investigation_engine.ingestion.service import DocumentStore, IngestionService
from investigation_engine.storage.memory import InMemoryDocumentStore
from investigation_engine.storage.postgres import PostgresDocumentStore


class StoreUnderTest(DocumentStore, Protocol):
    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

    def sources_of_many(self, chunk_ids: Sequence[str]) -> dict[str, list[SourceRef]]: ...

    def document_ids(self) -> set[str]: ...

    def chunk_ids(self) -> set[str]: ...

    def chunk(self, chunk_id: str) -> Chunk: ...

    def owners_of(self, chunk_id: str) -> dict[str, int]: ...


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> StoreUnderTest:
    if request.param == "memory":
        return InMemoryDocumentStore()
    make_database = request.getfixturevalue("make_database")
    return PostgresDocumentStore(make_database(2))


def make_service(store: StoreUnderTest, max_words: int = 3) -> IngestionService:
    return IngestionService(store, SentenceChunker(max_words=max_words))


def make_document(document_id: str, text: str, title: str = "") -> Document:
    return Document(
        id=document_id, text=text, title=title or document_id, source=f"{document_id}.txt"
    )


def test_ingest_without_documents_reports_nothing(store: StoreUnderTest) -> None:
    service = make_service(store)

    report = service.ingest([])

    assert (report.added, report.updated, report.unchanged) == ([], [], [])
    assert report.changes.chunks_added == []
    assert store.chunk_ids() == set()


def test_ingest_new_document_is_reported_as_added_and_its_chunks_are_stored(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)

    report = service.ingest([make_document("a", "A one. B two.")])

    assert report.added == ["a"]
    assert (report.updated, report.unchanged) == ([], [])
    assert [chunk.id for chunk in report.changes.chunks_added] == ["a:0", "a:1"]
    assert store.chunk_ids() == {"a:0", "a:1"}


def test_ingest_identical_document_again_is_reported_as_unchanged(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one.")])

    report = service.ingest([make_document("a", "A one.")])

    assert report.unchanged == ["a"]
    assert report.added == []
    assert report.changes.chunks_added == []
    assert report.changes.chunks_removed == []


def test_ingest_changed_document_with_fewer_chunks_removes_the_extra_ones(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one. B two.")])

    report = service.ingest([make_document("a", "A one.")])

    assert report.updated == ["a"]
    assert [chunk.id for chunk in report.changes.chunks_added] == ["a:0"]
    assert report.changes.chunks_removed == ["a:1"]
    assert store.chunk_ids() == {"a:0"}


def test_ingest_changed_text_replaces_the_stored_chunk_text(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one.")])

    service.ingest([make_document("a", "Z nine.")])

    assert store.chunk("a:0").text == "Z nine."


def test_ingest_changed_title_alone_counts_as_an_update(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one.", title="Old")])

    report = service.ingest([make_document("a", "A one.", title="New")])

    assert report.updated == ["a"]
    assert store.chunk("a:0").title == "New"


def test_ingest_same_text_in_two_documents_stores_one_chunk_with_two_owners(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)

    report = service.ingest([make_document("a", "Same text."), make_document("b", "same   TEXT.")])

    assert store.chunk_ids() == {"a:0"}
    assert store.owners_of("a:0") == {"a": 0, "b": 0}
    assert report.duplicate_chunks == 1
    assert [chunk.id for chunk in report.changes.chunks_added] == ["a:0"]


def test_ingest_text_already_stored_by_an_earlier_run_is_not_stored_again(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text.")])

    report = service.ingest([make_document("b", "Same text.")])

    assert report.duplicate_chunks == 1
    assert report.changes.chunks_added == []
    assert store.owners_of("a:0") == {"a": 0, "b": 0}


def test_ingest_repeated_text_inside_one_document_is_stored_once(store: StoreUnderTest) -> None:
    service = make_service(store, max_words=1)

    report = service.ingest([make_document("a", "Same. Same.")])

    assert store.chunk_ids() == {"a:0"}
    assert store.owners_of("a:0") == {"a": 0}
    assert report.duplicate_chunks == 1


def test_ingest_same_document_twice_in_one_batch_keeps_only_the_latest_version(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)

    report = service.ingest([make_document("a", "A one."), make_document("a", "Z nine.")])

    assert report.added == ["a"]
    assert report.updated == ["a"]
    assert [chunk.text for chunk in report.changes.chunks_added] == ["Z nine."]
    assert report.changes.chunks_removed == []
    assert store.chunk("a:0").text == "Z nine."


def test_ingest_parsed_files_keeps_their_titles_on_the_chunks(store: StoreUnderTest) -> None:
    service = make_service(store)
    documents = parse_bytes("notes.md", b"# Outage\n\nPayments failed. Refunds late.")

    report = service.ingest(documents)

    assert report.added == ["notes.md"]
    assert {chunk.title for chunk in report.changes.chunks_added} == {"Outage"}


def test_delete_document_that_only_shares_a_chunk_leaves_the_chunk_in_place(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text."), make_document("b", "Same text.")])

    changes = service.delete("b")

    assert changes is not None
    assert changes.chunks_removed == []
    assert changes.chunks_added == []
    assert store.chunk_ids() == {"a:0"}
    assert store.owners_of("a:0") == {"a": 0}


def test_delete_the_document_a_shared_chunk_belongs_to_moves_it_to_the_next_owner(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text."), make_document("b", "Same text.")])

    changes = service.delete("a")

    assert changes is not None
    assert changes.chunks_removed == ["a:0"]
    assert [chunk.id for chunk in changes.chunks_added] == ["b:0"]
    moved = store.chunk("b:0")
    assert (moved.document_id, moved.title, moved.source) == ("b", "b", "b.txt")
    assert store.chunk_ids() == {"b:0"}
    assert store.owners_of("b:0") == {"b": 0}


def test_delete_the_last_owner_removes_the_chunk(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one. B two.")])

    changes = service.delete("a")

    assert changes is not None
    assert changes.chunks_removed == ["a:0", "a:1"]
    assert changes.chunks_added == []
    assert store.chunk_ids() == set()
    assert store.document_ids() == set()


def test_delete_unknown_document_returns_nothing(store: StoreUnderTest) -> None:
    service = make_service(store)

    assert service.delete("missing") is None


def test_readding_a_deleted_document_with_new_text_does_not_clash_with_the_moved_chunk(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text."), make_document("b", "Same text.")])
    service.delete("a")

    service.ingest([make_document("a", "Other text.")])

    assert store.chunk_ids() == {"a:0", "b:0"}
    assert store.chunk("a:0").text == "Other text."
    assert store.chunk("b:0").text == "Same text."


def test_readding_a_deleted_document_with_the_shared_text_becomes_an_owner_again(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text."), make_document("b", "Same text.")])
    service.delete("a")

    service.ingest([make_document("a", "Same text.")])

    assert store.chunk_ids() == {"b:0"}
    assert store.owners_of("b:0") == {"b": 0, "a": 0}


def test_updating_a_document_that_shares_a_chunk_keeps_the_other_owner(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "Same text."), make_document("b", "Same text.")])

    service.ingest([make_document("a", "Different words.")])

    assert store.chunk("b:0").text == "Same text."
    assert store.owners_of("b:0") == {"b": 0}
    assert store.chunk("a:0").text == "Different words."


def test_store_rejects_saving_the_same_document_twice(store: StoreUnderTest) -> None:
    document = make_document("a", "A one.")
    store.save(document, "hash", [], {})

    with pytest.raises(ValueError, match="already stored"):
        store.save(document, "hash", [], {})


def test_sources_of_many_lists_every_document_that_contains_each_chunk(
    store: StoreUnderTest,
) -> None:
    service = make_service(store)
    service.ingest(
        [
            make_document("a", "Same text. Own words."),
            make_document("b", "Same text."),
        ]
    )

    assert store.sources_of_many(["a:0", "a:1"]) == {
        "a:0": [SourceRef("a", "a", "a.txt"), SourceRef("b", "b", "b.txt")],
        "a:1": [SourceRef("a", "a", "a.txt")],
    }


def test_sources_of_many_leaves_out_chunks_it_does_not_know(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one.")])

    assert store.sources_of_many(["ghost:0", "a:0"]) == {"a:0": [SourceRef("a", "a", "a.txt")]}
    assert store.sources_of_many([]) == {}


def test_get_chunks_returns_only_the_ids_the_store_knows(store: StoreUnderTest) -> None:
    service = make_service(store)
    service.ingest([make_document("a", "A one. B two.")])

    found = store.get_chunks(["a:1", "ghost", "a:0"])

    assert list(found) == ["a:1", "a:0"]
    assert found["a:1"].text == "B two."
