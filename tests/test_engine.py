import threading
from collections.abc import Sequence

import pytest

from fakes import (
    FakeClient,
    HashingEmbedder,
    make_engine,
    make_tool_completion,
    search_call,
    submit_call,
)
from investigation_engine.app.engine import NothingToIndexError, ProtectedDocumentError
from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.parsers import UnsupportedFormatError
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.llm.errors import LlmUnavailable
from investigation_engine.llm.factory import ConfigurationError
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.storage.memory import InMemoryDocumentStore

OUTAGE = b"Payments failed on 5 March because of a bad deploy."
REFUNDS = b"Refunds were delayed by two days."
WAIT_SECONDS = 5.0
SHORT_WAIT_SECONDS = 0.3


class GatedEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.first_entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.first_entered.set()
        self.release.wait(WAIT_SECONDS)
        try:
            return super().embed_documents(texts)
        finally:
            with self._guard:
                self.active -= 1


def test_an_uploaded_file_is_reported_as_added_and_becomes_searchable() -> None:
    engine = make_engine()

    result = engine.upload("outage.txt", OUTAGE)

    assert result.filename == "outage.txt"
    assert result.added == ["outage.txt"]
    assert (result.updated, result.unchanged) == ([], [])
    assert result.chunks_added == 1
    hits = engine.search("bad deploy", 3)
    assert [hit.chunk.document_id for hit in hits] == ["outage.txt"]


def test_uploading_the_same_file_again_changes_nothing() -> None:
    engine = make_engine()
    engine.upload("outage.txt", OUTAGE)

    result = engine.upload("outage.txt", OUTAGE)

    assert result.unchanged == ["outage.txt"]
    assert (result.added, result.chunks_added) == ([], 0)
    assert engine.stats() == (1, 1)


def test_uploading_changed_content_replaces_what_was_searchable_before() -> None:
    engine = make_engine()
    engine.upload("notes.txt", b"Alpha report about gateways.")

    result = engine.upload("notes.txt", b"Omega report about refunds.")

    assert result.updated == ["notes.txt"]
    assert (result.chunks_added, result.chunks_removed) == (1, 0)
    texts = [hit.chunk.text for hit in engine.search("gateways", 5)]
    assert "Alpha report about gateways." not in texts


def test_a_file_in_an_unsupported_format_is_rejected() -> None:
    with pytest.raises(UnsupportedFormatError):
        make_engine().upload("photo.png", b"data")


def test_a_file_without_any_text_is_rejected() -> None:
    with pytest.raises(NothingToIndexError, match="empty.txt"):
        make_engine().upload("empty.txt", b"   \n")


def test_deleting_a_document_removes_it_from_search_and_reports_success() -> None:
    engine = make_engine()
    engine.upload("outage.txt", OUTAGE)
    engine.upload("refunds.txt", REFUNDS)

    deleted = engine.delete_document("outage.txt")

    assert deleted is True
    assert {hit.chunk.document_id for hit in engine.search("payments deploy refunds", 5)} == {
        "refunds.txt"
    }
    assert engine.stats() == (1, 1)


def test_deleting_an_unknown_document_reports_that_nothing_was_deleted() -> None:
    assert make_engine().delete_document("ghost.txt") is False


def test_documents_are_listed_newest_first_with_the_total() -> None:
    engine = make_engine()
    engine.upload("outage.txt", OUTAGE)
    engine.upload("refunds.txt", REFUNDS)

    page = engine.documents(limit=1)

    assert [summary.id for summary in page.items] == ["refunds.txt"]
    assert page.total == 2


def test_the_document_list_can_be_filtered_by_name() -> None:
    engine = make_engine()
    engine.upload("outage.txt", OUTAGE)
    engine.upload("refunds.txt", REFUNDS)

    page = engine.documents(containing="outage")

    assert [summary.id for summary in page.items] == ["outage.txt"]
    assert page.total == 1


def test_asking_without_a_language_model_explains_what_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="language model"):
        make_engine().ask("What failed?")


def test_asking_returns_the_cited_answer_the_investigator_produced() -> None:
    client = FakeClient(
        "m1",
        make_tool_completion(search_call("payments failed deploy")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )
    engine = make_engine(client)
    engine.upload("outage.txt", OUTAGE)

    result = engine.ask("Why did payments fail?")

    assert result.status == "answered"
    assert result.citations[0].sources[0].document_id == "outage.txt"


def test_a_failing_language_model_error_reaches_the_caller() -> None:
    engine = make_engine(FakeClient("m1", LlmUnavailable("down")))

    with pytest.raises(LlmUnavailable):
        engine.ask("Why did payments fail?")


def test_the_keyword_index_is_rebuilt_from_the_store_after_a_restart() -> None:
    store = InMemoryDocumentStore()
    make_engine(store=store).upload("outage.txt", OUTAGE)
    restarted_keywords = Bm25Index()
    restarted = make_engine(store=store, keywords=restarted_keywords)

    loaded = restarted.load_keyword_index()

    assert loaded == 1
    assert [item_id for item_id, _ in restarted_keywords.search("deploy", 3)] == ["outage.txt:0"]


def test_uploads_from_several_threads_are_all_stored_without_corrupting_the_indexes() -> None:
    engine = make_engine()
    names = [f"doc{number}.txt" for number in range(12)]

    def upload(name: str) -> None:
        engine.upload(name, f"Report {name} describes incident {name}.".encode())

    threads = [threading.Thread(target=upload, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert engine.stats() == (12, 12)
    assert {hit.chunk.document_id for hit in engine.search("report incident", 20)} == set(names)


def test_a_second_upload_waits_for_the_first_to_finish_indexing() -> None:
    embedder = GatedEmbedder()
    engine = make_engine(embedder=embedder)
    first = threading.Thread(target=engine.upload, args=("a.txt", OUTAGE))
    second = threading.Thread(target=engine.upload, args=("b.txt", REFUNDS))
    first.start()
    assert embedder.first_entered.wait(WAIT_SECONDS)

    second.start()
    second.join(SHORT_WAIT_SECONDS)
    embedder.release.set()
    first.join(WAIT_SECONDS)
    second.join(WAIT_SECONDS)

    assert embedder.max_active == 1
    assert engine.stats() == (2, 2)


def test_uploaded_documents_are_marked_as_uploads() -> None:
    store = InMemoryDocumentStore()
    engine = make_engine(store=store)

    engine.upload("outage.txt", OUTAGE)

    assert store.origins_of(["outage.txt"]) == {"outage.txt": "upload"}


def seed_built_in_document(store: InMemoryDocumentStore) -> None:
    IngestionService(store, SentenceChunker(max_words=30)).ingest(
        [
            Document(
                id="runbook.txt", text="Restart the gateway.", title="Runbook", source="runbook.txt"
            )
        ]
    )


def test_a_built_in_document_cannot_be_deleted() -> None:
    store = InMemoryDocumentStore()
    seed_built_in_document(store)
    engine = make_engine(store=store)

    with pytest.raises(ProtectedDocumentError, match="runbook.txt"):
        engine.delete_document("runbook.txt")

    assert store.document_ids() == {"runbook.txt"}


def test_an_upload_cannot_replace_a_built_in_document() -> None:
    store = InMemoryDocumentStore()
    seed_built_in_document(store)
    engine = make_engine(store=store)

    with pytest.raises(ProtectedDocumentError, match="runbook.txt"):
        engine.upload("runbook.txt", b"Something else entirely.")

    assert store.chunk("runbook.txt:0").text == "Restart the gateway."


def test_an_uploaded_document_can_still_be_updated_and_deleted() -> None:
    engine = make_engine()
    engine.upload("notes.txt", b"First version.")

    updated = engine.upload("notes.txt", b"Second version.")
    deleted = engine.delete_document("notes.txt")

    assert updated.updated == ["notes.txt"]
    assert deleted is True


def test_protection_can_be_turned_off_for_local_work() -> None:
    store = InMemoryDocumentStore()
    seed_built_in_document(store)
    engine = make_engine(store=store, protect_built_in=False)

    assert engine.delete_document("runbook.txt") is True
