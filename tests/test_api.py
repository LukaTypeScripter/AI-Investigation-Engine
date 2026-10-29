import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fakes import (
    FakeClient,
    make_engine,
    make_tool_completion,
    search_call,
    submit_call,
)
from investigation_engine.app.api import ApiLimits, create_app
from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.llm.errors import (
    LlmBadGeneration,
    LlmMalformedResponse,
    LlmRateLimited,
    LlmRejected,
    LlmUnavailable,
)
from investigation_engine.llm.types import Completion
from investigation_engine.storage.memory import InMemoryDocumentStore

OUTAGE = b"Payments failed on 5 March because of a bad deploy."
REFUNDS = b"Refunds were delayed by two days."
LIMITS = ApiLimits(max_upload_bytes=200, max_question_chars=100)
API_LOGGER = "investigation_engine.app.api"


def open_client(llm: FakeClient | None = None) -> TestClient:
    engine = make_engine(llm)
    return TestClient(create_app(lambda: engine, LIMITS))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with open_client() as opened:
        yield opened


def upload(client: TestClient, filename: str, content: bytes) -> dict[str, object]:
    response = client.post("/api/documents", params={"filename": filename}, content=content)
    assert response.status_code == 201
    return response.json()


def answering_client(*outcomes: Completion | Exception) -> TestClient:
    return open_client(FakeClient("m1", *outcomes))


def test_health_reports_the_counts_and_whether_answers_are_available(client: TestClient) -> None:
    upload(client, "outage.txt", OUTAGE)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 1, "chunks": 1, "can_answer": False}


def test_an_uploaded_file_is_created_and_reports_what_was_stored(client: TestClient) -> None:
    response = client.post("/api/documents", params={"filename": "outage.txt"}, content=OUTAGE)

    assert response.status_code == 201
    assert response.json() == {
        "filename": "outage.txt",
        "added": ["outage.txt"],
        "updated": [],
        "unchanged": [],
        "duplicate_chunks": 0,
        "chunks_added": 1,
        "chunks_removed": 0,
    }


def test_uploaded_documents_are_listed_with_their_chunk_counts(client: TestClient) -> None:
    upload(client, "outage.txt", OUTAGE)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {"id": "outage.txt", "title": "outage", "source": "outage.txt", "chunk_count": 1}
        ],
        "total": 1,
    }


def test_the_document_list_accepts_paging_and_a_name_filter(client: TestClient) -> None:
    upload(client, "outage.txt", OUTAGE)
    upload(client, "refunds.txt", REFUNDS)

    paged = client.get("/api/documents", params={"limit": 1, "offset": 1}).json()
    filtered = client.get("/api/documents", params={"q": "refund"}).json()

    assert [item["id"] for item in paged["items"]] == ["outage.txt"]
    assert paged["total"] == 2
    assert [item["id"] for item in filtered["items"]] == ["refunds.txt"]
    assert filtered["total"] == 1


def test_a_file_in_an_unsupported_format_is_rejected_with_415(client: TestClient) -> None:
    response = client.post("/api/documents", params={"filename": "photo.png"}, content=b"data")

    assert response.status_code == 415
    assert ".png" in response.json()["detail"]


def test_a_file_without_text_is_rejected_with_422(client: TestClient) -> None:
    response = client.post("/api/documents", params={"filename": "empty.txt"}, content=b"  \n")

    assert response.status_code == 422
    assert "empty.txt" in response.json()["detail"]


def test_a_malformed_file_is_rejected_with_422_and_the_reason(client: TestClient) -> None:
    response = client.post("/api/documents", params={"filename": "data.json"}, content=b'{"text": ')

    assert response.status_code == 422
    assert "invalid JSON" in response.json()["detail"]


def test_a_file_over_the_size_limit_is_rejected_with_413(client: TestClient) -> None:
    response = client.post(
        "/api/documents",
        params={"filename": "big.txt"},
        content=b"x" * (LIMITS.max_upload_bytes + 1),
    )

    assert response.status_code == 413
    assert client.get("/api/health").json()["documents"] == 0


def test_an_upload_without_a_filename_is_rejected_with_422(client: TestClient) -> None:
    assert client.post("/api/documents", content=OUTAGE).status_code == 422


def test_deleting_a_document_removes_it_and_a_second_delete_finds_nothing(
    client: TestClient,
) -> None:
    upload(client, "outage.txt", OUTAGE)

    first = client.delete("/api/documents/outage.txt")
    second = client.delete("/api/documents/outage.txt")

    assert first.status_code == 204
    assert second.status_code == 404
    assert client.get("/api/health").json()["documents"] == 0


def test_documents_whose_ids_contain_special_characters_can_be_deleted(client: TestClient) -> None:
    upload(client, "data.json", b'[{"id": "inc-1", "text": "Body of the record"}]')

    response = client.delete("/api/documents/data.json%23inc-1")

    assert response.status_code == 204


def test_search_returns_ranked_chunks_with_their_sources(client: TestClient) -> None:
    upload(client, "outage.txt", OUTAGE)
    upload(client, "refunds.txt", REFUNDS)

    response = client.post("/api/search", json={"query": "bad deploy", "limit": 2})

    assert response.status_code == 200
    first = response.json()["hits"][0]
    assert first["document_id"] == "outage.txt"
    assert first["text"] == OUTAGE.decode()
    assert first["sources"] == [
        {"document_id": "outage.txt", "title": "outage", "source": "outage.txt"}
    ]
    assert first["keyword_rank"] == 1
    assert isinstance(first["score"], float)


@pytest.mark.parametrize(
    "body", [{"query": ""}, {"query": "x", "limit": 0}, {"query": "x", "limit": 99}]
)
def test_search_rejects_an_empty_query_and_an_out_of_range_limit(
    client: TestClient, body: dict[str, object]
) -> None:
    assert client.post("/api/search", json=body).status_code == 422


def test_asking_without_a_language_model_returns_503_and_says_why(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "Why did payments fail?"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_asking_returns_the_answer_with_its_citations_and_cost() -> None:
    with answering_client(
        make_tool_completion(search_call("payments failed deploy")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    ) as client:
        upload(client, "outage.txt", OUTAGE)

        response = client.post("/api/ask", json={"question": "Why did payments fail?"})

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["answer"] == "A bad deploy."
    assert body["citations"][0]["evidence_id"] == "E1"
    assert body["citations"][0]["sources"][0]["document_id"] == "outage.txt"
    assert body["searches"] == ["payments failed deploy"]
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    assert body["llm_calls"] == 2
    assert body["abstain_reason"] is None
    assert body["elapsed_ms"] >= 0


def test_asking_can_return_a_conflict_between_two_sources() -> None:
    conflict = {"description": "The causes differ.", "evidence": ["E1", "E2"]}
    with answering_client(
        make_tool_completion(search_call("payments")),
        make_tool_completion(
            submit_call("conflicting_evidence", "Sources disagree.", [], [conflict])
        ),
    ) as client:
        upload(client, "outage.txt", b"Payments failed because of a bad deploy.")
        upload(client, "status.txt", b"Payments degraded because of a provider issue.")

        body = client.post("/api/ask", json={"question": "Why did payments fail?"}).json()

    assert body["status"] == "conflicting_evidence"
    assert body["conflicts"] == [
        {"description": "The causes differ.", "evidence_ids": ["E1", "E2"]}
    ]


def test_asking_can_return_an_abstention() -> None:
    with answering_client(
        make_tool_completion(search_call("weather")),
        make_tool_completion(submit_call("insufficient_evidence", "The documents do not say.")),
    ) as client:
        body = client.post("/api/ask", json={"question": "What is the weather?"}).json()

    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LlmRateLimited("slow down"), 429),
        (LlmUnavailable("down"), 503),
        (LlmRejected("bad key"), 502),
        (LlmMalformedResponse("garbage"), 502),
        (LlmBadGeneration("garbage"), 502),
    ],
)
def test_language_model_failures_become_clear_http_errors(error: Exception, status: int) -> None:
    with answering_client(error) as client:
        response = client.post("/api/ask", json={"question": "Why did payments fail?"})

    assert response.status_code == status
    assert "bad key" not in response.text
    assert "garbage" not in response.text


@pytest.mark.parametrize("question", ["", "   ", "x" * 101])
def test_empty_and_overlong_questions_are_rejected_with_422(question: str) -> None:
    with answering_client() as client:
        response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422


def test_the_root_serves_the_web_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Investigation Engine" in response.text


def test_unknown_api_paths_return_404(client: TestClient) -> None:
    assert client.get("/api/nothing").status_code == 404


def test_a_language_model_failure_is_written_to_the_log_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with answering_client(LlmRejected("m1: HTTP 404: model not found")) as client:
        response = client.post("/api/ask", json={"question": "Why did payments fail?"})

    assert "model not found" not in response.text
    assert "model not found" in caplog.text


def test_an_answered_question_is_logged_with_its_outcome_and_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger=API_LOGGER),
        answering_client(
            make_tool_completion(search_call("payments failed deploy")),
            make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
        ) as client,
    ):
        upload(client, "outage.txt", OUTAGE)
        client.post("/api/ask", json={"question": "Why did payments fail?"})

    assert "Why did payments fail?" in caplog.text
    assert "status=answered" in caplog.text
    assert "searches=1" in caplog.text
    assert "llm_calls=2" in caplog.text


def test_a_failed_question_is_named_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    with answering_client(LlmRateLimited("m1: rate limited")) as client:
        client.post("/api/ask", json={"question": "Why did payments fail?"})

    assert "Why did payments fail?" in caplog.text
    assert "m1: rate limited" in caplog.text


def test_a_long_question_is_shortened_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    with answering_client(LlmUnavailable("m1: down")) as client:
        client.post("/api/ask", json={"question": "q" * 100})

    assert "q" * 80 + "..." in caplog.text
    assert "q" * 81 not in caplog.text


def test_uploads_and_deletes_are_logged(
    caplog: pytest.LogCaptureFixture, client: TestClient
) -> None:
    with caplog.at_level(logging.INFO, logger=API_LOGGER):
        upload(client, "outage.txt", OUTAGE)
        client.delete("/api/documents/outage.txt")

    assert "uploaded outage.txt: added=1 updated=0 unchanged=0" in caplog.text
    assert "deleted outage.txt" in caplog.text


def test_a_rejected_upload_is_logged_with_the_reason(
    caplog: pytest.LogCaptureFixture, client: TestClient
) -> None:
    client.post("/api/documents", params={"filename": "tool.exe"}, content=b"x")

    assert "upload of tool.exe rejected (415)" in caplog.text


def test_the_engine_being_ready_is_logged_with_its_size(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=API_LOGGER), open_client():
        pass

    assert "engine ready: documents=0 chunks=0 can_answer=False" in caplog.text


def client_with_built_in_document() -> TestClient:
    store = InMemoryDocumentStore()
    IngestionService(store, SentenceChunker(max_words=30)).ingest(
        [Document(id="runbook.txt", text="Restart.", title="Runbook", source="runbook.txt")]
    )
    engine = make_engine(store=store)
    return TestClient(create_app(lambda: engine, LIMITS))


def test_deleting_a_built_in_document_is_forbidden() -> None:
    with client_with_built_in_document() as client:
        response = client.delete("/api/documents/runbook.txt")

        assert response.status_code == 403
        assert "built-in" in response.json()["detail"]
        assert client.get("/api/health").json()["documents"] == 1


def test_uploading_over_a_built_in_document_is_forbidden() -> None:
    with client_with_built_in_document() as client:
        response = client.post(
            "/api/documents", params={"filename": "runbook.txt"}, content=b"Other text."
        )

        assert response.status_code == 403
        assert "built-in" in response.json()["detail"]
