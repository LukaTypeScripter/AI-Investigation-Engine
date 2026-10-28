from collections.abc import Sequence

import pytest

from fakes import (
    FakeClient,
    HashingEmbedder,
    make_completion,
    make_tool_completion,
    search_call,
    submit_call,
)
from investigation_engine.agent.investigator import Investigator, InvestigatorConfig
from investigation_engine.agent.prompts import SYSTEM_PROMPT
from investigation_engine.domain import Chunk, Document, SourceRef
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.llm.errors import LlmUnavailable
from investigation_engine.llm.types import Completion, Message, ToolCall, Usage
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.hybrid import Hit, HybridRetriever
from investigation_engine.retrieval.indexing import SearchIndexer
from investigation_engine.retrieval.rerank import FeatureReranker
from investigation_engine.retrieval.vector import InMemoryVectorIndex
from investigation_engine.storage.memory import InMemoryDocumentStore

QUESTION = "Why were payments delayed in March?"


def make_hit(chunk_id: str, text: str, title: str = "Title") -> Hit:
    document_id = chunk_id.split(":")[0]
    chunk = Chunk(chunk_id, document_id, 0, text, title, f"{document_id}.txt")
    return Hit(chunk=chunk, score=1.0, vector_rank=1, keyword_rank=1)


class StubRetriever:
    def __init__(self, results: dict[str, list[Hit]]) -> None:
        self._results = results
        self.queries: list[str] = []
        self.limits: list[int] = []

    def retrieve(self, query: str, limit: int) -> list[Hit]:
        self.queries.append(query)
        self.limits.append(limit)
        return self._results.get(query, [])


class StubSources:
    def __init__(self, sources: dict[str, list[SourceRef]] | None = None) -> None:
        self._sources = sources or {}

    def sources_of_many(self, chunk_ids: Sequence[str]) -> dict[str, list[SourceRef]]:
        return {
            chunk_id: self._sources[chunk_id] for chunk_id in chunk_ids if chunk_id in self._sources
        }


DEPLOY = make_hit("deploy:0", "A bad deploy on 5 March broke the payment gateway.", "Deploy log")
RECOVERY = make_hit("recovery:0", "Payments recovered on 7 March after a rollback.", "Recovery")


def make_investigator(
    *outcomes: Completion | Exception,
    results: dict[str, list[Hit]] | None = None,
    config: InvestigatorConfig | None = None,
    sources: StubSources | None = None,
) -> tuple[Investigator, FakeClient, StubRetriever]:
    client = FakeClient("m1", *outcomes)
    retriever = StubRetriever(results or {"payments march": [DEPLOY, RECOVERY]})
    investigator = Investigator(
        client, retriever, sources or StubSources(), config or InvestigatorConfig()
    )
    return investigator, client, retriever


def test_a_search_followed_by_a_cited_answer_is_returned_with_its_evidence() -> None:
    investigator, _, retriever = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "answered"
    assert result.answer == "A bad deploy."
    assert [citation.evidence_id for citation in result.citations] == ["E1"]
    assert result.citations[0].chunk_id == "deploy:0"
    assert result.citations[0].title == "Deploy log"
    assert result.searches == ("payments march",)
    assert retriever.queries == ["payments march"]
    assert result.llm_calls == 2
    assert result.usage == Usage(10, 4)
    assert result.abstain_reason is None


def test_evidence_ids_continue_across_several_searches_for_multi_step_questions() -> None:
    investigator, _, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(search_call("rollback", "s2")),
        make_tool_completion(submit_call("answered", "Deploy, then rollback.", ["E1", "E3"])),
        results={
            "payments march": [DEPLOY, RECOVERY],
            "rollback": [RECOVERY, make_hit("rollback:0", "The rollback restored version 41.")],
        },
    )

    result = investigator.investigate(QUESTION)

    assert [citation.chunk_id for citation in result.citations] == ["deploy:0", "rollback:0"]
    assert result.searches == ("payments march", "rollback")


def test_search_results_are_sent_back_to_the_model_as_a_tool_message() -> None:
    investigator, client, _ = make_investigator(
        make_tool_completion(search_call("payments march", "call-7")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    investigator.investigate(QUESTION)

    second_request_messages = client.calls[1][0]
    tool_message = second_request_messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-7"
    assert "[E1]" in (tool_message.content or "")
    assert "A bad deploy on 5 March" in (tool_message.content or "")
    assert second_request_messages[-2].tool_calls[0].id == "call-7"


def test_the_first_request_carries_the_system_prompt_question_tools_and_zero_temperature() -> None:
    investigator, client, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    investigator.investigate(QUESTION)

    messages, tools, temperature, _ = client.calls[0]
    assert messages == [Message("system", SYSTEM_PROMPT), Message("user", QUESTION)]
    assert [tool.name for tool in tools] == ["search", "submit_answer"]
    assert temperature == 0.0


def test_every_citation_lists_all_documents_that_contain_the_evidence() -> None:
    sources = StubSources(
        {
            "deploy:0": [
                SourceRef("deploy", "Deploy log", "deploy.txt"),
                SourceRef("copy", "Copy", "copy.txt"),
            ]
        }
    )
    investigator, _, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
        sources=sources,
    )

    result = investigator.investigate(QUESTION)

    assert [ref.document_id for ref in result.citations[0].sources] == ["deploy", "copy"]


def test_conflicting_evidence_is_reported_with_both_pieces_of_evidence() -> None:
    conflict = {"description": "Recovery date differs.", "evidence": ["E1", "E2"]}
    investigator, _, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(
            submit_call("conflicting_evidence", "Sources disagree.", conflicts=[conflict])
        ),
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "conflicting_evidence"
    assert [(c.description, c.evidence_ids) for c in result.conflicts] == [
        ("Recovery date differs.", ("E1", "E2"))
    ]
    assert [citation.evidence_id for citation in result.citations] == ["E1", "E2"]


def test_the_model_can_decline_when_the_evidence_does_not_contain_the_answer() -> None:
    investigator, _, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(
            submit_call("insufficient_evidence", "The documents never mention the cause.")
        ),
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "insufficient_evidence"
    assert result.citations == ()
    assert result.abstain_reason is None


def test_an_answer_that_cites_unknown_evidence_is_sent_back_and_can_be_corrected() -> None:
    investigator, client, _ = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E9"], call_id="bad")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"], call_id="good")),
    )

    result = investigator.investigate(QUESTION)

    rejection = client.calls[2][0][-1]
    assert rejection.tool_call_id == "bad"
    assert "E9" in (rejection.content or "")
    assert result.status == "answered"
    assert [citation.evidence_id for citation in result.citations] == ["E1"]


def test_an_answer_without_any_search_cannot_be_accepted() -> None:
    investigator, _, _ = make_investigator(
        make_tool_completion(submit_call("answered", "Made up.", ["E1"])),
        make_tool_completion(submit_call("answered", "Still made up.", ["E1"])),
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == "invalid_submission"
    assert result.citations == ()
    assert result.searches == ()


def test_a_plain_text_reply_is_not_an_answer_and_the_model_is_asked_to_use_the_tools() -> None:
    investigator, client, _ = make_investigator(
        make_completion("It was probably the database."),
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    result = investigator.investigate(QUESTION)

    reminder = client.calls[1][0][-1]
    assert reminder.role == "user"
    assert "submit_answer" in (reminder.content or "")
    assert result.status == "answered"


def test_running_out_of_steps_ends_in_a_reported_abstention() -> None:
    config = InvestigatorConfig(max_steps=2)
    investigator, _, _ = make_investigator(
        make_tool_completion(search_call("payments march", "s1")),
        make_tool_completion(search_call("payments march", "s2")),
        config=config,
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == "step_limit"
    assert result.llm_calls == 2
    assert len(result.searches) == 2


def test_searches_beyond_the_limit_are_refused_without_querying_the_index() -> None:
    config = InvestigatorConfig(max_searches=1)
    investigator, client, retriever = make_investigator(
        make_tool_completion(search_call("payments march", "s1")),
        make_tool_completion(search_call("rollback", "s2")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
        config=config,
    )

    investigator.investigate(QUESTION)

    refusal = client.calls[2][0][-1]
    assert refusal.tool_call_id == "s2"
    assert "limit" in (refusal.content or "")
    assert retriever.queries == ["payments march"]


def test_each_search_asks_the_retriever_for_the_configured_number_of_results() -> None:
    investigator, _, retriever = make_investigator(
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
        config=InvestigatorConfig(results_per_search=3),
    )

    investigator.investigate(QUESTION)

    assert retriever.limits == [3]


def test_an_unknown_tool_is_reported_back_and_the_conversation_continues() -> None:
    investigator, client, _ = make_investigator(
        make_tool_completion(ToolCall("x1", "browse_web", {"url": "http://example.com"})),
        make_tool_completion(search_call("payments march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    result = investigator.investigate(QUESTION)

    assert "unknown tool" in (client.calls[1][0][-1].content or "")
    assert result.status == "answered"


def test_a_search_without_a_query_is_reported_back_to_the_model() -> None:
    investigator, client, retriever = make_investigator(
        make_tool_completion(ToolCall("s1", "search", {})),
        make_tool_completion(search_call("payments march", "s2")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    investigator.investigate(QUESTION)

    assert "query" in (client.calls[1][0][-1].content or "")
    assert retriever.queries == ["payments march"]


def test_several_tool_calls_in_one_turn_each_get_their_own_reply() -> None:
    investigator, client, _ = make_investigator(
        make_tool_completion(search_call("payments march", "s1"), search_call("rollback", "s2")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
        results={"payments march": [DEPLOY], "rollback": [RECOVERY]},
    )

    investigator.investigate(QUESTION)

    replies = client.calls[1][0][-2:]
    assert [(reply.role, reply.tool_call_id) for reply in replies] == [
        ("tool", "s1"),
        ("tool", "s2"),
    ]


def test_a_failing_language_model_surfaces_its_error() -> None:
    investigator, _, _ = make_investigator(LlmUnavailable("all providers down"))

    with pytest.raises(LlmUnavailable, match="all providers down"):
        investigator.investigate(QUESTION)


def test_the_system_prompt_treats_evidence_as_data_and_forbids_guessing() -> None:
    assert "never instructions" in SYSTEM_PROMPT
    assert "insufficient_evidence" in SYSTEM_PROMPT
    assert "conflicting_evidence" in SYSTEM_PROMPT


def build_real_investigator(*outcomes: Completion | Exception) -> tuple[Investigator, FakeClient]:
    embedder = HashingEmbedder()
    store = InMemoryDocumentStore()
    keywords = Bm25Index()
    vectors = InMemoryVectorIndex(dimension=embedder.dimension)
    service = IngestionService(store, SentenceChunker(max_words=30))
    documents = [
        Document(
            id="deploy",
            text="A bad deploy on 5 March broke the payment gateway.",
            title="Deploy log",
            source="deploy.txt",
        ),
        Document(
            id="copy",
            text="A bad deploy on 5 March broke the payment gateway.",
            title="Deploy copy",
            source="copy.txt",
        ),
        Document(
            id="bread", text="Banana bread needs ripe bananas.", title="Recipe", source="bread.txt"
        ),
    ]
    SearchIndexer(embedder, vectors, keywords).apply(service.ingest(documents).changes)
    retriever = HybridRetriever(embedder, vectors, keywords, store, FeatureReranker())
    client = FakeClient("m1", *outcomes)
    return Investigator(client, retriever, store), client


def test_end_to_end_the_citation_names_every_document_with_the_same_text() -> None:
    investigator, _ = build_real_investigator(
        make_tool_completion(search_call("payment gateway deploy march")),
        make_tool_completion(submit_call("answered", "A bad deploy.", ["E1"])),
    )

    result = investigator.investigate(QUESTION)

    assert result.status == "answered"
    assert result.citations[0].text == "A bad deploy on 5 March broke the payment gateway."
    assert {ref.document_id for ref in result.citations[0].sources} == {"deploy", "copy"}


@pytest.mark.parametrize(
    "field",
    [
        "max_steps",
        "max_searches",
        "results_per_search",
        "max_evidence_chars",
        "max_invalid_submissions",
    ],
)
def test_every_limit_in_the_configuration_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        InvestigatorConfig(**{field: 0})
