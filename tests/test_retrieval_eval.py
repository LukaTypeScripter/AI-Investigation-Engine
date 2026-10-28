import json

import pytest

from fakes import HashingEmbedder
from investigation_engine.domain import Document
from investigation_engine.evaluation.questions import EvalQuestion, QuestionCategory
from investigation_engine.evaluation.retrieval_eval import (
    DEFAULT_CONFIGS,
    RetrievalConfig,
    SearchStack,
    build_stack,
    document_ranking,
    evaluate_question,
    limit_documents,
    make_retriever,
    render_table,
    report_to_dict,
    run_configs,
    summarize,
)
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestReport

DOCUMENTS = [
    Document(
        "alpha", "The payments gateway failed during the march outage.", "Gateway outage", "a.txt"
    ),
    Document("beta", "Banana bread needs ripe bananas and a little sugar.", "Recipe", "b.txt"),
    Document("gamma", "The ledger backlog was drained with a command.", "Ledger runbook", "c.txt"),
]


def make_question(
    question_id: str, text: str, gold: tuple[str, ...], category: QuestionCategory = "simple"
) -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        text=text,
        category=category,
        gold_answers=("x",),
        gold_document_ids=gold,
        expected_status="answered",
        origin="test",
    )


def make_default_stack(documents: list[Document] | None = None) -> tuple[SearchStack, IngestReport]:
    corpus = DOCUMENTS if documents is None else documents
    return build_stack(corpus, SentenceChunker(max_words=50), HashingEmbedder())


def test_the_default_configurations_are_the_four_compared_settings() -> None:
    assert [config.name for config in DEFAULT_CONFIGS] == [
        "bm25_only",
        "vector_only",
        "hybrid_rrf",
        "hybrid_rerank",
    ]


def test_build_stack_ingests_and_indexes_every_document() -> None:
    stack, report = make_default_stack()

    assert sorted(report.added) == ["alpha", "beta", "gamma"]
    assert len(stack.keywords) == 3
    assert len(stack.vectors) == 3
    assert stack.store.document_ids() == {"alpha", "beta", "gamma"}
    assert report.duplicate_chunks == 0
    assert stack.embedder.dimension == 64


def test_a_gold_document_found_first_scores_one_everywhere() -> None:
    stack, _ = make_default_stack()
    retriever = make_retriever(stack, RetrievalConfig("hybrid_rrf"))
    question = make_question("q1", "why did the payments gateway fail", ("alpha",))

    result = evaluate_question(retriever, stack.store, question, (1, 5))

    assert result.ranked_documents[0] == "alpha"
    assert result.metrics == {
        "recall@1": 1.0,
        "full_recall@1": 1.0,
        "recall@5": 1.0,
        "full_recall@5": 1.0,
        "mrr": 1.0,
    }


def test_a_gold_document_missing_from_the_corpus_scores_zero() -> None:
    stack, _ = make_default_stack()
    retriever = make_retriever(stack, RetrievalConfig("hybrid_rrf"))
    question = make_question("q1", "why did the payments gateway fail", ("hotpot:Nowhere",))

    result = evaluate_question(retriever, stack.store, question, (1, 5))

    assert result.metrics["recall@5"] == 0.0
    assert result.metrics["mrr"] == 0.0


def test_a_chunk_shared_by_duplicate_documents_credits_every_document() -> None:
    copies = [
        Document("a", "The payments gateway failed during the march outage.", "T", "a.txt"),
        Document("b", "The payments gateway failed during the march outage.", "T", "b.txt"),
    ]
    stack, report = make_default_stack(copies)
    retriever = make_retriever(stack, RetrievalConfig("hybrid_rrf"))

    hits = retriever.retrieve("payments gateway", 5)

    assert report.duplicate_chunks == 1
    assert document_ranking(hits, stack.store) == ["a", "b"]


def test_turning_off_vectors_removes_the_vector_rank_from_hits() -> None:
    stack, _ = make_default_stack()

    hits = make_retriever(stack, RetrievalConfig("bm25_only", use_vectors=False)).retrieve(
        "payments gateway", 3
    )

    assert hits
    assert all(hit.vector_rank is None for hit in hits)
    assert all(hit.keyword_rank is not None for hit in hits)


def test_turning_off_keywords_removes_the_keyword_rank_from_hits() -> None:
    stack, _ = make_default_stack()

    hits = make_retriever(stack, RetrievalConfig("vector_only", use_keywords=False)).retrieve(
        "payments gateway", 3
    )

    assert hits
    assert all(hit.keyword_rank is None for hit in hits)


def test_run_configs_produces_one_report_per_configuration() -> None:
    stack, ingest_report = make_default_stack()
    questions = [
        make_question("q1", "why did the payments gateway fail", ("alpha",)),
        make_question("q2", "how is banana bread made", ("beta",), "multi_step"),
        make_question("q3", "what is the revenue loss", (), "unanswerable"),
    ]

    reports = run_configs(stack, ingest_report, questions, DEFAULT_CONFIGS, "sentence-50", (1, 5))

    assert [report.config for report in reports] == [config.name for config in DEFAULT_CONFIGS]
    first = reports[0]
    assert (first.chunker, first.documents, first.chunks) == ("sentence-50", 3, 3)
    assert [result.question_id for result in first.questions] == ["q1", "q2"]
    assert first.mean_latency_ms >= 0.0


def test_summarize_averages_per_category_and_overall() -> None:
    stack, ingest_report = make_default_stack()
    questions = [
        make_question("q1", "why did the payments gateway fail", ("alpha",), "simple"),
        make_question("q2", "how is banana bread made", ("beta",), "multi_step"),
    ]
    report = run_configs(
        stack, ingest_report, questions, [RetrievalConfig("hybrid_rrf")], "c", (1, 5)
    )[0]

    summary = summarize(report.questions)

    assert set(summary) == {"all", "simple", "multi_step"}
    assert summary["all"]["recall@5"] == 1.0
    assert summary["simple"]["mrr"] == 1.0


def test_limit_documents_keeps_the_required_ones_and_fills_up_reproducibly() -> None:
    documents = [Document(f"d{n}", f"text {n}", f"T{n}", "s") for n in range(50)]

    chosen = limit_documents(documents, required_ids={"d7", "d33"}, limit=10, seed=3)
    again = limit_documents(list(reversed(documents)), required_ids={"d7", "d33"}, limit=10, seed=3)

    assert len(chosen) == 10
    assert {"d7", "d33"} <= {document.id for document in chosen}
    assert [d.id for d in chosen] == [d.id for d in again]


def test_limit_documents_returns_everything_when_the_limit_is_large_enough() -> None:
    documents = [Document(f"d{n}", f"text {n}", f"T{n}", "s") for n in range(5)]

    assert len(limit_documents(documents, required_ids=set(), limit=100, seed=1)) == 5


def test_a_report_converts_to_plain_json() -> None:
    stack, ingest_report = make_default_stack()
    questions = [make_question("q1", "why did the payments gateway fail", ("alpha",))]
    report = run_configs(
        stack, ingest_report, questions, [RetrievalConfig("hybrid_rrf")], "c", (5,)
    )[0]

    payload = json.loads(json.dumps(report_to_dict(report)))

    assert payload["config"] == "hybrid_rrf"
    assert payload["metrics"]["all"]["recall@5"] == 1.0
    assert payload["questions"][0]["question_id"] == "q1"


def test_render_table_lists_each_configuration_with_its_metrics() -> None:
    stack, ingest_report = make_default_stack()
    questions = [make_question("q1", "why did the payments gateway fail", ("alpha",))]
    reports = run_configs(stack, ingest_report, questions, DEFAULT_CONFIGS, "c", (1, 5))

    table = render_table(reports, "all", ("recall@1", "recall@5", "mrr"))

    lines = table.splitlines()
    assert lines[0] == "| config | recall@1 | recall@5 | mrr |"
    assert lines[1] == "| --- | --- | --- | --- |"
    assert [line.split("|")[1].strip() for line in lines[2:]] == [c.name for c in DEFAULT_CONFIGS]
    assert "1.000" in lines[2]


def test_retrieval_config_rejects_a_configuration_with_no_search() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RetrievalConfig("nothing", use_vectors=False, use_keywords=False)
