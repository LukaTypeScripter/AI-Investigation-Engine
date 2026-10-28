import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import httpx

from investigation_engine.domain import Document
from investigation_engine.embeddings import (
    CachedEmbedder,
    Embedder,
    FastEmbedEmbedder,
    SqliteEmbeddingCache,
)
from investigation_engine.evaluation.hotpot import (
    build_documents,
    download_validation,
    load_examples,
)
from investigation_engine.evaluation.questions import (
    EvalQuestion,
    read_questions,
    sample_hotpot_questions,
    write_questions,
)
from investigation_engine.evaluation.retrieval_eval import (
    DEFAULT_CONFIGS,
    RetrievalConfig,
    RetrievalReport,
    build_stack,
    limit_documents,
    render_table,
    report_to_dict,
    run_configs,
)
from investigation_engine.ingestion.chunking import Chunker, FixedWindowChunker, SentenceChunker
from investigation_engine.ingestion.parsers import parse_path

CHUNKERS: dict[str, Chunker] = {
    "sentence-120": SentenceChunker(max_words=120),
    "sentence-60": SentenceChunker(max_words=60),
    "window-60-15": FixedWindowChunker(size=60, overlap=15),
}
DEFAULT_CHUNKER = "sentence-120"
DEFAULT_SEED = 42
DEFAULT_QUESTION_COUNT = 30
DEFAULT_HOTPOT_PATH = Path("data/hotpot/validation.jsonl")
DEFAULT_CACHE_PATH = Path("data/embeddings.sqlite")
OVERVIEW_METRICS = ("recall@1", "recall@5", "recall@10", "full_recall@5", "full_recall@10", "mrr")
CATEGORY_METRICS = ("recall@5", "full_recall@5", "mrr")
DOWNLOAD_ATTEMPTS = 10
DOWNLOAD_PAUSE_SECONDS = 1.5
DOWNLOAD_TIMEOUT_SECONDS = 60.0
CHUNKING_CONFIGS = (RetrievalConfig("hybrid_rrf"), RetrievalConfig("hybrid_rerank", rerank=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="investigation-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download the HotpotQA validation split")
    download.add_argument("--out", type=Path, default=DEFAULT_HOTPOT_PATH)

    build = commands.add_parser("build-questions", help="sample HotpotQA evaluation questions")
    build.add_argument("--hotpot", type=Path, default=DEFAULT_HOTPOT_PATH)
    build.add_argument("--count", type=int, default=DEFAULT_QUESTION_COUNT)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--out", type=Path, required=True)

    for name, help_text in (
        ("retrieval", "compare retrieval configurations"),
        ("chunking", "compare chunking strategies"),
    ):
        experiment = commands.add_parser(name, help=help_text)
        experiment.add_argument("--hotpot", type=Path, default=DEFAULT_HOTPOT_PATH)
        experiment.add_argument("--samples", type=Path)
        experiment.add_argument("--questions", type=Path, nargs="+", required=True)
        experiment.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
        experiment.add_argument("--out", type=Path, required=True)
        experiment.add_argument("--max-documents", type=int)
        experiment.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "retrieval":
            experiment.add_argument("--chunker", choices=sorted(CHUNKERS), default=DEFAULT_CHUNKER)
    return parser


def run_download(arguments: argparse.Namespace) -> int:
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
        count = download_validation(
            client, arguments.out, attempts=DOWNLOAD_ATTEMPTS, pause=DOWNLOAD_PAUSE_SECONDS
        )
    print(f"downloaded {count} rows to {arguments.out}")
    return 0


def run_build_questions(arguments: argparse.Namespace) -> int:
    examples = load_examples(arguments.hotpot)
    questions = sample_hotpot_questions(examples, arguments.count, arguments.seed)
    write_questions(arguments.out, questions)
    print(f"wrote {len(questions)} questions to {arguments.out}")
    return 0


def load_corpus(arguments: argparse.Namespace, questions: Sequence[EvalQuestion]) -> list[Document]:
    documents = build_documents(load_examples(arguments.hotpot))
    if arguments.max_documents:
        required = {
            document_id for question in questions for document_id in question.gold_document_ids
        }
        documents = limit_documents(documents, required, arguments.max_documents, arguments.seed)
    if arguments.samples:
        for path in sorted(arguments.samples.iterdir()):
            documents.extend(parse_path(path))
    return documents


def print_tables(reports: Sequence[RetrievalReport]) -> None:
    print(render_table(reports, "all", OVERVIEW_METRICS))
    categories = sorted({group for report in reports for group in report.metrics} - {"all"})
    for category in categories:
        print(f"\n{category}\n")
        print(render_table(reports, category, CATEGORY_METRICS))


def write_reports(arguments: argparse.Namespace, reports: Sequence[RetrievalReport]) -> None:
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": arguments.seed,
        "max_documents": arguments.max_documents,
        "reports": [report_to_dict(report) for report in reports],
    }
    arguments.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(arguments: argparse.Namespace, embedder: Embedder) -> int:
    questions = [question for path in arguments.questions for question in read_questions(path)]
    documents = load_corpus(arguments, questions)
    if arguments.command == "retrieval":
        chunkers, configs = {arguments.chunker: CHUNKERS[arguments.chunker]}, DEFAULT_CONFIGS
    else:
        chunkers, configs = CHUNKERS, CHUNKING_CONFIGS
    reports: list[RetrievalReport] = []
    for name, chunker in chunkers.items():
        print(f"indexing {len(documents)} documents with {name}", file=sys.stderr)
        stack, ingest_report = build_stack(documents, chunker, embedder)
        for report in run_configs(stack, ingest_report, questions, configs, name):
            labelled = replace(report, config=f"{name}/{report.config}")
            reports.append(labelled if arguments.command == "chunking" else report)
    write_reports(arguments, reports)
    print_tables(reports)
    return 0


def open_embedder(cache_path: Path) -> CachedEmbedder:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return CachedEmbedder(FastEmbedEmbedder(), SqliteEmbeddingCache(cache_path))


def main(argv: Sequence[str] | None = None, embedder: Embedder | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "download":
        return run_download(arguments)
    if arguments.command == "build-questions":
        return run_build_questions(arguments)
    if embedder is not None:
        return run_experiment(arguments, embedder)
    cached = open_embedder(arguments.cache)
    try:
        return run_experiment(arguments, cached)
    finally:
        cached.close()


if __name__ == "__main__":
    sys.exit(main())
