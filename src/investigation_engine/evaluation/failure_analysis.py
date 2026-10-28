from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from investigation_engine.evaluation.questions import EvalQuestion


@dataclass(frozen=True, slots=True)
class RetrievalFailure:
    question_id: str
    category: str
    text: str
    gold: tuple[str, ...]
    missing: tuple[str, ...]
    retrieved: tuple[str, ...]


def find_failures(
    report: Mapping[str, object], questions: Sequence[EvalQuestion], k: int
) -> list[RetrievalFailure]:
    by_id = {question.id: question for question in questions}
    results = report["questions"]
    if not isinstance(results, list):
        raise ValueError("the report has no list of question results")
    failures: list[RetrievalFailure] = []
    for result in results:
        question = by_id[result["question_id"]]
        retrieved = tuple(result["ranked_documents"][:k])
        missing = tuple(
            document for document in question.gold_document_ids if document not in retrieved
        )
        if missing:
            failures.append(
                RetrievalFailure(
                    question_id=question.id,
                    category=question.category,
                    text=question.text,
                    gold=question.gold_document_ids,
                    missing=missing,
                    retrieved=retrieved,
                )
            )
    return failures


def render_failures(failures: Sequence[RetrievalFailure]) -> str:
    if not failures:
        return "No retrieval failures."
    blocks = [
        "\n".join(
            [
                f"### {failure.question_id} ({failure.category})",
                failure.text,
                f"Missing: {', '.join(failure.missing)}",
                f"Retrieved: {', '.join(failure.retrieved)}",
            ]
        )
        for failure in failures
    ]
    return "\n\n".join(blocks)
