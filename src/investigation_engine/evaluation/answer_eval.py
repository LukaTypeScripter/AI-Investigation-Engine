from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from investigation_engine.agent.models import Citation, Investigation
from investigation_engine.evaluation.metrics import contains_answer, exact_match, mean, token_f1
from investigation_engine.evaluation.questions import EvalQuestion
from investigation_engine.llm.errors import LlmError

ALL_GROUP = "all"
ERROR_STATUS = "error"
EXPECTS_ANSWER = "answered"
EXPECTS_ABSTENTION = "insufficient_evidence"
TABLE_METRICS = (
    "accuracy",
    "f1",
    "citation_precision",
    "citation_recall",
    "hallucination_rate",
    "searches",
    "tokens",
)


class AnswerInvestigator(Protocol):
    def investigate(self, question: str) -> Investigation: ...


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question_id: str
    category: str
    expected_status: str
    status: str
    answer: str
    correct: bool
    exact_match: float
    f1: float
    citation_precision: float
    citation_recall: float
    cited_documents: tuple[str, ...]
    searches: int
    llm_calls: int
    total_tokens: int
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_citations(citations: Sequence[Citation], gold: Sequence[str]) -> tuple[float, float]:
    gold_set = set(gold)
    if not citations or not gold_set:
        return 0.0, 0.0
    cited_per_citation = [{ref.document_id for ref in citation.sources} for citation in citations]
    correct = sum(1 for documents in cited_per_citation if documents & gold_set)
    covered = set().union(*cited_per_citation) & gold_set
    return correct / len(citations), len(covered) / len(gold_set)


def is_correct(question: EvalQuestion, investigation: Investigation) -> bool:
    if investigation.status != question.expected_status:
        return False
    if question.expected_status != EXPECTS_ANSWER:
        return True
    return any(contains_answer(investigation.answer, gold) for gold in question.gold_answers)


def best_over(metric: Callable[[str, str], float], answer: str, golds: Sequence[str]) -> float:
    return max((metric(answer, gold) for gold in golds), default=0.0)


def score_answer(question: EvalQuestion, investigation: Investigation) -> AnswerResult:
    is_scored_text = investigation.status == EXPECTS_ANSWER
    gold_answers = question.gold_answers if is_scored_text else ()
    precision, recall = score_citations(investigation.citations, question.gold_document_ids)
    cited = dict.fromkeys(
        ref.document_id for citation in investigation.citations for ref in citation.sources
    )
    return AnswerResult(
        question_id=question.id,
        category=question.category,
        expected_status=question.expected_status,
        status=investigation.status,
        answer=investigation.answer,
        correct=is_correct(question, investigation),
        exact_match=best_over(exact_match, investigation.answer, gold_answers),
        f1=best_over(token_f1, investigation.answer, gold_answers),
        citation_precision=precision,
        citation_recall=recall,
        cited_documents=tuple(cited),
        searches=len(investigation.searches),
        llm_calls=investigation.llm_calls,
        total_tokens=investigation.usage.total_tokens,
        error=None,
    )


def error_result(question: EvalQuestion, message: str) -> AnswerResult:
    return AnswerResult(
        question_id=question.id,
        category=question.category,
        expected_status=question.expected_status,
        status=ERROR_STATUS,
        answer="",
        correct=False,
        exact_match=0.0,
        f1=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        cited_documents=(),
        searches=0,
        llm_calls=0,
        total_tokens=0,
        error=message,
    )


def evaluate_answers(
    investigator: AnswerInvestigator, questions: Iterable[EvalQuestion]
) -> list[AnswerResult]:
    results: list[AnswerResult] = []
    for question in questions:
        try:
            investigation = investigator.investigate(question.text)
        except LlmError as error:
            results.append(error_result(question, str(error)))
        else:
            results.append(score_answer(question, investigation))
    return results


def summarize_group(members: Sequence[AnswerResult]) -> dict[str, float]:
    expects_answer = [r for r in members if r.expected_status == EXPECTS_ANSWER]
    expects_abstention = [r for r in members if r.expected_status == EXPECTS_ABSTENTION]
    has_gold = [r for r in members if r.expected_status != EXPECTS_ABSTENTION]
    return {
        "accuracy": mean(float(r.correct) for r in members),
        "exact_match": mean(r.exact_match for r in expects_answer),
        "f1": mean(r.f1 for r in expects_answer),
        "citation_precision": mean(r.citation_precision for r in has_gold),
        "citation_recall": mean(r.citation_recall for r in has_gold),
        "hallucination_rate": mean(float(r.status == EXPECTS_ANSWER) for r in expects_abstention),
        "error_rate": mean(float(r.error is not None) for r in members),
        "searches": mean(float(r.searches) for r in members),
        "llm_calls": mean(float(r.llm_calls) for r in members),
        "tokens": mean(float(r.total_tokens) for r in members),
    }


def summarize_answers(results: Sequence[AnswerResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[AnswerResult]] = {ALL_GROUP: list(results)}
    for result in results:
        groups.setdefault(result.category, []).append(result)
    return {group: summarize_group(members) for group, members in groups.items()}


def render_answer_table(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]], group: str
) -> str:
    lines = [
        "| config | " + " | ".join(TABLE_METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in TABLE_METRICS) + " |",
    ]
    for config, summary in summaries.items():
        values = summary.get(group, {})
        cells = " | ".join(f"{values.get(metric, 0.0):.3f}" for metric in TABLE_METRICS)
        lines.append(f"| {config} | {cells} |")
    return "\n".join(lines)
