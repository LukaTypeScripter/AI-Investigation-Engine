import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from investigation_engine.agent.models import ANSWER_STATUSES, AnswerStatus
from investigation_engine.evaluation.hotpot import HotpotExample, gold_document_ids

QuestionCategory = Literal["simple", "multi_step", "multi_document", "unanswerable", "conflicting"]
CATEGORIES: tuple[QuestionCategory, ...] = (
    "simple",
    "multi_step",
    "multi_document",
    "unanswerable",
    "conflicting",
)
HOTPOT_ORIGIN = "hotpotqa"
BRIDGE_SHARE = 2 / 3
CATEGORY_BY_HOTPOT_TYPE: dict[str, QuestionCategory] = {
    "bridge": "multi_step",
    "comparison": "multi_document",
}


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    text: str
    category: QuestionCategory
    gold_answers: tuple[str, ...]
    gold_document_ids: tuple[str, ...]
    expected_status: AnswerStatus
    origin: str


def read_category(value: object) -> QuestionCategory:
    for category in CATEGORIES:
        if value == category:
            return category
    raise ValueError(f"unknown question category: {value!r}")


def read_status(value: object) -> AnswerStatus:
    for status in ANSWER_STATUSES:
        if value == status:
            return status
    raise ValueError(f"unknown expected status: {value!r}")


def question_from_record(record: Mapping[str, object]) -> EvalQuestion:
    gold_answers = record.get("gold_answers", [])
    gold_documents = record.get("gold_document_ids", [])
    if not isinstance(gold_answers, list) or not isinstance(gold_documents, list):
        raise ValueError("gold_answers and gold_document_ids must be lists")
    return EvalQuestion(
        id=str(record["id"]),
        text=str(record["text"]),
        category=read_category(record.get("category")),
        gold_answers=tuple(str(answer) for answer in gold_answers),
        gold_document_ids=tuple(str(document_id) for document_id in gold_documents),
        expected_status=read_status(record.get("expected_status")),
        origin=str(record.get("origin", "")),
    )


def write_questions(path: Path, questions: Iterable[EvalQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(question), ensure_ascii=False) for question in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_questions(path: Path) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [question_from_record(json.loads(line)) for line in handle if line.strip()]


def question_from_example(example: HotpotExample) -> EvalQuestion:
    return EvalQuestion(
        id=example.id,
        text=example.question,
        category=CATEGORY_BY_HOTPOT_TYPE[example.type],
        gold_answers=(example.answer,),
        gold_document_ids=gold_document_ids(example),
        expected_status="answered",
        origin=HOTPOT_ORIGIN,
    )


def seeded_order(examples: Iterable[HotpotExample], seed: int) -> list[HotpotExample]:
    def sort_key(example: HotpotExample) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{example.id}".encode()).hexdigest()
        return digest, example.id

    return sorted(examples, key=sort_key)


def sample_hotpot_questions(
    examples: Iterable[HotpotExample], count: int, seed: int
) -> list[EvalQuestion]:
    ordered = seeded_order(examples, seed)
    bridges = [example for example in ordered if example.type == "bridge"]
    comparisons = [example for example in ordered if example.type == "comparison"]
    bridge_count = round(count * BRIDGE_SHARE)
    comparison_count = count - bridge_count
    if len(bridges) < bridge_count or len(comparisons) < comparison_count:
        raise ValueError(
            f"only {len(bridges)} bridge and {len(comparisons)} comparison examples "
            f"are available, {bridge_count} and {comparison_count} are needed"
        )
    chosen = bridges[:bridge_count] + comparisons[:comparison_count]
    return [question_from_example(example) for example in chosen]
