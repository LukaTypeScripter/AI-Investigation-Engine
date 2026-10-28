from collections.abc import Mapping
from dataclasses import dataclass

from investigation_engine.agent.evidence import EvidenceBoard, normalize_evidence_id
from investigation_engine.agent.models import ANSWER_STATUSES, AnswerStatus, Conflict
from investigation_engine.llm.types import ToolSpec

SEARCH_TOOL = ToolSpec(
    name="search",
    description=(
        "Search the document collection. Returns numbered evidence items such as [E1]. "
        "Call it several times with focused queries when the question needs more than one document."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for."}},
        "required": ["query"],
    },
)

SUBMIT_TOOL = ToolSpec(
    name="submit_answer",
    description="Submit the final answer. Call it exactly once, after searching.",
    parameters={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
            "answer": {"type": "string", "description": "The answer, or what is missing."},
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evidence ids such as E1 that support the answer.",
            },
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["description", "evidence"],
                },
            },
        },
        "required": ["status", "answer"],
    },
)

TOOLS = (SEARCH_TOOL, SUBMIT_TOOL)


class ToolArgumentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Submission:
    status: AnswerStatus
    answer: str
    evidence_ids: tuple[str, ...]
    conflicts: tuple[Conflict, ...]


def read_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentError(f"'{key}' must be a non-empty string")
    return value.strip()


def read_status(value: object) -> AnswerStatus:
    for status in ANSWER_STATUSES:
        if value == status:
            return status
    raise ToolArgumentError(f"'status' must be one of: {', '.join(ANSWER_STATUSES)}")


def read_evidence_ids(raw: object, board: EvidenceBoard, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ToolArgumentError(f"'{key}' must be a list of evidence ids such as [\"E1\"]")
    evidence_ids = tuple(dict.fromkeys(normalize_evidence_id(item) for item in raw))
    unknown = board.unknown(evidence_ids)
    if unknown:
        raise ToolArgumentError(
            f"unknown evidence ids {', '.join(unknown)}; cite only ids returned by search"
        )
    return evidence_ids


def read_conflicts(raw: object, board: EvidenceBoard) -> tuple[Conflict, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ToolArgumentError("'conflicts' must be a list of conflict objects")
    conflicts: list[Conflict] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ToolArgumentError(
                "each conflict must be an object with 'description' and 'evidence'"
            )
        evidence_ids = read_evidence_ids(item.get("evidence"), board, "evidence")
        if len(evidence_ids) < 2:
            raise ToolArgumentError("each conflict must reference at least two evidence ids")
        conflicts.append(Conflict(read_text(item, "description"), evidence_ids))
    return tuple(conflicts)


def parse_search_query(arguments: Mapping[str, object]) -> str:
    return read_text(arguments, "query")


def parse_submission(arguments: Mapping[str, object], board: EvidenceBoard) -> Submission:
    status = read_status(arguments.get("status"))
    answer = read_text(arguments, "answer")
    cited = read_evidence_ids(arguments.get("citations"), board, "citations")
    conflicts = read_conflicts(arguments.get("conflicts"), board)
    if status == "answered" and not cited:
        raise ToolArgumentError("an answered submission must cite at least one evidence id")
    if status == "conflicting_evidence" and not conflicts:
        raise ToolArgumentError("conflicting_evidence needs at least one conflict")
    conflict_ids = (evidence_id for conflict in conflicts for evidence_id in conflict.evidence_ids)
    evidence_ids = tuple(dict.fromkeys([*cited, *conflict_ids]))
    return Submission(status, answer, evidence_ids, conflicts)
