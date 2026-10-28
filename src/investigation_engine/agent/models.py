from dataclasses import dataclass
from typing import Literal

from investigation_engine.domain import SourceRef
from investigation_engine.llm.types import Usage

AnswerStatus = Literal["answered", "insufficient_evidence", "conflicting_evidence"]
ANSWER_STATUSES: tuple[AnswerStatus, ...] = (
    "answered",
    "insufficient_evidence",
    "conflicting_evidence",
)


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    chunk_id: str
    title: str
    text: str
    sources: tuple[SourceRef, ...]


@dataclass(frozen=True, slots=True)
class Conflict:
    description: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Investigation:
    question: str
    status: AnswerStatus
    answer: str
    citations: tuple[Citation, ...]
    conflicts: tuple[Conflict, ...]
    searches: tuple[str, ...]
    usage: Usage
    llm_calls: int
    abstain_reason: str | None = None
