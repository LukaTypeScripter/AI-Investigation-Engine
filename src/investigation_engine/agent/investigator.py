from dataclasses import dataclass, field
from typing import Protocol

from investigation_engine.agent.evidence import EvidenceBoard, SourceLookup
from investigation_engine.agent.models import Investigation
from investigation_engine.agent.prompts import ABSTENTION_MESSAGES, NUDGE, SYSTEM_PROMPT
from investigation_engine.agent.tools import (
    SEARCH_TOOL,
    SUBMIT_TOOL,
    TOOLS,
    Submission,
    ToolArgumentError,
    parse_search_query,
    parse_submission,
)
from investigation_engine.llm.client import ChatClient
from investigation_engine.llm.types import Completion, Message, ToolCall, Usage
from investigation_engine.retrieval.hybrid import Hit

SEARCH_LIMIT_REPLY = (
    "Error: the search limit was reached. Submit your answer now with submit_answer."
)
SUBMISSION_ACCEPTED_REPLY = "Answer accepted."


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int) -> list[Hit]: ...


@dataclass(frozen=True, slots=True)
class InvestigatorConfig:
    max_steps: int = 6
    max_searches: int = 5
    results_per_search: int = 5
    max_evidence_chars: int = 1200
    max_invalid_submissions: int = 2

    def __post_init__(self) -> None:
        limits = (
            self.max_steps,
            self.max_searches,
            self.results_per_search,
            self.max_evidence_chars,
            self.max_invalid_submissions,
        )
        if min(limits) <= 0:
            raise ValueError("every investigator limit must be positive")


@dataclass(slots=True)
class RunState:
    searches: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    llm_calls: int = 0
    invalid_submissions: int = 0

    def record(self, completion: Completion) -> None:
        self.usage = self.usage + completion.usage
        self.llm_calls += 1


class Investigator:
    def __init__(
        self,
        llm: ChatClient,
        retriever: Retriever,
        sources: SourceLookup,
        config: InvestigatorConfig | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._sources = sources
        self._config = config or InvestigatorConfig()

    def investigate(self, question: str) -> Investigation:
        board = EvidenceBoard(self._sources, self._config.max_evidence_chars)
        state = RunState()
        messages = [Message("system", SYSTEM_PROMPT), Message("user", question)]
        for _ in range(self._config.max_steps):
            completion = self._llm.complete(messages, tools=TOOLS, temperature=0.0)
            state.record(completion)
            messages.append(completion.message)
            if not completion.tool_calls:
                messages.append(Message("user", NUDGE))
                continue
            for call in completion.tool_calls:
                reply, submission = self._handle(call, board, state)
                messages.append(Message("tool", reply, tool_call_id=call.id))
                if submission is not None:
                    return self._finish(question, submission, board, state)
            if state.invalid_submissions >= self._config.max_invalid_submissions:
                return self._abstain(question, "invalid_submission", state)
        return self._abstain(question, "step_limit", state)

    def _handle(
        self, call: ToolCall, board: EvidenceBoard, state: RunState
    ) -> tuple[str, Submission | None]:
        try:
            if call.name == SEARCH_TOOL.name:
                return self._search(call.arguments, board, state), None
            if call.name == SUBMIT_TOOL.name:
                return SUBMISSION_ACCEPTED_REPLY, parse_submission(call.arguments, board)
        except ToolArgumentError as error:
            if call.name == SUBMIT_TOOL.name:
                state.invalid_submissions += 1
            return f"Error: {error}", None
        return f"Error: unknown tool '{call.name}'. Use search or submit_answer.", None

    def _search(self, arguments: dict[str, object], board: EvidenceBoard, state: RunState) -> str:
        query = parse_search_query(arguments)
        if len(state.searches) >= self._config.max_searches:
            return SEARCH_LIMIT_REPLY
        state.searches.append(query)
        hits = self._retriever.retrieve(query, self._config.results_per_search)
        return board.render(query, board.add(hits))

    def _finish(
        self, question: str, submission: Submission, board: EvidenceBoard, state: RunState
    ) -> Investigation:
        return Investigation(
            question=question,
            status=submission.status,
            answer=submission.answer,
            citations=tuple(board.citation(evidence_id) for evidence_id in submission.evidence_ids),
            conflicts=submission.conflicts,
            searches=tuple(state.searches),
            usage=state.usage,
            llm_calls=state.llm_calls,
        )

    def _abstain(self, question: str, reason: str, state: RunState) -> Investigation:
        return Investigation(
            question=question,
            status="insufficient_evidence",
            answer=ABSTENTION_MESSAGES[reason],
            citations=(),
            conflicts=(),
            searches=tuple(state.searches),
            usage=state.usage,
            llm_calls=state.llm_calls,
            abstain_reason=reason,
        )
