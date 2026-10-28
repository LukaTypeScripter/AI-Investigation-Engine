import zlib
from collections.abc import Mapping, Sequence

from investigation_engine.llm.types import Completion, Message, ToolCall, ToolSpec, Usage
from investigation_engine.retrieval.bm25 import tokenize


def make_completion(text: str = "Paris", model: str = "m1") -> Completion:
    return Completion(
        message=Message("assistant", text),
        usage=Usage(prompt_tokens=5, completion_tokens=2),
        model=model,
        finish_reason="stop",
    )


def make_tool_completion(*calls: ToolCall, model: str = "m1") -> Completion:
    return Completion(
        message=Message("assistant", None, tool_calls=tuple(calls)),
        usage=Usage(prompt_tokens=5, completion_tokens=2),
        model=model,
        finish_reason="tool_calls",
    )


def search_call(query: str, call_id: str = "s1") -> ToolCall:
    return ToolCall(id=call_id, name="search", arguments={"query": query})


def submit_call(
    status: str,
    answer: str,
    citations: Sequence[str] = (),
    conflicts: Sequence[Mapping[str, object]] = (),
    call_id: str = "f1",
) -> ToolCall:
    arguments: dict[str, object] = {
        "status": status,
        "answer": answer,
        "citations": list(citations),
    }
    if conflicts:
        arguments["conflicts"] = list(conflicts)
    return ToolCall(id=call_id, name="submit_answer", arguments=arguments)


class FakeClient:
    def __init__(self, model: str = "m1", *outcomes: Completion | Exception) -> None:
        self.model = model
        self.calls: list[tuple[Sequence[Message], Sequence[ToolSpec], float, int | None]] = []
        self._outcomes = list(outcomes)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append((list(messages), tools, temperature, max_tokens))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HashingEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.document_batches.append(batch)
        return [self._vector(text) for text in batch]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            vector[zlib.crc32(token.encode("utf-8")) % self.dimension] += 1.0
        return vector
