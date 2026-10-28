from collections.abc import Sequence

from investigation_engine.llm.types import Completion, Message, ToolSpec, Usage


def make_completion(text: str = "Paris", model: str = "m1") -> Completion:
    return Completion(
        message=Message("assistant", text),
        usage=Usage(prompt_tokens=5, completion_tokens=2),
        model=model,
        finish_reason="stop",
    )


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
        self.calls.append((messages, tools, temperature, max_tokens))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
