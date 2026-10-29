import logging
from collections.abc import Sequence

from investigation_engine.llm.client import ChatClient
from investigation_engine.llm.errors import LlmMalformedResponse, LlmUnavailable, TransientLlmError
from investigation_engine.llm.types import Completion, Message, ToolSpec

logger = logging.getLogger(__name__)

RECOVERABLE_ERRORS = (TransientLlmError, LlmMalformedResponse)


class FallbackChatClient:
    def __init__(self, clients: Sequence[ChatClient]) -> None:
        if not clients:
            raise ValueError("at least one client is required")
        self.model = clients[0].model
        self.fallbacks = 0
        self._clients = list(clients)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        failures: list[Exception] = []
        for position, client in enumerate(self._clients):
            try:
                completion = client.complete(messages, tools, temperature, max_tokens)
            except RECOVERABLE_ERRORS as error:
                logger.warning("provider failed: %s", error)
                failures.append(error)
                continue
            if position > 0:
                self.fallbacks += 1
            return completion
        details = "; ".join(str(failure) for failure in failures)
        raise LlmUnavailable(f"all providers failed: {details}") from failures[-1]
