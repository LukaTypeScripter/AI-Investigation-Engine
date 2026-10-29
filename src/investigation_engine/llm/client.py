import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from investigation_engine.llm.errors import (
    LlmBadGeneration,
    LlmMalformedResponse,
    LlmRateLimited,
    LlmRejected,
    LlmUnavailable,
    TransientLlmError,
)
from investigation_engine.llm.types import Completion, Message, ToolSpec
from investigation_engine.llm.wire import build_request_body, parse_completion

logger = logging.getLogger(__name__)

BAD_REQUEST_STATUS = 400
RATE_LIMITED_STATUS = 429
UNAVAILABLE_STATUSES = frozenset({500, 502, 503, 504})
GENERATION_FAILURE_MARKERS = (
    "tool_use_failed",
    "tool call validation failed",
    "failed to parse tool call",
    "could not be parsed",
)
ERROR_TEXT_LIMIT = 200


class ChatClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion: ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.base_delay < 0 or self.max_delay < self.base_delay:
            raise ValueError("invalid retry policy")

    def delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        return min(self.max_delay, self.base_delay * 2 ** (attempt - 1))


DEFAULT_RETRY_POLICY = RetryPolicy()


def parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def provider_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:ERROR_TEXT_LIMIT]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return response.text[:ERROR_TEXT_LIMIT]


def is_generation_failure(response: httpx.Response) -> bool:
    body = response.text.lower()
    return any(marker in body for marker in GENERATION_FAILURE_MARKERS)


class OpenAiCompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        http: httpx.Client,
        retry: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http
        self._retry = retry
        self._sleep = sleep

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        body = build_request_body(self.model, messages, tools, temperature, max_tokens)
        for attempt in range(1, self._retry.max_attempts):
            try:
                return self._send(body)
            except TransientLlmError as failure:
                delay = self._retry.delay(attempt, failure.retry_after)
                logger.warning(
                    "attempt %d of %d failed, retrying in %.1fs: %s",
                    attempt,
                    self._retry.max_attempts,
                    delay,
                    failure,
                )
                self._sleep(delay)
        return self._send(body)

    def _send(self, body: dict[str, object]) -> Completion:
        try:
            response = self._http.post(self._url, json=body, headers=self._headers)
        except httpx.TransportError as error:
            raise LlmUnavailable(
                f"{self.model}: {type(error).__name__} while contacting the provider"
            ) from error
        if not response.is_success:
            raise self._error_for(response)
        try:
            payload = response.json()
        except ValueError as error:
            raise LlmMalformedResponse(f"{self.model}: response body is not JSON") from error
        return parse_completion(payload)

    def _error_for(self, response: httpx.Response) -> Exception:
        status = response.status_code
        detail = provider_message(response)
        retry_after = parse_retry_after(response)
        if status == RATE_LIMITED_STATUS:
            return LlmRateLimited(f"{self.model}: rate limited: {detail}", retry_after)
        if status in UNAVAILABLE_STATUSES:
            return LlmUnavailable(f"{self.model}: HTTP {status}: {detail}", retry_after)
        if status == BAD_REQUEST_STATUS and is_generation_failure(response):
            return LlmBadGeneration(f"{self.model}: broken tool call: {detail}", retry_after)
        return LlmRejected(f"{self.model}: HTTP {status}: {detail}")
