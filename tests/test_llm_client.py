import json

import httpx
import pytest

from investigation_engine.llm.client import OpenAiCompatibleClient, RetryPolicy
from investigation_engine.llm.errors import (
    LlmBadGeneration,
    LlmMalformedResponse,
    LlmRateLimited,
    LlmRejected,
    LlmUnavailable,
)
from investigation_engine.llm.types import Message, ToolSpec

USER_QUESTION = [Message("user", "Capital of France?")]


class ScriptedTransport:
    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok_response(content: str = "Paris") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m1",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )


def make_client(
    transport: ScriptedTransport,
    sleeps: list[float] | None = None,
    retry: RetryPolicy | None = None,
) -> OpenAiCompatibleClient:
    recorded = sleeps if sleeps is not None else []
    return OpenAiCompatibleClient(
        base_url="https://llm.test/v1/",
        api_key="secret-key",
        model="m1",
        http=httpx.Client(transport=httpx.MockTransport(transport)),
        retry=retry or RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=10.0),
        sleep=recorded.append,
    )


def test_complete_posts_to_chat_completions_with_bearer_authentication() -> None:
    transport = ScriptedTransport(ok_response("Paris"))

    completion = make_client(transport).complete(USER_QUESTION)

    request = transport.requests[0]
    body = json.loads(request.content)
    assert completion.text == "Paris"
    assert completion.usage.total_tokens == 7
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    assert body["model"] == "m1"
    assert body["messages"] == [{"role": "user", "content": "Capital of France?"}]
    assert body["temperature"] == 0.0


def test_complete_sends_tools_and_token_limit() -> None:
    transport = ScriptedTransport(ok_response())
    tool = ToolSpec("search", "Search the corpus", {"type": "object", "properties": {}})

    make_client(transport).complete(USER_QUESTION, tools=[tool], max_tokens=64)

    body = json.loads(transport.requests[0].content)
    assert body["max_tokens"] == 64
    assert body["tools"][0]["function"]["name"] == "search"


def test_complete_returns_tool_calls_requested_by_the_model() -> None:
    tool_call_response = httpx.Response(
        200,
        json={
            "model": "m1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query": "x"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )

    completion = make_client(ScriptedTransport(tool_call_response)).complete(USER_QUESTION)

    assert [(call.name, call.arguments) for call in completion.tool_calls] == [
        ("search", {"query": "x"})
    ]


def test_complete_waits_for_the_retry_after_header_when_rate_limited() -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}),
        ok_response(),
    )

    completion = make_client(transport, sleeps).complete(USER_QUESTION)

    assert completion.text == "Paris"
    assert sleeps == [2.0]
    assert len(transport.requests) == 2


def test_complete_logs_each_retry_with_the_delay_and_the_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = ScriptedTransport(
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}),
        ok_response(),
    )

    make_client(transport).complete(USER_QUESTION)

    assert "attempt 1 of 3 failed, retrying in 2.0s" in caplog.text
    assert "slow down" in caplog.text


def test_complete_backs_off_exponentially_on_server_errors() -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(httpx.Response(503), httpx.Response(502), ok_response())

    make_client(transport, sleeps).complete(USER_QUESTION)

    assert sleeps == [1.0, 2.0]


def test_complete_gives_up_after_the_maximum_number_of_attempts() -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(httpx.Response(500), httpx.Response(500), httpx.Response(500))

    with pytest.raises(LlmUnavailable, match="HTTP 500"):
        make_client(transport, sleeps).complete(USER_QUESTION)

    assert len(transport.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_complete_reports_persistent_rate_limiting_with_the_advised_delay() -> None:
    throttled = [httpx.Response(429, headers={"retry-after": "2"}) for _ in range(3)]

    with pytest.raises(LlmRateLimited) as raised:
        make_client(ScriptedTransport(*throttled)).complete(USER_QUESTION)

    assert raised.value.retry_after == 2.0


def test_complete_never_waits_longer_than_the_maximum_delay() -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(
        httpx.Response(429, headers={"retry-after": "120"}), ok_response()
    )

    make_client(transport, sleeps).complete(USER_QUESTION)

    assert sleeps == [10.0]


def test_complete_does_not_retry_authentication_failures() -> None:
    transport = ScriptedTransport(
        httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )

    with pytest.raises(LlmRejected, match="Invalid API key") as raised:
        make_client(transport).complete(USER_QUESTION)

    assert len(transport.requests) == 1
    assert "secret-key" not in str(raised.value)


def test_complete_does_not_retry_invalid_requests() -> None:
    transport = ScriptedTransport(httpx.Response(400, json={"error": {"message": "bad tools"}}))

    with pytest.raises(LlmRejected, match="bad tools"):
        make_client(transport).complete(USER_QUESTION)

    assert len(transport.requests) == 1


TOOL_CALL_FAILURES = [
    "Tool call validation failed: attempted to call tool 'open' which was not in request.tools",
    "Failed to parse tool call arguments as JSON",
    "Parsing failed. The model generated output that could not be parsed.",
]


@pytest.mark.parametrize("message", TOOL_CALL_FAILURES)
def test_complete_retries_when_the_model_generates_a_broken_tool_call(message: str) -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(
        httpx.Response(400, json={"error": {"message": message}}), ok_response()
    )

    completion = make_client(transport, sleeps).complete(USER_QUESTION)

    assert completion.text == "Paris"
    assert len(transport.requests) == 2
    assert sleeps == [1.0]


def test_complete_reports_a_broken_tool_call_that_keeps_happening() -> None:
    broken = [
        httpx.Response(400, json={"error": {"message": TOOL_CALL_FAILURES[1]}}) for _ in range(3)
    ]
    transport = ScriptedTransport(*broken)

    with pytest.raises(LlmBadGeneration, match="Failed to parse tool call"):
        make_client(transport).complete(USER_QUESTION)

    assert len(transport.requests) == 3


def test_complete_retries_connection_failures() -> None:
    sleeps: list[float] = []
    transport = ScriptedTransport(httpx.ConnectError("boom"), ok_response())

    completion = make_client(transport, sleeps).complete(USER_QUESTION)

    assert completion.text == "Paris"
    assert sleeps == [1.0]


def test_complete_reports_timeouts_as_unavailable_after_retrying() -> None:
    timeouts = [httpx.ReadTimeout("slow") for _ in range(3)]

    with pytest.raises(LlmUnavailable, match="ReadTimeout"):
        make_client(ScriptedTransport(*timeouts)).complete(USER_QUESTION)


def test_complete_includes_the_response_text_when_the_error_body_is_not_json() -> None:
    failures = [httpx.Response(500, text="upstream exploded") for _ in range(3)]

    with pytest.raises(LlmUnavailable, match="upstream exploded"):
        make_client(ScriptedTransport(*failures)).complete(USER_QUESTION)


def test_complete_rejects_a_successful_response_that_is_not_json_without_retrying() -> None:
    transport = ScriptedTransport(httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(LlmMalformedResponse):
        make_client(transport).complete(USER_QUESTION)

    assert len(transport.requests) == 1


def test_client_exposes_its_model_name() -> None:
    assert make_client(ScriptedTransport()).model == "m1"


def test_retry_policy_doubles_the_delay_up_to_the_maximum() -> None:
    policy = RetryPolicy(max_attempts=6, base_delay=1.0, max_delay=5.0)

    assert [policy.delay(attempt, None) for attempt in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


@pytest.mark.parametrize(
    ("max_attempts", "base_delay", "max_delay"), [(0, 1.0, 5.0), (3, -1.0, 5.0), (3, 2.0, 1.0)]
)
def test_retry_policy_rejects_invalid_values(
    max_attempts: int, base_delay: float, max_delay: float
) -> None:
    with pytest.raises(ValueError, match="retry"):
        RetryPolicy(max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay)
