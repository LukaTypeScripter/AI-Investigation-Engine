import pytest

from fakes import FakeClient, make_completion
from investigation_engine.llm.errors import (
    LlmMalformedResponse,
    LlmRateLimited,
    LlmRejected,
    LlmUnavailable,
)
from investigation_engine.llm.fallback import FallbackChatClient
from investigation_engine.llm.types import Message

QUESTION = [Message("user", "Capital of France?")]


def test_fallback_uses_the_first_client_when_it_answers() -> None:
    primary = FakeClient("gemini", make_completion("Paris", "gemini"))
    secondary = FakeClient("groq", make_completion("Paris", "groq"))

    answer = FallbackChatClient([primary, secondary]).complete(QUESTION)

    assert answer.model == "gemini"
    assert secondary.calls == []


@pytest.mark.parametrize(
    "failure",
    [LlmRateLimited("slow down"), LlmUnavailable("down"), LlmMalformedResponse("garbled")],
)
def test_fallback_moves_to_the_next_client_on_recoverable_failures(failure: Exception) -> None:
    primary = FakeClient("gemini", failure)
    secondary = FakeClient("groq", make_completion("Paris", "groq"))
    client = FallbackChatClient([primary, secondary])

    answer = client.complete(QUESTION)

    assert answer.model == "groq"
    assert client.fallbacks == 1


def test_fallback_forwards_every_request_parameter_to_the_next_client() -> None:
    primary = FakeClient("gemini", LlmUnavailable("down"))
    secondary = FakeClient("groq", make_completion())

    FallbackChatClient([primary, secondary]).complete(QUESTION, temperature=0.4, max_tokens=30)

    _, _, temperature, max_tokens = secondary.calls[0]
    assert (temperature, max_tokens) == (0.4, 30)


def test_fallback_does_not_hide_configuration_errors() -> None:
    primary = FakeClient("gemini", LlmRejected("Invalid API key"))
    secondary = FakeClient("groq", make_completion())

    with pytest.raises(LlmRejected, match="Invalid API key"):
        FallbackChatClient([primary, secondary]).complete(QUESTION)

    assert secondary.calls == []


def test_fallback_reports_every_failure_when_all_clients_fail() -> None:
    last = LlmRateLimited("groq: rate limited")
    primary = FakeClient("gemini", LlmUnavailable("gemini: HTTP 503"))
    secondary = FakeClient("groq", last)

    with pytest.raises(LlmUnavailable) as raised:
        FallbackChatClient([primary, secondary]).complete(QUESTION)

    assert "gemini: HTTP 503" in str(raised.value)
    assert "groq: rate limited" in str(raised.value)
    assert raised.value.__cause__ is last


def test_fallback_counts_only_requests_that_left_the_first_client() -> None:
    primary = FakeClient("gemini", make_completion(), LlmUnavailable("down"))
    secondary = FakeClient("groq", make_completion("Paris", "groq"))
    client = FallbackChatClient([primary, secondary])

    client.complete(QUESTION)
    client.complete(QUESTION)

    assert client.fallbacks == 1


def test_fallback_reports_the_name_of_the_first_client() -> None:
    assert FallbackChatClient([FakeClient("gemini"), FakeClient("groq")]).model == "gemini"


def test_fallback_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackChatClient([])
