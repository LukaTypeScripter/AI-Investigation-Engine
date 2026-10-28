import json

import httpx
import pytest

from investigation_engine.llm.cache import InMemoryResponseCache
from investigation_engine.llm.client import RetryPolicy
from investigation_engine.llm.factory import (
    ConfigurationError,
    ProviderSettings,
    build_chat_client,
    build_http_client,
)
from investigation_engine.llm.fallback import FallbackChatClient
from investigation_engine.llm.types import Message
from investigation_engine.settings import Settings

QUESTION = [Message("user", "Capital of France?")]
NO_RETRY = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("GEMINI_API_KEY", "GROQ_API_KEY", "GEMINI_MODEL", "GROQ_MODEL"):
        monkeypatch.delenv(variable, raising=False)


class Provider:
    def __init__(self, *outcomes: int) -> None:
        self._statuses = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self._statuses.pop(0) if self._statuses else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "failed"}})
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [
                    {"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )


def client_for(provider: Provider) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(provider))


def hosts(provider: Provider) -> list[str]:
    return [request.url.host for request in provider.requests]


def test_without_any_key_no_client_can_be_built() -> None:
    with pytest.raises(ConfigurationError, match="no LLM provider"):
        build_chat_client(Settings(_env_file=None), client_for(Provider()), InMemoryResponseCache())


def test_a_single_configured_provider_is_used_with_its_own_url_model_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("GROQ_MODEL", "llama-test")
    provider = Provider()

    client = build_chat_client(
        Settings(_env_file=None), client_for(provider), InMemoryResponseCache()
    )
    completion = client.complete(QUESTION)

    request = provider.requests[0]
    assert completion.text == "Paris"
    assert client.model == "llama-test"
    assert request.url.host == "api.groq.com"
    assert request.headers["authorization"] == "Bearer groq-secret"


def test_with_two_providers_the_second_takes_over_when_the_first_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    provider = Provider(429)

    client = build_chat_client(
        Settings(_env_file=None), client_for(provider), InMemoryResponseCache(), retry=NO_RETRY
    )
    completion = client.complete(QUESTION)

    assert isinstance(client, FallbackChatClient)
    assert completion.text == "Paris"
    assert hosts(provider) == ["generativelanguage.googleapis.com", "api.groq.com"]


def test_a_named_provider_is_used_alone_even_when_both_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    provider = Provider(429)

    client = build_chat_client(
        Settings(_env_file=None),
        client_for(provider),
        InMemoryResponseCache(),
        provider="groq",
        retry=NO_RETRY,
    )

    assert not isinstance(client, FallbackChatClient)
    with pytest.raises(Exception, match="rate limited"):
        client.complete(QUESTION)
    assert hosts(provider) == ["api.groq.com"]


def test_a_named_provider_without_a_key_names_the_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_chat_client(
            Settings(_env_file=None),
            client_for(Provider()),
            InMemoryResponseCache(),
            provider="gemini",
        )


def test_an_unknown_provider_name_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown provider"):
        build_chat_client(
            Settings(_env_file=None),
            client_for(Provider()),
            InMemoryResponseCache(),
            provider="openai",
        )


def test_repeated_requests_are_answered_from_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    provider = Provider()
    client = build_chat_client(
        Settings(_env_file=None), client_for(provider), InMemoryResponseCache()
    )

    first = client.complete(QUESTION)
    second = client.complete(QUESTION)

    assert first == second
    assert len(provider.requests) == 1


def test_the_http_client_uses_the_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    client = build_http_client(Settings(_env_file=None))

    assert client.timeout.read == 30.0
    client.close()


def test_provider_settings_never_show_the_key_in_their_text_form() -> None:
    shown = repr(ProviderSettings("groq", "https://x.test", "super-secret-key", "model"))

    assert "super-secret-key" not in shown
