from pathlib import Path

import pytest

from fakes import FakeClient, make_completion
from investigation_engine.llm.cache import (
    CachedChatClient,
    InMemoryResponseCache,
    SqliteResponseCache,
    cache_key,
)
from investigation_engine.llm.errors import LlmUnavailable
from investigation_engine.llm.types import Completion, Message, ToolCall, ToolSpec, Usage

QUESTION = [Message("user", "Capital of France?")]
SEARCH_TOOL = ToolSpec("search", "Search the corpus", {"type": "object", "properties": {}})


def key_for(
    model: str = "m1",
    messages: list[Message] | None = None,
    tools: list[ToolSpec] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    return cache_key(model, messages or QUESTION, tools or [], temperature, max_tokens)


def tool_call_completion() -> Completion:
    call = ToolCall("c1", "search", {"query": "payments", "limit": 3})
    return Completion(
        message=Message("assistant", None, tool_calls=(call,)),
        usage=Usage(7, 3),
        model="m1",
        finish_reason="tool_calls",
    )


def test_cache_key_is_stable_for_identical_requests() -> None:
    assert key_for() == key_for()


CHANGED_KEYS = {
    "model": key_for(model="m2"),
    "messages": key_for(messages=[Message("user", "Capital of Spain?")]),
    "tools": key_for(tools=[SEARCH_TOOL]),
    "temperature": key_for(temperature=0.7),
    "max_tokens": key_for(max_tokens=50),
}


@pytest.mark.parametrize("changed_part", CHANGED_KEYS)
def test_cache_key_changes_when_any_request_part_changes(changed_part: str) -> None:
    assert CHANGED_KEYS[changed_part] != key_for()


def test_cache_key_ignores_the_order_of_tool_call_arguments() -> None:
    first = Message("assistant", None, tool_calls=(ToolCall("c1", "s", {"a": 1, "b": 2}),))
    second = Message("assistant", None, tool_calls=(ToolCall("c1", "s", {"b": 2, "a": 1}),))

    assert key_for(messages=[first]) == key_for(messages=[second])


def test_in_memory_cache_returns_what_was_stored() -> None:
    cache = InMemoryResponseCache()
    completion = make_completion()

    assert cache.get("k") is None
    cache.put("k", completion)

    assert cache.get("k") == completion


def test_sqlite_cache_survives_being_reopened(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    first = SqliteResponseCache(path)
    first.put("k", tool_call_completion())
    first.close()

    second = SqliteResponseCache(path)

    assert second.get("k") == tool_call_completion()
    second.close()


def test_sqlite_cache_miss_returns_nothing_and_put_replaces_older_values() -> None:
    cache = SqliteResponseCache(":memory:")

    assert cache.get("k") is None
    cache.put("k", make_completion("old"))
    cache.put("k", make_completion("new"))

    assert cache.get("k") == make_completion("new")
    cache.close()


def test_cached_client_serves_a_repeated_request_without_calling_the_provider() -> None:
    inner = FakeClient("m1", make_completion("Paris"))
    client = CachedChatClient(inner, InMemoryResponseCache())

    first = client.complete(QUESTION)
    second = client.complete(QUESTION)

    assert first == second
    assert len(inner.calls) == 1
    assert (client.hits, client.misses) == (1, 1)


def test_cached_client_calls_the_provider_again_for_a_different_request() -> None:
    inner = FakeClient("m1", make_completion("Paris"), make_completion("Madrid"))
    client = CachedChatClient(inner, InMemoryResponseCache())

    client.complete(QUESTION)
    other = client.complete([Message("user", "Capital of Spain?")])

    assert other.text == "Madrid"
    assert len(inner.calls) == 2


def test_cached_client_forwards_every_request_parameter() -> None:
    inner = FakeClient("m1", make_completion())

    CachedChatClient(inner, InMemoryResponseCache()).complete(
        QUESTION, tools=[SEARCH_TOOL], temperature=0.3, max_tokens=40
    )

    messages, tools, temperature, max_tokens = inner.calls[0]
    assert (list(messages), list(tools), temperature, max_tokens) == (
        QUESTION,
        [SEARCH_TOOL],
        0.3,
        40,
    )


def test_cached_client_does_not_cache_failures() -> None:
    inner = FakeClient("m1", LlmUnavailable("down"), make_completion("Paris"))
    client = CachedChatClient(inner, InMemoryResponseCache())

    with pytest.raises(LlmUnavailable):
        client.complete(QUESTION)
    recovered = client.complete(QUESTION)
    again = client.complete(QUESTION)

    assert recovered == again
    assert len(inner.calls) == 2


def test_cached_client_mirrors_the_model_name_of_the_wrapped_client() -> None:
    assert CachedChatClient(FakeClient("gemini-x"), InMemoryResponseCache()).model == "gemini-x"


def test_cached_client_answers_from_disk_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    before_restart = SqliteResponseCache(path)
    CachedChatClient(FakeClient("m1", make_completion("Paris")), before_restart).complete(QUESTION)
    before_restart.close()
    after_restart = SqliteResponseCache(path)
    offline = FakeClient("m1")

    answer = CachedChatClient(offline, after_restart).complete(QUESTION)

    assert answer.text == "Paris"
    assert offline.calls == []
    after_restart.close()
