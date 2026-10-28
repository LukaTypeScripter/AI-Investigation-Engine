from collections.abc import Mapping

import pytest

from investigation_engine.llm.errors import LlmMalformedResponse
from investigation_engine.llm.types import Message, ToolCall, ToolSpec, Usage
from investigation_engine.llm.wire import build_request_body, parse_completion


def completion_payload(
    message: Mapping[str, object], finish_reason: str = "stop"
) -> dict[str, object]:
    return {
        "model": "m1",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }


def test_request_body_minimal_contains_model_messages_and_temperature() -> None:
    body = build_request_body("m1", [Message("user", "hi")], [], 0.0, None)

    assert body == {
        "model": "m1",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }


def test_request_body_includes_tools_and_token_limit_when_given() -> None:
    tool = ToolSpec(
        name="search",
        description="Search the corpus",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    body = build_request_body("m1", [Message("user", "hi")], [tool], 0.2, 300)

    assert body["max_tokens"] == 300
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the corpus",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]


def test_request_body_echoes_assistant_tool_calls_and_tool_results() -> None:
    call = ToolCall(id="c1", name="search", arguments={"query": "payments", "limit": 3})
    messages = [
        Message("assistant", None, tool_calls=(call,)),
        Message("tool", "found 2 chunks", tool_call_id="c1"),
    ]

    body = build_request_body("m1", messages, [], 0.0, None)

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"limit": 3, "query": "payments"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "found 2 chunks", "tool_call_id": "c1"},
    ]


def test_parse_completion_reads_text_usage_model_and_finish_reason() -> None:
    completion = parse_completion(completion_payload({"role": "assistant", "content": "Paris"}))

    assert completion.text == "Paris"
    assert completion.usage == Usage(prompt_tokens=11, completion_tokens=4)
    assert completion.usage.total_tokens == 15
    assert completion.model == "m1"
    assert completion.finish_reason == "stop"
    assert completion.tool_calls == ()


def test_parse_completion_decodes_tool_call_arguments() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "payments"}'},
            }
        ],
    }

    completion = parse_completion(completion_payload(message, "tool_calls"))

    assert completion.text == ""
    assert completion.tool_calls == (ToolCall("c1", "search", {"query": "payments"}),)
    assert completion.finish_reason == "tool_calls"


def test_parse_completion_invents_an_id_for_tool_calls_without_one() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
    }

    completion = parse_completion(completion_payload(message))

    assert completion.tool_calls[0].id == "call_0"


def test_parse_completion_without_usage_reports_zero_tokens() -> None:
    payload = completion_payload({"role": "assistant", "content": "x"})
    del payload["usage"]

    assert parse_completion(payload).usage == Usage(0, 0)


def test_parse_completion_rejects_tool_arguments_that_are_not_json() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{oops"}}],
    }

    with pytest.raises(LlmMalformedResponse, match="arguments"):
        parse_completion(completion_payload(message))


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}, [], "text", None])
def test_parse_completion_rejects_responses_without_a_message(payload: object) -> None:
    with pytest.raises(LlmMalformedResponse):
        parse_completion(payload)
