import json
from collections.abc import Mapping, Sequence

from investigation_engine.llm.errors import LlmMalformedResponse
from investigation_engine.llm.types import Completion, Message, ToolCall, ToolSpec, Usage


def tool_call_to_wire(call: ToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments, sort_keys=True)},
    }


def message_to_wire(message: Message) -> dict[str, object]:
    wire: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [tool_call_to_wire(call) for call in message.tool_calls]
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def tool_to_wire(tool: ToolSpec) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def build_request_body(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    temperature: float,
    max_tokens: int | None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": [message_to_wire(message) for message in messages],
        "temperature": temperature,
    }
    if tools:
        body["tools"] = [tool_to_wire(tool) for tool in tools]
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def require_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LlmMalformedResponse(f"expected an object at {where}")
    return value


def parse_arguments(raw: object, tool_name: str) -> dict[str, object]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LlmMalformedResponse(
                f"arguments of tool call '{tool_name}' are not valid JSON"
            ) from error
        if isinstance(decoded, dict):
            return decoded
    raise LlmMalformedResponse(f"arguments of tool call '{tool_name}' must be a JSON object")


def parse_tool_calls(raw: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        entry = require_mapping(item, f"tool_calls[{index}]")
        function = require_mapping(entry.get("function"), f"tool_calls[{index}].function")
        name = function.get("name")
        if not isinstance(name, str):
            raise LlmMalformedResponse(f"tool_calls[{index}] has no function name")
        call_id = entry.get("id")
        calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                name=name,
                arguments=parse_arguments(function.get("arguments"), name),
            )
        )
    return tuple(calls)


def as_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def parse_usage(raw: object) -> Usage:
    if not isinstance(raw, Mapping):
        return Usage()
    return Usage(
        prompt_tokens=as_count(raw.get("prompt_tokens")),
        completion_tokens=as_count(raw.get("completion_tokens")),
    )


def parse_completion(payload: object) -> Completion:
    root = require_mapping(payload, "the response")
    choices = root.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmMalformedResponse("the response has no choices")
    choice = require_mapping(choices[0], "choices[0]")
    raw_message = require_mapping(choice.get("message"), "choices[0].message")
    content = raw_message.get("content")
    message = Message(
        role="assistant",
        content=content if isinstance(content, str) else None,
        tool_calls=parse_tool_calls(raw_message.get("tool_calls")),
    )
    return Completion(
        message=message,
        usage=parse_usage(root.get("usage")),
        model=str(root.get("model") or ""),
        finish_reason=str(choice.get("finish_reason") or ""),
    )


def completion_to_wire(completion: Completion) -> dict[str, object]:
    return {
        "model": completion.model,
        "choices": [
            {
                "message": message_to_wire(completion.message),
                "finish_reason": completion.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        },
    }
