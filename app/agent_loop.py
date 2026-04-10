"""Shared agent loop orchestration for terminal and MCP modes."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

LoopStatus = Literal["final", "denied", "max_iterations"]


@dataclass
class AgentLoopOutcome:
    status: LoopStatus
    final_text: str | None = None


@dataclass
class _StreamedFunction:
    name: str
    arguments: str


@dataclass
class _StreamedToolCall:
    id: str
    function: _StreamedFunction


@dataclass
class _StreamedMessage:
    content: str
    tool_calls: list[_StreamedToolCall]


@dataclass
class _StreamedChoice:
    message: _StreamedMessage


@dataclass
class _StreamedResponse:
    choices: list[_StreamedChoice]


def _consume_streaming_completion(stream: Any, on_text_delta: Callable[[str], None]) -> _StreamedResponse:
    content_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            on_text_delta(content)

        for tool_delta in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_delta, "index", 0) or 0
            state = tool_call_parts.setdefault(index, {"id": None, "name": None, "arguments_parts": []})

            tool_call_id = getattr(tool_delta, "id", None)
            if tool_call_id:
                state["id"] = tool_call_id

            function_delta = getattr(tool_delta, "function", None)
            if function_delta is None:
                continue

            function_name = getattr(function_delta, "name", None)
            if function_name:
                state["name"] = function_name

            function_arguments = getattr(function_delta, "arguments", None)
            if function_arguments:
                state["arguments_parts"].append(function_arguments)

    tool_calls = [
        _StreamedToolCall(
            id=state["id"] or f"call_{index}",
            function=_StreamedFunction(
                name=state["name"] or "",
                arguments="".join(state["arguments_parts"]),
            ),
        )
        for index, state in sorted(tool_call_parts.items())
        if state["name"]
    ]

    return _StreamedResponse(
        choices=[
            _StreamedChoice(
                message=_StreamedMessage(
                    content="".join(content_parts),
                    tool_calls=tool_calls,
                )
            )
        ]
    )

def run_agent_loop(
    task: str,
    llm_call: Callable[..., Any],
    messages: list[dict],
    tools: list[dict],
    execute_tool: Callable[[str, dict[str, Any]], str],
    parse_text_tool_calls: Callable[[str, set[str], dict[str, str]], list[dict]],
    valid_tools: set[str],
    first_param: dict[str, str],
    max_iterations: int,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    compress_messages: Callable[[list[dict]], list[dict]] | None = None,
) -> AgentLoopOutcome:
    def emit(event_name: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event(event_name, payload)

    messages.append({"role": "user", "content": task})

    for _ in range(max_iterations):
        if compress_messages:
            messages = compress_messages(messages)

        if on_text_delta:
            response = _consume_streaming_completion(
                llm_call(messages, tools, tool_choice="auto", stream=True),
                on_text_delta,
            )
        else:
            response = llm_call(messages, tools, tool_choice="auto")
        msg = response.choices[0].message

        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in msg.tool_calls
            ]
        messages.append(entry)

        if not msg.tool_calls:
            text_calls = parse_text_tool_calls(msg.content or "", valid_tools, first_param)
            if not text_calls:
                final_text = (msg.content or "").strip()
                if final_text:
                    emit("final_text", {"text": final_text})
                return AgentLoopOutcome("final", final_text or None)

            result_parts: list[str] = []
            for text_call in text_calls:
                name, args = text_call["name"], text_call["arguments"]
                emit("tool_call", {"name": name, "args": args, "source": "text"})
                result = execute_tool(name, args)
                emit("tool_result", {"name": name, "result": result, "source": "text"})

                if result == "Operation denied by user.":
                    emit("denied", {"name": name, "args": args, "source": "text"})
                    return AgentLoopOutcome("denied")

                result_parts.append(f"[{name}]\n{result}")

            messages.append({
                "role": "user",
                "content": "Tool results:\n" + "\n---\n".join(result_parts),
            })
            continue

        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            emit("tool_call", {"name": tool_call.function.name, "args": args, "source": "structured"})
            result = execute_tool(tool_call.function.name, args)
            emit(
                "tool_result",
                {"name": tool_call.function.name, "result": result, "source": "structured"},
            )

            if result == "Operation denied by user.":
                emit("denied", {"name": tool_call.function.name, "args": args, "source": "structured"})
                return AgentLoopOutcome("denied")

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    emit("max_iterations", {})
    return AgentLoopOutcome("max_iterations")