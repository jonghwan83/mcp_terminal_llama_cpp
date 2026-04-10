import json
from dataclasses import dataclass
from typing import Any

from app.agent_loop import run_agent_loop


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


@dataclass
class _FakeDeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _FakeDeltaToolCall:
    index: int = 0
    id: str | None = None
    function: _FakeDeltaFunction | None = None


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[_FakeDeltaToolCall] | None = None


@dataclass
class _FakeChunkChoice:
    delta: _FakeDelta


@dataclass
class _FakeChunk:
    choices: list[_FakeChunkChoice]


class _SeqLLM:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = messages
        self._index = 0

    def __call__(self, messages_to_send: list[dict], tools: list[dict], **kwargs: Any) -> Any:
        msg = self._messages[self._index]
        self._index += 1
        if kwargs.get("stream"):
            return iter(msg)  # type: ignore[arg-type]
        return _FakeResponse(choices=[_FakeChoice(message=msg)])


def test_agent_loop_returns_final_text_without_tools() -> None:
    llm = _SeqLLM([_FakeMessage(content="done", tool_calls=None)])
    messages = [{"role": "system", "content": "sys"}]
    events: list[tuple[str, dict]] = []

    outcome = run_agent_loop(
        task="hello",
        llm_call=llm,
        messages=messages,
        tools=[],
        execute_tool=lambda name, args: "unused",
        parse_text_tool_calls=lambda content, valid, first: [],
        valid_tools={"bash_exec"},
        first_param={"bash_exec": "command"},
        max_iterations=2,
        on_event=lambda event, payload: events.append((event, payload)),
    )

    assert outcome.status == "final"
    assert outcome.final_text == "done"
    assert events[-1] == ("final_text", {"text": "done"})


def test_agent_loop_executes_structured_tool_then_finishes() -> None:
    llm = _SeqLLM(
        [
            _FakeMessage(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        id="tc1",
                        function=_FakeFunction(
                            name="read_file",
                            arguments=json.dumps({"path": "README.md"}),
                        ),
                    )
                ],
            ),
            _FakeMessage(content="all done", tool_calls=None),
        ]
    )
    messages = [{"role": "system", "content": "sys"}]

    outcome = run_agent_loop(
        task="read",
        llm_call=llm,
        messages=messages,
        tools=[],
        execute_tool=lambda name, args: "file content",
        parse_text_tool_calls=lambda content, valid, first: [],
        valid_tools={"read_file"},
        first_param={"read_file": "path"},
        max_iterations=3,
    )

    assert outcome.status == "final"
    assert outcome.final_text == "all done"
    assert any(m.get("role") == "tool" for m in messages)


def test_agent_loop_stops_on_denied_tool_result() -> None:
    llm = _SeqLLM(
        [
            _FakeMessage(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        id="tc1",
                        function=_FakeFunction(name="bash_exec", arguments=json.dumps({"command": "pwd"})),
                    )
                ],
            )
        ]
    )

    outcome = run_agent_loop(
        task="run",
        llm_call=llm,
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        execute_tool=lambda name, args: "Operation denied by user.",
        parse_text_tool_calls=lambda content, valid, first: [],
        valid_tools={"bash_exec"},
        first_param={"bash_exec": "command"},
        max_iterations=2,
    )

    assert outcome.status == "denied"


def test_agent_loop_uses_text_fallback_calls() -> None:
    llm = _SeqLLM(
        [
            _FakeMessage(content="<read_file>README.md</read_file>", tool_calls=None),
            _FakeMessage(content="done", tool_calls=None),
        ]
    )

    outcome = run_agent_loop(
        task="read",
        llm_call=llm,
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        execute_tool=lambda name, args: "content",
        parse_text_tool_calls=lambda content, valid, first: [{"name": "read_file", "arguments": {"path": "README.md"}}] if "read_file" in content else [],
        valid_tools={"read_file"},
        first_param={"read_file": "path"},
        max_iterations=3,
    )

    assert outcome.status == "final"
    assert outcome.final_text == "done"


def test_agent_loop_streams_final_text_deltas() -> None:
    llm = _SeqLLM(
        [
            [
                _FakeChunk(choices=[_FakeChunkChoice(delta=_FakeDelta(content="hel"))]),
                _FakeChunk(choices=[_FakeChunkChoice(delta=_FakeDelta(content="lo"))]),
            ]
        ]
    )
    streamed: list[str] = []

    outcome = run_agent_loop(
        task="hello",
        llm_call=llm,
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        execute_tool=lambda name, args: "unused",
        parse_text_tool_calls=lambda content, valid, first: [],
        valid_tools={"bash_exec"},
        first_param={"bash_exec": "command"},
        max_iterations=2,
        on_text_delta=streamed.append,
    )

    assert outcome.status == "final"
    assert outcome.final_text == "hello"
    assert streamed == ["hel", "lo"]
