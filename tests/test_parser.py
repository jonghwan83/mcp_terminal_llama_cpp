from app.parsers.tool_call_parser import DEFAULT_FIRST_PARAM, parse_text_tool_calls


VALID_TOOLS = set(DEFAULT_FIRST_PARAM.keys())


def test_parse_tool_name_tag_with_json_body() -> None:
    content = '<bash_exec>{"command":"pwd"}</bash_exec>'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == [{"name": "bash_exec", "arguments": {"command": "pwd"}}]


def test_parse_tool_call_json_wrapper() -> None:
    content = '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == [{"name": "read_file", "arguments": {"path": "README.md"}}]


def test_parse_react_action_input_format() -> None:
    content = 'Action: list_dir\nAction Input: {"path": "."}'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == [{"name": "list_dir", "arguments": {"path": "."}}]


def test_parse_bare_json_object() -> None:
    content = 'I will call this now: {"name":"search_code","arguments":{"query":"ToolExecutor"}}'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == [{"name": "search_code", "arguments": {"query": "ToolExecutor"}}]


def test_parse_plain_text_body_maps_to_first_param() -> None:
    content = '<read_file>README.md</read_file>'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == [{"name": "read_file", "arguments": {"path": "README.md"}}]


def test_parse_unknown_tool_returns_empty_list() -> None:
    content = '<unknown_tool>{"foo":"bar"}</unknown_tool>'
    calls = parse_text_tool_calls(content, VALID_TOOLS)
    assert calls == []
