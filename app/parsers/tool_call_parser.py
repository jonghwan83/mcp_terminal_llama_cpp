"""Fallback parsing for LLM-emitted tool calls."""

import json
import re
from typing import Any

DEFAULT_FIRST_PARAM = {
    "bash_exec": "command",
    "read_file": "path",
    "write_file": "path",
    "list_dir": "path",
    "find_files": "pattern",
    "search_code": "query",
    "replace_in_file": "path",
}


def try_parse_tool_json(data: dict[str, Any], valid_tools: set[str]) -> dict | None:
    """Extract {name, arguments} from a JSON object if it matches a known tool."""
    name = data.get("name") or data.get("function") or data.get("tool")
    if not isinstance(name, str) or name not in valid_tools:
        return None
    args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
    if not isinstance(args, dict):
        try:
            args = json.loads(args)
        except (TypeError, json.JSONDecodeError):
            args = {}
    return {"name": name, "arguments": args}


def extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects from text using bracket counting."""
    results = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start : i + 1])
                start = -1
    return results


def parse_text_tool_calls(
    content: str,
    valid_tools: set[str],
    first_param: dict[str, str] | None = None,
) -> list[dict]:
    """Fallback parser for models that emit tool calls as plain text.

    Tried in order:
    0. <tool_name>{...}</tool_name>
    1. <tool_call>{"name":...,"arguments":...}</tool_call>
    2. <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
    3. Action: NAME / Action Input: {...}
    4. Bare JSON object with a known tool name
    """
    first_param = first_param or DEFAULT_FIRST_PARAM
    results: list[dict] = []

    for name in valid_tools:
        for match in re.finditer(rf"<{name}>(.*?)</{name}>", content, re.DOTALL):
            body = match.group(1).strip()
            if body.startswith("{{") and body.endswith("}}"):
                body = body[1:-1]
            if body.startswith("{"):
                try:
                    data = json.loads(body)
                    if "arguments" in data:
                        args = data["arguments"]
                    elif "parameters" in data:
                        args = data["parameters"]
                    elif "args" in data:
                        args = data["args"]
                    else:
                        args = {k: v for k, v in data.items() if k not in ("name", "function", "tool")}
                    if not args and "name" in data and data["name"] not in valid_tools:
                        args = {first_param[name]: data["name"]}
                    results.append({"name": name, "arguments": args})
                    continue
                except json.JSONDecodeError:
                    pass
            if body and name in first_param:
                results.append({"name": name, "arguments": {first_param[name]: body}})

    if results:
        return results

    for tc_match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        body = tc_match.group(1).strip()

        if body.startswith("{"):
            if body.startswith("{{") and body.endswith("}}"):
                body = body[1:-1]
            try:
                parsed = try_parse_tool_json(json.loads(body), valid_tools)
                if parsed:
                    results.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass

        fn_match = re.search(r"<function=(\w+)>(.*?)</function>", body, re.DOTALL)
        if fn_match and fn_match.group(1) in valid_tools:
            args: dict[str, str] = {}
            for param_match in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", fn_match.group(2), re.DOTALL
            ):
                args[param_match.group(1)] = param_match.group(2)
            results.append({"name": fn_match.group(1), "arguments": args})

    if results:
        return results

    for match in re.finditer(
        r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", content, re.DOTALL
    ):
        if match.group(1) not in valid_tools:
            continue
        try:
            results.append({"name": match.group(1), "arguments": json.loads(match.group(2))})
        except json.JSONDecodeError:
            pass

    if results:
        return results

    for candidate in extract_json_objects(content):
        try:
            parsed = try_parse_tool_json(json.loads(candidate), valid_tools)
            if parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            pass

    return results