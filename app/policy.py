"""Policy and permission rules for tool execution."""

from typing import Any, Literal

PolicyDecision = Literal["allow", "deny", "require_confirm"]
ToolRisk = Literal["low", "high", "unknown"]

LOW_RISK_TOOLS = {"read_file", "list_dir", "find_files", "search_code"}
HIGH_RISK_TOOLS = {"bash_exec", "write_file", "replace_in_file"}
KNOWN_TOOLS = LOW_RISK_TOOLS | HIGH_RISK_TOOLS


def get_tool_risk(tool_name: str) -> ToolRisk:
    if tool_name in LOW_RISK_TOOLS:
        return "low"
    if tool_name in HIGH_RISK_TOOLS:
        return "high"
    return "unknown"


def describe_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "bash_exec":
        return f"Command: {args.get('command', '')}"
    if tool_name == "read_file":
        return f"File: {args.get('path', '')}"
    if tool_name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        content_preview = content[:50] + ("…" if len(content) > 50 else "")
        return f"File: {path}\nContent: {content_preview}"
    if tool_name == "list_dir":
        return f"Directory: {args.get('path', '.')}"
    if tool_name == "find_files":
        return f"Path: {args.get('path', '.')}\nPattern: {args.get('pattern', '*.py')}"
    if tool_name == "search_code":
        return (
            f"Path: {args.get('path', '.')}\n"
            f"Query: {args.get('query', '')}\n"
            f"Regex: {args.get('is_regex', False)}"
        )
    if tool_name == "replace_in_file":
        return f"File: {args.get('path', '')}\nReplace text in first occurrence"
    return str(args)


def evaluate_tool_policy(tool_name: str, ask_permission: bool) -> PolicyDecision:
    if tool_name not in KNOWN_TOOLS:
        return "deny"
    if not ask_permission:
        return "allow"
    if get_tool_risk(tool_name) == "high":
        return "require_confirm"
    return "allow"


def build_permission_request(tool_name: str, args: dict[str, Any]) -> str:
    details = describe_tool_args(tool_name, args)
    return (
        f"⚠️  Permission Required\n\n"
        f"Tool: {tool_name}\n"
        f"{details}\n\n"
        f"This operation requires approval. Reply with 'approved' to proceed."
    )
