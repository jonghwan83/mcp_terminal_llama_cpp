"""Validation helpers for tool arguments and paths."""

import re
from pathlib import Path
from typing import Any

MAX_SEARCH_RESULTS = 1000
_DANGEROUS_COMMAND_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\brm\s+-fr\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\s*\)\s*\{",
    r"\bformat\s+",
)


def format_validation_error(tool_name: str, field: str, message: str) -> str:
    return f"ValidationError(tool={tool_name}, field={field}, message={message})"


def is_within_workspace(path: Path, workspace_root: Path) -> bool:
    try:
        path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False


def validate_shell_command(command: str) -> tuple[bool, str | None]:
    stripped = command.strip()
    if not stripped:
        return False, format_validation_error("bash_exec", "command", "command is required")

    lowered = stripped.lower()
    for pattern in _DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, lowered):
            return False, format_validation_error(
                "bash_exec",
                "command",
                f"destructive shell pattern blocked: {pattern}",
            )

    return True, None


def validate_tool_args(
    tool_name: str,
    args: dict[str, Any],
    workspace_root: Path,
    current_cwd: Path | None = None,
) -> tuple[bool, str | None]:
    base_dir = current_cwd or workspace_root

    if tool_name == "bash_exec":
        command = args.get("command")
        if not isinstance(command, str):
            return False, format_validation_error(tool_name, "command", "command must be a string")
        return validate_shell_command(command)

    if tool_name == "read_file":
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path is required")
        return True, None

    if tool_name == "write_file":
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path is required")
        if not isinstance(content, str):
            return False, format_validation_error(tool_name, "content", "content must be a string")
        return True, None

    if tool_name == "list_dir":
        path = args.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path must be a non-empty string")
        return True, None

    if tool_name == "find_files":
        pattern = args.get("pattern", "*.py")
        path = args.get("path", ".")
        if not isinstance(pattern, str) or not pattern.strip():
            return False, format_validation_error(tool_name, "pattern", "pattern must be a non-empty string")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path must be a non-empty string")
        return True, None

    if tool_name == "search_code":
        query = args.get("query")
        path = args.get("path", ".")
        max_results = args.get("max_results", 100)
        if not isinstance(query, str) or not query.strip():
            return False, format_validation_error(tool_name, "query", "query is required")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path must be a non-empty string")
        if not isinstance(max_results, int) or max_results <= 0:
            return False, format_validation_error(tool_name, "max_results", "max_results must be a positive integer")
        if max_results > MAX_SEARCH_RESULTS:
            return False, format_validation_error(
                tool_name,
                "max_results",
                f"max_results must be <= {MAX_SEARCH_RESULTS}",
            )
        return True, None

    if tool_name == "replace_in_file":
        path = args.get("path")
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        if not isinstance(path, str) or not path.strip():
            return False, format_validation_error(tool_name, "path", "path is required")
        if not isinstance(old_text, str) or old_text == "":
            return False, format_validation_error(tool_name, "old_text", "old_text is required")
        if not isinstance(new_text, str):
            return False, format_validation_error(tool_name, "new_text", "new_text must be a string")
        return True, None

    return False, format_validation_error(tool_name, "tool_name", "unknown tool")
