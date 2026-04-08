#!/usr/bin/env python3
"""MCP Terminal Server with llama.cpp bridge.

Starts a llama_cpp.server subprocess (OpenAI-compatible API),
then exposes terminal tools via MCP stdio transport.
The special `run_task` tool sends a natural-language task to the
local LLM which orchestrates the other terminal tools itself.
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.theme import Theme

# stderr-only console so MCP stdio transport (stdout) is never polluted
_THEME = Theme({"ok": "bold green", "warn": "yellow", "err": "bold red", "label": "dim", "model": "bold yellow"})
ui = Console(stderr=True, theme=_THEME, highlight=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://localhost:{LLAMA_PORT}/v1"
MAX_AGENT_ITERATIONS = 10

# Permission checking - set via MCP_ASK_PERMISSION environment variable
ASK_PERMISSION = os.getenv("MCP_ASK_PERMISSION", "false").lower() in ("true", "1", "yes")

# Tool schemas shared by both MCP and the LLM function-calling API
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file (creates parent dirs as needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by glob pattern under a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (default '*.py')"},
                    "path": {"type": "string", "description": "Base directory (default '.')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search text in files under a directory (supports regex).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex pattern to search"},
                    "path": {"type": "string", "description": "Base directory (default '.')"},
                    "is_regex": {"type": "boolean", "description": "Treat query as regex (default false)"},
                    "max_results": {"type": "integer", "description": "Maximum matches to return (default 100)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace text in a file once (simple code editing helper).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {"type": "string", "description": "Text to find"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def find_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.gguf"))


def select_model(models: list[Path]) -> Path:
    if not models:
        ui.print("\n  [err]No .gguf models found in ./models/[/err]")
        sys.exit(1)

    if len(models) == 1:
        ui.print(f"\n  Model: [model]{models[0].name}[/model]")
        return models[0]

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Model", style="model")
    table.add_column("Size", justify="right", style="label")

    for i, m in enumerate(models, 1):
        mb = m.stat().st_size / (1024 ** 2)
        size = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
        table.add_row(str(i), m.name, size)

    ui.print()
    ui.print(table)

    while True:
        raw = Prompt.ask("  Select model", console=ui, default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        ui.print(f"  [warn]Enter 1–{len(models)}.[/warn]")


# ---------------------------------------------------------------------------
# llama.cpp server management
# ---------------------------------------------------------------------------

def _print_server_error(log_path: str, returncode: int | None) -> None:
    if returncode is not None:
        ui.print(f"  [err]Server process exited (code {returncode})[/err]")
    else:
        ui.print("  [err]Server did not become ready within 60 s[/err]")

    try:
        lines = Path(log_path).read_text().splitlines()
        if lines:
            ui.print("\n  [warn]Last server output:[/warn]")
            for line in lines[-25:]:
                ui.print(f"  [label]{line}[/label]")
            if len(lines) > 25:
                ui.print(f"  [label]  … ({len(lines) - 25} earlier lines)[/label]")
        else:
            ui.print("  [label](no output captured)[/label]")
    except OSError:
        pass

    ui.print(f"\n  [label]Full log: {log_path}[/label]")


def start_llama_server(model_path: Path) -> subprocess.Popen:
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="llama_server_")
    log_file = os.fdopen(log_fd, "w")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_path),
            "--port", str(LLAMA_PORT),
            "--n_gpu_layers", "-1",
            "--n_ctx", "4096",
        ],
        stdout=log_file,
        stderr=log_file,
    )

    probe = OpenAI(base_url=LLAMA_BASE_URL, api_key="none")
    t0 = time.time()

    with Live(
        Spinner("dots", text="  Starting llama.cpp server …"),
        console=ui,
        refresh_per_second=12,
        transient=True,
    ):
        for _ in range(60):
            if proc.poll() is not None:
                log_file.close()
                _print_server_error(log_path, proc.returncode)
                sys.exit(1)
            try:
                probe.models.list()
                elapsed = time.time() - t0
                log_file.close()
                os.unlink(log_path)
                ui.print(
                    f"  [ok]Server ready[/ok]  "
                    f"[label]port {LLAMA_PORT}  ({elapsed:.1f}s)[/label]"
                )
                return proc
            except Exception:
                time.sleep(1)

    log_file.close()
    proc.kill()
    _print_server_error(log_path, None)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Tool execution (local, no LLM involved)
# ---------------------------------------------------------------------------

def exec_bash(command: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout
        if result.stderr:
            out += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            out += f"\nExit code: {result.returncode}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def exec_read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except Exception as e:
        return f"Error: {e}"


def exec_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def exec_list_dir(path: str = ".") -> str:
    try:
        entries = sorted(Path(path).iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [("[dir] " if e.is_dir() else "      ") + e.name for e in entries]
        return "\n".join(lines) or "(empty)"
    except Exception as e:
        return f"Error: {e}"


def exec_find_files(pattern: str = "*.py", path: str = ".") -> str:
    try:
        base = Path(path)
        if not base.exists() or not base.is_dir():
            return f"Error: directory not found: {base}"
        files = sorted(p for p in base.rglob(pattern) if p.is_file())
        if not files:
            return "(no files found)"
        lines = [str(p.relative_to(base)) for p in files]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def exec_search_code(
    query: str,
    path: str = ".",
    is_regex: bool = False,
    max_results: int = 100,
) -> str:
    try:
        base = Path(path)
        if not base.exists() or not base.is_dir():
            return f"Error: directory not found: {base}"

        if max_results <= 0:
            max_results = 100

        matcher = re.compile(query) if is_regex else None
        results: list[str] = []

        for f in sorted(p for p in base.rglob("*") if p.is_file()):
            try:
                text = f.read_text()
            except Exception:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                matched = bool(matcher.search(line)) if matcher else (query in line)
                if matched:
                    rel = f.relative_to(base)
                    results.append(f"{rel}:{i}: {line.strip()}")
                    if len(results) >= max_results:
                        return "\n".join(results)

        return "\n".join(results) if results else "(no matches)"
    except re.error as e:
        return f"Error: invalid regex: {e}"
    except Exception as e:
        return f"Error: {e}"


def exec_replace_in_file(path: str, old_text: str, new_text: str) -> str:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return f"Error: file not found: {path}"

        content = p.read_text()
        if old_text not in content:
            return "Error: old_text not found in file"

        updated = content.replace(old_text, new_text, 1)
        p.write_text(updated)
        return f"Updated {path} (replaced first occurrence)"
    except Exception as e:
        return f"Error: {e}"


def _format_tool_details(name: str, args: dict[str, Any]) -> str:
    """Format tool call details for permission display."""
    if name == "bash_exec":
        return f"Command: {args.get('command', '')}"
    elif name == "read_file":
        return f"File: {args.get('path', '')}"
    elif name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        content_preview = content[:100] + ("…" if len(content) > 100 else "")
        return f"File: {path}\nContent preview: {content_preview}"
    elif name == "list_dir":
        return f"Directory: {args.get('path', '.')}"
    elif name == "find_files":
        return f"Path: {args.get('path', '.')}\nPattern: {args.get('pattern', '*.py')}"
    elif name == "search_code":
        return (
            f"Path: {args.get('path', '.')}\n"
            f"Query: {args.get('query', '')}\n"
            f"Regex: {args.get('is_regex', False)}"
        )
    elif name == "replace_in_file":
        return f"File: {args.get('path', '')}\nReplace text in first occurrence"
    return str(args)


def _get_permission_request(name: str, args: dict[str, Any]) -> str:
    """Generate a permission request message."""
    details = _format_tool_details(name, args)
    return (
        f"⚠️  Permission Required\n\n"
        f"Tool: {name}\n"
        f"{details}\n\n"
        f"This operation requires approval. Reply with 'approved' to proceed."
    )


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    if ASK_PERMISSION:
        return _get_permission_request(name, args)
    
    if name == "bash_exec":
        return exec_bash(args["command"], int(args.get("timeout", 30)))
    if name == "read_file":
        return exec_read_file(args["path"])
    if name == "write_file":
        return exec_write_file(args["path"], args["content"])
    if name == "list_dir":
        return exec_list_dir(args.get("path", "."))
    if name == "find_files":
        return exec_find_files(args.get("pattern", "*.py"), args.get("path", "."))
    if name == "search_code":
        return exec_search_code(
            args["query"],
            args.get("path", "."),
            bool(args.get("is_regex", False)),
            int(args.get("max_results", 100)),
        )
    if name == "replace_in_file":
        return exec_replace_in_file(args["path"], args["old_text"], args["new_text"])
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# LLM call wrapper — handles models that don't support system role (e.g. Gemma)
# ---------------------------------------------------------------------------

_no_system_role: bool = False


def _merge_system(messages: list[dict]) -> list[dict]:
    """Fold the system message into the first user message content."""
    system_text = next(
        (m.get("content") or "" for m in messages if m["role"] == "system"), ""
    )
    result = [m for m in messages if m["role"] != "system"]
    if system_text:
        idx = next((i for i, m in enumerate(result) if m["role"] == "user"), None)
        if idx is not None:
            result[idx] = dict(result[idx], content=f"{system_text}\n\n{result[idx]['content']}")
    return result


def _llm_call(llm: OpenAI, messages: list[dict], tools: list[dict], **kwargs) -> Any:
    global _no_system_role
    msgs = _merge_system(messages) if _no_system_role else messages
    try:
        return llm.chat.completions.create(
            model="local", messages=msgs, tools=tools, tool_choice="auto", **kwargs
        )
    except Exception as e:
        if not _no_system_role and "system role not supported" in str(e).lower():
            _no_system_role = True
            return llm.chat.completions.create(
                model="local",
                messages=_merge_system(messages),
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
        raise


# ---------------------------------------------------------------------------
# Text-based tool call parser (fallback for models that don't use structured tool_calls)
# ---------------------------------------------------------------------------

_VALID_TOOLS = {s["function"]["name"] for s in _TOOL_SCHEMAS}


def _try_parse_tool_json(data: dict) -> dict | None:
    name = data.get("name") or data.get("function") or data.get("tool")
    if not isinstance(name, str) or name not in _VALID_TOOLS:
        return None
    args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
    if not isinstance(args, dict):
        try:
            args = json.loads(args)
        except (TypeError, json.JSONDecodeError):
            args = {}
    return {"name": name, "arguments": args}


_FIRST_PARAM = {
    "bash_exec": "command",
    "read_file": "path",
    "write_file": "path",
    "list_dir": "path",
    "find_files": "pattern",
    "search_code": "query",
    "replace_in_file": "path",
}


def _parse_text_tool_calls(content: str) -> list[dict]:
    """Fallback parser for models that emit tool calls as plain text.

    Tried in order:
    0. <bash_exec>{"command":"..."}</bash_exec>              — tool name as tag
    1. <tool_call>{"name":...,"arguments":...}</tool_call>   — Qwen 2.5 JSON
    2. <tool_call><function=NAME><parameter=K>V</parameter>  — Qwen 3.x XML
    3. Action: NAME / Action Input: {...}                    — ReAct format
    4. Bare JSON object with a known tool name               — generic fallback
    """
    results: list[dict] = []

    # ── 0: <tool_name>…</tool_name> ───────────────────────────────────────────
    for name in _VALID_TOOLS:
        for m in re.finditer(rf"<{name}>(.*?)</{name}>", content, re.DOTALL):
            body = m.group(1).strip()
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
                    if not args and "name" in data and data["name"] not in _VALID_TOOLS:
                        args = {_FIRST_PARAM[name]: data["name"]}
                    results.append({"name": name, "arguments": args})
                    continue
                except json.JSONDecodeError:
                    pass
            if body and name in _FIRST_PARAM:
                results.append({"name": name, "arguments": {_FIRST_PARAM[name]: body}})

    if results:
        return results

    for tc_m in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        body = tc_m.group(1).strip()

        if body.startswith("{"):
            if body.startswith("{{") and body.endswith("}}"):
                body = body[1:-1]
            try:
                parsed = _try_parse_tool_json(json.loads(body))
                if parsed:
                    results.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass

        fn_m = re.search(r"<function=(\w+)>(.*?)</function>", body, re.DOTALL)
        if fn_m and fn_m.group(1) in _VALID_TOOLS:
            args: dict[str, str] = {}
            for p_m in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", fn_m.group(2), re.DOTALL
            ):
                args[p_m.group(1)] = p_m.group(2)
            results.append({"name": fn_m.group(1), "arguments": args})

    if results:
        return results

    for m in re.finditer(
        r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", content, re.DOTALL
    ):
        if m.group(1) not in _VALID_TOOLS:
            continue
        try:
            results.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass

    if results:
        return results

    for candidate in _extract_json_objects(content):
        try:
            parsed = _try_parse_tool_json(json.loads(candidate))
            if parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            pass

    return results


def _extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects using bracket counting (handles nested braces)."""
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


# ---------------------------------------------------------------------------
# LLM agent loop (bridge: MCP task → llama.cpp → tool calls → result)
# ---------------------------------------------------------------------------

def run_agent_task(task: str, llm: OpenAI) -> str:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a terminal assistant. Complete the user's task using the available tools.\n"
                "Be concise. Only use tools when necessary.\n\n"
                "Available tools:\n"
                "- bash_exec(command, timeout=30) : execute a shell command\n"
                "- read_file(path)                : read a file\n"
                "- write_file(path, content)      : write a file\n"
                "- list_dir(path=\".\")             : list directory contents\n"
                "- find_files(pattern=\"*.py\", path=\".\") : find files by glob pattern\n"
                "- search_code(query, path=\".\", is_regex=False, max_results=100) : search code text\n"
                "- replace_in_file(path, old_text, new_text) : replace text once in a file\n\n"
                "To call a tool, output its name and arguments as JSON:\n"
                "{\"name\": \"bash_exec\", \"arguments\": {\"command\": \"ls -la\"}}\n\n"
                "Call one tool at a time and wait for the result before continuing."
            ),
        },
        {"role": "user", "content": task},
    ]

    for _ in range(MAX_AGENT_ITERATIONS):
        response = _llm_call(llm, messages, _TOOL_SCHEMAS)

        choice = response.choices[0]
        msg = choice.message

        # Append assistant turn
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # If no structured tool calls, check for text-based fallback
        if not msg.tool_calls:
            text_calls = _parse_text_tool_calls(msg.content or "")
            if not text_calls:
                return msg.content or "(no response)"

            result_parts: list[str] = []
            for tc in text_calls:
                result = dispatch_tool(tc["name"], tc["arguments"])
                result_parts.append(f"[{tc['name']}]\n{result}")

            messages.append({
                "role": "user",
                "content": "Tool results:\n" + "\n---\n".join(result_parts),
            })
            continue

        # Execute each structured tool call and feed results back
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = dispatch_tool(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "Reached maximum iteration limit without a final answer."


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def build_mcp_server(llm: OpenAI) -> Server:
    server = Server("terminal-llm")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="bash_exec",
                description="Execute a shell command and return stdout/stderr.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["command"],
                },
            ),
            types.Tool(
                name="read_file",
                description="Read the contents of a file.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="write_file",
                description="Write text content to a file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
            types.Tool(
                name="list_dir",
                description="List directory contents.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                },
            ),
            types.Tool(
                name="find_files",
                description="Find files by glob pattern under a directory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "default": "*.py"},
                        "path": {"type": "string", "default": "."},
                    },
                },
            ),
            types.Tool(
                name="search_code",
                description="Search text in files under a directory (supports regex).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "is_regex": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 100},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="replace_in_file",
                description="Replace text in a file once.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            ),
            types.Tool(
                name="run_task",
                description=(
                    "Send a natural-language task to the local LLM. "
                    "The LLM will autonomously call available tools as needed "
                    "and return the final result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description in natural language",
                        }
                    },
                    "required": ["task"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent]:
        if name == "run_task":
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, run_agent_task, arguments["task"], llm
            )
        else:
            text = dispatch_tool(name, arguments)

        return [types.TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def amain() -> None:
    ui.print(
        Panel.fit(
            "[bold cyan]MCP Terminal Server[/bold cyan]  [label]llama.cpp bridge[/label]\n"
            "[label]Exposes terminal tools via MCP stdio transport.[/label]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    models = find_models()
    model_path = select_model(models)
    ui.print()

    if not Confirm.ask(
        f"  Start llama.cpp server on port [cyan]{LLAMA_PORT}[/cyan]?",
        console=ui,
        default=True,
    ):
        ui.print("  [label]Aborted.[/label]")
        return

    llama_proc = start_llama_server(model_path)
    llm = OpenAI(base_url=LLAMA_BASE_URL, api_key="none")

    def _shutdown(sig=None, frame=None) -> None:
        ui.print("\n  [label]Shutting down …[/label]")
        llama_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server = build_mcp_server(llm)

    ui.print(
        f"\n  [ok]MCP server running[/ok]  "
        f"[label]model: {model_path.name}[/label]"
    )
    ui.print("  [label]Waiting for MCP client to connect via stdio …[/label]\n")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    _shutdown()


if __name__ == "__main__":
    asyncio.run(amain())
