#!/usr/bin/env python3
"""
terminal.py — Interactive terminal bridge for local LLM + terminal tools.

Starts a llama_cpp.server subprocess, then opens an interactive REPL where
you type natural-language tasks.  The local LLM decides which terminal tools
to call (bash_exec / read_file / write_file / list_dir) and returns a final
answer.

Usage:
    python terminal.py
"""

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

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.theme import Theme
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://localhost:{LLAMA_PORT}/v1"
MAX_ITERATIONS = 20
VERSION = "0.1.0"

SUMMARY_TOKEN_THRESHOLD = 2000   # approximate tokens; summarize when exceeded
SUMMARY_KEEP_LAST = 2            # keep this many recent message pairs after summary

# Permission checking
ASK_PERMISSION = True            # Set to False to disable permission prompts

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "integer", "default": 30},
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
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file (creates parent dirs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
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
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Rich console
# ─────────────────────────────────────────────────────────────────────────────

THEME = Theme(
    {
        "banner": "bold cyan",
        "label": "dim",
        "model": "bold yellow",
        "tool.name": "bold magenta",
        "tool.arg": "cyan",
        "tool.out": "dim white",
        "answer": "bright_white",
        "status": "dim cyan",
        "ok": "bold green",
        "warn": "yellow",
        "err": "bold red",
    }
)

console = Console(theme=THEME, highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# Session working directory (persists across bash_exec calls)
# ─────────────────────────────────────────────────────────────────────────────

_session_cwd: str = str(Path.cwd())
_CWD_MARKER = "__TERMINAL_CWD__:"

# ─────────────────────────────────────────────────────────────────────────────
# LLM call wrapper — handles models that don't support system role (e.g. Gemma)
# ─────────────────────────────────────────────────────────────────────────────

_no_system_role: bool = False  # flipped on first "system role not supported" error


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
    """Call chat completions with automatic fallback for unsupported system role."""
    global _no_system_role
    msgs = _merge_system(messages) if _no_system_role else messages
    try:
        return llm.chat.completions.create(
            model="local", messages=msgs, tools=tools, tool_choice="auto", **kwargs
        )
    except Exception as e:
        if not _no_system_role and "system role not supported" in str(e).lower():
            _no_system_role = True
            console.print("  [label]Model does not support system role — merging into user message.[/label]")
            return llm.chat.completions.create(
                model="local",
                messages=_merge_system(messages),
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
        raise


def exec_bash(command: str, timeout: int = 30) -> str:
    global _session_cwd
    # Append a sentinel so we can capture the final cwd after the command runs
    wrapped = f"{command}\necho '{_CWD_MARKER}'\"$PWD\""
    try:
        r = subprocess.run(
            wrapped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_session_cwd,
        )
        stdout = r.stdout

        # Extract and strip the cwd sentinel line
        lines = stdout.splitlines()
        cwd_line = next((l for l in reversed(lines) if l.startswith(_CWD_MARKER)), None)
        if cwd_line:
            new_cwd = cwd_line[len(_CWD_MARKER):]
            if Path(new_cwd).is_dir():
                _session_cwd = new_cwd
            lines = [l for l in lines if not l.startswith(_CWD_MARKER)]

        out = "\n".join(lines)
        if r.stderr:
            out += f"\nSTDERR:\n{r.stderr}"
        if r.returncode != 0:
            out += f"\nExit code: {r.returncode}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _resolve(path: str) -> Path:
    """Resolve a path relative to the current session cwd."""
    p = Path(path)
    return p if p.is_absolute() else Path(_session_cwd) / p


def exec_read_file(path: str) -> str:
    try:
        return _resolve(path).read_text()
    except Exception as e:
        return f"Error: {e}"


def exec_write_file(path: str, content: str) -> str:
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error: {e}"


def exec_list_dir(path: str = ".") -> str:
    try:
        entries = sorted(_resolve(path).iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [("[dir] " if e.is_dir() else "      ") + e.name for e in entries]
        return "\n".join(lines) or "(empty)"
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
        content_preview = content[:50] + ("…" if len(content) > 50 else "")
        return f"File: {path}\nContent: {content_preview}"
    elif name == "list_dir":
        return f"Directory: {args.get('path', '.')}"
    return str(args)


def ask_tool_permission(name: str, args: dict[str, Any]) -> bool:
    """Ask user permission to execute a tool. Returns True if approved."""
    if not ASK_PERMISSION:
        return True
    
    console.print()
    console.print(f"  [warn]Tool Request:[/warn] [tool.name]{name}[/tool.name]")
    details = _format_tool_details(name, args)
    for line in details.split("\n"):
        console.print(f"    {line}")
    
    try:
        return Confirm.ask(
            "  Allow this operation",
            default=False,
            console=console
        )
    except EOFError:
        # If EOF reached during confirmation, deny permission
        return False


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    if not ask_tool_permission(name, args):
        return "Operation denied by user."
    
    if name == "bash_exec":
        return exec_bash(args["command"], int(args.get("timeout", 30)))
    if name == "read_file":
        return exec_read_file(args["path"])
    if name == "write_file":
        return exec_write_file(args["path"], args["content"])
    if name == "list_dir":
        return exec_list_dir(args.get("path", "."))
    return f"Unknown tool: {name}"


def _run_tool_with_ui(name: str, args: dict[str, Any]) -> str:
    """Run a tool with UI behavior that won't block permission prompts."""
    if ASK_PERMISSION:
        # Permission prompt uses stdin interaction; avoid running it under Live spinner.
        return dispatch_tool(name, args)

    with Live(
        Spinner("line", text="  [status]executing …[/status]"),
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        return dispatch_tool(name, args)


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────


def find_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.gguf"))


def prompt_select_model(models: list[Path]) -> Path:
    if not models:
        console.print("\n  [err]No .gguf models found in ./models/[/err]")
        sys.exit(1)

    if len(models) == 1:
        console.print(f"\n  Model: [model]{models[0].name}[/model]")
        return models[0]

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Model", style="model")
    table.add_column("Size", justify="right", style="label")

    for i, m in enumerate(models, 1):
        mb = m.stat().st_size / (1024**2)
        size = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
        table.add_row(str(i), m.name, size)

    console.print()
    console.print(table)

    while True:
        raw = Prompt.ask("  Select model", default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        console.print(f"  [warn]Enter a number between 1 and {len(models)}.[/warn]")


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp server lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_llama_proc: subprocess.Popen | None = None


def _print_server_error(log_path: str, returncode: int | None) -> None:
    if returncode is not None:
        console.print(f"  [err]Server process exited (code {returncode})[/err]")
    else:
        console.print("  [err]Server did not become ready within 60 s[/err]")

    try:
        lines = Path(log_path).read_text().splitlines()
        if lines:
            console.print("\n  [warn]Last server output:[/warn]")
            for line in lines[-25:]:
                console.print(f"  [tool.out]{line}[/tool.out]")
            if len(lines) > 25:
                console.print(f"  [label]  … ({len(lines) - 25} earlier lines)[/label]")
        else:
            console.print("  [label](no output captured)[/label]")
    except OSError:
        pass

    console.print(f"\n  [label]Full log: {log_path}[/label]")


def start_llama_server(model_path: Path) -> subprocess.Popen:
    global _llama_proc

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="llama_server_")
    log_file = os.fdopen(log_fd, "w")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(model_path),
            "--port",
            str(LLAMA_PORT),
            "--n_gpu_layers",
            "-1",
            "--n_ctx",
            "4096",
        ],
        stdout=log_file,
        stderr=log_file,
    )
    _llama_proc = proc

    probe = OpenAI(base_url=LLAMA_BASE_URL, api_key="none")
    t0 = time.time()

    with Live(
        Spinner("dots", text="  [status]Starting llama.cpp server …[/status]"),
        console=console,
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
                console.print(
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


def stop_llama_server() -> None:
    if _llama_proc and _llama_proc.poll() is None:
        _llama_proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────


_VALID_TOOLS = {s["function"]["name"] for s in TOOL_SCHEMAS}
_FIRST_PARAM = {
    "bash_exec": "command",
    "read_file": "path",
    "write_file": "path",
    "list_dir": "path",
}


def _try_parse_tool_json(data: dict) -> dict | None:
    """Extract {name, arguments} from a JSON object if it matches a known tool."""
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
            # JSON arguments body
            if body.startswith("{"):
                try:
                    data = json.loads(body)
                    # Prefer explicit arguments/parameters/args key (check existence, not truthiness)
                    if "arguments" in data:
                        args = data["arguments"]
                    elif "parameters" in data:
                        args = data["parameters"]
                    elif "args" in data:
                        args = data["args"]
                    else:
                        args = {k: v for k, v in data.items() if k not in ("name", "function", "tool")}
                    # Edge case: model wrote the command in "name" and left arguments empty
                    # e.g. <bash_exec>{"name": "pwd", "arguments": {}}</bash_exec>
                    if not args and "name" in data and data["name"] not in _VALID_TOOLS:
                        args = {_FIRST_PARAM[name]: data["name"]}
                    results.append({"name": name, "arguments": args})
                    continue
                except json.JSONDecodeError:
                    pass
            # Plain text body — treat as the first required argument
            if body and name in _FIRST_PARAM:
                results.append({"name": name, "arguments": {_FIRST_PARAM[name]: body}})

    if results:
        return results

    # ── 1 & 2: <tool_call>…</tool_call> ──────────────────────────────────────
    for tc_m in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        body = tc_m.group(1).strip()

        # Format 1: JSON body  (some models wrap with double braces: {{...}})
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

        # Format 2: XML body  <function=NAME><parameter=K>V</parameter></function>
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

    # ── 3: ReAct  Action: NAME / Action Input: {...} ──────────────────────────
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

    # ── 4: Bare JSON object anywhere in the text ──────────────────────────────
    for candidate in _extract_json_objects(content):
        try:
            parsed = _try_parse_tool_json(json.loads(candidate))
            if parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            pass

    return results


def _extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects from text using bracket counting.
    Handles nested braces correctly, unlike a simple regex."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Session memory + auto-summarize
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a terminal assistant. Complete the user's task using the available tools.
Be concise. Only use tools when necessary.

Available tools:
- bash_exec(command, timeout=30) : execute a shell command
- read_file(path)                : read a file
- write_file(path, content)      : write a file
- list_dir(path=".")             : list directory contents

To call a tool, output its name and arguments as JSON:
{"name": "bash_exec", "arguments": {"command": "ls -la"}}

Call one tool at a time and wait for the result before continuing.\
"""


def _make_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _count_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total characters / 4."""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for tc in m.get("tool_calls", []):
            total += len(tc.get("function", {}).get("arguments", ""))
    return total // 4


def _summarize(messages: list[dict], llm: OpenAI) -> str:
    """Ask the LLM to summarize the conversation history."""
    history_text = []
    for m in messages[1:]:  # skip system
        role = m["role"]
        content = m.get("content") or ""
        if role == "user":
            history_text.append(f"User: {content}")
        elif role == "assistant":
            if content:
                history_text.append(f"Assistant: {content}")
            for tc in m.get("tool_calls", []):
                history_text.append(
                    f"Assistant called {tc['function']['name']}({tc['function']['arguments']})"
                )
        elif role == "tool":
            short = content[:300] + ("…" if len(content) > 300 else "")
            history_text.append(f"Tool result: {short}")

    prompt = (
        "Summarize the following conversation concisely. "
        "Focus on what was accomplished, key facts discovered, and any important context "
        "needed to continue the work. Be brief.\n\n"
        + "\n".join(history_text)
    )

    resp = _llm_call(llm, [{"role": "user", "content": prompt}], [], max_tokens=512)
    return resp.choices[0].message.content or ""


def maybe_compress(messages: list[dict], llm: OpenAI) -> list[dict]:
    """If token count exceeds threshold, summarize history and return compressed messages."""
    if _count_tokens(messages) < SUMMARY_TOKEN_THRESHOLD:
        return messages

    with Live(
        Spinner("dots2", text="  [status]Summarizing conversation history …[/status]"),
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        summary = _summarize(messages, llm)

    # Keep only system + summary + last N non-system messages
    recent = [m for m in messages[1:] if m["role"] != "system"]
    keep = recent[-(SUMMARY_KEEP_LAST * 2):]

    compressed = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": f"[Previous session summary]\n{summary}"},
        *keep,
    ]

    tokens_before = _count_tokens(messages)
    tokens_after = _count_tokens(compressed)
    console.print(
        f"  [status]Memory compressed[/status]  "
        f"[label]~{tokens_before} → ~{tokens_after} tokens[/label]"
    )
    return compressed


def _args_preview(tool_name: str, args: dict) -> str:
    if tool_name == "bash_exec":
        return args.get("command", "")
    if tool_name in ("read_file", "write_file"):
        return args.get("path", "")
    if tool_name == "list_dir":
        return args.get("path", ".")
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _print_result(result: str, max_lines: int = 20) -> None:
    lines = result.splitlines()
    for line in lines[:max_lines]:
        console.print(f"    [tool.out]{line}[/tool.out]")
    if len(lines) > max_lines:
        console.print(f"    [label]… {len(lines) - max_lines} more lines[/label]")


def run_task(task: str, llm: OpenAI, messages: list[dict]) -> None:
    messages.append({"role": "user", "content": task})

    for _ in range(MAX_ITERATIONS):
        messages = maybe_compress(messages, llm)
        # ── LLM inference ──────────────────────────────────────────────────
        with Live(
            Spinner("dots", text="  [status]Thinking …[/status]"),
            console=console,
            refresh_per_second=12,
            transient=True,
        ):
            response = _llm_call(llm, messages, TOOL_SCHEMAS)

        msg = response.choices[0].message

        # Build assistant history entry
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
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
        messages.append(entry)

        # ── Final answer or text-based tool calls (fallback) ──────────────
        if not msg.tool_calls:
            text_calls = _parse_text_tool_calls(msg.content or "")
            if not text_calls:
                text = (msg.content or "").strip()
                if text:
                    console.print()
                    console.print(
                        Panel(text, border_style="cyan", padding=(0, 1), expand=False)
                    )
                return

            # Model emitted tool calls as raw text — execute them and inject results
            result_parts: list[str] = []
            for tc in text_calls:
                name, args = tc["name"], tc["arguments"]
                preview = _args_preview(name, args)
                console.print(
                    f"\n  [tool.name]{name}[/tool.name]  [tool.arg]{preview}[/tool.arg]"
                    f"  [label](text fallback)[/label]"
                )
                result = _run_tool_with_ui(name, args)
                _print_result(result)

                if result == "Operation denied by user.":
                    console.print("  [warn]Task stopped: tool execution was denied.[/warn]")
                    return

                result_parts.append(f"[{name}]\n{result}")

            messages.append({
                "role": "user",
                "content": "Tool results:\n" + "\n---\n".join(result_parts),
            })
            continue

        # ── Structured tool calls ──────────────────────────────────────────
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            preview = _args_preview(tc.function.name, args)
            console.print(
                f"\n  [tool.name]{tc.function.name}[/tool.name]"
                f"  [tool.arg]{preview}[/tool.arg]"
            )

            result = _run_tool_with_ui(tc.function.name, args)

            _print_result(result)

            if result == "Operation denied by user.":
                console.print("  [warn]Task stopped: tool execution was denied.[/warn]")
                return

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    console.print("  [warn]Reached maximum iteration limit.[/warn]")


# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold]Commands[/bold]
  [cyan]/help[/cyan]     Show this help
  [cyan]/clear[/cyan]    Clear the screen
  [cyan]/reset[/cyan]    Clear conversation memory and start fresh
  [cyan]/exit[/cyan]     Quit  (also Ctrl-C or Ctrl-D)

Type any task in natural language — the local LLM will handle it.
"""


def handle_slash(cmd: str, messages: list[dict]) -> bool:
    """Return True if it was a slash-command (so the REPL skips task execution)."""
    c = cmd.strip().lower()
    if c in ("/exit", "/quit", "/q"):
        raise KeyboardInterrupt
    if c == "/help":
        console.print(HELP_TEXT)
        return True
    if c == "/clear":
        console.clear()
        return True
    if c == "/reset":
        messages.clear()
        messages.extend(_make_messages())
        console.print("  [ok]Memory cleared.[/ok]")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    # Banner
    console.print(
        Panel.fit(
            f"[banner]Terminal LLM Bridge[/banner]  [label]v{VERSION}[/label]\n"
            "[label]llama.cpp · MCP terminal tools · local inference[/label]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    # Model selection
    models = find_models()
    model_path = prompt_select_model(models)
    console.print()

    # Confirm server start
    if not Confirm.ask(
        f"  Start llama.cpp server on port [cyan]{LLAMA_PORT}[/cyan]?",
        default=True,
    ):
        console.print("  [label]Aborted.[/label]")
        return

    start_llama_server(model_path)
    llm = OpenAI(base_url=LLAMA_BASE_URL, api_key="none")
    messages = _make_messages()

    console.print()
    console.print(Rule("[label]ready — type a task or /help[/label]", style="cyan"))
    console.print()

    # REPL
    try:
        while True:
            cwd_display = _session_cwd.replace(str(Path.home()), "~")
            try:
                task = Prompt.ask(f"[label]{cwd_display}[/label] [bold green]>[/bold green]")
            except EOFError:
                break

            task = task.strip()
            if not task:
                continue
            if handle_slash(task, messages):
                continue

            run_task(task, llm, messages)
            console.print()

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n  [label]Shutting down …[/label]")
        stop_llama_server()


if __name__ == "__main__":
    main()
