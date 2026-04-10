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
import signal
import subprocess
import sys
import tempfile
import time
import argparse
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

from app.agent_loop import run_agent_loop
from app.executor import ToolExecutor
import app.memory as session_memory
from app.memory import make_messages, maybe_compress_messages
from app.policy import describe_tool_args, evaluate_tool_policy
from app.parsers.tool_call_parser import DEFAULT_FIRST_PARAM, parse_text_tool_calls
from app.tool_registry import TOOL_NAMES, TOOL_SCHEMAS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://localhost:{LLAMA_PORT}/v1"
MAX_ITERATIONS = 20
VERSION = "0.1.0"

SUMMARY_TOKEN_THRESHOLD = 2000   # approximate tokens; summarize when exceeded
SUMMARY_KEEP_LAST = 2            # keep this many recent message pairs after summary
DEFAULT_N_CTX = 4096
SERVER_LOG_TAIL_LINES = 100

# Permission checking
ASK_PERMISSION = True            # Set to False to disable permission prompts

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

_tool_executor = ToolExecutor(mode="terminal", initial_cwd=Path.cwd())
_session_cwd: str = _tool_executor.cwd

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
    if "tool_choice" not in kwargs:
        kwargs["tool_choice"] = "auto"
    try:
        return llm.chat.completions.create(
            model="local", messages=msgs, tools=tools, **kwargs
        )
    except Exception as e:
        if not _no_system_role and "system role not supported" in str(e).lower():
            _no_system_role = True
            console.print("  [label]Model does not support system role — merging into user message.[/label]")
            return llm.chat.completions.create(
                model="local",
                messages=_merge_system(messages),
                tools=tools,
                **kwargs,
            )
        raise


def ask_tool_permission(name: str, args: dict[str, Any]) -> bool:
    """Ask user permission to execute a tool. Returns True if approved."""
    policy_decision = evaluate_tool_policy(name, ASK_PERMISSION)
    if policy_decision == "allow":
        return True
    if policy_decision == "deny":
        console.print()
        console.print(f"  [err]Tool denied by policy:[/err] [tool.name]{name}[/tool.name]")
        details = describe_tool_args(name, args)
        for line in details.split("\n"):
            console.print(f"    {line}")
        return False

    console.print()
    console.print(f"  [warn]Tool Request:[/warn] [tool.name]{name}[/tool.name]")
    details = describe_tool_args(name, args)
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
    global _session_cwd
    if not ask_tool_permission(name, args):
        return "Operation denied by user."

    result = _tool_executor.dispatch_tool(name, args)
    _session_cwd = _tool_executor.cwd
    return result


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
            for line in lines[-SERVER_LOG_TAIL_LINES:]:
                console.print(f"  [tool.out]{line}[/tool.out]")
            if len(lines) > SERVER_LOG_TAIL_LINES:
                console.print(
                    f"  [label]  … ({len(lines) - SERVER_LOG_TAIL_LINES} earlier lines)[/label]"
                )
        else:
            console.print("  [label](no output captured)[/label]")
    except OSError:
        pass

    console.print(f"\n  [label]Full log: {log_path}[/label]")


def start_llama_server(model_path: Path, n_ctx: int = DEFAULT_N_CTX) -> subprocess.Popen:
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
            str(n_ctx),
        ],
        stdout=log_file,
        stderr=log_file,
    )
    _llama_proc = proc

    launch_cmd = (
        f"{sys.executable} -m llama_cpp.server --model {model_path} --port {LLAMA_PORT} "
        f"--n_gpu_layers -1 --n_ctx {n_ctx}"
    )
    console.print(f"  [label]Server command:[/label] {launch_cmd}")
    console.print(f"  [label]Probe URL:[/label] {LLAMA_BASE_URL}")

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


_VALID_TOOLS = TOOL_NAMES


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


def run_task(task: str, llm: OpenAI, messages: list[dict], streaming_enabled: bool = True) -> None:
    streamed_answer_started = False

    def llm_call(messages_to_send: list[dict], tools: list[dict], **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _llm_call(llm, messages_to_send, tools, **kwargs)

        with Live(
            Spinner("dots", text="  [status]Thinking …[/status]"),
            console=console,
            refresh_per_second=12,
            transient=True,
        ):
            return _llm_call(llm, messages_to_send, tools, **kwargs)

    def on_text_delta(chunk: str) -> None:
        nonlocal streamed_answer_started
        if not streaming_enabled:
            return
        if not streamed_answer_started:
            console.print()
            streamed_answer_started = True
        console.print(chunk, end="", markup=False, soft_wrap=True)

    def on_compressed(tokens_before: int, tokens_after: int) -> None:
        console.print(
            f"  [status]Memory compressed[/status]  "
            f"[label]~{tokens_before} → ~{tokens_after} tokens[/label]"
        )

    def on_event(event_name: str, payload: dict[str, Any]) -> None:
        if event_name == "tool_call":
            name = payload["name"]
            args = payload["args"]
            preview = _args_preview(name, args)
            suffix = "  [label](text fallback)[/label]" if payload.get("source") == "text" else ""
            console.print(f"\n  [tool.name]{name}[/tool.name]  [tool.arg]{preview}[/tool.arg]{suffix}")
        elif event_name == "tool_result":
            _print_result(payload["result"])
        elif event_name == "final_text":
            if streaming_enabled:
                console.print()
            else:
                console.print()
                console.print(Panel(payload["text"], border_style="cyan", padding=(0, 1), expand=False))
        elif event_name == "denied":
            console.print("  [warn]Task stopped: tool execution was denied.[/warn]")
        elif event_name == "max_iterations":
            console.print("  [warn]Reached maximum iteration limit.[/warn]")

    run_agent_loop(
        task=task,
        llm_call=llm_call,
        messages=messages,
        tools=TOOL_SCHEMAS,
        execute_tool=_run_tool_with_ui,
        parse_text_tool_calls=parse_text_tool_calls,
        valid_tools=_VALID_TOOLS,
        first_param=DEFAULT_FIRST_PARAM,
        max_iterations=MAX_ITERATIONS,
        on_event=on_event,
        on_text_delta=on_text_delta if streaming_enabled else None,
        compress_messages=lambda current_messages: maybe_compress_messages(
            current_messages,
            lambda msgs, tools, **kwargs: _llm_call(llm, msgs, tools, **kwargs),
            on_compressed=on_compressed,
        ),
    )


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
        messages.extend(make_messages())
        console.print("  [ok]Memory cleared.[/ok]")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive terminal bridge for local LLM + terminal tools")
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=int(os.getenv("LLAMA_N_CTX", str(DEFAULT_N_CTX))),
        help="Context window size for llama.cpp server (default: 4096, env: LLAMA_N_CTX)",
    )
    parser.add_argument(
        "--summary-token-threshold",
        type=int,
        default=int(os.getenv("SUMMARY_TOKEN_THRESHOLD", str(SUMMARY_TOKEN_THRESHOLD))),
        help="Approx token threshold to trigger history summarization (default: 2000)",
    )
    parser.add_argument(
        "--summary-keep-last",
        type=int,
        default=int(os.getenv("SUMMARY_KEEP_LAST", str(SUMMARY_KEEP_LAST))),
        help="Number of recent message pairs to keep after summarization (default: 2)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=False,
        help="Stream assistant text tokens as they are generated (default: off)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable assistant text streaming (default)",
    )
    args = parser.parse_args()
    if args.n_ctx < 512:
        parser.error("--n-ctx must be at least 512")
    if args.summary_token_threshold < 200:
        parser.error("--summary-token-threshold must be at least 200")
    if args.summary_keep_last < 0:
        parser.error("--summary-keep-last must be >= 0")

    session_memory.SUMMARY_TOKEN_THRESHOLD = args.summary_token_threshold
    session_memory.SUMMARY_KEEP_LAST = args.summary_keep_last

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

    start_llama_server(model_path, n_ctx=args.n_ctx)
    llm = OpenAI(base_url=LLAMA_BASE_URL, api_key="none")
    messages = make_messages()

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

            run_task(task, llm, messages, streaming_enabled=args.stream)
            console.print()

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n  [label]Shutting down …[/label]")
        stop_llama_server()


if __name__ == "__main__":
    main()
