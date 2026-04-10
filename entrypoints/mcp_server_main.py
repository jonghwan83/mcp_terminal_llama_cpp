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
import signal
import subprocess
import sys
import tempfile
import time
import argparse
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

from app.agent_loop import run_agent_loop
from app.executor import ToolExecutor
from app.policy import build_permission_request, describe_tool_args, evaluate_tool_policy
from app.parsers.tool_call_parser import DEFAULT_FIRST_PARAM, parse_text_tool_calls
from app.tool_registry import TOOL_NAMES, TOOL_SCHEMAS, build_mcp_tool_specs

# stderr-only console so MCP stdio transport (stdout) is never polluted
_THEME = Theme({"ok": "bold green", "warn": "yellow", "err": "bold red", "label": "dim", "model": "bold yellow"})
ui = Console(stderr=True, theme=_THEME, highlight=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://localhost:{LLAMA_PORT}/v1"
MAX_AGENT_ITERATIONS = 10
DEFAULT_N_CTX = 4096
SERVER_LOG_TAIL_LINES = 100

# Permission checking - set via MCP_ASK_PERMISSION environment variable
ASK_PERMISSION = os.getenv("MCP_ASK_PERMISSION", "false").lower() in ("true", "1", "yes")


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
            for line in lines[-SERVER_LOG_TAIL_LINES:]:
                ui.print(f"  [label]{line}[/label]")
            if len(lines) > SERVER_LOG_TAIL_LINES:
                ui.print(f"  [label]  … ({len(lines) - SERVER_LOG_TAIL_LINES} earlier lines)[/label]")
        else:
            ui.print("  [label](no output captured)[/label]")
    except OSError:
        pass

    ui.print(f"\n  [label]Full log: {log_path}[/label]")


def start_llama_server(model_path: Path, n_ctx: int = DEFAULT_N_CTX) -> subprocess.Popen:
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="llama_server_")
    log_file = os.fdopen(log_fd, "w")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_path),
            "--port", str(LLAMA_PORT),
            "--n_gpu_layers", "-1",
            "--n_ctx", str(n_ctx),
        ],
        stdout=log_file,
        stderr=log_file,
    )

    launch_cmd = (
        f"{sys.executable} -m llama_cpp.server --model {model_path} --port {LLAMA_PORT} "
        f"--n_gpu_layers -1 --n_ctx {n_ctx}"
    )
    ui.print(f"  [label]Server command:[/label] {launch_cmd}")
    ui.print(f"  [label]Probe URL:[/label] {LLAMA_BASE_URL}")

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

_tool_executor = ToolExecutor(mode="mcp", initial_cwd=Path.cwd())


def _format_tool_details(name: str, args: dict[str, Any]) -> str:
    return describe_tool_args(name, args)


def _get_permission_request(name: str, args: dict[str, Any]) -> str:
    """Generate a permission request message."""
    return build_permission_request(name, args)


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    policy_decision = evaluate_tool_policy(name, ASK_PERMISSION)
    if policy_decision == "deny":
        return "Operation denied by policy."
    if policy_decision == "require_confirm":
        return _get_permission_request(name, args)

    return _tool_executor.dispatch_tool(name, args)


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
    if "tool_choice" not in kwargs:
        kwargs["tool_choice"] = "auto"
    try:
        return llm.chat.completions.create(
            model="local", messages=msgs, tools=tools, **kwargs
        )
    except Exception as e:
        if not _no_system_role and "system role not supported" in str(e).lower():
            _no_system_role = True
            return llm.chat.completions.create(
                model="local",
                messages=_merge_system(messages),
                tools=tools,
                **kwargs,
            )
        raise


_VALID_TOOLS = TOOL_NAMES


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
    ]

    def llm_call(messages_to_send: list[dict], tools: list[dict], **kwargs: Any) -> Any:
        return _llm_call(llm, messages_to_send, tools, **kwargs)

    outcome = run_agent_loop(
        task=task,
        llm_call=llm_call,
        messages=messages,
        tools=TOOL_SCHEMAS,
        execute_tool=dispatch_tool,
        parse_text_tool_calls=parse_text_tool_calls,
        valid_tools=_VALID_TOOLS,
        first_param=DEFAULT_FIRST_PARAM,
        max_iterations=MAX_AGENT_ITERATIONS,
    )

    if outcome.status == "final":
        return outcome.final_text or "(no response)"
    if outcome.status == "denied":
        return "Operation denied by user."
    return "Reached maximum iteration limit without a final answer."


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def build_mcp_server(llm: OpenAI) -> Server:
    server = Server("terminal-llm")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tool_defs = [types.Tool(**spec) for spec in build_mcp_tool_specs()]
        tool_defs.append(
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
            )
        )
        return tool_defs

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
    parser = argparse.ArgumentParser(description="MCP terminal server with llama.cpp bridge")
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=int(os.getenv("LLAMA_N_CTX", str(DEFAULT_N_CTX))),
        help="Context window size for llama.cpp server (default: 4096, env: LLAMA_N_CTX)",
    )
    args = parser.parse_args()
    if args.n_ctx < 512:
        parser.error("--n-ctx must be at least 512")

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

    llama_proc = start_llama_server(model_path, n_ctx=args.n_ctx)
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
