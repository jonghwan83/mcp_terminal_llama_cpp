"""Shared tool execution entry point."""

import re
import subprocess
from pathlib import Path
from typing import Literal

from app.validators import format_validation_error, validate_tool_args

ExecutorMode = Literal["terminal", "mcp"]


class ToolExecutor:
    """Shared stateful tool executor for terminal and MCP entry points."""

    _CWD_MARKER = "__TERMINAL_CWD__:"

    def __init__(self, mode: ExecutorMode, initial_cwd: str | Path | None = None):
        self.mode = mode
        self._workspace_root = Path(initial_cwd or Path.cwd()).resolve()
        self._cwd = str(self._workspace_root)

    @property
    def cwd(self) -> str:
        return self._cwd

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        base = p if p.is_absolute() else Path(self._cwd) / p
        return base.resolve()

    def _display_path(self, original: str, resolved: Path) -> str:
        if self.mode == "terminal":
            return str(resolved)
        return original

    def _validate(self, tool_name: str, args: dict) -> str | None:
        ok, error = validate_tool_args(
            tool_name,
            args,
            self._workspace_root,
            Path(self._cwd).resolve(),
        )
        if not ok:
            return error or format_validation_error(tool_name, "tool_name", "invalid tool arguments")
        return None

    def exec_bash(self, command: str, timeout: int = 30) -> str:
        error = self._validate("bash_exec", {"command": command, "timeout": timeout})
        if error:
            return error

        if self.mode == "terminal":
            wrapped = f"{command}\necho '{self._CWD_MARKER}'\"$PWD\""
            try:
                result = subprocess.run(
                    wrapped,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self._cwd,
                )
                lines = result.stdout.splitlines()
                cwd_line = next(
                    (line for line in reversed(lines) if line.startswith(self._CWD_MARKER)),
                    None,
                )
                if cwd_line:
                    new_cwd = cwd_line[len(self._CWD_MARKER):]
                    if Path(new_cwd).is_dir():
                        self._cwd = new_cwd
                    lines = [line for line in lines if not line.startswith(self._CWD_MARKER)]

                out = "\n".join(lines)
                if result.stderr:
                    out += f"\nSTDERR:\n{result.stderr}"
                if result.returncode != 0:
                    out += f"\nExit code: {result.returncode}"
                return out.strip() or "(no output)"
            except subprocess.TimeoutExpired:
                return f"Error: timed out after {timeout}s"
            except Exception as e:
                return f"Error: {e}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._cwd,
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

    def exec_read_file(self, path: str) -> str:
        error = self._validate("read_file", {"path": path})
        if error:
            return error

        try:
            return self._resolve(path).read_text()
        except Exception as e:
            return f"Error: {e}"

    def exec_write_file(self, path: str, content: str) -> str:
        error = self._validate("write_file", {"path": path, "content": content})
        if error:
            return error

        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Wrote {len(content)} bytes to {self._display_path(path, p)}"
        except Exception as e:
            return f"Error: {e}"

    def exec_list_dir(self, path: str = ".") -> str:
        error = self._validate("list_dir", {"path": path})
        if error:
            return error

        try:
            entries = sorted(self._resolve(path).iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = [('[dir] ' if e.is_dir() else '      ') + e.name for e in entries]
            return "\n".join(lines) or "(empty)"
        except Exception as e:
            return f"Error: {e}"

    def exec_find_files(self, pattern: str = "*.py", path: str = ".") -> str:
        error = self._validate("find_files", {"pattern": pattern, "path": path})
        if error:
            return error

        try:
            base = self._resolve(path)
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
        self,
        query: str,
        path: str = ".",
        is_regex: bool = False,
        max_results: int = 100,
    ) -> str:
        error = self._validate(
            "search_code",
            {"query": query, "path": path, "is_regex": is_regex, "max_results": max_results},
        )
        if error:
            return error

        try:
            base = self._resolve(path)
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

    def exec_replace_in_file(self, path: str, old_text: str, new_text: str) -> str:
        error = self._validate(
            "replace_in_file",
            {"path": path, "old_text": old_text, "new_text": new_text},
        )
        if error:
            return error

        try:
            p = self._resolve(path)
            if not p.exists() or not p.is_file():
                return f"Error: file not found: {self._display_path(path, p)}"

            content = p.read_text()
            if old_text not in content:
                return "Error: old_text not found in file"

            updated = content.replace(old_text, new_text, 1)
            p.write_text(updated)
            return f"Updated {self._display_path(path, p)} (replaced first occurrence)"
        except Exception as e:
            return f"Error: {e}"

    def dispatch_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "bash_exec":
            return self.exec_bash(args["command"], int(args.get("timeout", 30)))
        if tool_name == "read_file":
            return self.exec_read_file(args["path"])
        if tool_name == "write_file":
            return self.exec_write_file(args["path"], args["content"])
        if tool_name == "list_dir":
            return self.exec_list_dir(args.get("path", "."))
        if tool_name == "find_files":
            return self.exec_find_files(args.get("pattern", "*.py"), args.get("path", "."))
        if tool_name == "search_code":
            return self.exec_search_code(
                args["query"],
                args.get("path", "."),
                bool(args.get("is_regex", False)),
                int(args.get("max_results", 100)),
            )
        if tool_name == "replace_in_file":
            return self.exec_replace_in_file(args["path"], args["old_text"], args["new_text"])
        return f"Unknown tool: {tool_name}"
