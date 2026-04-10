from pathlib import Path

from app.executor import ToolExecutor


def test_find_files_returns_relative_matches(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "src" / "b.txt").write_text("hello\n")

    executor = ToolExecutor(mode="terminal", initial_cwd=tmp_path)
    result = executor.exec_find_files("*.py", "src")

    assert result == "a.py"


def test_search_code_plain_text_and_regex(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("alpha\nbeta\nalpha_beta\n")

    executor = ToolExecutor(mode="terminal", initial_cwd=tmp_path)

    plain = executor.exec_search_code("alpha", path=".", is_regex=False, max_results=10)
    regex = executor.exec_search_code(r"alpha_\w+", path=".", is_regex=True, max_results=10)

    assert "sample.py:1: alpha" in plain
    assert "sample.py:3: alpha_beta" in plain
    assert regex.strip() == "sample.py:3: alpha_beta"


def test_replace_in_file_success_and_missing_old_text(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world\n")

    executor = ToolExecutor(mode="terminal", initial_cwd=tmp_path)

    ok = executor.exec_replace_in_file("notes.txt", "world", "there")
    fail = executor.exec_replace_in_file("notes.txt", "missing", "x")

    assert "Updated" in ok
    assert file_path.read_text() == "hello there\n"
    assert fail == "Error: old_text not found in file"


def test_write_file_allows_outside_workspace(tmp_path: Path) -> None:
    outside_target = tmp_path.parent / "outside_test_allow.txt"
    if outside_target.exists():
        outside_target.unlink()

    executor = ToolExecutor(mode="terminal", initial_cwd=tmp_path)
    result = executor.exec_write_file("../outside_test_allow.txt", "allowed")

    assert "Wrote" in result
    assert outside_target.read_text() == "allowed"

    outside_target.unlink()


def test_dispatch_unknown_tool_returns_error(tmp_path: Path) -> None:
    executor = ToolExecutor(mode="terminal", initial_cwd=tmp_path)
    result = executor.dispatch_tool("unknown_tool", {})
    assert result == "Unknown tool: unknown_tool"
