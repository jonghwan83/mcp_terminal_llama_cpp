import inspect
from pathlib import Path

import mcp_server
import terminal
from app.tool_registry import TOOL_NAMES, TOOL_SCHEMAS, build_mcp_tool_specs
from entrypoints import mcp_server_main, terminal_main


def test_wrapper_exports_entrypoint_callables() -> None:
    assert callable(terminal.main)
    assert callable(mcp_server.amain)
    assert inspect.iscoroutinefunction(mcp_server.amain)


def test_tool_registry_and_mcp_specs_are_consistent() -> None:
    schema_names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    mcp_specs = build_mcp_tool_specs()
    mcp_names = [spec["name"] for spec in mcp_specs]

    assert set(schema_names) == TOOL_NAMES
    assert set(mcp_names) == TOOL_NAMES
    assert len(schema_names) == len(mcp_names)


def test_entrypoint_model_dirs_point_to_repo_models() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = repo_root / "models"

    assert terminal_main.MODELS_DIR == expected
    assert mcp_server_main.MODELS_DIR == expected
