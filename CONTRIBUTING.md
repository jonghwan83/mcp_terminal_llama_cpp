# 툴 추가 가이드

새 툴을 추가할 때 수정해야 할 파일과 위치를 정리합니다.

---

## 수정 파일 목록

툴 하나를 추가하려면 **두 파일을 각각 4곳씩** 수정합니다.

| 파일 | 수정 위치 | 역할 |
|---|---|---|
| `terminal.py` | `TOOL_SCHEMAS` | LLM에게 툴 설명 전달 |
| `terminal.py` | `exec_*` 함수 | 실제 실행 로직 |
| `terminal.py` | `dispatch_tool` | 이름 → 함수 라우팅 |
| `terminal.py` | `_args_preview` | 터미널 UI 미리보기 |
| `mcp_server.py` | `_TOOL_SCHEMAS` | LLM에게 툴 설명 전달 |
| `mcp_server.py` | `exec_*` 함수 | 실제 실행 로직 |
| `mcp_server.py` | `dispatch_tool` | 이름 → 함수 라우팅 |
| `mcp_server.py` | `list_tools()` | MCP 클라이언트에 툴 노출 |

---

## 예시: `http_get` 툴 추가

URL을 받아 HTTP GET 요청 결과를 반환하는 툴을 추가하는 전체 예시입니다.

### 1. `terminal.py` — `TOOL_SCHEMAS`에 스키마 추가

```python
TOOL_SCHEMAS: list[dict] = [
    # ... 기존 툴들 ...
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Send an HTTP GET request and return the response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["url"],
            },
        },
    },
]
```

### 2. `terminal.py` — 실행 함수 추가

`exec_list_dir` 아래에 추가합니다.

```python
def exec_http_get(url: str, timeout: int = 10) -> str:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode(errors="replace")[:4000]
    except Exception as e:
        return f"Error: {e}"
```

### 3. `terminal.py` — `dispatch_tool`에 라우팅 추가

```python
def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    if name == "bash_exec":
        return exec_bash(args["command"], int(args.get("timeout", 30)))
    if name == "read_file":
        return exec_read_file(args["path"])
    if name == "write_file":
        return exec_write_file(args["path"], args["content"])
    if name == "list_dir":
        return exec_list_dir(args.get("path", "."))
    if name == "http_get":                              # ← 추가
        return exec_http_get(args["url"], int(args.get("timeout", 10)))
    return f"Unknown tool: {name}"
```

### 4. `terminal.py` — `_args_preview`에 미리보기 추가

터미널 UI에서 `  http_get  https://example.com` 처럼 표시됩니다.

```python
def _args_preview(tool_name: str, args: dict) -> str:
    if tool_name == "bash_exec":
        return args.get("command", "")
    if tool_name in ("read_file", "write_file"):
        return args.get("path", "")
    if tool_name == "list_dir":
        return args.get("path", ".")
    if tool_name == "http_get":                         # ← 추가
        return args.get("url", "")
    return ", ".join(f"{k}={v}" for k, v in args.items())
```

### 5. `mcp_server.py` — `_TOOL_SCHEMAS`에 스키마 추가

`terminal.py`의 1번과 동일한 내용을 `_TOOL_SCHEMAS`에 추가합니다.

### 6. `mcp_server.py` — 실행 함수 추가

`terminal.py`의 2번과 동일한 함수를 추가합니다.

### 7. `mcp_server.py` — `dispatch_tool`에 라우팅 추가

`terminal.py`의 3번과 동일하게 추가합니다.

### 8. `mcp_server.py` — `list_tools()`에 MCP 툴 정의 추가

`build_mcp_server` 함수 안의 `list_tools()`에 추가합니다.

```python
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ... 기존 툴들 ...
        types.Tool(
            name="http_get",
            description="Send an HTTP GET request and return the response body.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["url"],
            },
        ),
    ]
```

---

## 체크리스트

툴 추가 후 확인:

- [ ] `terminal.py` — `TOOL_SCHEMAS` 추가
- [ ] `terminal.py` — `exec_*` 함수 추가
- [ ] `terminal.py` — `dispatch_tool` 라우팅 추가
- [ ] `terminal.py` — `_args_preview` 미리보기 추가
- [ ] `mcp_server.py` — `_TOOL_SCHEMAS` 추가
- [ ] `mcp_server.py` — `exec_*` 함수 추가
- [ ] `mcp_server.py` — `dispatch_tool` 라우팅 추가
- [ ] `mcp_server.py` — `list_tools()` 추가
- [ ] `README.md` — 툴 목록 표 업데이트

---

## 참고: `_VALID_TOOLS`는 자동 갱신

`_VALID_TOOLS`는 `TOOL_SCHEMAS` / `_TOOL_SCHEMAS`에서 자동으로 생성되므로 별도 수정이 필요 없습니다.

```python
# terminal.py / mcp_server.py — 수동 수정 불필요
_VALID_TOOLS = {s["function"]["name"] for s in TOOL_SCHEMAS}
```
