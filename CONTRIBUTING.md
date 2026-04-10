# 툴 추가 가이드 (현재 구조 기준)

현재 프로젝트는 thin wrapper + shared module 구조입니다.

- 실행 파일: `terminal.py`, `mcp_server.py` (wrapper)
- 실제 런타임: `entrypoints/terminal_main.py`, `entrypoints/mcp_server_main.py`
- 공통 툴 정의/실행: `app/*`

즉, 새 툴은 보통 `terminal.py`, `mcp_server.py`를 직접 수정하지 않고
`app/tool_registry.py` + `app/executor.py` + `app/validators.py` + `app/policy.py`를 수정합니다.

---

## 수정 파일 목록

| 파일 | 수정 위치 | 역할 |
|---|---|---|
| `app/tool_registry.py` | `TOOL_SCHEMAS` | 툴 스키마(LLM/MCP 공통) 등록 |
| `app/executor.py` | `exec_*` 메서드 | 실제 실행 로직 구현 |
| `app/executor.py` | `dispatch_tool` | 이름 → 실행 메서드 라우팅 |
| `app/validators.py` | `validate_tool_args` | 인자 타입/범위/경로 검증 |
| `app/policy.py` | `LOW_RISK_TOOLS` / `HIGH_RISK_TOOLS` | 권한 정책(allow/confirm/deny) |
| `tests/*` | 테스트 추가 | 회귀 방지 |
| `README.md` | 툴 목록 문서 갱신 | 사용자 안내 |

참고:
- MCP 노출은 `entrypoints/mcp_server_main.py`가 `build_mcp_tool_specs()`를 사용하므로,
  일반 툴은 `app/tool_registry.py`만 수정해도 자동 반영됩니다.
- `run_task`는 MCP 전용 특수 툴이라 entrypoint에서 별도로 append됩니다.

---

## 예시: `http_get` 툴 추가

### 1) `app/tool_registry.py`에 스키마 추가

```python
TOOL_SCHEMAS = [
    # ... existing ...
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Send an HTTP GET request and return response text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 10)",
                        "default": 10,
                    },
                },
                "required": ["url"],
            },
        },
    },
]
```

### 2) `app/executor.py`에 실행 메서드 + 라우팅 추가

```python
def exec_http_get(self, url: str, timeout: int = 10) -> str:
    error = self._validate("http_get", {"url": url, "timeout": timeout})
    if error:
        return error
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode(errors="replace")[:4000]
    except Exception as e:
        return f"Error: {e}"

def dispatch_tool(self, tool_name: str, args: dict) -> str:
    # ... existing ...
    if tool_name == "http_get":
        return self.exec_http_get(args["url"], int(args.get("timeout", 10)))
```

### 3) `app/validators.py`에 검증 규칙 추가

```python
if tool_name == "http_get":
    url = args.get("url")
    timeout = args.get("timeout", 10)
    if not isinstance(url, str) or not url.strip():
        return False, format_validation_error(tool_name, "url", "url is required")
    if not isinstance(timeout, int) or timeout <= 0:
        return False, format_validation_error(tool_name, "timeout", "timeout must be positive")
    return True, None
```

### 4) `app/policy.py`에 리스크 등급 반영

예: 읽기 성격이면 low-risk

```python
LOW_RISK_TOOLS = {"read_file", "list_dir", "find_files", "search_code", "http_get"}
```

---

## 테스트 추가 가이드

최소 권장:

- `tests/test_executor.py`: 실행 성공/실패, 경계 케이스
- `tests/test_policy.py`: 새 툴의 allow/confirm/deny 결과

필요 시:

- `tests/test_parser.py`: 파서 입력 포맷 영향이 있을 때
- `tests/test_integration_runtime.py`: 스키마 노출 일관성 확인

실행:

```bash
conda run -n mcp_dev pytest -q
```

---

## 최종 체크리스트

- [ ] `app/tool_registry.py`에 새 툴 스키마 추가
- [ ] `app/executor.py`에 `exec_*` 구현 + `dispatch_tool` 라우팅
- [ ] `app/validators.py`에 인자 검증 추가
- [ ] `app/policy.py` 리스크 등급 반영
- [ ] 테스트 추가/수정 후 `pytest` 통과
- [ ] `README.md` 툴 목록 업데이트

---

## 참고

- `_VALID_TOOLS`는 `TOOL_NAMES`에서 파생되어 자동 반영됩니다.
- 일반 툴은 `entrypoints/mcp_server_main.py`의 `list_tools()`를 직접 수정할 필요가 없습니다.
