# MCP Terminal LLM Bridge

로컬 LLM(llama.cpp)으로 터미널 작업을 수행하는 도구.  
두 가지 사용 방식을 지원합니다:

| 방식 | 실행 파일 | 실제 엔트리포인트 | 설명 |
|---|---|---|---|
| 대화형 터미널 | `terminal.py` | `entrypoints/terminal_main.py` | Ollama처럼 터미널에서 직접 대화 |
| MCP 서버 | `mcp_server.py` | `entrypoints/mcp_server_main.py` | Claude Code 등 MCP 클라이언트와 연결 |

---

## 아키텍처

```
[대화형 터미널]          [MCP 클라이언트 (Claude Code 등)]
  terminal.py               mcp_server.py
       │                          │
       └──────────┬───────────────┘
                  ↕  OpenAI-compatible REST API
           llama_cpp.server  (port 8080)
                  ↕
             ./models/*.gguf  (로컬 모델)
```

- 시작 시 `./models/` 디렉토리를 스캔해 `.gguf` 모델을 선택합니다.
- `llama_cpp.server`를 서브프로세스로 실행하고 OpenAI API로 통신합니다.
- LLM이 터미널 툴(`bash_exec`, `read_file`, `write_file`, `list_dir`, `find_files`, `search_code`, `replace_in_file`)을 자율적으로 호출합니다.
- `bash_exec`의 디렉토리 변경(`cd`)은 세션 전체에 유지되며, 다른 툴(`read_file`, `find_files`, `search_code`, `replace_in_file` 등)도 현재 디렉토리 기준으로 상대 경로를 해석합니다.
- 모델이 tool call을 텍스트로 출력하는 경우 자동으로 파싱해 실행합니다 (Qwen 2.5 / Qwen 3.x / ReAct 등 포맷 무관).

---

## 권한 확인 (Permission Checking)

MCP 도구 실행 전에 사용자 승인을 받도록 설정할 수 있습니다.

### terminal.py에서 권한 확인 활성화
실제 설정 위치는 `entrypoints/terminal_main.py`이며, 기본값으로 권한 확인이 **활성화**되어 있습니다.
각 도구 실행 전에 사용자에게 승인을 요청합니다.

권한 확인을 비활성화하려면 `entrypoints/terminal_main.py`의 다음 줄을 수정하세요:
```python
ASK_PERMISSION = False  # 권한 확인 비활성화
```

### mcp_server.py에서 권한 확인 활성화
MCP 서버 모드에서는 기본값으로 권한 확인이 **비활성화**되어 있습니다.
환경변수를 통해 권한 확인을 활성화할 수 있습니다:

```bash
# 권한 확인 활성화
export MCP_ASK_PERMISSION=true
python mcp_server.py

# 또는
MCP_ASK_PERMISSION=true python mcp_server.py
```

권한 확인이 활성화되면, 도구 사용 시 퍼미션 요청 메시지가 반환되며,
클라이언트(Claude 등)가 'approved'로 응답할 때까지 도구가 실행되지 않습니다.

---

## 설치

```bash
# 기본
pip install -r requirements.txt

# GPU 가속 (macOS Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python[server]

# GPU 가속 (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server]
```

모델은 `./models/` 디렉토리에 `.gguf` 파일로 넣어두면 됩니다.

---

## 사용법

### 1. 대화형 터미널 (`terminal.py`)

```bash
python terminal.py
```

터미널 모드에서는 기본적으로 assistant 응답을 한 번에 보여줍니다.
원하면 스트리밍을 켤 수 있습니다:

```bash
python terminal.py --stream
```

컨텍스트 윈도우와 자동 요약 임계값은 CLI 인자로 조정할 수 있습니다:

```bash
python terminal.py --n-ctx 8192 --summary-token-threshold 3200 --summary-keep-last 3
```

실행 흐름:

```
╭─ Terminal LLM Bridge  v0.1 ──────────────────────────╮
│  llama.cpp · MCP terminal tools · local inference    │
╰──────────────────────────────────────────────────────╯

  #   Model                                    Size
  1   Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M        5.7 GB

  Select model [1]:
  Start llama.cpp server on port 8080? [Y/n]:
  ⠋ Starting llama.cpp server …
  Server ready  port 8080  (14.2s)        ← 실패 시 원인과 로그 경로 표시

──────── ready — type a task or /help ────────

> 현재 디렉토리의 Python 파일 목록 보여줘

  find_files  pattern="*.py", path="."
    ./terminal.py
    ./mcp_server.py
    ./basic_test.py

╭──────────────────────────────────────╮
│ 3개의 Python 파일이 있습니다.          │
╰──────────────────────────────────────╯

>
```

프롬프트에 현재 디렉토리가 표시되고, `cd` 명령이 세션 전체에서 유지됩니다:

```
~/Documents > cd ~/Downloads
~/Downloads > ls
```

**슬래시 명령어:**

| 명령어 | 설명 |
|---|---|
| `/help` | 도움말 표시 |
| `/clear` | 화면 지우기 |
| `/reset` | 대화 메모리 초기화 |
| `/exit` | 종료 (Ctrl-C / Ctrl-D도 가능) |

---

### 세션 메모리 & 자동 요약

대화 히스토리는 세션 전체에서 유지됩니다. 이전 대화 내용을 기억한 채로 이어서 작업할 수 있습니다.

토큰 수가 임계치(기본 2000 토큰)를 초과하면 LLM이 자동으로 히스토리를 요약하고 압축합니다:

```
  Memory compressed  ~2400 → ~320 tokens
```

런타임에서 아래 인자로 조정할 수 있습니다:

```bash
python terminal.py --summary-token-threshold 2000 --summary-keep-last 2
```

`--summary-token-threshold`: 요약을 시작할 토큰 임계값(기본 2000)  
`--summary-keep-last`: 요약 후 유지할 최근 대화 쌍 수(기본 2)

---

### 2. MCP 서버 (`mcp_server.py`)

Claude Code 또는 다른 MCP 클라이언트에서 로컬 LLM을 터미널 에이전트로 사용합니다.

```bash
python mcp_server.py
```

MCP 서버도 컨텍스트 윈도우를 CLI 인자로 조정할 수 있습니다:

```bash
python mcp_server.py --n-ctx 8192
```

**Claude Code 연결 설정** (`~/.claude/claude.json`):

```json
{
  "mcpServers": {
    "terminal-llm": {
      "command": "conda",
      "args": ["run", "-n", "mcp_dev", "python", "/path/to/mcp_server.py"]
    }
  }
}
```

**노출 툴:**

| 툴 | 설명 |
|---|---|
| `bash_exec` | 쉘 명령어 실행 |
| `read_file` | 파일 읽기 |
| `write_file` | 파일 쓰기 |
| `list_dir` | 디렉토리 목록 |
| `find_files` | 글롭 패턴으로 파일 검색 (예: `*.py`) |
| `search_code` | 코드/텍스트 검색 (일반 문자열 또는 정규식) |
| `replace_in_file` | 파일 내 텍스트 1회 치환 (간단 코드 수정) |
| `run_task` | 자연어 태스크 → LLM이 위 툴들을 자율 호출 |

---

## 모델 호환성

`llama_cpp.server`가 모델의 tool call 출력을 OpenAI 구조화 형식으로 변환하지 못해도, 브리지가 텍스트를 직접 파싱해 실행합니다. 다음 포맷을 순서대로 시도합니다:

| 순서 | 포맷 | 대표 모델 |
|---|---|---|
| 1 | `<tool_call>{"name":…, "arguments":{…}}</tool_call>` | Qwen 2.5 |
| 2 | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` | Qwen 3.x |
| 3 | `Action: NAME` / `Action Input: {…}` | ReAct 계열 |
| 4 | 응답 내 bare JSON (알려진 툴 이름만 인정) | 기타 |

---

## 서버 시작 실패 시

서버가 시작되지 않으면 종료 코드와 마지막 출력 25줄, 전체 로그 경로를 출력합니다:

```
  Server process exited (code 1)

  Last server output:
  llama_model_load: error loading model './models/...'
  error: unable to load model
  ...

  Full log: /tmp/llama_server_xxxxxx.log
```

자주 발생하는 원인:

| 증상 | 원인 |
|---|---|
| `error loading model` | 모델 파일 경로 오류 또는 파일 손상 |
| `address already in use` | 포트 8080이 이미 사용 중 |
| `out of memory` | VRAM/RAM 부족 — 더 작은 모델 사용 |
| `ModuleNotFoundError` | `llama-cpp-python[server]` 미설치 |

---

## 파일 구조

```
mcp_terminal_llama_cpp/
├── terminal.py         # thin wrapper -> entrypoints/terminal_main.py
├── mcp_server.py       # thin wrapper -> entrypoints/mcp_server_main.py
├── entrypoints/
│   ├── terminal_main.py
│   └── mcp_server_main.py
├── app/
│   ├── executor.py
│   ├── policy.py
│   ├── validators.py
│   ├── tool_registry.py
│   ├── memory.py
│   ├── agent_loop.py
│   └── parsers/tool_call_parser.py
├── tests/
│   ├── test_policy.py
│   ├── test_parser.py
│   └── test_executor.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md     # 툴 추가 가이드
└── models/
    └── *.gguf          # 여기에 모델 파일을 넣으세요
```

새 툴을 추가하려면 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

---

## Migration Notes (2026-04)

- `terminal.py`, `mcp_server.py`는 실행 편의를 위한 thin wrapper로 유지됩니다.
- 실제 런타임 구현은 `entrypoints/`와 `app/` 하위 공유 모듈로 분리되었습니다.

### Known Limitation

- 자동화 환경(비대화형 stdin)에서 `python terminal.py` 실행 시 `Prompt.ask`가 EOF를 반환해 종료될 수 있습니다.
- 실제 수동 검증은 TTY가 있는 터미널에서 수행해야 합니다.

---

## 의존성

```
llama-cpp-python[server]   # llama.cpp + OpenAI 호환 서버
mcp                        # Anthropic MCP Python SDK
openai                     # llama_cpp.server API 호출
rich                       # 터미널 UI
```
