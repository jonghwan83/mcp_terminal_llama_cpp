# MCP Terminal LLM Bridge

로컬 LLM(llama.cpp)으로 터미널 작업을 수행하는 도구.  
두 가지 사용 방식을 지원합니다:

| 방식 | 파일 | 설명 |
|---|---|---|
| 대화형 터미널 | `terminal.py` | Ollama처럼 터미널에서 직접 대화 |
| MCP 서버 | `mcp_server.py` | Claude Code 등 MCP 클라이언트와 연결 |

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
- LLM이 터미널 툴(`bash_exec`, `read_file`, `write_file`, `list_dir`)을 자율적으로 호출합니다.
- `bash_exec`의 디렉토리 변경(`cd`)은 세션 전체에 유지되며, 다른 툴(read_file 등)도 현재 디렉토리 기준으로 상대 경로를 해석합니다.
- 모델이 tool call을 텍스트로 출력하는 경우 자동으로 파싱해 실행합니다 (Qwen 2.5 / Qwen 3.x / ReAct 등 포맷 무관).

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

  bash_exec  find . -name "*.py"
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

`terminal.py` 상단의 상수로 조정할 수 있습니다:

```python
SUMMARY_TOKEN_THRESHOLD = 2000  # 이 토큰 수를 넘으면 요약
SUMMARY_KEEP_LAST = 2           # 요약 후 유지할 최근 대화 쌍 수
```

---

### 2. MCP 서버 (`mcp_server.py`)

Claude Code 또는 다른 MCP 클라이언트에서 로컬 LLM을 터미널 에이전트로 사용합니다.

```bash
python mcp_server.py
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
├── terminal.py         # 대화형 터미널 브리지
├── mcp_server.py       # MCP stdio 서버
├── requirements.txt
├── README.md
├── CLAUDE.md
├── CONTRIBUTING.md     # 툴 추가 가이드
└── models/
    └── *.gguf          # 여기에 모델 파일을 넣으세요
```

새 툴을 추가하려면 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

---

## 의존성

```
llama-cpp-python[server]   # llama.cpp + OpenAI 호환 서버
mcp                        # Anthropic MCP Python SDK
openai                     # llama_cpp.server API 호출
rich                       # 터미널 UI
```
