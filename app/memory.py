"""Session memory and summarization helpers."""

from typing import Any, Callable

SYSTEM_PROMPT = """\
You are a terminal assistant. Complete the user's task using the available tools.
Be concise. Only use tools when necessary.

Available tools:
- bash_exec(command, timeout=30) : execute a shell command
- read_file(path)                : read a file
- write_file(path, content)      : write a file
- list_dir(path=".")             : list directory contents
- find_files(pattern="*.py", path=".") : find files by glob pattern
- search_code(query, path=".", is_regex=False, max_results=100) : search code text
- replace_in_file(path, old_text, new_text) : replace text once in a file

To call a tool, output its name and arguments as JSON:
{"name": "bash_exec", "arguments": {"command": "ls -la"}}

Call one tool at a time and wait for the result before continuing.\
"""

SUMMARY_TOKEN_THRESHOLD = 2000
SUMMARY_KEEP_LAST = 2


def make_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def count_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total characters / 4."""
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for tool_call in message.get("tool_calls", []):
            total += len(tool_call.get("function", {}).get("arguments", ""))
    return total // 4


def summarize_messages(llm_call: Callable[..., Any], messages: list[dict]) -> str:
    """Ask the LLM to summarize the conversation history."""
    history_text = []
    for message in messages[1:]:
        role = message["role"]
        content = message.get("content") or ""
        if role == "user":
            history_text.append(f"User: {content}")
        elif role == "assistant":
            if content:
                history_text.append(f"Assistant: {content}")
            for tool_call in message.get("tool_calls", []):
                history_text.append(
                    f"Assistant called {tool_call['function']['name']}({tool_call['function']['arguments']})"
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

    response = llm_call([{"role": "user", "content": prompt}], [], max_tokens=512)
    return response.choices[0].message.content or ""


def maybe_compress_messages(
    messages: list[dict],
    llm_call: Callable[..., Any],
    on_compressed: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Summarize old history when the token estimate exceeds the threshold."""
    if count_tokens(messages) < SUMMARY_TOKEN_THRESHOLD:
        return messages

    summary = summarize_messages(llm_call, messages)

    recent = [message for message in messages[1:] if message["role"] != "system"]
    keep = recent[-(SUMMARY_KEEP_LAST * 2):]

    compressed = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": f"[Previous session summary]\n{summary}"},
        *keep,
    ]

    if on_compressed:
        on_compressed(count_tokens(messages), count_tokens(compressed))

    return compressed