"""Tool schema registry shared across entry points."""


TOOL_SCHEMAS: list[dict] = [
	{
		"type": "function",
		"function": {
			"name": "bash_exec",
			"description": "Execute a shell command and return stdout/stderr.",
			"parameters": {
				"type": "object",
				"properties": {
					"command": {"type": "string", "description": "Shell command to execute"},
					"timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
				},
				"required": ["command"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "read_file",
			"description": "Read the contents of a file.",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "File path to read"},
				},
				"required": ["path"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "write_file",
			"description": "Write text content to a file (creates parent dirs as needed).",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "File path to write"},
					"content": {"type": "string", "description": "Content to write"},
				},
				"required": ["path", "content"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "list_dir",
			"description": "List the contents of a directory.",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "Directory path (default '.')", "default": "."},
				},
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "find_files",
			"description": "Find files by glob pattern under a directory.",
			"parameters": {
				"type": "object",
				"properties": {
					"pattern": {"type": "string", "description": "Glob pattern (default '*.py')", "default": "*.py"},
					"path": {"type": "string", "description": "Base directory (default '.')", "default": "."},
				},
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "search_code",
			"description": "Search text in files under a directory (supports regex).",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "Text or regex pattern to search"},
					"path": {"type": "string", "description": "Base directory (default '.')", "default": "."},
					"is_regex": {"type": "boolean", "description": "Treat query as regex (default false)", "default": False},
					"max_results": {"type": "integer", "description": "Maximum matches to return (default 100)", "default": 100},
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "replace_in_file",
			"description": "Replace text in a file once (simple code editing helper).",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "File path"},
					"old_text": {"type": "string", "description": "Text to find"},
					"new_text": {"type": "string", "description": "Replacement text"},
				},
				"required": ["path", "old_text", "new_text"],
			},
		},
	},
]

TOOL_NAMES: set[str] = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def build_mcp_tool_specs() -> list[dict]:
	"""Return MCP tool specs derived from the shared function-calling schema."""
	specs: list[dict] = []
	for schema in TOOL_SCHEMAS:
		fn = schema["function"]
		specs.append(
			{
				"name": fn["name"],
				"description": fn["description"],
				"inputSchema": fn["parameters"],
			}
		)
	return specs
