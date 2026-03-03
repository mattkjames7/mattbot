# MattBot

[![Tests](https://github.com/mattkjames7/mattbot/actions/workflows/test.yml/badge.svg)](https://github.com/mattkjames7/mattbot/actions/workflows/test.yml)

MattBot - An AI agent system with tool-calling capabilities.

## Features

- **LLM Integration**: Ollama client for chat/generate with streaming support
- **Tool Calling**: OpenAI-compatible function calling format
- **Comprehensive Testing**: 106 unit tests with 100% pass rate
- **CI/CD**: Automated testing on pull requests

## Available Tools

### All Tools Implemented (8/8) ✅
✅ **run_bash_command** - Execute shell commands with working directory support  
✅ **read_file** - Read files with optional line range selection  
✅ **write_file** - Create/overwrite files with auto parent directory creation  
✅ **edit_file** - In-place file editing with find/replace  
✅ **list_directory** - List files and directories (recursive optional)  
✅ **grep_search** - Regex text search with file pattern filtering  
✅ **semantic_search** - AI-powered semantic code search using embeddings  
✅ **web_search** - DuckDuckGo web search for real-time internet information

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/mattkjames7/mattbot.git
cd mattbot

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### As a Package

```bash
# Install from source
pip install git+https://github.com/mattkjames7/mattbot.git

# Or install in editable mode for development
pip install -e .

# With dev dependencies (pytest, coverage)
pip install -e ".[dev]"
```

## Configuration

MattBot creates a config file on first run (or with `--init-config`):

- `~/.config/mattbot/config.toml`

Example:

```toml
[agent]
model = "gpt-oss:latest"
ollama_url = "http://localhost:11434"
max_context_tokens = 8192
temperature = 0.7
embedding_model = "all-minilm:l6-v2"
history_length = 1000
```

Override precedence is:

1. CLI args (`--model`, `--ollama-url`, `--max-context-tokens`, `--temperature`, `--history-length`)
2. Environment variables (`MATTBOT_MODEL`, `MATTBOT_OLLAMA_URL`, `MATTBOT_MAX_CONTEXT_TOKENS`, `MATTBOT_TEMPERATURE`, `MATTBOT_EMBEDDING_MODEL`, `MATTBOT_HISTORY_LENGTH`)
3. Config file
4. Built-in defaults

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_tool_executor.py -v

# Skip integration tests (recommended for CI)
pytest tests/ -k "not integration"
```

## Usage

### Interactive CLI

The easiest way to use the agent is through the interactive CLI:

```bash
# Start the agent in the current directory
mattbot

# Use a different model
mattbot --model llama2
```

The CLI provides an interactive chat interface where:
- The agent can call tools automatically based on your requests
- All file operations work from your current working directory
- Press **Ctrl+D** to exit gracefully
- Tool executions are displayed in real-time

Example session:
```
You: List all Python files in this directory

🔧 Calling tool: list_directory
   Arguments: {...}
✓ Tool completed successfully
```

### Programmatic Usage

```python
from mattbot.llm import OllamaClient
from mattbot.tool_executor import execute_tool
from mattbot.tools import TOOLS

# Initialize LLM client
client = OllamaClient()

# Execute a tool directly
result = execute_tool('semantic_search', 
                     query='HTTP client code', 
                     path='mattbot', 
                     limit=3)

# Use with LLM (tool calling)
response = client.chat(
    model="gpt-oss:latest",
    messages=[{"role": "user", "content": "List Python files"}],
    tools=TOOLS
)
```

## Architecture

- `mattbot/cli.py` - Interactive CLI interface
- `mattbot/llm.py` - OllamaClient for API communication
- `mattbot/tools.py` - Tool definitions in OpenAI format
- `mattbot/tool_executor.py` - Tool implementations
- `tests/` - Comprehensive test suite

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Triggers**: Pull requests to main/master branch and manual workflow dispatch
- **Test Exclusions**: Integration tests requiring external Ollama API are skipped in CI
- **Python Version**: Tests run on Python 3.12
- **Test Command**: `pytest tests/ -k "not integration"`

To run the same tests locally:
```bash
pytest tests/ -k "not integration" -v
```

## Dependencies

- Python 3.12+
- requests - HTTP client
- ddgs - DuckDuckGo web search
- pytest - Testing framework
- numpy - For embeddings calculations (lightweight)

**Note**: Semantic search uses Ollama's embeddings API with the `all-minilm:l6-v2` model, avoiding the need for heavy ML dependencies like PyTorch.
