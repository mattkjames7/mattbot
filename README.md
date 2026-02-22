# Agent OSS

An AI agent system with tool-calling capabilities, similar to Copilot/Cursor.

## Features

- **LLM Integration**: Ollama client for chat/generate with streaming support
- **Tool Calling**: OpenAI-compatible function calling format
- **Comprehensive Testing**: 134 unit tests with 100% pass rate

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

## Setup

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your Ollama server URL in `agent/llm.py` (default: `http://192.168.0.34:11434`)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_tool_executor.py -v

# Skip integration tests
pytest tests/ -k "not integration"
```

## Usage Example

```python
from agent.llm import OllamaClient
from agent.tool_executor import execute_tool
from agent.tools import TOOLS

# Initialize LLM client
client = OllamaClient()

# Execute a tool
result = execute_tool('semantic_search', 
                     query='HTTP client code', 
                     path='agent', 
                     limit=3)

# Use with LLM (tool calling)
response = client.chat(
    model="gpt-oss:latest",
    messages=[{"role": "user", "content": "List Python files"}],
    tools=TOOLS
)
```

## Architecture

- `agent/llm.py` - OllamaClient for API communication
- `agent/tools.py` - Tool definitions in OpenAI format
- `agent/tool_executor.py` - Tool implementations
- `tests/` - Comprehensive test suite

## Dependencies

- Python 3.12+
- requests - HTTP client
- sentence-transformers - Semantic search embeddings
- ddgs - DuckDuckGo web search
- pytest - Testing framework
