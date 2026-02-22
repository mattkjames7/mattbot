# Agent OSS CLI Demo

This demonstrates the interactive CLI in action.

## Quick Start

```bash
# Activate your environment
source env/bin/activate

# Start the agent
agent
```

## Example Usage

Once the agent starts, you can interact with it naturally:

### Example 1: File Operations
```
You: Show me what Python files are in the agent directory

# The agent will call list_directory automatically
# and show you the results
```

### Example 2: Code Search
```
You: Find where the OllamaClient is defined

# The agent will use semantic_search or grep_search
# to locate the code for you
```

### Example 3: Web Research
```
You: What's the latest version of FastAPI?

# The agent will use web_search to find current info
```

### Example 4: File Editing
```
You: Add a docstring to the execute_tool function

# The agent will read the file, understand the context,
# and make the edit for you
```

### Example 5: Command Execution
```
You: Run the tests for the tool_executor module

# The agent will execute: pytest tests/test_tool_executor.py
```

## Tips

- **Be Natural**: Just describe what you want - the agent will figure out which tools to use
- **Context Matters**: The agent works from your current directory
- **Interrupt Anytime**: Press Ctrl+C to stop the current operation
- **Exit Gracefully**: Press Ctrl+D when you're done
- **Tool Feedback**: Watch as tools are called and see their results in real-time

## Advanced Usage

### Use Different Models
```bash
agent --model llama2
agent --model codellama
```

### Change Working Directory
```bash
cd /your/project/directory
agent
```

The agent will now work within that directory context.

## Safety Notes

- The agent can execute shell commands - review tool calls before confirming important operations
- File edits are performed directly - consider using version control
- Web searches use DuckDuckGo - results depend on current internet content

## Troubleshooting

If the agent doesn't respond:
1. Check your Ollama server is running: `curl http://192.168.0.34:11434/api/tags`
2. Verify the model exists: `ollama list`
3. Check the server URL in `agent/llm.py`

If a tool fails:
- Read the error message - it will explain what went wrong
- The agent can often recover and try a different approach
- You can rephrase your request to guide the agent differently
