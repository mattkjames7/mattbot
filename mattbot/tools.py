"""
Tool definitions for the agent.

This module defines the tools available to the agent in the format expected by
Ollama/gpt-oss models (OpenAI-compatible function calling format).
"""

from typing import List, Dict, Any


# Tool definitions following OpenAI/Ollama function calling format
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Execute a bash command in the terminal. Use this to run shell commands, install packages, compile code, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "The working directory to execute the command in (optional)"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional: line number to start reading from (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional: line number to stop reading at (inclusive, 1-indexed)"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a specific section of a file by replacing old content with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to edit"
                    },
                    "old_content": {
                        "type": "string",
                        "description": "The exact content to replace (must match exactly)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content to insert in place of old_content"
                    }
                },
                "required": ["file_path", "old_content", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional directory path to list (defaults to current directory)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list files recursively (tree view)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern in files using grep-like functionality. Fast text search in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern (can be regex)"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory or file path to search in"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional: glob pattern to filter files (e.g., '*.py')"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case sensitive (default: false)"
                    }
                },
                "required": ["pattern", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search for code or text using semantic similarity. Better for finding code by meaning rather than exact keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional: limit search to specific directory"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional: maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns relevant web search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def get_tools() -> List[Dict[str, Any]]:
    """
    Get the list of available tools.
    
    Returns:
        List of tool definitions in OpenAI/Ollama function calling format
    """
    return TOOLS


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """
    Get a specific tool definition by name.
    
    Args:
        name: The name of the tool to retrieve
        
    Returns:
        Tool definition dict or None if not found
    """
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """
    Get a list of all available tool names.
    
    Returns:
        List of tool names
    """
    return [tool["function"]["name"] for tool in TOOLS]
