"""
Tool implementations for the agent.

This module contains the actual implementations of the tools defined in tools.py.
"""

import subprocess
import os
from typing import Dict, Any, Optional


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    pass


def run_bash_command(command: str, working_directory: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a bash command in the terminal.
    
    Args:
        command: The bash command to execute
        working_directory: Optional working directory to execute the command in
        
    Returns:
        Dictionary containing:
            - success: bool indicating if command succeeded
            - stdout: standard output from the command
            - stderr: standard error from the command
            - exit_code: exit code from the command
            - command: the command that was executed
            
    Raises:
        ToolExecutionError: If the working directory doesn't exist
    """
    # Validate working directory if provided
    if working_directory and not os.path.isdir(working_directory):
        raise ToolExecutionError(f"Working directory does not exist: {working_directory}")
    
    # Use current directory if not specified
    cwd = working_directory or os.getcwd()
    
    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "command": command,
            "working_directory": cwd
        }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out after 300 seconds",
            "exit_code": -1,
            "command": command,
            "working_directory": cwd
        }
    
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "command": command,
            "working_directory": cwd
        }


def read_file(file_path: str, start_line: Optional[int] = None, 
              end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed, inclusive)
        
    Returns:
        Dictionary containing:
            - success: bool indicating if read succeeded
            - content: file contents (or selected lines)
            - file_path: the file path that was read
            - lines_read: number of lines read
            - error: error message if failed
    """
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            return {
                "success": False,
                "content": "",
                "file_path": file_path,
                "lines_read": 0,
                "error": f"File not found: {file_path}"
            }
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # If no line range specified, return entire file
        if start_line is None and end_line is None:
            content = ''.join(lines)
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "lines_read": len(lines),
                "total_lines": len(lines)
            }
        
        # Validate line numbers
        total_lines = len(lines)
        start = start_line if start_line is not None else 1
        end = end_line if end_line is not None else total_lines
        
        # Validate range
        if start < 1:
            return {
                "success": False,
                "content": "",
                "file_path": file_path,
                "lines_read": 0,
                "error": f"Invalid start_line: {start}. Line numbers start at 1."
            }
        
        if end < start:
            return {
                "success": False,
                "content": "",
                "file_path": file_path,
                "lines_read": 0,
                "error": f"Invalid range: end_line ({end}) must be >= start_line ({start})"
            }
        
        # Adjust for 0-indexing and slice the lines
        start_idx = start - 1
        end_idx = min(end, total_lines)
        selected_lines = lines[start_idx:end_idx]
        content = ''.join(selected_lines)
        
        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "lines_read": len(selected_lines),
            "total_lines": total_lines,
            "start_line": start,
            "end_line": end_idx
        }
    
    except UnicodeDecodeError:
        return {
            "success": False,
            "content": "",
            "file_path": file_path,
            "lines_read": 0,
            "error": "File contains non-UTF-8 content and cannot be read as text"
        }
    
    except PermissionError:
        return {
            "success": False,
            "content": "",
            "file_path": file_path,
            "lines_read": 0,
            "error": f"Permission denied: {file_path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "content": "",
            "file_path": file_path,
            "lines_read": 0,
            "error": f"Error reading file: {str(e)}"
        }


def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Dictionary containing:
            - success: bool indicating if write succeeded
            - file_path: the file path that was written
            - bytes_written: number of bytes written
            - error: error message if failed
    """
    try:
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Write the content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return {
            "success": True,
            "file_path": file_path,
            "bytes_written": file_size,
            "lines_written": content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        }
    
    except PermissionError:
        return {
            "success": False,
            "file_path": file_path,
            "bytes_written": 0,
            "error": f"Permission denied: {file_path}"
        }
    
    except IsADirectoryError:
        return {
            "success": False,
            "file_path": file_path,
            "bytes_written": 0,
            "error": f"Cannot write to directory: {file_path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "file_path": file_path,
            "bytes_written": 0,
            "error": f"Error writing file: {str(e)}"
        }


def edit_file(file_path: str, old_content: str, new_content: str) -> Dict[str, Any]:
    """
    Edit a file by replacing old content with new content.
    
    Args:
        file_path: Path to the file to edit
        old_content: Content to replace
        new_content: New content to insert
        
    Returns:
        Dictionary containing:
            - success: bool indicating if edit succeeded
            - file_path: the file path that was edited
            - error: error message if failed
    """
    # TODO: Implement in next step
    raise NotImplementedError("edit_file not yet implemented")


def list_directory(path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    List files and directories in a path.
    
    Args:
        path: Directory path to list
        recursive: Whether to list recursively
        
    Returns:
        Dictionary containing:
            - success: bool indicating if listing succeeded
            - entries: list of files/directories
            - path: the path that was listed
            - error: error message if failed
    """
    # TODO: Implement in next step
    raise NotImplementedError("list_directory not yet implemented")


def grep_search(pattern: str, path: str, file_pattern: Optional[str] = None,
                case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Search for a pattern in files.
    
    Args:
        pattern: Search pattern (can be regex)
        path: Directory or file path to search
        file_pattern: Optional glob pattern to filter files
        case_sensitive: Whether search is case sensitive
        
    Returns:
        Dictionary containing:
            - success: bool indicating if search succeeded
            - matches: list of matches with file, line number, and content
            - pattern: the pattern that was searched
            - error: error message if failed
    """
    # TODO: Implement in next step
    raise NotImplementedError("grep_search not yet implemented")


def semantic_search(query: str, path: Optional[str] = None, 
                    limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Search for code using semantic similarity.
    
    Args:
        query: Natural language search query
        path: Optional path to limit search
        limit: Optional maximum number of results
        
    Returns:
        Dictionary containing:
            - success: bool indicating if search succeeded
            - results: list of relevant code snippets
            - query: the query that was searched
            - error: error message if failed
    """
    # TODO: Implement in next step
    raise NotImplementedError("semantic_search not yet implemented")


def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dictionary containing:
            - success: bool indicating if search succeeded
            - results: list of search results with title, url, snippet
            - query: the query that was searched
            - error: error message if failed
    """
    # TODO: Implement in next step
    raise NotImplementedError("web_search not yet implemented")


# Tool registry mapping tool names to functions
TOOL_IMPLEMENTATIONS = {
    "run_bash_command": run_bash_command,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_directory": list_directory,
    "grep_search": grep_search,
    "semantic_search": semantic_search,
    "web_search": web_search,
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a tool by name with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool
        
    Returns:
        Result dictionary from the tool execution
        
    Raises:
        ValueError: If tool name is not recognized
    """
    if tool_name not in TOOL_IMPLEMENTATIONS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_func = TOOL_IMPLEMENTATIONS[tool_name]
    return tool_func(**kwargs)
