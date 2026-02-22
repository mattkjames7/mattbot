"""
Tool implementations for the agent.

This module contains the actual implementations of the tools defined in tools.py.
"""

import subprocess
import os
import re
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
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            return {
                "success": False,
                "file_path": file_path,
                "error": f"File not found: {file_path}"
            }
        
        # Read the current content
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Check if old_content exists in the file
        if old_content not in current_content:
            return {
                "success": False,
                "file_path": file_path,
                "error": "Old content not found in file. Make sure the content matches exactly."
            }
        
        # Count occurrences
        occurrences = current_content.count(old_content)
        
        # Replace old content with new content
        new_file_content = current_content.replace(old_content, new_content)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)
        
        return {
            "success": True,
            "file_path": file_path,
            "replacements_made": occurrences,
            "old_content_length": len(old_content),
            "new_content_length": len(new_content)
        }
    
    except UnicodeDecodeError:
        return {
            "success": False,
            "file_path": file_path,
            "error": "File contains non-UTF-8 content and cannot be edited as text"
        }
    
    except PermissionError:
        return {
            "success": False,
            "file_path": file_path,
            "error": f"Permission denied: {file_path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "file_path": file_path,
            "error": f"Error editing file: {str(e)}"
        }


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
    try:
        # Check if path exists
        if not os.path.exists(path):
            return {
                "success": False,
                "entries": [],
                "path": path,
                "error": f"Path does not exist: {path}"
            }
        
        # Check if path is a directory
        if not os.path.isdir(path):
            return {
                "success": False,
                "entries": [],
                "path": path,
                "error": f"Path is not a directory: {path}"
            }
        
        entries = []
        
        if recursive:
            # Recursive listing - walk the directory tree
            for root, dirs, files in os.walk(path):
                # Get relative path from base
                rel_root = os.path.relpath(root, path)
                if rel_root == '.':
                    rel_root = ''
                
                # Add directories
                for dir_name in sorted(dirs):
                    rel_path = os.path.join(rel_root, dir_name) if rel_root else dir_name
                    full_path = os.path.join(root, dir_name)
                    entries.append({
                        "name": dir_name,
                        "path": rel_path,
                        "type": "directory",
                        "size": None
                    })
                
                # Add files
                for file_name in sorted(files):
                    rel_path = os.path.join(rel_root, file_name) if rel_root else file_name
                    full_path = os.path.join(root, file_name)
                    try:
                        size = os.path.getsize(full_path)
                    except (OSError, PermissionError):
                        size = None
                    
                    entries.append({
                        "name": file_name,
                        "path": rel_path,
                        "type": "file",
                        "size": size
                    })
        else:
            # Non-recursive listing - just immediate children
            items = sorted(os.listdir(path))
            
            for item in items:
                full_path = os.path.join(path, item)
                
                if os.path.isdir(full_path):
                    entries.append({
                        "name": item,
                        "path": item,
                        "type": "directory",
                        "size": None
                    })
                elif os.path.isfile(full_path):
                    try:
                        size = os.path.getsize(full_path)
                    except (OSError, PermissionError):
                        size = None
                    
                    entries.append({
                        "name": item,
                        "path": item,
                        "type": "file",
                        "size": size
                    })
                else:
                    # Other types (symlinks, etc.)
                    entries.append({
                        "name": item,
                        "path": item,
                        "type": "other",
                        "size": None
                    })
        
        return {
            "success": True,
            "entries": entries,
            "path": path,
            "total_entries": len(entries),
            "files": sum(1 for e in entries if e["type"] == "file"),
            "directories": sum(1 for e in entries if e["type"] == "directory")
        }
    
    except PermissionError:
        return {
            "success": False,
            "entries": [],
            "path": path,
            "error": f"Permission denied: {path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "entries": [],
            "path": path,
            "error": f"Error listing directory: {str(e)}"
        }


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
    try:
        # Check if path exists
        if not os.path.exists(path):
            return {
                "success": False,
                "matches": [],
                "pattern": pattern,
                "error": f"Path does not exist: {path}"
            }
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                "success": False,
                "matches": [],
                "pattern": pattern,
                "error": f"Invalid regex pattern: {str(e)}"
            }
        
        matches = []
        files_searched = 0
        
        # Determine files to search
        if os.path.isfile(path):
            # Single file
            files_to_search = [path]
        else:
            # Directory - find all files
            files_to_search = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Apply file pattern filter if provided
                    if file_pattern:
                        import fnmatch
                        if not fnmatch.fnmatch(file, file_pattern):
                            continue
                    
                    files_to_search.append(file_path)
        
        # Search each file
        for file_path in files_to_search:
            try:
                # Skip binary files by trying to read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                files_searched += 1
                
                # Search each line
                for line_num, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append({
                            "file": file_path,
                            "line_number": line_num,
                            "line_content": line.rstrip('\n\r'),
                            "column": regex.search(line).start() if regex.search(line) else 0
                        })
            
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read as text or don't have permissions
                continue
            except Exception:
                # Skip any other problematic files
                continue
        
        return {
            "success": True,
            "matches": matches,
            "pattern": pattern,
            "total_matches": len(matches),
            "files_searched": files_searched,
            "case_sensitive": case_sensitive
        }
    
    except PermissionError:
        return {
            "success": False,
            "matches": [],
            "pattern": pattern,
            "error": f"Permission denied: {path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "matches": [],
            "pattern": pattern,
            "error": f"Error searching: {str(e)}"
        }


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
    try:
        import requests
        import numpy as np
        
        # Use default path if not provided
        if path is None:
            path = "."
        
        # Check if path exists
        if not os.path.exists(path):
            return {
                "success": False,
                "results": [],
                "query": query,
                "error": f"Path does not exist: {path}"
            }
        
        # Ollama embeddings endpoint
        ollama_url = "http://192.168.0.34:11434/api/embeddings"
        embedding_model = "all-minilm:l6-v2"
        
        def get_embedding(text: str) -> list:
            """Get embedding from Ollama API."""
            try:
                response = requests.post(
                    ollama_url,
                    json={
                        "model": embedding_model,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception as e:
                raise Exception(f"Failed to get embedding: {str(e)}")
        
        # Collect text chunks from files
        chunks = []
        
        if os.path.isfile(path):
            files_to_search = [path]
        else:
            files_to_search = []
            # Only search text files (common code extensions)
            text_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
                             '.go', '.rs', '.rb', '.php', '.txt', '.md', '.json',
                             '.yaml', '.yml', '.xml', '.html', '.css', '.sh'}
            
            for root, dirs, files in os.walk(path):
                # Skip common non-code directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', 
                                                         '.venv', 'venv', 'env', 'dist', 'build'}]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    if ext in text_extensions:
                        files_to_search.append(file_path)
        
        # Read and chunk files
        for file_path in files_to_search[:100]:  # Limit to 100 files for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks (by function/class or fixed size)
                lines = content.split('\n')
                
                # Create chunks of ~20 lines each
                chunk_size = 20
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i:i+chunk_size]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    if chunk_text.strip():  # Skip empty chunks
                        chunks.append({
                            'file': file_path,
                            'start_line': i + 1,
                            'end_line': min(i + chunk_size, len(lines)),
                            'content': chunk_text
                        })
            
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not chunks:
            return {
                "success": True,
                "results": [],
                "query": query,
                "total_results": 0,
                "message": "No text files found to search"
            }
        
        # Get query embedding
        query_embedding = np.array(get_embedding(query))
        
        # Get embeddings for all chunks
        chunk_embeddings = []
        for chunk in chunks:
            try:
                embedding = get_embedding(chunk['content'])
                chunk_embeddings.append(embedding)
            except Exception:
                # Skip chunks that fail to embed
                continue
        
        if not chunk_embeddings:
            return {
                "success": True,
                "results": [],
                "query": query,
                "total_results": 0,
                "message": "Failed to generate embeddings for content"
            }
        
        chunk_embeddings = np.array(chunk_embeddings)
        # Filter chunks to match embeddings (in case some failed)
        chunks = chunks[:len(chunk_embeddings)]
        
        # Calculate cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        max_results = limit if limit else 5
        top_indices = np.argsort(similarities)[::-1][:max_results]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            # Only include results with reasonable similarity
            if similarity > 0.1:  # Threshold to filter very low matches
                chunk = chunks[idx]
                results.append({
                    'file': chunk['file'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'content': chunk['content'],
                    'similarity': round(similarity, 3)
                })
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "total_results": len(results),
            "files_searched": len(files_to_search)
        }
    
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "query": query,
            "error": f"Error performing semantic search: {str(e)}"
        }


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
    try:
        # Import DuckDuckGo search
        try:
            from ddgs import DDGS
        except ImportError:
            return {
                "success": False,
                "results": [],
                "query": query,
                "error": "ddgs not installed. Run: pip install ddgs"
            }
        
        # Validate inputs
        if not query or not query.strip():
            return {
                "success": False,
                "results": [],
                "query": query,
                "error": "Search query cannot be empty"
            }
        
        if num_results < 1:
            return {
                "success": False,
                "results": [],
                "query": query,
                "error": "num_results must be at least 1"
            }
        
        # Perform search
        results = []
        
        with DDGS() as ddgs:
            # Use text search
            search_results = ddgs.text(
                query,
                max_results=min(num_results, 20)  # Cap at 20 to avoid rate limits
            )
            
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": result.get("href", "").split('/')[2] if result.get("href") else ""
                })
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "total_results": len(results)
        }
    
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "query": query,
            "error": f"Error performing web search: {str(e)}"
        }


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
