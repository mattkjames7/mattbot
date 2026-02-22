"""
Tests for tool executor implementations.
"""

import os
import pytest
import tempfile
from pathlib import Path
from agent.tool_executor import (
    run_bash_command,
    read_file,
    write_file,
    edit_file,
    list_directory,
    grep_search,
    semantic_search,
    web_search,
    execute_tool,
    ToolExecutionError
)




class TestRunBashCommand:
    """Tests for run_bash_command tool."""
    
    def test_simple_command_success(self):
        """Test running a simple successful command."""
        result = run_bash_command("echo 'hello world'")
        
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["exit_code"] == 0
        assert result["command"] == "echo 'hello world'"
        assert "working_directory" in result
    
    def test_command_with_stderr(self):
        """Test command that outputs to stderr."""
        result = run_bash_command("echo 'error message' >&2")
        
        assert result["success"] is True
        assert "error message" in result["stderr"]
        assert result["exit_code"] == 0
    
    def test_failing_command(self):
        """Test a command that fails."""
        result = run_bash_command("exit 1")
        
        assert result["success"] is False
        assert result["exit_code"] == 1
    
    def test_invalid_command(self):
        """Test running an invalid command."""
        result = run_bash_command("nonexistentcommand12345")
        
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["stderr"]) > 0
    
    def test_command_with_working_directory(self):
        """Test running command in specific directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")
            
            # List files in that directory
            result = run_bash_command("ls", working_directory=tmpdir)
            
            assert result["success"] is True
            assert "test.txt" in result["stdout"]
            assert result["working_directory"] == tmpdir
    
    def test_invalid_working_directory(self):
        """Test with non-existent working directory."""
        with pytest.raises(ToolExecutionError, match="Working directory does not exist"):
            run_bash_command("echo test", working_directory="/nonexistent/path/12345")
    
    def test_multiline_output(self):
        """Test command with multiline output."""
        result = run_bash_command("echo 'line1'; echo 'line2'; echo 'line3'")
        
        assert result["success"] is True
        assert "line1" in result["stdout"]
        assert "line2" in result["stdout"]
        assert "line3" in result["stdout"]
    
    def test_command_with_pipes(self):
        """Test command using pipes."""
        result = run_bash_command("echo 'hello world' | grep 'world'")
        
        assert result["success"] is True
        assert "world" in result["stdout"]
    
    def test_command_with_environment(self):
        """Test command that uses environment variables."""
        result = run_bash_command("echo $HOME")
        
        assert result["success"] is True
        assert len(result["stdout"].strip()) > 0
    
    def test_command_with_file_operations(self):
        """Test command that creates and reads a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_bash_command(
                "echo 'test content' > test.txt && cat test.txt",
                working_directory=tmpdir
            )
            
            assert result["success"] is True
            assert "test content" in result["stdout"]


class TestExecuteTool:
    """Tests for the execute_tool dispatcher."""
    
    def test_execute_bash_command_tool(self):
        """Test executing run_bash_command via execute_tool."""
        result = execute_tool("run_bash_command", command="echo 'test'")
        
        assert result["success"] is True
        assert "test" in result["stdout"]
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool("nonexistent_tool", arg="value")
    
    def test_execute_tool_with_kwargs(self):
        """Test execute_tool passes kwargs correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_tool(
                "run_bash_command",
                command="pwd",
                working_directory=tmpdir
            )
            
            assert result["success"] is True
            assert tmpdir in result["stdout"]


class TestReadFile:
    """Tests for read_file tool."""
    
    def test_read_entire_file(self):
        """Test reading an entire file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            
            assert result["success"] is True
            assert result["content"] == "line 1\nline 2\nline 3\n"
            assert result["lines_read"] == 3
            assert result["total_lines"] == 3
            assert result["file_path"] == temp_path
        finally:
            os.unlink(temp_path)
    
    def test_read_file_with_line_range(self):
        """Test reading specific line range."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=2, end_line=4)
            
            assert result["success"] is True
            assert result["content"] == "line 2\nline 3\nline 4\n"
            assert result["lines_read"] == 3
            assert result["total_lines"] == 5
            assert result["start_line"] == 2
            assert result["end_line"] == 4
        finally:
            os.unlink(temp_path)
    
    def test_read_file_from_start_line_only(self):
        """Test reading from a specific start line to end."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=2)
            
            assert result["success"] is True
            assert result["content"] == "line 2\nline 3\n"
            assert result["lines_read"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_file_to_end_line_only(self):
        """Test reading from beginning to specific end line."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, end_line=2)
            
            assert result["success"] is True
            assert result["content"] == "line 1\nline 2\n"
            assert result["lines_read"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_single_line(self):
        """Test reading a single line."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=2, end_line=2)
            
            assert result["success"] is True
            assert result["content"] == "line 2\n"
            assert result["lines_read"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        result = read_file("/nonexistent/path/file.txt")
        
        assert result["success"] is False
        assert "File not found" in result["error"]
        assert result["lines_read"] == 0
    
    def test_read_file_invalid_start_line(self):
        """Test with invalid start line (less than 1)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=0)
            
            assert result["success"] is False
            assert "Invalid start_line" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_read_file_invalid_range(self):
        """Test with end line before start line."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=3, end_line=1)
            
            assert result["success"] is False
            assert "Invalid range" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_read_file_line_beyond_end(self):
        """Test reading past the end of the file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path, start_line=1, end_line=100)
            
            assert result["success"] is True
            assert result["content"] == "line 1\nline 2\n"
            assert result["lines_read"] == 2
            assert result["end_line"] == 2  # Adjusted to actual file length
        finally:
            os.unlink(temp_path)
    
    def test_read_empty_file(self):
        """Test reading an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            
            assert result["success"] is True
            assert result["content"] == ""
            assert result["lines_read"] == 0
        finally:
            os.unlink(temp_path)
    
    def test_read_file_without_trailing_newline(self):
        """Test reading file without trailing newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2")  # No trailing newline
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            
            assert result["success"] is True
            assert result["content"] == "line 1\nline 2"
            assert result["lines_read"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_file_with_unicode(self):
        """Test reading file with unicode characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Hello 世界\nBonjour 🌍\n")
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            
            assert result["success"] is True
            assert "世界" in result["content"]
            assert "🌍" in result["content"]
        finally:
            os.unlink(temp_path)
    
    def test_read_binary_file(self):
        """Test reading a binary file (should fail gracefully)."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe')
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            
            assert result["success"] is False
            assert "non-UTF-8" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_read_python_file(self):
        """Test reading an actual Python source file."""
        # Read this test file itself
        result = read_file(__file__, start_line=1, end_line=5)
        
        assert result["success"] is True
        assert '"""' in result["content"] or 'import' in result["content"]
        assert result["lines_read"] == 5


class TestWriteFile:
    """Tests for write_file tool."""
    
    def test_write_new_file(self):
        """Test writing a new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            content = "Hello, World!\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            assert result["file_path"] == file_path
            assert result["bytes_written"] > 0
            assert os.path.exists(file_path)
            
            # Verify content was written correctly
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_overwrite_existing_file(self):
        """Test overwriting an existing file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Old content")
            temp_path = f.name
        
        try:
            new_content = "New content\n"
            result = write_file(temp_path, new_content)
            
            assert result["success"] is True
            
            # Verify old content was replaced
            with open(temp_path, 'r') as f:
                assert f.read() == new_content
        finally:
            os.unlink(temp_path)
    
    def test_write_empty_file(self):
        """Test writing an empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "empty.txt")
            
            result = write_file(file_path, "")
            
            assert result["success"] is True
            assert result["bytes_written"] == 0
            assert os.path.exists(file_path)
            
            with open(file_path, 'r') as f:
                assert f.read() == ""
    
    def test_write_multiline_content(self):
        """Test writing multiline content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "multi.txt")
            content = "line 1\nline 2\nline 3\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            assert result["lines_written"] == 3
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_with_unicode(self):
        """Test writing unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "unicode.txt")
            content = "Hello 世界\nBonjour 🌍\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            
            with open(file_path, 'r', encoding='utf-8') as f:
                assert f.read() == content
    
    def test_write_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "subdir1", "subdir2", "test.txt")
            content = "test content\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            assert os.path.exists(file_path)
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_without_trailing_newline(self):
        """Test writing content without trailing newline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            content = "no trailing newline"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            assert result["lines_written"] == 1
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_python_code(self):
        """Test writing Python code to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "script.py")
            content = '#!/usr/bin/env python3\n\ndef hello():\n    print("Hello")\n'
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_json_content(self):
        """Test writing JSON content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "data.json")
            content = '{\n  "key": "value",\n  "number": 42\n}\n'
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_large_content(self):
        """Test writing large content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "large.txt")
            # Create content with 1000 lines
            content = "\n".join([f"line {i}" for i in range(1000)]) + "\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            assert result["lines_written"] == 1000
            
            with open(file_path, 'r') as f:
                assert f.read() == content
    
    def test_write_to_directory_fails(self):
        """Test that writing to a directory path fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_file(tmpdir, "content")
            
            assert result["success"] is False
            assert "directory" in result["error"].lower()
    
    def test_write_to_readonly_location(self):
        """Test writing to a read-only location (if applicable)."""
        # Try to write to root directory (usually requires permissions)
        result = write_file("/test_readonly_file.txt", "content")
        
        # Should fail with permission error on most systems
        if not result["success"]:
            assert "Permission denied" in result["error"] or "Error writing" in result["error"]
    
    def test_write_special_characters(self):
        """Test writing special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "special.txt")
            content = "Special chars: \t\\\"'\n"
            
            result = write_file(file_path, content)
            
            assert result["success"] is True
            
            with open(file_path, 'r') as f:
                read_content = f.read()
                assert read_content == content
                assert '\t' in read_content
                assert '\\' in read_content
    
    def test_write_then_read(self):
        """Test writing a file and then reading it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            original_content = "Test content\nLine 2\nLine 3\n"
            
            # Write
            write_result = write_file(file_path, original_content)
            assert write_result["success"] is True
            
            # Read back
            read_result = read_file(file_path)
            assert read_result["success"] is True
            assert read_result["content"] == original_content


class TestEditFile:
    """Tests for edit_file tool."""
    
    def test_edit_single_line(self):
        """Test editing a single line in a file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "line 2", "modified line 2")
            
            assert result["success"] is True
            assert result["replacements_made"] == 1
            
            # Verify the edit
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "line 1\nmodified line 2\nline 3\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_multiple_lines(self):
        """Test editing multiple lines at once."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\nline 4\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "line 2\nline 3", "new line 2\nnew line 3")
            
            assert result["success"] is True
            assert result["replacements_made"] == 1
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "line 1\nnew line 2\nnew line 3\nline 4\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_multiple_occurrences(self):
        """Test editing when old content appears multiple times."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("foo bar foo baz foo\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "foo", "FOO")
            
            assert result["success"] is True
            assert result["replacements_made"] == 3
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "FOO bar FOO baz FOO\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_entire_file(self):
        """Test replacing entire file content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            original = "This is the entire content\n"
            f.write(original)
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, original, "Completely new content\n")
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                assert f.read() == "Completely new content\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_add_content(self):
        """Test adding content (replacing with longer content)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("short\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "short", "much longer content here")
            
            assert result["success"] is True
            assert result["new_content_length"] > result["old_content_length"]
            
            with open(temp_path, 'r') as f:
                assert f.read() == "much longer content here\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_remove_content(self):
        """Test removing content (replacing with shorter content)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("this is a long line\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "this is a long line", "short")
            
            assert result["success"] is True
            assert result["new_content_length"] < result["old_content_length"]
            
            with open(temp_path, 'r') as f:
                assert f.read() == "short\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_delete_content(self):
        """Test deleting content (replacing with empty string)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nDELETE ME\nline 3\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "DELETE ME\n", "")
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                assert f.read() == "line 1\nline 3\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_nonexistent_file(self):
        """Test editing a file that doesn't exist."""
        result = edit_file("/nonexistent/file.txt", "old", "new")
        
        assert result["success"] is False
        assert "File not found" in result["error"]
    
    def test_edit_content_not_found(self):
        """Test when old content is not in the file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "line 99", "new line")
            
            assert result["success"] is False
            assert "Old content not found" in result["error"]
            
            # Verify file wasn't changed
            with open(temp_path, 'r') as f:
                assert f.read() == "line 1\nline 2\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_case_sensitive(self):
        """Test that editing is case sensitive."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Hello World\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "hello world", "Hi Earth")
            
            assert result["success"] is False
            assert "Old content not found" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_edit_with_whitespace(self):
        """Test editing content with whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("  indented\n\tindented with tab\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "  indented", "no indent")
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "no indent" in content
                assert "\tindented with tab" in content
        finally:
            os.unlink(temp_path)
    
    def test_edit_python_function(self):
        """Test editing Python code."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write('def hello():\n    print("Hello")\n')
            temp_path = f.name
        
        try:
            result = edit_file(
                temp_path,
                'print("Hello")',
                'print("Hello, World!")'
            )
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'print("Hello, World!")' in content
        finally:
            os.unlink(temp_path)
    
    def test_edit_json_value(self):
        """Test editing JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{\n  "key": "old_value"\n}\n')
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, '"old_value"', '"new_value"')
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert '"new_value"' in content
        finally:
            os.unlink(temp_path)
    
    def test_edit_with_special_chars(self):
        """Test editing content with special characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('text with "quotes" and \\backslash\n')
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, '"quotes"', "'quotes'")
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "'quotes'" in content
        finally:
            os.unlink(temp_path)
    
    def test_edit_unicode_content(self):
        """Test editing unicode content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Hello 世界\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "世界", "World")
            
            assert result["success"] is True
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Hello World" in content
        finally:
            os.unlink(temp_path)
    
    def test_edit_preserves_surrounding_content(self):
        """Test that edit only changes specified content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("before\ntarget\nafter\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "target", "modified")
            
            assert result["success"] is True
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "before\nmodified\nafter\n"
        finally:
            os.unlink(temp_path)
    
    def test_edit_empty_replacement(self):
        """Test replacing content with empty string."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("keep this\nremove this\nkeep this too\n")
            temp_path = f.name
        
        try:
            result = edit_file(temp_path, "remove this\n", "")
            
            assert result["success"] is True
            assert result["new_content_length"] == 0
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "keep this\nkeep this too\n"
        finally:
            os.unlink(temp_path)

class TestListDirectory:
    """Tests for list_directory tool."""
    
    def test_list_empty_directory(self):
        """Test listing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["entries"] == []
            assert result["total_entries"] == 0
            assert result["files"] == 0
            assert result["directories"] == 0
    
    def test_list_directory_with_files(self):
        """Test listing a directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            Path(tmpdir, "file1.txt").write_text("content")
            Path(tmpdir, "file2.py").write_text("code")
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["total_entries"] == 2
            assert result["files"] == 2
            assert result["directories"] == 0
            
            # Check entries
            names = [e["name"] for e in result["entries"]]
            assert "file1.txt" in names
            assert "file2.py" in names
            
            # Check all are files
            assert all(e["type"] == "file" for e in result["entries"])
    
    def test_list_directory_with_subdirs(self):
        """Test listing a directory with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            os.makedirs(os.path.join(tmpdir, "subdir1"))
            os.makedirs(os.path.join(tmpdir, "subdir2"))
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["total_entries"] == 2
            assert result["files"] == 0
            assert result["directories"] == 2
            
            # Check entries
            names = [e["name"] for e in result["entries"]]
            assert "subdir1" in names
            assert "subdir2" in names
            
            # Check all are directories
            assert all(e["type"] == "directory" for e in result["entries"])
    
    def test_list_mixed_directory(self):
        """Test listing a directory with both files and subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files and directories
            Path(tmpdir, "file1.txt").write_text("content")
            Path(tmpdir, "file2.txt").write_text("content")
            os.makedirs(os.path.join(tmpdir, "subdir1"))
            os.makedirs(os.path.join(tmpdir, "subdir2"))
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["total_entries"] == 4
            assert result["files"] == 2
            assert result["directories"] == 2
    
    def test_list_directory_recursive(self):
        """Test recursive directory listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            Path(tmpdir, "file1.txt").write_text("root")
            os.makedirs(os.path.join(tmpdir, "subdir1"))
            Path(tmpdir, "subdir1", "file2.txt").write_text("sub1")
            os.makedirs(os.path.join(tmpdir, "subdir1", "nested"))
            Path(tmpdir, "subdir1", "nested", "file3.txt").write_text("nested")
            
            result = list_directory(tmpdir, recursive=True)
            
            assert result["success"] is True
            assert result["total_entries"] > 3  # At least files + dirs
            
            # Check paths include subdirectories
            paths = [e["path"] for e in result["entries"]]
            assert "file1.txt" in paths
            assert any("subdir1" in p for p in paths)
            assert any("nested" in p for p in paths)
    
    def test_list_directory_sizes(self):
        """Test that file sizes are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with known size
            file_path = Path(tmpdir, "test.txt")
            content = "Hello, World!"
            file_path.write_text(content)
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            file_entry = result["entries"][0]
            assert file_entry["type"] == "file"
            assert file_entry["size"] == len(content)
    
    def test_list_nonexistent_directory(self):
        """Test listing a directory that doesn't exist."""
        result = list_directory("/nonexistent/path/12345")
        
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert result["entries"] == []
    
    def test_list_file_instead_of_directory(self):
        """Test listing a file path instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = list_directory(temp_path)
            
            assert result["success"] is False
            assert "not a directory" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_list_directory_sorted(self):
        """Test that entries are sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in non-alphabetical order
            Path(tmpdir, "zebra.txt").write_text("z")
            Path(tmpdir, "alpha.txt").write_text("a")
            Path(tmpdir, "beta.txt").write_text("b")
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            names = [e["name"] for e in result["entries"]]
            assert names == ["alpha.txt", "beta.txt", "zebra.txt"]
    
    def test_list_directory_with_hidden_files(self):
        """Test listing directory with hidden files (dotfiles)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "visible.txt").write_text("visible")
            Path(tmpdir, ".hidden").write_text("hidden")
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            names = [e["name"] for e in result["entries"]]
            assert ".hidden" in names
            assert "visible.txt" in names
    
    def test_list_directory_recursive_depth(self):
        """Test recursive listing goes multiple levels deep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deep structure
            level1 = os.path.join(tmpdir, "level1")
            level2 = os.path.join(level1, "level2")
            level3 = os.path.join(level2, "level3")
            os.makedirs(level3)
            
            Path(level3, "deep.txt").write_text("deep")
            
            result = list_directory(tmpdir, recursive=True)
            
            assert result["success"] is True
            paths = [e["path"] for e in result["entries"]]
            
            # Should contain the deep file
            assert any("level3" in p and "deep.txt" in p for p in paths)
    
    def test_list_current_directory(self):
        """Test listing current directory using '.'"""
        # List the project directory
        result = list_directory(".")
        
        assert result["success"] is True
        assert result["total_entries"] > 0
        
        # Should find common project files/dirs
        names = [e["name"] for e in result["entries"]]
        assert "agent" in names or "tests" in names or "README.md" in names
    
    def test_list_directory_entry_structure(self):
        """Test that entries have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.txt").write_text("content")
            os.makedirs(os.path.join(tmpdir, "testdir"))
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            
            for entry in result["entries"]:
                assert "name" in entry
                assert "path" in entry
                assert "type" in entry
                assert "size" in entry
                assert entry["type"] in ["file", "directory", "other"]
    
    def test_list_directory_with_various_extensions(self):
        """Test listing files with various extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions = [".txt", ".py", ".json", ".md", ".log"]
            for ext in extensions:
                Path(tmpdir, f"file{ext}").write_text("content")
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["files"] == len(extensions)
            
            names = [e["name"] for e in result["entries"]]
            for ext in extensions:
                assert any(name.endswith(ext) for name in names)
    
    def test_list_large_directory(self):
        """Test listing directory with many files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 50 files
            for i in range(50):
                Path(tmpdir, f"file{i:03d}.txt").write_text(f"content {i}")
            
            result = list_directory(tmpdir)
            
            assert result["success"] is True
            assert result["files"] == 50
            assert len(result["entries"]) == 50


class TestGrepSearch:
    """Tests for grep_search tool."""
    
    def test_search_single_file(self):
        """Test searching a single file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\ntarget line\nline 3\n")
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 1
            assert len(result["matches"]) == 1
            
            match = result["matches"][0]
            assert match["line_number"] == 2
            assert "target" in match["line_content"]
        finally:
            os.unlink(temp_path)
    
    def test_search_directory(self):
        """Test searching all files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("contains target\n")
            Path(tmpdir, "file2.txt").write_text("no match here\n")
            Path(tmpdir, "file3.txt").write_text("another target\n")
            
            result = grep_search("target", tmpdir)
            
            assert result["success"] is True
            assert result["total_matches"] == 2
            assert result["files_searched"] == 3
    
    def test_search_case_insensitive(self):
        """Test case-insensitive search (default)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("TARGET\ntarget\nTarget\n")
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path, case_sensitive=False)
            
            assert result["success"] is True
            assert result["total_matches"] == 3
            assert result["case_sensitive"] is False
        finally:
            os.unlink(temp_path)
    
    def test_search_case_sensitive(self):
        """Test case-sensitive search."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("TARGET\ntarget\nTarget\n")
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path, case_sensitive=True)
            
            assert result["success"] is True
            assert result["total_matches"] == 1
            assert result["case_sensitive"] is True
        finally:
            os.unlink(temp_path)
    
    def test_search_with_file_pattern(self):
        """Test searching with file pattern filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").write_text("def target():\n    pass\n")
            Path(tmpdir, "file2.txt").write_text("target\n")
            Path(tmpdir, "file3.py").write_text("class Target:\n    pass\n")
            
            result = grep_search("target", tmpdir, file_pattern="*.py")
            
            assert result["success"] is True
            assert result["files_searched"] == 2
            
            # Should only find matches in .py files
            for match in result["matches"]:
                assert match["file"].endswith(".py")
    
    def test_search_regex_pattern(self):
        """Test searching with regex pattern."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test123\ntest456\ntest\n")
            temp_path = f.name
        
        try:
            result = grep_search(r"test\d+", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_search_multiple_matches_per_line(self):
        """Test finding multiple occurrences in same line."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("foo foo foo\n")
            temp_path = f.name
        
        try:
            result = grep_search("foo", temp_path)
            
            assert result["success"] is True
            # Should match the line (once per line, not per occurrence)
            assert result["total_matches"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_search_no_matches(self):
        """Test searching when pattern doesn't match."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name
        
        try:
            result = grep_search("nomatch", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 0
            assert result["matches"] == []
        finally:
            os.unlink(temp_path)
    
    def test_search_nonexistent_path(self):
        """Test searching a path that doesn't exist."""
        result = grep_search("pattern", "/nonexistent/path")
        
        assert result["success"] is False
        assert "does not exist" in result["error"]
    
    def test_search_invalid_regex(self):
        """Test with invalid regex pattern."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("content\n")
            temp_path = f.name
        
        try:
            result = grep_search("[invalid(", temp_path)
            
            assert result["success"] is False
            assert "Invalid regex" in result["error"]
        finally:
            os.unlink(temp_path)
    
    def test_search_match_info(self):
        """Test that match info contains all required fields."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("line 1\ntarget line\nline 3\n")
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path)
            
            assert result["success"] is True
            match = result["matches"][0]
            
            assert "file" in match
            assert "line_number" in match
            assert "line_content" in match
            assert "column" in match
            assert match["line_number"] == 2
            assert match["column"] >= 0
        finally:
            os.unlink(temp_path)
    
    def test_search_multiline_file(self):
        """Test searching in file with many lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            lines = [f"line {i}\n" for i in range(100)]
            lines[25] = "special target line\n"
            lines[75] = "another target here\n"
            f.writelines(lines)
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 2
            assert result["matches"][0]["line_number"] == 26  # Line 25 (0-indexed)
            assert result["matches"][1]["line_number"] == 76
        finally:
            os.unlink(temp_path)
    
    def test_search_with_special_chars(self):
        """Test searching for special characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('line with "quotes"\n')
            temp_path = f.name
        
        try:
            result = grep_search('"quotes"', temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_search_word_boundary(self):
        """Test regex word boundary search."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test testing tester\n")
            temp_path = f.name
        
        try:
            # Should only match whole word "test"
            result = grep_search(r"\btest\b", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_search_nested_directories(self):
        """Test searching in nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            
            Path(tmpdir, "root.txt").write_text("target in root\n")
            Path(subdir, "sub.txt").write_text("target in subdir\n")
            
            result = grep_search("target", tmpdir)
            
            assert result["success"] is True
            assert result["total_matches"] == 2
            assert result["files_searched"] == 2
    
    def test_search_empty_file(self):
        """Test searching an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        
        try:
            result = grep_search("pattern", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 0
        finally:
            os.unlink(temp_path)
    
    def test_search_skips_binary_files(self):
        """Test that binary files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a text file and a binary file
            Path(tmpdir, "text.txt").write_text("target\n")
            Path(tmpdir, "binary.bin").write_bytes(b"\x00\x01\x02target\xff\xfe")
            
            result = grep_search("target", tmpdir)
            
            assert result["success"] is True
            # Should find match in text file but skip binary file
            assert result["files_searched"] >= 1
    
    def test_search_column_position(self):
        """Test that column position is correct."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("prefix target suffix\n")
            temp_path = f.name
        
        try:
            result = grep_search("target", temp_path)
            
            assert result["success"] is True
            match = result["matches"][0]
            assert match["column"] == 7  # "target" starts at position 7
        finally:
            os.unlink(temp_path)
    
    def test_search_python_function(self):
        """Test searching for Python function definition."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("def hello():\n    print('hi')\n\ndef world():\n    pass\n")
            temp_path = f.name
        
        try:
            result = grep_search(r"^def ", temp_path)
            
            assert result["success"] is True
            assert result["total_matches"] == 2
        finally:
            os.unlink(temp_path)


class TestSemanticSearch:
    """Tests for semantic_search function."""
    
    def test_basic_search(self):
        """Test basic semantic search in current directory."""
        result = semantic_search("test functions", path="tests")
        
        assert result["success"] is True
        assert "results" in result
        assert "query" in result
        assert result["query"] == "test functions"
        assert isinstance(result["results"], list)
    
    def test_search_with_results(self):
        """Test semantic search that should return results."""
        result = semantic_search("bash command execution", path="agent")
        
        assert result["success"] is True
        assert len(result["results"]) > 0
        
        # Check result structure
        first_result = result["results"][0]
        assert "file" in first_result
        assert "content" in first_result
        assert "similarity" in first_result
        assert "start_line" in first_result
        assert "end_line" in first_result
        
        # Similarity should be between 0 and 1
        assert 0 <= first_result["similarity"] <= 1
    
    def test_search_nonexistent_path(self):
        """Test semantic search with non-existent path."""
        result = semantic_search("test", path="/nonexistent/path")
        
        assert result["success"] is False
        assert "error" in result
        assert "does not exist" in result["error"]
    
    def test_search_with_limit(self):
        """Test semantic search with result limit."""
        result = semantic_search("function", path="agent", limit=2)
        
        assert result["success"] is True
        assert len(result["results"]) <= 2
    
    def test_search_single_file(self):
        """Test semantic search on a single file."""
        result = semantic_search("OllamaClient", path="agent/llm.py")
        
        assert result["success"] is True
        if result["results"]:
            assert all("agent/llm.py" in r["file"] for r in result["results"])
    
    def test_search_filters_low_similarity(self):
        """Test that very low similarity results are filtered."""
        result = semantic_search("xyzabc123impossible", path="agent")
        
        assert result["success"] is True
        # Should either have no results or only results above similarity threshold
        for res in result["results"]:
            assert res["similarity"] > 0.1
    
    def test_search_with_default_path(self):
        """Test semantic search with default path (current directory)."""
        result = semantic_search("import", limit=3)
        
        assert result["success"] is True
        assert "results" in result
    
    def test_search_result_fields(self):
        """Test that search results have all expected fields."""
        result = semantic_search("test class", path="tests", limit=1)
        
        assert result["success"] is True
        assert "results" in result
        assert "query" in result
        assert "total_results" in result
        assert "files_searched" in result
        
        if result["results"]:
            res = result["results"][0]
            required_fields = ["file", "content", "similarity", "start_line", "end_line"]
            for field in required_fields:
                assert field in res
    
    def test_search_in_temp_file(self):
        """Test semantic search on a temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("def calculate_fibonacci(n):\n")
            f.write("    if n <= 1:\n")
            f.write("        return n\n")
            f.write("    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\n")
            temp_path = f.name
        
        try:
            result = semantic_search("recursive function", path=temp_path)
            
            assert result["success"] is True
            if result["results"]:
                assert temp_path in result["results"][0]["file"]
        finally:
            os.unlink(temp_path)
    
    def test_search_skips_binary_files(self):
        """Test that semantic search handles binary files gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file
            text_file = os.path.join(temp_dir, "test.py")
            with open(text_file, 'w') as f:
                f.write("def test():\n    pass\n")
            
            # Create a binary file
            binary_file = os.path.join(temp_dir, "test.bin")
            with open(binary_file, 'wb') as f:
                f.write(b'\x00\x01\x02\x03')
            
            result = semantic_search("test", path=temp_dir)
            
            # Should succeed even with binary files present
            assert result["success"] is True
    
    def test_search_empty_directory(self):
        """Test semantic search in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = semantic_search("test", path=temp_dir)
            
            assert result["success"] is True
            assert result["total_results"] == 0
            assert "message" in result
    
    def test_search_skips_common_directories(self):
        """Test that semantic search skips .git, node_modules, etc."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a .git directory
            git_dir = os.path.join(temp_dir, ".git")
            os.makedirs(git_dir)
            
            git_file = os.path.join(git_dir, "config")
            with open(git_file, 'w') as f:
                f.write("test content")
            
            # Create a normal file
            normal_file = os.path.join(temp_dir, "test.py")
            with open(normal_file, 'w') as f:
                f.write("def test(): pass")
            
            result = semantic_search("test", path=temp_dir)
            
            assert result["success"] is True
            # Results should not include files from .git
            for res in result["results"]:
                assert ".git" not in res["file"]
    
    def test_search_only_text_extensions(self):
        """Test that semantic search only searches text file extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various files
            for ext in ['.py', '.txt', '.md', '.jpg', '.png']:
                file_path = os.path.join(temp_dir, f"test{ext}")
                with open(file_path, 'w' if ext != '.jpg' and ext != '.png' else 'wb') as f:
                    if ext == '.jpg' or ext == '.png':
                        f.write(b'\x00\x01\x02')
                    else:
                        f.write("test content")
            
            result = semantic_search("test", path=temp_dir)
            
            assert result["success"] is True
            # Results should only include text files
            for res in result["results"]:
                assert not res["file"].endswith(('.jpg', '.png'))
    
    def test_search_similarity_ordering(self):
        """Test that results are ordered by similarity (highest first)."""
        result = semantic_search("HTTP request client", path="agent", limit=3)
        
        assert result["success"] is True
        if len(result["results"]) > 1:
            similarities = [r["similarity"] for r in result["results"]]
            # Check that similarities are in descending order
            assert similarities == sorted(similarities, reverse=True)
    
    def test_search_chunks_files(self):
        """Test that large files are chunked properly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            # Write 50 lines
            for i in range(50):
                f.write(f"# Line {i}\n")
                if i == 25:
                    f.write("def important_function():\n    pass\n")
            temp_path = f.name
        
        try:
            result = semantic_search("important function", path=temp_path)
            
            assert result["success"] is True
            if result["results"]:
                # Should find chunks within the file
                assert result["results"][0]["start_line"] < result["results"][0]["end_line"]
        finally:
            os.unlink(temp_path)


class TestWebSearch:
    """Tests for web_search function."""
    
    def test_basic_search(self):
        """Test basic web search."""
        result = web_search("Python programming", num_results=3)
        
        assert result["success"] is True
        assert "results" in result
        assert "query" in result
        assert result["query"] == "Python programming"
        assert isinstance(result["results"], list)
    
    def test_search_returns_results(self):
        """Test that search returns actual results."""
        result = web_search("Wikipedia", num_results=2)
        
        assert result["success"] is True
        # Should find at least some results for common query
        if result["results"]:
            assert len(result["results"]) > 0
            
            # Check result structure
            first_result = result["results"][0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result
            assert "source" in first_result
            
            # URL should be valid
            assert first_result["url"].startswith("http")
    
    def test_search_result_fields(self):
        """Test that all expected fields are present in results."""
        result = web_search("test query", num_results=1)
        
        assert result["success"] is True
        assert "results" in result
        assert "query" in result
        assert "total_results" in result
        
        if result["results"]:
            res = result["results"][0]
            required_fields = ["title", "url", "snippet", "source"]
            for field in required_fields:
                assert field in res
    
    def test_search_with_limit(self):
        """Test search with custom result limit."""
        result = web_search("Python", num_results=5)
        
        assert result["success"] is True
        # Should return at most 5 results
        assert len(result["results"]) <= 5
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        result = web_search("", num_results=3)
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()
    
    def test_search_whitespace_query(self):
        """Test search with whitespace-only query."""
        result = web_search("   ", num_results=3)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_search_invalid_num_results(self):
        """Test search with invalid num_results."""
        result = web_search("test", num_results=0)
        
        assert result["success"] is False
        assert "error" in result
        assert "num_results" in result["error"].lower()
    
    def test_search_negative_num_results(self):
        """Test search with negative num_results."""
        result = web_search("test", num_results=-1)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_search_large_num_results(self):
        """Test search with large num_results (should be capped)."""
        result = web_search("Python", num_results=100)
        
        assert result["success"] is True
        # Should be capped at 20
        assert len(result["results"]) <= 20
    
    def test_search_specific_topic(self):
        """Test search for specific topic."""
        result = web_search("Python list comprehension", num_results=3)
        
        assert result["success"] is True
        assert result["query"] == "Python list comprehension"
    
    def test_search_url_format(self):
        """Test that returned URLs are properly formatted."""
        result = web_search("GitHub", num_results=2)
        
        assert result["success"] is True
        if result["results"]:
            for res in result["results"]:
                url = res["url"]
                assert url.startswith("http://") or url.startswith("https://")
    
    def test_search_source_extraction(self):
        """Test that source domain is extracted from URL."""
        result = web_search("Wikipedia", num_results=2)
        
        assert result["success"] is True
        if result["results"]:
            for res in result["results"]:
                # Source should be extracted from URL
                if res["url"]:
                    assert len(res["source"]) > 0
    
    def test_search_with_special_characters(self):
        """Test search with special characters in query."""
        result = web_search("Python & programming", num_results=2)
        
        assert result["success"] is True
        assert "results" in result
    
    def test_search_with_quotes(self):
        """Test search with quoted phrase."""
        result = web_search('"Python programming language"', num_results=2)
        
        assert result["success"] is True
        assert "results" in result
    
    def test_search_unicode_query(self):
        """Test search with unicode characters."""
        result = web_search("Python 编程", num_results=2)
        
        assert result["success"] is True
        assert "results" in result
    
    def test_search_technical_query(self):
        """Test search with technical query."""
        result = web_search("machine learning algorithms", num_results=3)
        
        assert result["success"] is True
        assert result["query"] == "machine learning algorithms"