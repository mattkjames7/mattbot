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