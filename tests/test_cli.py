"""Tests for the CLI module."""
import pytest
from unittest.mock import patch

from mattbot.cli import AgentCLI
from mattbot.system_prompt import SYSTEM_PROMPT


class TestAgentCLI:
    """Test the AgentCLI class."""
    
    def test_initialization(self):
        """Test that AgentCLI initializes correctly."""
        cli = AgentCLI(model="test-model", history_length=123, artifact_store_enabled=False)
        assert cli.model == "test-model"
        assert cli.history_length == 123
        assert cli.messages == []
        assert cli.cwd is not None
        
    def test_print_separator(self, capsys):
        """Test that print_separator outputs correctly."""
        cli = AgentCLI(artifact_store_enabled=False)
        cli.print_separator()
        captured = capsys.readouterr()
        assert "─" in captured.out
        
    def test_cwd_is_set(self):
        """Test that current working directory is captured."""
        cli = AgentCLI(artifact_store_enabled=False)
        assert isinstance(cli.cwd, str)
        assert len(cli.cwd) > 0

    def test_exit_command_returns_exit(self):
        """Test that /exit triggers CLI exit action."""
        cli = AgentCLI(artifact_store_enabled=False)
        result = cli._handle_command("/exit")
        assert result == "exit"

    def test_set_command_updates_temperature(self):
        """Test that /set updates runtime settings."""
        cli = AgentCLI(temperature=0.7, artifact_store_enabled=False)
        result = cli._handle_command("/set temperature 0.25")
        assert result == "continue"
        assert cli.temperature == 0.25

    def test_shell_command_executes_without_llm(self):
        """Test that /shell invokes local shell execution."""
        cli = AgentCLI(artifact_store_enabled=False)

        with patch("mattbot.commands.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {
                "success": True,
                "stdout": "ok\n",
                "stderr": "",
                "exit_code": 0,
            }
            result = cli._handle_command("/shell echo ok")

        assert result == "continue"
        mock_execute_tool.assert_called_once_with(
            "run_bash_command",
            command="echo ok",
            working_directory=cli.cwd,
        )

    def test_new_session_clears_messages(self):
        """Test that /new clears chat history and starts a new session."""
        cli = AgentCLI(logging_enabled=True, artifact_store_enabled=False)
        cli.messages = [{"role": "user", "content": "hi"}]
        old_logger = cli.session_logger

        result = cli._handle_command("/new")

        assert result == "continue"
        assert cli.messages == []
        assert cli.session_logger is not old_logger

    def test_large_tool_result_is_compacted_to_artifact_reference(self, tmp_path):
        """Large tool outputs should be summarized with artifact handle for context efficiency."""
        cli = AgentCLI(
            artifact_store_enabled=True,
            artifact_dir=str(tmp_path / "artifacts"),
            artifact_inline_char_limit=200,
        )

        result = {
            "success": True,
            "stdout": "x" * 1000,
            "stderr": "",
            "exit_code": 0,
        }

        formatted = cli._format_tool_result_for_model(tool_name="run_bash_command", result=result)

        assert "artifact_id" in formatted
        assert "truncated" in formatted
        assert "raw_size_chars" in formatted

    def test_messages_for_llm_includes_system_prompt(self):
        """Outbound model context should include a dedicated system prompt."""
        cli = AgentCLI(artifact_store_enabled=False)
        cli.messages = [{"role": "user", "content": "hello"}]

        llm_messages = cli._messages_for_llm()

        assert llm_messages[0]["role"] == "system"
        assert SYSTEM_PROMPT in llm_messages[0]["content"]
        assert "AVAILABLE TOOL NAMES" in llm_messages[0]["content"]
        assert llm_messages[1:] == cli.messages

    def test_handle_tool_calls_supports_wrapped_functions_tool(self):
        """Some models wrap real tool calls inside a 'functions' tool payload."""
        cli = AgentCLI(artifact_store_enabled=False)
        tool_calls = [{
            "id": "call_1",
            "function": {
                "name": "functions",
                "arguments": {
                    "name": "read_file",
                    "arguments": {"file_path": "README.md", "start_line": 1, "end_line": 1},
                },
            },
        }]

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {
                "success": True,
                "content": "# MattBot\n",
                "file_path": "README.md",
                "lines_read": 1,
            }
            results = cli.handle_tool_calls(tool_calls)

        mock_execute_tool.assert_called_once_with(
            "read_file", file_path="README.md", start_line=1, end_line=1
        )
        assert len(results) == 1
        assert results[0]["role"] == "tool"
