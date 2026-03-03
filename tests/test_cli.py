"""Tests for the CLI module."""
import pytest
from unittest.mock import patch

from mattbot.cli import AgentCLI


class TestAgentCLI:
    """Test the AgentCLI class."""
    
    def test_initialization(self):
        """Test that AgentCLI initializes correctly."""
        cli = AgentCLI(model="test-model", history_length=123)
        assert cli.model == "test-model"
        assert cli.history_length == 123
        assert cli.messages == []
        assert cli.cwd is not None
        
    def test_print_separator(self, capsys):
        """Test that print_separator outputs correctly."""
        cli = AgentCLI()
        cli.print_separator()
        captured = capsys.readouterr()
        assert "─" in captured.out
        
    def test_cwd_is_set(self):
        """Test that current working directory is captured."""
        cli = AgentCLI()
        assert isinstance(cli.cwd, str)
        assert len(cli.cwd) > 0

    def test_exit_command_returns_exit(self):
        """Test that /exit triggers CLI exit action."""
        cli = AgentCLI()
        result = cli._handle_command("/exit")
        assert result == "exit"

    def test_set_command_updates_temperature(self):
        """Test that /set updates runtime settings."""
        cli = AgentCLI(temperature=0.7)
        result = cli._handle_command("/set temperature 0.25")
        assert result == "continue"
        assert cli.temperature == 0.25

    def test_shell_command_executes_without_llm(self):
        """Test that /shell invokes local shell execution."""
        cli = AgentCLI()

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
        cli = AgentCLI(logging_enabled=True)
        cli.messages = [{"role": "user", "content": "hi"}]
        old_logger = cli.session_logger

        result = cli._handle_command("/new")

        assert result == "continue"
        assert cli.messages == []
        assert cli.session_logger is not old_logger
