"""Tests for the CLI module."""
import pytest
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
