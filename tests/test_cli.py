"""Tests for the CLI module."""
import pytest
from types import SimpleNamespace
from unittest.mock import patch

import mattbot.cli as cli_module
from mattbot.cli import AgentCLI


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


class TestCliMain:
    """Tests for CLI entrypoint behavior."""

    def test_main_prompt_file_runs_single_turn(self, tmp_path, monkeypatch):
        """`--prompt-file` should run one turn and exit without interactive loop."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello from benchmark", encoding="utf-8")

        fake_config = SimpleNamespace(
            model="gpt-oss:latest",
            ollama_url="http://localhost:11434",
            max_context_tokens=8192,
            temperature=0.7,
            embedding_model="all-minilm:l6-v2",
            history_length=100,
            logging_enabled=False,
            log_dir="~/.mattbot/logs",
            artifact_store_enabled=False,
            artifact_dir="~/.mattbot/artifacts",
            artifact_ttl_days=7,
            artifact_max_sessions=20,
            artifact_inline_char_limit=8000,
        )

        class FakeCLI:
            chat_called = False
            run_called = False
            close_called = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def chat(self, prompt):
                assert prompt == "hello from benchmark"
                FakeCLI.chat_called = True
                return "ok"

            def run(self):
                FakeCLI.run_called = True

            def close(self, reason="session_end"):
                FakeCLI.close_called = True

        monkeypatch.setattr(
            cli_module,
            "resolve_config",
            lambda cli_args, config_path: fake_config,
        )
        monkeypatch.setattr(cli_module, "AgentCLI", FakeCLI)
        monkeypatch.setattr(
            "sys.argv",
            ["mattbot", "--prompt-file", str(prompt_file)],
        )

        exit_code = cli_module.main()

        assert exit_code == 0
        assert FakeCLI.chat_called is True
        assert FakeCLI.close_called is True
        assert FakeCLI.run_called is False

    def test_main_emits_run_summary_json(self, tmp_path, monkeypatch, capsys):
        """`--emit-run-summary-json` should print summary marker with JSON payload."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello from benchmark", encoding="utf-8")

        fake_config = SimpleNamespace(
            model="gpt-oss:latest",
            ollama_url="http://localhost:11434",
            max_context_tokens=8192,
            temperature=0.7,
            embedding_model="all-minilm:l6-v2",
            history_length=100,
            logging_enabled=False,
            log_dir="~/.mattbot/logs",
            artifact_store_enabled=False,
            artifact_dir="~/.mattbot/artifacts",
            artifact_ttl_days=7,
            artifact_max_sessions=20,
            artifact_inline_char_limit=8000,
        )

        class FakeCLI:
            def __init__(self, **kwargs):
                self.last_run_summary = {
                    "success": True,
                    "tool_call_total": 3,
                    "tool_call_counts": {"read_file": 2, "edit_file": 1},
                }

            def chat(self, prompt):
                return "ok"

            def run(self):
                raise AssertionError("run() should not be called")

            def close(self, reason="session_end"):
                return None

        monkeypatch.setattr(
            cli_module,
            "resolve_config",
            lambda cli_args, config_path: fake_config,
        )
        monkeypatch.setattr(cli_module, "AgentCLI", FakeCLI)
        monkeypatch.setattr(
            "sys.argv",
            ["mattbot", "--prompt-file", str(prompt_file), "--emit-run-summary-json"],
        )

        exit_code = cli_module.main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "MATTBOT_RUN_SUMMARY_JSON:" in captured.out
        assert '"tool_call_total": 3' in captured.out
