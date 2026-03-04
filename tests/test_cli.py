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
        assert llm_messages[1:] == cli.messages

    def test_messages_for_llm_includes_runtime_instruction(self):
        """Optional runtime instruction should be appended to system content."""
        cli = AgentCLI(artifact_store_enabled=False)
        llm_messages = cli._messages_for_llm(extra_instruction="Do X now")

        assert "RUNTIME INSTRUCTION" in llm_messages[0]["content"]
        assert "Do X now" in llm_messages[0]["content"]

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

    def test_handle_tool_calls_normalizes_tool_name_suffix(self):
        """Malformed tool names with channel suffixes should still execute."""
        cli = AgentCLI(artifact_store_enabled=False)
        tool_calls = [{
            "id": "call_2",
            "function": {
                "name": "edit_file<|channel|>commentary",
                "arguments": {
                    "file_path": "README.md",
                    "old_content": "a",
                    "new_content": "b",
                },
            },
        }]

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {"success": True, "file_path": "README.md"}
            cli.handle_tool_calls(tool_calls)

        mock_execute_tool.assert_called_once_with(
            "edit_file", file_path="README.md", old_content="a", new_content="b"
        )

    def test_handle_tool_calls_sanitizes_arguments(self):
        """Unexpected args should be dropped and empty list_directory path normalized."""
        cli = AgentCLI(artifact_store_enabled=False)

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {"success": True, "entries": [], "path": "."}
            cli.handle_tool_calls([
                {
                    "id": "call_3",
                    "function": {
                        "name": "list_directory",
                        "arguments": {"path": "", "recursive": True},
                    },
                }
            ])

        mock_execute_tool.assert_called_once_with("list_directory", path=".", recursive=False)

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {
                "success": True,
                "content": "x",
                "file_path": "README.md",
                "lines_read": 1,
            }
            cli.handle_tool_calls([
                {
                    "id": "call_4",
                    "function": {
                        "name": "read_file",
                        "arguments": {
                            "file_path": "README.md",
                            "lines_read": 50,
                            "total_lines": 203,
                        },
                    },
                }
            ])

        mock_execute_tool.assert_called_once_with("read_file", file_path="README.md")

    def test_list_directory_keeps_recursive_when_user_requested(self):
        """Recursive listing should be preserved if the user explicitly asks for it."""
        cli = AgentCLI(artifact_store_enabled=False)
        cli.active_user_input = "Please recursively list the whole repo as a tree"

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {"success": True, "entries": [], "path": "."}
            cli.handle_tool_calls([
                {
                    "id": "call_5",
                    "function": {
                        "name": "list_directory",
                        "arguments": {"path": ".", "recursive": True},
                    },
                }
            ])

        mock_execute_tool.assert_called_once_with("list_directory", path=".", recursive=True)

    def test_is_clarification_response_detection(self):
        """Clarification-style assistant replies should be detected for fallback handling."""
        cli = AgentCLI(artifact_store_enabled=False)

        assert cli._is_clarification_response(
            "I’m ready to make the change, but I need more information. Which file would you like to edit?"
        )
        assert not cli._is_clarification_response("Done. I removed the section from README.md.")

    def test_inline_json_edit_payload_executes(self):
        """Raw JSON edit payload in assistant content should be executed as edit_file."""
        cli = AgentCLI(artifact_store_enabled=False)
        payload = (
            '{"old_content":"A","new_content":"B","file_path":"README.md"}'
        )

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {"success": True, "file_path": "README.md", "replacements_made": 1}
            result = cli._maybe_execute_inline_edit_from_content(payload)

        mock_execute_tool.assert_called_once_with(
            "edit_file", file_path="README.md", old_content="A", new_content="B"
        )
        assert result is not None
        assert result["success"] is True

    def test_inline_json_edit_payload_supports_fenced_json(self):
        """Fenced json payloads should also be parsed and executed."""
        cli = AgentCLI(artifact_store_enabled=False)
        payload = """```json
{"old_content":"A","new_content":"B","file_path":"README.md"}
```"""

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.return_value = {"success": True, "file_path": "README.md"}
            result = cli._maybe_execute_inline_edit_from_content(payload)

        mock_execute_tool.assert_called_once()
        assert result is not None
        assert result["success"] is True

    def test_build_local_edit_summary(self):
        """Local summary should deterministically describe edited files."""
        cli = AgentCLI(artifact_store_enabled=False)
        summary = cli._build_local_edit_summary([
            {"tool_name": "edit_file", "file_path": "README.md", "replacements_made": 1},
            {"tool_name": "write_file", "file_path": "notes.txt", "bytes_written": 42},
        ])

        assert "Applied the requested edit(s):" in summary
        assert "README.md (replacements: 1)" in summary
        assert "notes.txt (bytes written: 42)" in summary

    def test_attempt_direct_section_removal(self):
        """Deterministic fallback should remove a section from any explicitly targeted file."""
        cli = AgentCLI(artifact_store_enabled=False)
        doc = (
            "# Title\n\n"
            "## Features\n\n"
            "- item 1\n"
            "- item 2\n\n"
            "## Usage\n\n"
            "text\n"
        )

        with patch("mattbot.cli.execute_tool") as mock_execute_tool:
            mock_execute_tool.side_effect = [
                {"success": True, "content": doc, "file_path": "docs/guide.md"},
                {"success": True, "file_path": "docs/guide.md", "replacements_made": 1},
            ]
            result = cli._attempt_direct_section_removal(
                "Please remove the Features section from docs/guide.md"
            )

        assert result is not None
        assert result["success"] is True
        assert result["tool_name"] == "edit_file"
        assert result["file_path"] == "docs/guide.md"

    def test_infer_target_file_from_request(self):
        """Target-file inference should handle explicit paths and README shorthand."""
        cli = AgentCLI(artifact_store_enabled=False)

        assert cli._infer_target_file_from_request("Edit docs/guide.md and remove section") == "docs/guide.md"
        assert cli._infer_target_file_from_request("Please edit the README in this project") == "README.md"

    def test_process_streamed_response_preserves_role_and_tool_calls(self):
        """Streaming parser should keep assistant role and retain tool calls."""
        cli = AgentCLI(artifact_store_enabled=False)
        response_stream = [
            {"message": {"role": "assistant", "content": "Hello"}},
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": {"file_path": "README.md"}},
                        }
                    ]
                }
            },
        ]

        parsed = cli._process_streamed_response(response_stream)

        assert parsed["message"]["role"] == "assistant"
        assert parsed["message"]["content"] == "Hello"
        assert len(parsed["message"]["tool_calls"]) == 1
        assert parsed["message"]["tool_calls"][0]["id"] == "call_1"
