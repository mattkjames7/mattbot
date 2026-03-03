"""Slash-command handling for the interactive MattBot CLI."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from mattbot.config import AgentConfig, write_config
from mattbot.tool_executor import execute_tool

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None

if TYPE_CHECKING:
    from mattbot.cli import AgentCLI


class CommandProcessor:
    """Parse and execute local slash commands for an ``AgentCLI`` instance."""

    def __init__(self, cli: "AgentCLI"):
        self.cli = cli

    def _build_agent_config(self) -> AgentConfig:
        """Create config object from current runtime settings."""
        return AgentConfig(
            model=self.cli.model,
            ollama_url=self.cli.ollama_url,
            max_context_tokens=self.cli.max_context_tokens,
            temperature=self.cli.temperature,
            embedding_model=self.cli.embedding_model,
            history_length=self.cli.history_length,
            logging_enabled=self.cli.logging_enabled,
            log_dir=self.cli.log_dir,
        )

    def _show_settings(self):
        """Print current runtime settings."""
        self.cli.console.print("[bold]Current settings[/bold]")
        self.cli.console.print(f"  model = [green]{self.cli.model}[/green]")
        self.cli.console.print(f"  ollama_url = [green]{self.cli.ollama_url}[/green]")
        self.cli.console.print(f"  max_context_tokens = [green]{self.cli.max_context_tokens}[/green]")
        self.cli.console.print(f"  temperature = [green]{self.cli.temperature}[/green]")
        self.cli.console.print(f"  embedding_model = [green]{self.cli.embedding_model}[/green]")
        self.cli.console.print(f"  history_length = [green]{self.cli.history_length}[/green]")
        self.cli.console.print(f"  logging_enabled = [green]{self.cli.logging_enabled}[/green]")
        self.cli.console.print(f"  log_dir = [green]{self.cli.log_dir}[/green]")
        if self.cli.config_path:
            self.cli.console.print(f"  config_path = [green]{self.cli.config_path}[/green]")

    def _update_setting(self, key: str, raw_value: str):
        """Update one runtime setting from command input."""
        normalized_key = key.strip().lower()
        bool_map = {
            "1": True,
            "true": True,
            "yes": True,
            "y": True,
            "on": True,
            "0": False,
            "false": False,
            "no": False,
            "n": False,
            "off": False,
        }

        if normalized_key == "model":
            self.cli.model = raw_value
            self.cli._rebuild_client()
        elif normalized_key == "ollama_url":
            self.cli.ollama_url = raw_value
            self.cli._rebuild_client()
        elif normalized_key == "max_context_tokens":
            parsed = int(raw_value)
            if parsed <= 0:
                raise ValueError("max_context_tokens must be > 0")
            self.cli.max_context_tokens = parsed
        elif normalized_key == "temperature":
            self.cli.temperature = float(raw_value)
        elif normalized_key == "embedding_model":
            self.cli.embedding_model = raw_value
        elif normalized_key == "history_length":
            parsed = int(raw_value)
            if parsed <= 0:
                raise ValueError("history_length must be > 0")
            self.cli.history_length = parsed
            if self.cli.readline_enabled and readline is not None:
                readline.set_history_length(self.cli.history_length)
        elif normalized_key == "logging_enabled":
            parsed_bool = bool_map.get(raw_value.strip().lower())
            if parsed_bool is None:
                raise ValueError("logging_enabled must be a bool (true/false)")
            self.cli.logging_enabled = parsed_bool
            self.cli.close(reason="settings_changed")
            self.cli.session_logger = self.cli._create_session_logger()
        elif normalized_key == "log_dir":
            self.cli.log_dir = raw_value
            self.cli.close(reason="settings_changed")
            self.cli.session_logger = self.cli._create_session_logger()
        else:
            valid = [
                "model",
                "ollama_url",
                "max_context_tokens",
                "temperature",
                "embedding_model",
                "history_length",
                "logging_enabled",
                "log_dir",
            ]
            raise ValueError(f"Unknown setting '{key}'. Valid keys: {', '.join(valid)}")

    def _show_command_help(self):
        """Display available local slash commands."""
        self.cli.console.print("[bold]Slash commands[/bold]")
        self.cli.console.print("  /help")
        self.cli.console.print("  /settings [show]")
        self.cli.console.print("  /settings set <key> <value>")
        self.cli.console.print("  /settings save [config_path]")
        self.cli.console.print("  /set <key> <value>  (alias)")
        self.cli.console.print("  /shell <command>    (alias: /bash)")
        self.cli.console.print("  /new                (start a new session)")
        self.cli.console.print("  /exit               (alias: /quit)")

    def handle(self, raw_input: str) -> str:
        """Handle local slash commands. Returns ``continue`` or ``exit``."""
        try:
            tokens = shlex.split(raw_input[1:])
        except ValueError as exc:
            self.cli.console.print(f"[red]Invalid command syntax:[/red] {exc}")
            return "continue"

        if not tokens:
            self.cli.console.print("[yellow]Empty command.[/yellow] Try /help")
            return "continue"

        command = tokens[0].lower()
        args = tokens[1:]

        if command in {"exit", "quit"}:
            self.cli.close(reason="command_exit")
            self.cli.console.print("[green]Goodbye! 👋[/green]")
            return "exit"

        if command in {"help", "?"}:
            self._show_command_help()
            return "continue"

        if command in {"new", "reset"}:
            self.cli._start_new_session()
            return "continue"

        if command in {"shell", "bash"}:
            if not args:
                self.cli.console.print("[yellow]Usage:[/yellow] /shell <command>")
                return "continue"

            shell_command = " ".join(args)
            self.cli.session_logger.log_tool_call(
                tool_name="run_bash_command",
                arguments={"command": shell_command, "working_directory": self.cli.cwd},
                tool_call_id="local-shell-command",
            )
            result = execute_tool("run_bash_command", command=shell_command, working_directory=self.cli.cwd)
            self.cli.session_logger.log_tool_result(
                tool_name="run_bash_command",
                result=result,
                tool_call_id="local-shell-command",
            )

            if result.get("stdout"):
                self.cli.console.print(result["stdout"], end="")
            if result.get("stderr"):
                self.cli.console.print(result["stderr"], style="red", end="")

            exit_code = result.get("exit_code", -1)
            if exit_code == 0:
                self.cli.console.print("[green]Command completed successfully.[/green]")
            else:
                self.cli.console.print(f"[yellow]Command exited with code {exit_code}.[/yellow]")

            return "continue"

        if command in {"settings", "set"}:
            if command == "set":
                args = ["set", *args]

            subcommand = args[0].lower() if args else "show"

            if subcommand in {"show", "list"}:
                self._show_settings()
                return "continue"

            if subcommand == "set":
                if len(args) < 3:
                    self.cli.console.print("[yellow]Usage:[/yellow] /settings set <key> <value>")
                    return "continue"

                key = args[1]
                value = " ".join(args[2:])
                try:
                    self._update_setting(key, value)
                except ValueError as exc:
                    self.cli.console.print(f"[red]Invalid setting:[/red] {exc}")
                    return "continue"

                self.cli.console.print(f"[green]Updated {key} to {value}[/green]")
                return "continue"

            if subcommand == "save":
                target_path = args[1] if len(args) > 1 else self.cli.config_path
                written_path = write_config(self._build_agent_config(), config_path=target_path)
                self.cli.config_path = str(written_path)
                self.cli.console.print(f"[green]Saved settings to {written_path}[/green]")
                return "continue"

            self.cli.console.print("[yellow]Usage:[/yellow] /settings [show|set|save]")
            return "continue"

        self.cli.console.print(f"[yellow]Unknown command:[/yellow] /{command}. Try /help")
        return "continue"
