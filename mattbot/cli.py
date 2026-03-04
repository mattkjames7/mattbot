#!/usr/bin/env python3
"""
CLI tool for the agent system.
Provides an interactive chat interface with tool-calling capabilities.
"""
import os
import json
import time
import threading
from typing import List, Dict, Any

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from mattbot.config import ensure_config_file, resolve_config
from mattbot.artifact_store import ArtifactStore
from mattbot.commands import CommandProcessor
from mattbot.rich_themes import DarkerOneDarkStyle
from mattbot.session_logger import SessionLogger
from mattbot.system_prompt import SYSTEM_PROMPT

from mattbot.llm import OllamaClient
from mattbot.tools import TOOLS
from mattbot.tool_executor import execute_tool


class AgentCLI:
    def __init__(
        self,
        model: str = "gpt-oss:latest",
        ollama_url: str = "http://localhost:11434",
        max_context_tokens: int = 8192,
        temperature: float = 0.7,
        embedding_model: str = "all-minilm:l6-v2",
        history_length: int = 1000,
        logging_enabled: bool = False,
        log_dir: str = "~/.mattbot/logs",
        artifact_store_enabled: bool = True,
        artifact_dir: str = "~/.mattbot/artifacts",
        artifact_ttl_days: int = 7,
        artifact_max_sessions: int = 20,
        artifact_inline_char_limit: int = 8000,
        config_path: str | None = None,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.history_length = history_length
        self.logging_enabled = logging_enabled
        self.log_dir = log_dir
        self.artifact_store_enabled = artifact_store_enabled
        self.artifact_dir = artifact_dir
        self.artifact_ttl_days = artifact_ttl_days
        self.artifact_max_sessions = artifact_max_sessions
        self.artifact_inline_char_limit = artifact_inline_char_limit
        self.config_path = config_path

        self.client = OllamaClient(
            base_url=self.ollama_url,
            model=self.model.split(":")[0]
        )  # Extract base model name
        self.messages: List[Dict[str, Any]] = []
        self.cwd = os.getcwd()
        self.console = Console(color_system="truecolor")
        self.session_logger = self._create_session_logger()
        self.artifact_store = self._create_artifact_store()
        
        # Spinner frames for animation
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        
        # Status tracking
        self.current_status = ""
        self.current_stage = "processing"
        self.last_committed_status = None
        self.live_display = None
        self.animation_thread = None
        self.animation_running = False
        self.status_lock = threading.Lock()
        self.readline_enabled = False
        self.command_processor = CommandProcessor(self)

        self._setup_readline()

    def _messages_for_llm(self) -> List[Dict[str, Any]]:
        """Build the message list sent to the model, including system prompt."""
        tool_names = [tool["function"]["name"] for tool in TOOLS if "function" in tool and "name" in tool["function"]]
        tool_instructions = (
            "\n\nAVAILABLE TOOL NAMES (use exact names only):\n"
            + "\n".join(f"- {name}" for name in tool_names)
            + "\nNever call a wrapper tool name like 'functions'. Call the target tool directly."
        )
        return [{"role": "system", "content": f"{SYSTEM_PROMPT}{tool_instructions}"}, *self.messages]

    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize malformed tool names emitted by some models."""
        name = (tool_name or "").strip()

        # Strip accidental channel/control suffixes like: edit_file<|channel|>commentary
        if "<|" in name:
            name = name.split("<|", 1)[0].strip()

        # Handle dotted wrappers like functions.edit_file
        if name not in {tool["function"]["name"] for tool in TOOLS} and "." in name:
            candidate = name.split(".")[-1].strip()
            if candidate in {tool["function"]["name"] for tool in TOOLS}:
                name = candidate

        return name

    def _sanitize_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Filter/normalize arguments so tool calls match schema and runtime expectations."""
        args = dict(arguments)

        # Common alias emitted by models
        if "filename" in args and "file_path" not in args:
            args["file_path"] = args["filename"]

        # Practical defaults for malformed empty values
        if tool_name == "list_directory" and not args.get("path"):
            args["path"] = "."
        if tool_name == "run_bash_command" and "working_directory" in args and not args.get("working_directory"):
            args["working_directory"] = self.cwd

        # Keep only schema-declared keys for the selected tool
        tool_def = next((tool for tool in TOOLS if tool.get("function", {}).get("name") == tool_name), None)
        if tool_def:
            props = tool_def.get("function", {}).get("parameters", {}).get("properties", {})
            allowed = set(props.keys())
            args = {k: v for k, v in args.items() if k in allowed}

        return args

    def _session_metadata(self) -> Dict[str, Any]:
        """Build metadata recorded at session start."""
        return {
            "model": self.model,
            "ollama_url": self.ollama_url,
            "cwd": self.cwd,
            "max_context_tokens": self.max_context_tokens,
            "temperature": self.temperature,
            "artifact_store_enabled": self.artifact_store_enabled,
            "artifact_dir": self.artifact_dir,
            "artifact_ttl_days": self.artifact_ttl_days,
            "artifact_max_sessions": self.artifact_max_sessions,
            "artifact_inline_char_limit": self.artifact_inline_char_limit,
        }

    def _create_session_logger(self) -> SessionLogger:
        """Create a fresh session logger instance from current settings."""
        return SessionLogger(
            enabled=self.logging_enabled,
            log_dir=self.log_dir,
            session_metadata=self._session_metadata(),
        )

    def _create_artifact_store(self) -> ArtifactStore:
        """Create artifact store from current settings."""
        return ArtifactStore(
            enabled=self.artifact_store_enabled,
            artifact_dir=self.artifact_dir,
            ttl_days=self.artifact_ttl_days,
            max_sessions=self.artifact_max_sessions,
        )

    def _rebuild_client(self):
        """Rebuild LLM client after model/url changes."""
        self.client = OllamaClient(
            base_url=self.ollama_url,
            model=self.model.split(":")[0],
        )

    def _start_new_session(self):
        """Reset conversational state and start a new session log."""
        self.close(reason="new_session")
        self.messages = []
        self.client.last_context_length = 0
        self.client.last_completion_tokens = 0
        self.session_logger = self._create_session_logger()
        if self.artifact_store_enabled:
            self.artifact_store.start_new_session()

        self.console.print("[green]Started a new session.[/green]")
        if self.session_logger.enabled and self.session_logger.file_path:
            self.console.print(f"[bold]Session log:[/bold] [green]{self.session_logger.file_path}[/green]")

    def _extract_tool_key_facts(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract compact, high-signal fields from a tool result."""
        keys_of_interest = [
            "success",
            "error",
            "exit_code",
            "command",
            "file_path",
            "lines_read",
            "total_lines",
            "bytes_written",
            "lines_written",
            "path",
            "matches",
            "num_results",
            "working_directory",
        ]
        facts: dict[str, Any] = {}
        for key in keys_of_interest:
            if key in result:
                facts[key] = result[key]
        return facts

    def _build_tool_summary(self, tool_name: str, result: dict[str, Any], max_chars: int = 500) -> str:
        """Build a compact textual summary for model context."""
        success = result.get("success")
        prefix = f"{tool_name}: "
        if success is True:
            prefix += "success"
        elif success is False:
            prefix += "failed"
        else:
            prefix += "completed"

        parts: list[str] = []
        if "error" in result and result.get("error"):
            parts.append(f"error={result.get('error')}")
        if "exit_code" in result:
            parts.append(f"exit_code={result.get('exit_code')}")
        if "file_path" in result:
            parts.append(f"file={result.get('file_path')}")
        if "lines_read" in result:
            parts.append(f"lines_read={result.get('lines_read')}")
        if "bytes_written" in result:
            parts.append(f"bytes_written={result.get('bytes_written')}")
        if "matches" in result and isinstance(result.get("matches"), list):
            parts.append(f"matches={len(result.get('matches', []))}")
        if "stdout" in result and isinstance(result.get("stdout"), str) and result.get("stdout"):
            stdout_preview = result["stdout"].strip().replace("\n", " ")
            parts.append(f"stdout_preview={stdout_preview[:120]}")

        summary = prefix
        if parts:
            summary += " | " + " | ".join(parts)

        if len(summary) > max_chars:
            return summary[: max_chars - 3] + "..."
        return summary

    def _format_tool_result_for_model(self, tool_name: str, result: dict[str, Any]) -> str:
        """Format tool result for model context with artifact-backed truncation."""
        serialized = json.dumps(result, ensure_ascii=False)
        if len(serialized) <= self.artifact_inline_char_limit:
            return serialized

        artifact = self.artifact_store.put(tool_name=tool_name, result=result)
        compact_payload = {
            "ok": bool(result.get("success", True)),
            "summary": self._build_tool_summary(tool_name, result),
            "key_facts": self._extract_tool_key_facts(result),
            "artifact_id": artifact.get("artifact_id"),
            "artifact_session_id": artifact.get("session_id"),
            "truncated": True,
            "raw_size_chars": len(serialized),
            "inline_char_limit": self.artifact_inline_char_limit,
        }
        return json.dumps(compact_payload, ensure_ascii=False)

    def _handle_command(self, raw_input: str) -> str:
        """Delegate slash commands to command processor."""
        return self.command_processor.handle(raw_input)

    def _setup_readline(self):
        """Enable terminal line editing and prompt history when available."""
        if readline is None:
            return

        self.readline_enabled = True
        readline.set_history_length(self.history_length)

    def _read_user_input(self, prompt: str = "You: ") -> str:
        """Read user input with optional in-session readline history."""
        user_input = input(prompt)

        if self.readline_enabled and user_input:
            history_len = readline.get_current_history_length()
            last_item = readline.get_history_item(history_len) if history_len > 0 else None
            if last_item != user_input:
                readline.add_history(user_input)

        return user_input.strip()
        
    def print_separator(self):
        """Print a visual separator."""
        print("\n" + "─" * 80 + "\n")
        
    def _animate_spinner(self):
        """Background thread that continuously animates the spinner."""
        while self.animation_running:
            with self.status_lock:
                self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
                self._render_status()
            time.sleep(0.1)  # 10 Hz animation
    
    def _render_status(self):
        """Render the current status text with the current spinner frame."""
        if not self.live_display:
            return
        
        spinner = self.spinner_frames[self.spinner_index]
        status_text = self._build_status_text(
            status=self.current_status,
            stage=self.current_stage,
            spinner=spinner,
            completed=False
        )
        self.live_display.update(status_text)

    def _build_status_text(
        self,
        status: str,
        stage: str,
        spinner: str | None,
        completed: bool
    ) -> Text:
        """Build status text for live or completed display."""
        status_text = Text()

        if stage == "llm":
            label = "🧠 LLM Generating"
            label_style = "cyan bold"
            spinner_style = "cyan"
        elif stage == "tool":
            label = "🔧 Executing Tool"
            label_style = "yellow bold"
            spinner_style = "yellow"
        elif stage == "waiting":
            label = "Waiting for response"
            label_style = "orange"
            spinner_style = "orange"
        elif stage == "done":
            label = status
            label_style = "green"
            spinner_style = "green"
        else:
            label = status
            label_style = "green"
            spinner_style = "green"

        if completed:
            status_text.append("✓ ", style="green")
        elif spinner and stage != "done":
            status_text.append(f"{spinner} ", style=spinner_style)
        elif stage == "done":
            status_text.append("⏳ ", style="green")

        status_text.append(label, style=label_style)

        if stage != "done" and status:
            status_text.append(f" | {status}", style="dim")

        return status_text

    def _commit_status(self, status: str, stage: str):
        """Print a completed status line to the console history."""
        if not status:
            return
        if self.last_committed_status == (status, stage):
            return

        status_text = self._build_status_text(
            status=status,
            stage=stage,
            spinner=None,
            completed=True
        )

        if self.live_display:
            self.live_display.console.print(status_text)
        else:
            self.console.print(status_text)

        self.last_committed_status = (status, stage)
    
    def update_status(self, status: str, stage: str = "processing"):
        """Update the status message and stage."""
        with self.status_lock:
            if self.current_status and (self.current_status, self.current_stage) != (status, stage):
                self._commit_status(self.current_status, self.current_stage)
            self.current_status = status
            self.current_stage = stage
            self._render_status()
    
    def start_status_display(self):
        """Start the live status display with animation thread."""
        self.live_display = Live(
            Text("⏳ Initializing...", style="dim"),
            console=self.console,
            refresh_per_second=10
        )
        self.live_display.__enter__()
        
        # Start animation thread
        self.animation_running = True
        self.animation_thread = threading.Thread(target=self._animate_spinner, daemon=True)
        self.animation_thread.start()
    
    def stop_status_display(self):
        """Stop the live status display and animation thread."""
        self.animation_running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1)
        
        if self.live_display:
            with self.status_lock:
                self._commit_status(self.current_status, self.current_stage)
            self.live_display.__exit__(None, None, None)
            self.live_display = None

    def _process_streamed_response(self, response_stream) -> Dict[str, Any]:
        """Stream thinking/content output with rich formatting and collect tool calls."""
        full_response = {"message": {"content": "", "tool_calls": []}}
        has_shown_thinking_header = False
        has_started_markdown_stream = False

        markdown_live = Live(
            Markdown("", code_theme=DarkerOneDarkStyle),
            console=self.console,
            refresh_per_second=20,
            auto_refresh=False
        )

        for chunk in response_stream:
            if "message" in chunk:
                message = chunk["message"]

                # Display thinking tokens in real-time (proper rich formatting)
                if "thinking" in message and message["thinking"]:
                    if not has_shown_thinking_header:
                        self.console.print()
                        self.console.print("💭 Thinking:", style="bold dim")
                        has_shown_thinking_header = True
                    self.console.print(message["thinking"], style="dim", end="")

                # Stream markdown output while preserving formatting
                if "content" in message and message["content"]:
                    if has_shown_thinking_header and not has_started_markdown_stream:
                        self.console.print()
                        self.console.print()

                    if not has_started_markdown_stream:
                        markdown_live.__enter__()
                        has_started_markdown_stream = True

                    full_response["message"]["content"] += message["content"]
                    markdown_live.update(
                        Markdown(
                            full_response["message"]["content"],
                            code_theme=DarkerOneDarkStyle
                        ),
                        refresh=True
                    )

            if "message" in chunk and "tool_calls" in chunk["message"]:
                full_response["message"]["tool_calls"] = chunk["message"].get("tool_calls", [])

        if has_started_markdown_stream:
            markdown_live.__exit__(None, None, None)
            self.console.print()
        elif has_shown_thinking_header:
            self.console.print()

        return full_response
        
    def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = self._normalize_tool_name(tool_call["function"]["name"])
            
            # Update status with tool name
            self.update_status(f"Running: {tool_name}", "tool")
            time.sleep(0.1)  # Small delay for visual effect
            
            # Parse arguments (handle both string and dict formats)
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif not isinstance(arguments, dict):
                arguments = {}

            # Compatibility shim for models that wrap calls as: {name: "<tool>", arguments: {...}}
            if tool_name == "functions" and isinstance(arguments, dict):
                wrapped_name = arguments.get("name")
                wrapped_arguments = arguments.get("arguments", {})
                if isinstance(wrapped_name, str) and wrapped_name in [t["function"]["name"] for t in TOOLS]:
                    tool_name = self._normalize_tool_name(wrapped_name)
                    arguments = wrapped_arguments if isinstance(wrapped_arguments, dict) else {}

            arguments = self._sanitize_tool_arguments(tool_name=tool_name, arguments=arguments)

            self.session_logger.log_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call.get("id"),
            )
            
            # Execute the tool
            try:
                # Add config values for tool calls
                if tool_name == "semantic_search":
                    arguments["embedding_model"] = self.embedding_model
                    arguments["ollama_url"] = self.ollama_url
                result = execute_tool(tool_name, **arguments)
                self.session_logger.log_tool_result(tool_name=tool_name, result=result, tool_call_id=tool_call.get("id"))
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": self._format_tool_result_for_model(tool_name=tool_name, result=result)
                })
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                self.session_logger.log_tool_result(
                    tool_name=tool_name,
                    result={"success": False, "error": error_msg},
                    tool_call_id=tool_call.get("id"),
                )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"error": error_msg})
                })
        
        return tool_results
    
    def chat(self, user_input: str) -> str:
        """Send a message and handle tool calls."""
        self.last_committed_status = None
        self.start_status_display()
        
        try:
            # Add user message
            self.messages.append({
                "role": "user",
                "content": user_input
            })
            self.session_logger.log_message("user", user_input)
            
            # Initial LLM call with streaming
            self.update_status("Sending request to LLM", "llm")
            response_stream = self.client.chat(
                model=self.model,
                messages=self._messages_for_llm(),
                tools=TOOLS,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_context_tokens,
            )
            
            # Stop animation temporarily to stream response
            self.animation_running = False
            if self.animation_thread:
                self.animation_thread.join(timeout=1)
            self.stop_status_display()
            full_response = self._process_streamed_response(response_stream)
            
            # Calculate completion tokens after streaming
            self.client.last_completion_tokens = len(
                self.client.tokenizer.encode(full_response["message"]["content"])
            )
            
            # Restart animation for tool handling
            self.start_status_display()
            
            # Add assistant response to history
            self.messages.append(full_response["message"])
            self.session_logger.log_message(
                "assistant",
                full_response["message"].get("content", ""),
                tool_call_count=len(full_response["message"].get("tool_calls", [])),
            )
            
            # Handle tool calls if present
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while full_response["message"].get("tool_calls") and iteration < max_iterations:
                iteration += 1
                
                # Execute tools
                tool_calls = full_response["message"]["tool_calls"]
                self.update_status(f"Executing {len(tool_calls)} tool(s)", "tool")
                tool_results = self.handle_tool_calls(tool_calls)
                
                # Add tool results to messages
                self.messages.extend(tool_results)
                for tool_result in tool_results:
                    self.session_logger.log_message(
                        "tool",
                        tool_result.get("content", ""),
                        tool_call_id=tool_result.get("tool_call_id"),
                    )
                
                # Get next response from LLM with streaming
                self.update_status("LLM processing tool results", "llm")
                
                response_stream = self.client.chat(
                    model=self.model,
                    messages=self._messages_for_llm(),
                    tools=TOOLS,
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.max_context_tokens,
                )
                
                # Stop animation to stream response
                self.animation_running = False
                if self.animation_thread:
                    self.animation_thread.join(timeout=1)
                self.stop_status_display()
                full_response = self._process_streamed_response(response_stream)
                
                # Calculate completion tokens
                self.client.last_completion_tokens = len(
                    self.client.tokenizer.encode(full_response["message"]["content"])
                )
                
                # Restart animation
                self.start_status_display()
                
                # Add assistant response to history
                self.messages.append(full_response["message"])
                self.session_logger.log_message(
                    "assistant",
                    full_response["message"].get("content", ""),
                    tool_call_count=len(full_response["message"].get("tool_calls", [])),
                )
            
            # Display context metrics before closing status
            stats = self.client.get_context_stats()
            final_status = (
                f"Complete | 📊 {stats['context_length']} ctx tokens "
                f"| {stats['completion_tokens']} out tokens"
            )
            self.update_status(final_status, "done")
            time.sleep(0.5)  # Brief pause to show completion
            
            # Return final content
            return full_response["message"].get("content", "")
        
        finally:
            self.stop_status_display()

    def close(self, reason: str = "session_end"):
        """Finalize session resources."""
        self.session_logger.finalize(reason=reason)
    
    def run(self):
        """Run the interactive CLI loop."""
        ascii_art = r"""
[bold #7a1f2a]    ___  ___      _   _   _____       _[/bold #7a1f2a]
[bold #9c2f3f]    |  \/  |     | | | | | ___ \     | |[/bold #9c2f3f]
[bold #c44536]    | .  . | __ _| |_| |_| |_/ / ___ | |_[/bold #c44536]
[bold #d96b2b]    | |\/| |/ _` | __| __| ___ \/ _ \| __|[/bold #d96b2b]
[bold #e68a2e]    | |  | | (_| | |_| |_| |_/ / (_) | |_[/bold #e68a2e]
[bold #f2a93b]    \_|  |_/\__,_|\__|\__\____/ \___/ \__|[/bold #f2a93b]
[bold #ffd166]              _ ._  _ , _ ._[/bold #ffd166]
[bold #ffd166]            (_ ' ( `  )_  .__)[/bold #ffd166]
[bold #ffd166]          ( (  (    )   `)  ) _)[/bold #ffd166]
[bold #ffd166]         (__ (_   (_ . _) _) ,__)[/bold #ffd166]
[bold #ffd166]             `~~`\ ' . /`~~`[/bold #ffd166]
[bold #ffd166]                  ;   ;[/bold #ffd166]
[bold #ffd166]                  /   \ [/bold #ffd166]
[bold #ffd166]______________..-`_____`-..______________[/bold #ffd166]

[dim]Interactive AI Assistant - Type your commands below
Use /help for local commands (no LLM call)
Press Ctrl+D to exit[/dim]
"""
        self.console.print(ascii_art)
        self.console.print(f"[bold]Working directory:[/bold] [green]{self.cwd}[/green]")
        self.console.print(f"[bold]Model:[/bold] [green]{self.model}[/green]")
        if self.session_logger.enabled and self.session_logger.file_path:
            self.console.print(f"[bold]Session log:[/bold] [green]{self.session_logger.file_path}[/green]")
        self.print_separator()
        
        while True:
            try:
                # Get user input
                user_input = self._read_user_input("You: ")
                
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self.print_separator()
                    command_result = self._handle_command(user_input)
                    self.print_separator()
                    if command_result == "exit":
                        break
                    continue
                
                self.print_separator()
                
                # Get response (response is already streamed and displayed in chat())
                self.chat(user_input)
                self.print_separator()
                
            except EOFError:
                # Ctrl+D pressed
                print("\n\nGoodbye! 👋")
                self.close(reason="eof")
                break
            except KeyboardInterrupt:
                # Ctrl+C pressed
                print("\n\nInterrupted. Goodbye! 👋")
                self.close(reason="keyboard_interrupt")
                break
            except Exception as e:
                self.session_logger.log_event("error", message=str(e))
                print(f"\n❌ Error: {e}")
                self.print_separator()


def main():
    """Entry point for the CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MattBot - Interactive AI Assistant")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model to use (overrides config file)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama server URL (overrides config file)"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum context tokens (overrides config file)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config file)"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Prompt history size for arrow-key recall (overrides config file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: ~/.config/mattbot/config.toml)"
    )
    parser.add_argument(
        "--log",
        dest="logging_enabled",
        action="store_true",
        default=None,
        help="Enable session logging to file"
    )
    parser.add_argument(
        "--no-log",
        dest="logging_enabled",
        action="store_false",
        default=None,
        help="Disable session logging to file"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for session logs (default: ~/.mattbot/logs)"
    )
    parser.add_argument(
        "--artifact-store",
        dest="artifact_store_enabled",
        action="store_true",
        default=None,
        help="Enable persistent artifact store for large tool outputs"
    )
    parser.add_argument(
        "--no-artifact-store",
        dest="artifact_store_enabled",
        action="store_false",
        default=None,
        help="Disable persistent artifact store"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory for stored tool output artifacts (default: ~/.mattbot/artifacts)"
    )
    parser.add_argument(
        "--artifact-ttl-days",
        type=int,
        default=None,
        help="Delete artifact sessions older than this many days (default: 7)"
    )
    parser.add_argument(
        "--artifact-max-sessions",
        type=int,
        default=None,
        help="Maximum number of artifact sessions to keep (default: 20)"
    )
    parser.add_argument(
        "--artifact-inline-char-limit",
        type=int,
        default=None,
        help="Max chars of raw tool result allowed in model context before artifacting (default: 8000)"
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create a default config file (if missing) and exit"
    )
    
    args = parser.parse_args()

    if args.init_config:
        config_path = ensure_config_file(args.config)
        print(f"Config ready at: {config_path}")
        return

    config = resolve_config(cli_args=args, config_path=args.config)
    
    cli = AgentCLI(
        model=config.model,
        ollama_url=config.ollama_url,
        max_context_tokens=config.max_context_tokens,
        temperature=config.temperature,
        embedding_model=config.embedding_model,
        history_length=config.history_length,
        logging_enabled=config.logging_enabled,
        log_dir=config.log_dir,
        artifact_store_enabled=config.artifact_store_enabled,
        artifact_dir=config.artifact_dir,
        artifact_ttl_days=config.artifact_ttl_days,
        artifact_max_sessions=config.artifact_max_sessions,
        artifact_inline_char_limit=config.artifact_inline_char_limit,
        config_path=args.config,
    )
    cli.run()


if __name__ == "__main__":
    main()
