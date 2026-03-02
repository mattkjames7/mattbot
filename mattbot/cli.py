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

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from mattbot.config import ensure_config_file, resolve_config
from mattbot.rich_themes import DarkerOneDarkStyle

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
    ):
        self.client = OllamaClient(
            base_url=ollama_url,
            model=model.split(":")[0]
        )  # Extract base model name
        self.model = model
        self.ollama_url = ollama_url
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.messages: List[Dict[str, Any]] = []
        self.cwd = os.getcwd()
        self.console = Console(color_system="truecolor")
        
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
            tool_name = tool_call["function"]["name"]
            
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
            
            # Execute the tool
            try:
                # Add config values for tool calls
                if tool_name == "semantic_search":
                    arguments["embedding_model"] = self.embedding_model
                    arguments["ollama_url"] = self.ollama_url
                result = execute_tool(tool_name, **arguments)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
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
            
            # Initial LLM call with streaming
            self.update_status("Sending request to LLM", "llm")
            response_stream = self.client.chat(
                model=self.model,
                messages=self.messages,
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
                
                # Get next response from LLM with streaming
                self.update_status("LLM processing tool results", "llm")
                
                response_stream = self.client.chat(
                    model=self.model,
                    messages=self.messages,
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
Press Ctrl+D to exit[/dim]
"""
        self.console.print(ascii_art)
        self.console.print(f"[bold]Working directory:[/bold] [green]{self.cwd}[/green]")
        self.console.print(f"[bold]Model:[/bold] [green]{self.model}[/green]")
        self.print_separator()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                self.print_separator()
                
                # Get response (response is already streamed and displayed in chat())
                self.chat(user_input)
                self.print_separator()
                
            except EOFError:
                # Ctrl+D pressed
                print("\n\nGoodbye! 👋")
                break
            except KeyboardInterrupt:
                # Ctrl+C pressed
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
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
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: ~/.config/mattbot/config.toml)"
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
    )
    cli.run()


if __name__ == "__main__":
    main()
