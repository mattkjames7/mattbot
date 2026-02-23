#!/usr/bin/env python3
"""
CLI tool for the agent system.
Provides an interactive chat interface with tool-calling capabilities.
"""
import os
import sys
import json
import time
import threading
from typing import List, Dict, Any
from itertools import cycle

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from mattbot.llm import OllamaClient
from mattbot.tools import TOOLS
from mattbot.tool_executor import execute_tool


class AgentCLI:
    def __init__(self, model: str = "gpt-oss:latest"):
        self.client = OllamaClient(model=model.split(":")[0])  # Extract base model name
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.cwd = os.getcwd()
        self.console = Console()
        
        # Spinner frames for animation
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        
        # Status tracking
        self.current_status = ""
        self.current_stage = "processing"
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
        status_text = Text()
        
        if self.current_stage == "llm":
            status_text.append(f"{spinner} ", style="cyan")
            status_text.append("🧠 LLM Generating", style="cyan bold")
        elif self.current_stage == "tool":
            status_text.append(f"{spinner} ", style="yellow")
            status_text.append("🔧 Executing Tool", style="yellow bold")
        elif self.current_stage == "waiting":
            status_text.append("⏳ ", style="blue")
            status_text.append("Waiting for response", style="blue")
        else:
            status_text.append(f"{spinner} ", style="green")
            status_text.append(self.current_status, style="green")
        
        status_text.append(f" | {self.current_status}", style="dim")
        self.live_display.update(status_text)
    
    def update_status(self, status: str, stage: str = "processing"):
        """Update the status message and stage."""
        with self.status_lock:
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
            self.live_display.__exit__(None, None, None)
            self.live_display = None
        
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
                stream=True
            )
            
            # Collect streamed response and display it
            full_response = {"message": {"content": "", "tool_calls": []}}
            has_shown_header = False
            has_shown_thinking_header = False
            
            # Stop animation temporarily to stream response
            self.animation_running = False
            if self.animation_thread:
                self.animation_thread.join(timeout=1)
            self.stop_status_display()
            
            # Stream and display the response
            for chunk in response_stream:
                if "message" in chunk:
                    message = chunk["message"]
                    
                    # Display thinking tokens in real-time
                    if "thinking" in message and message["thinking"]:
                        if not has_shown_thinking_header:
                            print("\n[dim][bold]💭 Thinking:[/bold]", flush=True)
                            has_shown_thinking_header = True
                        print(message["thinking"], end="", flush=True)
                    
                    # Collect content tokens (will render as markdown after streaming completes)
                    if "content" in message and message["content"]:
                        content = message["content"]
                        
                        if not has_shown_header:
                            if has_shown_thinking_header:
                                print("[/dim]\n", flush=True)
                            has_shown_header = True
                        
                        full_response["message"]["content"] += content
                
                # Collect tool calls if present
                if "message" in chunk and "tool_calls" in chunk["message"]:
                    full_response["message"]["tool_calls"] = chunk["message"].get("tool_calls", [])
            
            print()  # Final newline after streaming
            
            # Render the full response as Markdown for proper formatting
            if full_response["message"]["content"]:
                markdown = Markdown(full_response["message"]["content"])
                self.console.print(markdown)
            
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
                    stream=True
                )
                
                # Collect streamed response
                full_response = {"message": {"content": "", "tool_calls": []}}
                has_shown_header = False
                has_shown_thinking_header = False
                
                # Stop animation to stream response
                self.animation_running = False
                if self.animation_thread:
                    self.animation_thread.join(timeout=1)
                self.stop_status_display()
                
                # Stream and display the response
                for chunk in response_stream:
                    if "message" in chunk:
                        message = chunk["message"]
                        
                        # Display thinking tokens in real-time
                        if "thinking" in message and message["thinking"]:
                            if not has_shown_thinking_header:
                                print("\n[dim][bold]💭 Thinking:[/bold]", flush=True)
                                has_shown_thinking_header = True
                            print(message["thinking"], end="", flush=True)
                        
                        # Collect content tokens (will render as markdown after streaming completes)
                        if "content" in message and message["content"]:
                            content = message["content"]
                            
                            if not has_shown_header:
                                if has_shown_thinking_header:
                                    print("[/dim]\n", flush=True)
                                has_shown_header = True
                            
                            full_response["message"]["content"] += content
                    
                    if "message" in chunk and "tool_calls" in chunk["message"]:
                        full_response["message"]["tool_calls"] = chunk["message"].get("tool_calls", [])
                
                print()  # Final newline after streaming
                
                # Render the full response as Markdown for proper formatting
                if full_response["message"]["content"]:
                    markdown = Markdown(full_response["message"]["content"])
                    self.console.print(markdown)
                
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
            final_status = f"✓ Complete | 📊 {stats['context_length']} ctx tokens | {stats['completion_tokens']} out tokens"
            self.update_status(final_status, "done")
            time.sleep(0.5)  # Brief pause to show completion
            
            # Return final content
            return full_response["message"].get("content", "")
        
        finally:
            self.stop_status_display()
    
    def run(self):
        """Run the interactive CLI loop."""
        ascii_art = r"""
[bold cyan]
    ___  ___      _   _   ______       _   
    |  \/  |     | | | | | ___ \     | |  
    | .  . | __ _| |_| |_| |_/ / ___ | |_ 
    | |\/| |/ _` | __| __| ___ \/ _ \| __|
    | |  | | (_| | |_| |_| |_/ / (_) | |_ 
    \_|  |_/\__,_|\__|\__\____/ \___/ \__|
[/bold cyan]
[yellow]
              _ ._  _ , _ ._
            (_ ' ( `  )_  .__)
          ( (  (    )   `)  ) _)
         (__ (_   (_ . _) _) ,__)
             `~~`\ ' . /`~~`
                  ;   ;
                  /   \
______________..-`_____`-..______________
[/yellow]

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
        default="gpt-oss:latest",
        help="Ollama model to use (default: gpt-oss:latest)"
    )
    
    args = parser.parse_args()
    
    cli = AgentCLI(model=args.model)
    cli.run()


if __name__ == "__main__":
    main()
