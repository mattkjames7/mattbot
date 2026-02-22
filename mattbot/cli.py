#!/usr/bin/env python3
"""
CLI tool for the agent system.
Provides an interactive chat interface with tool-calling capabilities.
"""
import os
import sys
import json
from typing import List, Dict, Any

from rich.console import Console
from rich.markdown import Markdown

from mattbot.llm import OllamaClient
from mattbot.tools import TOOLS
from mattbot.tool_executor import execute_tool


class AgentCLI:
    def __init__(self, model: str = "gpt-oss:latest"):
        self.client = OllamaClient()
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.cwd = os.getcwd()
        self.console = Console()
        
    def print_separator(self):
        """Print a visual separator."""
        print("\n" + "─" * 80 + "\n")
        
    def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            
            # Parse arguments (handle both string and dict formats)
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif not isinstance(arguments, dict):
                arguments = {}
            
            print(f"🔧 Calling tool: {tool_name}")
            if arguments:
                print(f"   Arguments: {json.dumps(arguments, indent=2)}")
            
            # Execute the tool
            try:
                result = execute_tool(tool_name, **arguments)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })
                print(f"✓ Tool completed successfully")
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"✗ {error_msg}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"error": error_msg})
                })
            
            self.print_separator()
        
        return tool_results
    
    def chat(self, user_input: str) -> str:
        """Send a message and handle tool calls."""
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Initial LLM call
        response = self.client.chat(
            model=self.model,
            messages=self.messages,
            tools=TOOLS
        )
        
        # Add assistant response to history
        self.messages.append(response["message"])
        
        # Handle tool calls if present
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while response["message"].get("tool_calls") and iteration < max_iterations:
            iteration += 1
            
            # Execute tools
            tool_results = self.handle_tool_calls(response["message"]["tool_calls"])
            
            # Add tool results to messages
            self.messages.extend(tool_results)
            
            # Get next response from LLM
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOLS
            )
            
            # Add assistant response to history
            self.messages.append(response["message"])
        
        # Return final content
        return response["message"].get("content", "")
    
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
         .--.
        |o_o |     Interactive AI Assistant
        |:_/ |     
       //   \ \    Type your commands below...
      (|     | )   Press Ctrl+D to exit
     /'\_   _/`\
     \___)=(___/
[/yellow]
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
                
                # Get response
                response = self.chat(user_input)
                
                # Display response
                if response:
                    self.console.print("\n[bold cyan]Assistant:[/bold cyan]")
                    markdown = Markdown(response)
                    self.console.print(markdown)
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
