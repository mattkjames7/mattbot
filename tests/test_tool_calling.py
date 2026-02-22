"""
Integration tests for tool calling with the LLM.
"""

import os
import pytest
from mattbot.llm import OllamaClient
from mattbot.tools import get_tools


# Test cases: (prompt, expected_tool_name, description)
TOOL_CALLING_TEST_CASES = [
    (
        "List all Python files in the current directory",
        "list_directory",
        "Should use list_directory for listing files"
    ),
    (
        "Read the contents of README.md",
        "read_file",
        "Should use read_file to read a file"
    ),
    (
        "Create a new file called test.txt with the content 'Hello World'",
        "write_file",
        "Should use write_file to create a file"
    ),
    (
        "Run the command 'ls -la' to see all files",
        "run_bash_command",
        "Should use run_bash_command to execute shell commands"
    ),
    (
        "Search for all TODO comments in the codebase",
        "grep_search",
        "Should use grep_search for text pattern search"
    ),
    (
        "Find functions related to authentication in the code",
        "semantic_search",
        "Should use semantic_search for semantic code search"
    ),
    (
        "Search online for the latest Python version",
        "web_search",
        "Should use web_search for internet searches"
    ),
    (
        "Change line 5 in config.py from 'debug=False' to 'debug=True'",
        "edit_file",
        "Should use edit_file to modify files"
    ),
]


@pytest.mark.skipif(
    not os.environ.get('OLLAMA_HOST'),
    reason="OLLAMA_HOST environment variable not set"
)
class TestToolCalling:
    """Integration tests for LLM tool calling."""
    
    @pytest.fixture
    def client(self):
        """Create an Ollama client configured for the test environment."""
        ollama_host = os.environ.get('OLLAMA_HOST')
        return OllamaClient(base_url=f"http://{ollama_host}")
    
    @pytest.fixture
    def tools(self):
        """Get the list of available tools."""
        return get_tools()
    
    def test_tools_are_properly_formatted(self, tools):
        """Verify that tools follow the correct format."""
        assert len(tools) > 0, "Should have at least one tool defined"
        
        for tool in tools:
            # Check structure
            assert "type" in tool, "Tool should have a 'type' field"
            assert tool["type"] == "function", "Tool type should be 'function'"
            assert "function" in tool, "Tool should have a 'function' field"
            
            func = tool["function"]
            assert "name" in func, "Function should have a 'name' field"
            assert "description" in func, "Function should have a 'description' field"
            assert "parameters" in func, "Function should have a 'parameters' field"
            
            params = func["parameters"]
            assert "type" in params, "Parameters should have a 'type' field"
            assert params["type"] == "object", "Parameters type should be 'object'"
            assert "properties" in params, "Parameters should have 'properties' field"
            assert "required" in params, "Parameters should have 'required' field"
    
    @pytest.mark.parametrize("prompt,expected_tool,description", TOOL_CALLING_TEST_CASES)
    def test_tool_calling_with_prompt(self, client, tools, prompt, expected_tool, description):
        """Test that the LLM calls the correct tool for a given prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. When given a task, call the appropriate tool to accomplish it. Only call tools, do not provide explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make the request with tools
        response = client.chat(
            model="gpt-oss:latest",
            messages=messages,
            tools=tools,
            stream=False
        )
        
        # Print debug info
        print(f"\n{'='*60}")
        print(f"Test: {description}")
        print(f"Prompt: {prompt}")
        print(f"Expected tool: {expected_tool}")
        print(f"Response: {response}")
        
        # Verify response structure
        assert "message" in response, "Response should contain a message"
        message = response["message"]
        assert "role" in message, "Message should have a role"
        assert message["role"] == "assistant", "Message role should be assistant"
        
        # Check if tool_calls exist
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"Tool calls found: {len(tool_calls)}")
            
            # Get the first tool call
            first_tool_call = tool_calls[0]
            print(f"First tool call: {first_tool_call}")
            
            # Verify structure
            assert "function" in first_tool_call, "Tool call should have a function"
            function = first_tool_call["function"]
            assert "name" in function, "Function should have a name"
            
            called_tool_name = function["name"]
            print(f"Called tool: {called_tool_name}")
            
            # Verify the correct tool was called
            assert called_tool_name == expected_tool, \
                f"Expected tool '{expected_tool}' but got '{called_tool_name}'"
            
            # Verify arguments exist and are valid
            assert "arguments" in function, "Function should have arguments"
            arguments = function["arguments"]
            print(f"Arguments: {arguments}")
            assert isinstance(arguments, dict), "Arguments should be a dictionary"
            
            print(f"✓ Test passed: Correct tool '{expected_tool}' was called")
        else:
            # If no tool calls, this is a failure
            pytest.fail(
                f"No tool calls found in response. Expected '{expected_tool}' to be called.\n"
                f"Response content: {message.get('content', 'No content')}"
            )
    
    def test_multiple_tool_scenario(self, client, tools):
        """Test a scenario that might require multiple tools."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. Call appropriate tools to accomplish tasks."
            },
            {
                "role": "user",
                "content": "First, list the files in the current directory, then read the README.md file"
            }
        ]
        
        response = client.chat(
            model="gpt-oss:latest",
            messages=messages,
            tools=tools,
            stream=False
        )
        
        print(f"\n{'='*60}")
        print(f"Multiple tool scenario test")
        print(f"Response: {response}")
        
        # Verify at least one tool was called
        assert "message" in response
        message = response["message"]
        
        # The model should call at least one tool
        # (it might call one at a time, requiring multiple turns)
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"Number of tool calls: {len(tool_calls)}")
            
            # Check if any of the expected tools were called
            called_tools = [tc["function"]["name"] for tc in tool_calls]
            print(f"Called tools: {called_tools}")
            
            # At least one should be list_directory or read_file
            assert any(tool in ["list_directory", "read_file"] for tool in called_tools), \
                "Should call either list_directory or read_file"
        else:
            pytest.fail("No tool calls found for multiple tool scenario")
    
    def test_invalid_tool_request(self, client, tools):
        """Test that the model handles requests that don't need tools appropriately."""
        messages = [
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ]
        
        response = client.chat(
            model="gpt-oss:latest",
            messages=messages,
            tools=tools,
            stream=False
        )
        
        print(f"\n{'='*60}")
        print(f"Invalid tool request test")
        print(f"Response: {response}")
        
        # For a simple math question, the model might not call tools
        # It should either respond directly or not call any tools
        assert "message" in response
        message = response["message"]
        
        # This is valid either way - with or without tool calls
        # If it responds with content, that's fine
        # If it doesn't call tools, that's also fine
        if "content" in message and message["content"]:
            print(f"Model responded directly: {message['content']}")
        elif "tool_calls" in message and message["tool_calls"]:
            print(f"Model called tools (which is unexpected but not wrong): {message['tool_calls']}")
