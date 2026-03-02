"""
Unit tests for the LLM client.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from mattbot.llm import OllamaClient


class TestOllamaClient:
    """Tests for OllamaClient."""
    
    def test_init_default_url(self):
        """Test client initialization with default URL."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.chat_url == "http://localhost:11434/api/chat"
        assert client.generate_url == "http://localhost:11434/api/generate"
    
    def test_init_custom_url(self):
        """Test client initialization with custom URL."""
        client = OllamaClient(base_url="http://localhost:11434")
        assert client.base_url == "http://localhost:11434"
        assert client.chat_url == "http://localhost:11434/api/chat"
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"
    
    @patch('mattbot.llm.requests.post')
    def test_chat_basic(self, mock_post):
        """Test basic chat request without streaming."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "llama2",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            }
        }
        mock_post.return_value = mock_response
        
        # Make request
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        response = client.chat(model="llama2", messages=messages)
        
        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "llama2"
        assert call_args[1]['json']['messages'] == messages
        assert call_args[1]['json']['stream'] is False
        assert response['message']['content'] == "Hello! How can I help you?"
    
    @patch('mattbot.llm.requests.post')
    def test_chat_with_temperature(self, mock_post):
        """Test chat request with temperature parameter."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "test"}}
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        client.chat(model="llama2", messages=messages, temperature=0.7)
        
        call_args = mock_post.call_args
        assert call_args[1]['json']['temperature'] == 0.7
    
    @patch('mattbot.llm.requests.post')
    def test_chat_with_max_tokens(self, mock_post):
        """Test chat request with max_tokens parameter."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "test"}}
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        client.chat(model="llama2", messages=messages, max_tokens=100)
        
        call_args = mock_post.call_args
        assert call_args[1]['json']['max_tokens'] == 100
    
    @patch('mattbot.llm.requests.post')
    def test_chat_streaming(self, mock_post):
        """Test streaming chat request."""
        # Mock streaming response
        mock_response = Mock()
        mock_chunks = [
            b'{"message": {"content": "Hello"}, "done": false}',
            b'{"message": {"content": " there"}, "done": false}',
            b'{"message": {"content": "!"}, "done": true}'
        ]
        mock_response.iter_lines.return_value = mock_chunks
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        stream = client.chat(model="llama2", messages=messages, stream=True)
        
        # Consume stream
        chunks = list(stream)
        
        # Verify
        assert len(chunks) == 3
        assert chunks[0]['message']['content'] == "Hello"
        assert chunks[1]['message']['content'] == " there"
        assert chunks[2]['done'] is True
        
        call_args = mock_post.call_args
        assert call_args[1]['json']['stream'] is True
        assert call_args[1]['stream'] is True
    
    @patch('mattbot.llm.requests.post')
    def test_generate_basic(self, mock_post):
        """Test basic generate request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "llama2",
            "response": "This is a generated response."
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        response = client.generate(model="llama2", prompt="Tell me a story")
        
        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "llama2"
        assert call_args[1]['json']['prompt'] == "Tell me a story"
        assert call_args[1]['json']['stream'] is False
        assert response['response'] == "This is a generated response."
    
    @patch('mattbot.llm.requests.post')
    def test_generate_streaming(self, mock_post):
        """Test streaming generate request."""
        mock_response = Mock()
        mock_chunks = [
            b'{"response": "Once", "done": false}',
            b'{"response": " upon", "done": false}',
            b'{"response": " a time", "done": true}'
        ]
        mock_response.iter_lines.return_value = mock_chunks
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        stream = client.generate(model="llama2", prompt="Tell me a story", stream=True)
        
        chunks = list(stream)
        
        assert len(chunks) == 3
        assert chunks[0]['response'] == "Once"
        assert chunks[2]['done'] is True
    
    @patch('mattbot.llm.requests.post')
    def test_chat_raises_on_error(self, mock_post):
        """Test that HTTP errors are raised."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        
        with pytest.raises(Exception, match="Server error"):
            client.chat(model="llama2", messages=messages)
    
    @patch('mattbot.llm.requests.post')
    def test_chat_with_additional_kwargs(self, mock_post):
        """Test that additional kwargs are passed through."""
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "test"}}
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        client.chat(
            model="llama2",
            messages=messages,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['top_p'] == 0.9
        assert payload['top_k'] == 40
        assert payload['repeat_penalty'] == 1.1


# Integration tests - only run if OLLAMA_HOST is set
@pytest.mark.skipif(
    not os.environ.get('OLLAMA_HOST'),
    reason="OLLAMA_HOST environment variable not set"
)
class TestOllamaClientIntegration:
    """Integration tests for OllamaClient with real API."""
    
    def test_chat_real_api(self):
        """Test real chat request to Ollama server."""
        ollama_host = os.environ.get('OLLAMA_HOST')
        client = OllamaClient(base_url=f"http://{ollama_host}")
        
        messages = [
            {"role": "user", "content": "Say 'Hello' and nothing else."}
        ]
        
        response = client.chat(model="gpt-oss:latest", messages=messages)
        
        # Verify response structure
        assert 'message' in response
        assert 'role' in response['message']
        assert 'content' in response['message']
        assert response['message']['role'] == 'assistant'
        assert len(response['message']['content']) > 0
        print(f"\nResponse: {response['message']['content']}")
    
    def test_chat_streaming_real_api(self):
        """Test real streaming chat request to Ollama server."""
        ollama_host = os.environ.get('OLLAMA_HOST')
        client = OllamaClient(base_url=f"http://{ollama_host}")
        
        messages = [
            {"role": "user", "content": "Count from 1 to 3, one number per word."}
        ]
        
        stream = client.chat(model="gpt-oss:latest", messages=messages, stream=True)
        chunks = list(stream)
        
        # Verify we got chunks
        assert len(chunks) > 0
        
        # Verify last chunk is marked as done
        assert chunks[-1].get('done') is True
        
        # Collect full response
        full_content = ""
        for chunk in chunks:
            if 'message' in chunk and 'content' in chunk['message']:
                full_content += chunk['message']['content']
        
        print(f"\nStreamed response: {full_content}")
        assert len(full_content) > 0
    
    def test_generate_real_api(self):
        """Test real generate request to Ollama server."""
        ollama_host = os.environ.get('OLLAMA_HOST')
        client = OllamaClient(base_url=f"http://{ollama_host}")
        
        response = client.generate(
            model="gpt-oss:latest",
            prompt="Say the word 'test' and nothing else."
        )
        
        # Verify response structure
        assert 'response' in response
        assert len(response['response']) > 0
        print(f"\nGenerated response: {response['response']}")
