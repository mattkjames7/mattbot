"""
LLM client for making chat requests to Ollama server.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Iterator
import tiktoken


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://192.168.0.34:11434", model: str = "gpt-oss"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL of the Ollama server
            model: Model name to use for tokenization (default: gpt-oss)
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        self.model = model
        # Try to get tokenizer for the model; fall back to cl100k_base for gpt-oss
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # For models like gpt-oss that might not be registered, use cl100k_base
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.last_context_length = 0
        self.last_completion_tokens = 0
    
    def calculate_context_length(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate the token count for the given messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Number of tokens in the messages
        """
        token_count = 0
        for message in messages:
            # Add tokens for the message role and content
            token_count += 4  # Overhead for message metadata
            token_count += len(self.tokenizer.encode(message.get("content", "")))
        
        # Add buffer for response overhead
        token_count += 2
        
        return token_count
    
    def get_context_stats(self) -> Dict[str, int]:
        """
        Get statistics about the last API call.
        
        Returns:
            Dictionary with context_length and completion_tokens
        """
        return {
            "context_length": self.last_context_length,
            "completion_tokens": self.last_completion_tokens
        }
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """
        Send a chat request to Ollama.
        
        Args:
            model: Name of the model to use (e.g., 'llama2', 'mistral')
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to Ollama
            
        Returns:
            Response dictionary or iterator of response chunks if streaming
        """
        # Calculate context length before sending
        self.last_context_length = self.calculate_context_length(messages)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            self.chat_url,
            json=payload,
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            return self._stream_response(response)
        else:
            response_data = response.json()
            # Extract completion tokens from response if available
            if "message" in response_data:
                completion_content = response_data["message"].get("content", "")
                self.last_completion_tokens = len(self.tokenizer.encode(completion_content))
            return response_data
    
    def _stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """
        Stream response chunks from Ollama.
        
        Args:
            response: Streaming HTTP response
            
        Yields:
            Parsed JSON chunks
        """
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """
        Generate completion from a prompt.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary or iterator of response chunks if streaming
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        response = requests.post(
            self.generate_url,
            json=payload,
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
