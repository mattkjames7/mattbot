"""
LLM client for making chat requests to Ollama server.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Iterator


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://192.168.0.34:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL of the Ollama server
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
    
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
            return response.json()
    
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
