"""
LLM client utilities for the HCC Prognosis Assessment system.
"""

from openai import OpenAI
from typing import Optional, Dict, Any, Iterator
import json

from config.config import config


class LLMClient:
    """
    LLM client wrapper for Claude API via OpenAI-compatible proxy.

    This class handles all communication with the LLM backend,
    including API configuration, retry logic, and response parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 120
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for authentication
            api_base: API base URL (for proxy)
            model_name: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or config.api.api_key
        self.api_base = api_base or config.api.api_base
        self.model_name = model_name or config.api.model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Initialize OpenAI client (compatible with proxy)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout
        )

    def _parse_sse_stream(self, stream) -> str:
        """
        Parse SSE stream and extract full response.

        Args:
            stream: SSE stream response

        Returns:
            Full response text
        """
        if isinstance(stream, str):
            # Already a string, try to parse it
            return self._parse_sse_text(stream)

        # Handle iterator/stream
        full_text = ""
        for line in stream:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if isinstance(line, str):
                full_text += self._parse_sse_text(line)
        return full_text

    def _parse_sse_text(self, text: str) -> str:
        """Parse SSE format text and extract content."""
        result = ""
        for line in text.split('\n'):
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    continue
                try:
                    data = json.loads(data_str)
                    # Handle different response formats
                    if 'choices' in data:
                        for choice in data['choices']:
                            if 'delta' in choice and 'content' in choice['delta']:
                                result += choice['delta']['content']
                            elif 'message' in choice and 'content' in choice['message']:
                                result += choice['message']['content']
                except json.JSONDecodeError:
                    continue
        return result

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stream: Whether to stream the response
            thinking: If False, disables extended thinking (default True)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Extra body parameters
        extra_body = kwargs.pop('extra_body', {})
        if not thinking:
            extra_body['thinking'] = {"type": "disabled"}

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            stream=False,  # Always use non-streaming for simplicity
            extra_body=extra_body if extra_body else None,
            **kwargs
        )

        # Handle response - API may return SSE-formatted string
        return self._parse_response(response)

    def _strip_thinking_tags(self, text: str) -> str:
        """Remove thinking tags and content from response."""
        import re
        # Remove <thinking>...</thinking> and <thinking>...</thinking> blocks
        text = re.sub(r'<thinking\b.*?</thinking>', '', text, flags=re.DOTALL)
        # Also handle <think>...</think> format
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_response(self, response) -> str:
        """
        Parse API response and extract content.

        Args:
            response: API response (can be dict, str, or SSE iterator)

        Returns:
            Extracted text content
        """
        # If it's a string, parse as SSE
        if isinstance(response, str):
            return self._strip_thinking_tags(self._parse_sse_text(response))

        # If it's an iterator/generator (SSE stream), collect and parse
        if hasattr(response, '__iter__') and not isinstance(response, (dict, list)):
            full_text = ""
            for chunk in response:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode('utf-8')
                if isinstance(chunk, str):
                    full_text += self._parse_sse_text(chunk)
            return self._strip_thinking_tags(full_text)

        # If it's a dict (already parsed JSON)
        if isinstance(response, dict):
            if 'choices' in response:
                for choice in response['choices']:
                    if 'message' in choice and 'content' in choice['message']:
                        return self._strip_thinking_tags(choice['message']['content'])
                    if 'delta' in choice and 'content' in choice['delta']:
                        return self._strip_thinking_tags(choice['delta']['content'])

        # If it has choices attribute (OpenAI response object)
        if hasattr(response, 'choices'):
            return self._strip_thinking_tags(response.choices[0].message.content)

        # Fallback
        return self._strip_thinking_tags(str(response))

    def generate_with_messages(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stream: Whether to stream the response
            thinking: If False, disables extended thinking (default True)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Extra body parameters
        extra_body = kwargs.pop('extra_body', {})
        if not thinking:
            extra_body['thinking'] = {"type": "disabled"}

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages,
            stream=False,  # Always use non-streaming for simplicity
            extra_body=extra_body if extra_body else None,
            **kwargs
        )

        # Handle response - API may return SSE-formatted string
        return self._parse_response(response)

    def extract_structured(
        self,
        prompt: str,
        response_format: type,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from LLM response.

        This is a simplified version - for production, consider using
        structured output features or parsing JSON from responses.

        Args:
            prompt: User prompt
            response_format: Expected response format class
            system_prompt: Optional system prompt

        Returns:
            Parsed structured response
        """
        response_text = self.generate(prompt, system_prompt)

        # For now, return as dict - can be enhanced with JSON parsing
        return {"raw_response": response_text}


# Global client instance (lazy initialization)
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get the global LLM client instance.

    Returns:
        LLMClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def init_llm_client(api_key: str, api_base: str) -> LLMClient:
    """
    Initialize the global LLM client with custom settings.

    Args:
        api_key: API key
        api_base: API base URL

    Returns:
        Initialized LLMClient
    """
    global _llm_client
    _llm_client = LLMClient(api_key=api_key, api_base=api_base)
    return _llm_client
