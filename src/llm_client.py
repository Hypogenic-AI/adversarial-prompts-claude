"""LLM API client for experiment execution."""

import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response data."""
    content: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    error: Optional[str] = None


class LLMClient:
    """Unified client for multiple LLM providers."""

    def __init__(self):
        """Initialize API clients."""
        self.openai_client = None
        self.openrouter_client = None

        if os.environ.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI()
            logger.info("OpenAI client initialized")

        if os.environ.get("OPENROUTER_API_KEY"):
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            logger.info("OpenRouter client initialized")

    def query_openai(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> LLMResponse:
        """Query OpenAI API.

        Args:
            model: Model identifier (e.g., 'gpt-4.1-2025-04-14')
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Number of retry attempts

        Returns:
            LLMResponse object
        """
        if not self.openai_client:
            return LLMResponse(
                content="",
                model=model,
                provider="openai",
                tokens_used=0,
                latency_ms=0,
                error="OpenAI client not initialized"
            )

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=model,
                    provider="openai",
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    latency_ms=latency_ms
                )

            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return LLMResponse(
                        content="",
                        model=model,
                        provider="openai",
                        tokens_used=0,
                        latency_ms=0,
                        error=str(e)
                    )

    def query_openrouter(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> LLMResponse:
        """Query OpenRouter API.

        Args:
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4')
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Number of retry attempts

        Returns:
            LLMResponse object
        """
        if not self.openrouter_client:
            return LLMResponse(
                content="",
                model=model,
                provider="openrouter",
                tokens_used=0,
                latency_ms=0,
                error="OpenRouter client not initialized"
            )

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=model,
                    provider="openrouter",
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    latency_ms=latency_ms
                )

            except Exception as e:
                logger.warning(f"OpenRouter API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return LLMResponse(
                        content="",
                        model=model,
                        provider="openrouter",
                        tokens_used=0,
                        latency_ms=0,
                        error=str(e)
                    )

    def query(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Unified query interface.

        Args:
            provider: 'openai' or 'openrouter'
            model: Model identifier
            system_prompt: System message
            user_prompt: User message
            **kwargs: Additional arguments passed to provider-specific method

        Returns:
            LLMResponse object
        """
        if provider == "openai":
            return self.query_openai(model, system_prompt, user_prompt, **kwargs)
        elif provider == "openrouter":
            return self.query_openrouter(model, system_prompt, user_prompt, **kwargs)
        else:
            return LLMResponse(
                content="",
                model=model,
                provider=provider,
                tokens_used=0,
                latency_ms=0,
                error=f"Unknown provider: {provider}"
            )


if __name__ == "__main__":
    # Test the client
    client = LLMClient()

    # Test OpenAI
    print("Testing OpenAI...")
    response = client.query_openai(
        model="gpt-4.1-2025-04-14",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, World!' and nothing else.",
        max_tokens=50
    )
    print(f"OpenAI Response: {response.content}")
    print(f"Tokens: {response.tokens_used}, Latency: {response.latency_ms:.0f}ms")
    if response.error:
        print(f"Error: {response.error}")

    print()

    # Test OpenRouter
    print("Testing OpenRouter (Claude)...")
    response = client.query_openrouter(
        model="anthropic/claude-sonnet-4",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, World!' and nothing else.",
        max_tokens=50
    )
    print(f"OpenRouter Response: {response.content}")
    print(f"Tokens: {response.tokens_used}, Latency: {response.latency_ms:.0f}ms")
    if response.error:
        print(f"Error: {response.error}")
