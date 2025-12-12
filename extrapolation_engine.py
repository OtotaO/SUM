"""
Extrapolation Engine - The Reverse Summarizer
Expands concepts, tags, and seeds into full text using LLMs.
"""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
import json
from llm_backend import llm_backend, LLMConfig, ModelProvider

logger = logging.getLogger(__name__)

@dataclass
class ExtrapolationConfig:
    """Configuration for text expansion"""
    target_format: str  # paragraph, article, essay, story, email
    style: str          # academic, creative, professional, casual
    tone: str           # neutral, optimistic, critical, humorous
    length_words: int   # Approx target length
    creativity: float = 0.7  # Temperature
    model: str = "gpt-3.5-turbo" # Default model

class ExtrapolationEngine:
    """
    Core engine for bi-directional knowledge transformation (Expansion side).
    """
    
    def __init__(self):
        pass
        
    def _construct_prompt(self, seed: str, config: ExtrapolationConfig) -> str:
        """Construct a prompt that guides the LLM to expand the seed."""
        
        return f"""
        You are a highly skilled writer and knowledge expander.
        
        TASK: Expand the following seed text into a {config.target_format}.
        
        SEED: "{seed}"
        
        REQUIREMENTS:
        - Format: {config.target_format}
        - Style: {config.style}
        - Tone: {config.tone}
        - Target Length: Approximately {config.length_words} words.
        
        INSTRUCTIONS:
        1. Take the core concepts from the seed.
        2. Extrapolate logical consequences, examples, and context.
        3. Maintain coherence and flow.
        4. Do not simply repeat the seed; evolve it.
        
        OUTPUT:
        """

    async def extrapolate(self, seed: str, config: ExtrapolationConfig) -> str:
        """
        Generate expanded text from a seed.
        """
        prompt = self._construct_prompt(seed, config)
        
        # Determine provider based on config or default
        # For extrapolation, we need a smart model.
        provider = ModelProvider.OPENAI # Defaulting to OpenAI for quality expansion
        
        # Check if we should use local if configured
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA
        
        result = await llm_backend.generate(
            prompt=prompt,
            provider=provider,
            model_name=config.model,
            temperature=config.creativity,
            max_tokens=int(config.length_words * 1.5) # Allow buffer
        )
        
        return result

    async def stream_extrapolate(self, seed: str, config: ExtrapolationConfig) -> AsyncGenerator[str, None]:
        """
        Stream the expansion generation.
        """
        prompt = self._construct_prompt(seed, config)
        
        provider = ModelProvider.OPENAI
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA

        async for chunk in llm_backend.stream_generate(
            prompt=prompt,
            provider=provider,
            model_name=config.model,
            temperature=config.creativity,
            max_tokens=int(config.length_words * 1.5)
        ):
            yield chunk
