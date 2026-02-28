"""
Extrapolation Engine - The Reverse Summarizer
Expands concepts, tags, and seeds into full text using LLMs.
Includes Recursive Book Generation with Transparent Logging.
"""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Dict, Any
import json
import asyncio
from llm_backend import llm_backend, LLMConfig, ModelProvider

logger = logging.getLogger(__name__)

@dataclass
class ExtrapolationConfig:
    """Configuration for text expansion"""
    target_format: str  # paragraph, article, essay, book, encyclopedia
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

    async def stream_extrapolate(self, seed: str, config: ExtrapolationConfig) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the expansion generation.
        Delegates huge formats (like books) to the memory-safe unlimited processor to avoid duplication,
        while handling shorter formats directly.
        """
        
        # Check if we have a capable LLM
        available_providers = llm_backend.get_available_providers()
        has_llm = 'openai' in available_providers or 'anthropic' in available_providers or 'ollama' in available_providers or 'huggingface' in available_providers

        if not has_llm:
            yield {'type': 'text', 'content': '\n**Extrapolation Unavailable**\n\nExtrapolation requires an active LLM (OpenAI, Anthropic, or Ollama). Please configure one or run a local Ollama instance.'}
            return

        # DELEGATE BOOK MODE TO UNLIMITED PROCESSOR
        if config.target_format in ['book', 'essay', 'encyclopedia', 'tome']:
             from unlimited_extrapolation_processor import get_unlimited_extrapolator
             processor = get_unlimited_extrapolator()
             async for event in processor.process_extrapolation_stream(seed, config):
                 yield event
             return

        # Standard handling for shorter formats
        prompt = self._construct_prompt(seed, config)
        
        provider = ModelProvider.OPENAI
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA

        yield {'type': 'log', 'content': f'Initializing {provider.value} model...'}
        yield {'type': 'log', 'content': f'Generating {config.target_format} with {config.style} style...'}

        async for chunk in llm_backend.stream_generate(
            prompt=prompt,
            provider=provider,
            model_name=config.model,
            temperature=config.creativity,
            max_tokens=int(config.length_words * 1.5)
        ):
            yield {'type': 'text', 'content': chunk}

        yield {'type': 'log', 'content': 'Generation complete.'}
        yield {'type': 'progress', 'status': 'done'}
