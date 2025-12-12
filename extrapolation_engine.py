"""
Extrapolation Engine - The Reverse Summarizer
Expands concepts, tags, and seeds into full text using LLMs.
Includes Recursive Book Generation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, List
import json
import asyncio
from llm_backend import llm_backend, LLMConfig, ModelProvider

logger = logging.getLogger(__name__)

@dataclass
class ExtrapolationConfig:
    """Configuration for text expansion"""
    target_format: str  # paragraph, article, essay, book
    style: str          # academic, creative, professional, casual
    tone: str           # neutral, optimistic, critical, humorous
    length_words: int   # Approx target length
    creativity: float = 0.7  # Temperature
    model: str = "gpt-3.5-turbo" # Default model

class ExtrapolationEngine:
    """
    Core engine for bi-directional knowledge transformation (Expansion side).
    Now with Recursive Book Generation.
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
        provider = ModelProvider.OPENAI 
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA
        
        result = await llm_backend.generate(
            prompt=prompt,
            provider=provider,
            model_name=config.model,
            temperature=config.creativity,
            max_tokens=int(config.length_words * 1.5)
        )
        
        return result

    async def stream_extrapolate(self, seed: str, config: ExtrapolationConfig) -> AsyncGenerator[str, None]:
        """
        Stream the expansion generation. Handles Book mode recursively.
        """
        
        # SPECIAL HANDLING FOR BOOK MODE
        if config.target_format == 'book' or config.target_format == 'essay':
             async for chunk in self._stream_recursive_book(seed, config):
                 yield chunk
             return

        # Standard handling for shorter formats
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

    async def _stream_recursive_book(self, seed: str, config: ExtrapolationConfig) -> AsyncGenerator[str, None]:
        """
        Recursively generates a book: Outline -> Chapters.
        """
        provider = ModelProvider.OPENAI
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA

        # 1. Generate Outline
        yield "**PHASE 1: Architecting Knowledge Structure...**\n\n"
        
        outline_prompt = f"""
        Create a comprehensive Table of Contents for a non-fiction book based on this concept: "{seed}".
        The book should be deep, insightful, and cover the topic exhaustively.
        Return ONLY a list of 5 chapter titles.
        """
        
        outline_text = await llm_backend.generate(
            prompt=outline_prompt, 
            provider=provider,
            max_tokens=500
        )
        
        chapters = [line.strip() for line in outline_text.split('\n') if line.strip() and (line[0].isdigit() or line.startswith('-'))]
        if not chapters:
            chapters = ["Chapter 1: The Concept", "Chapter 2: The Implications", "Chapter 3: The Future"]
            
        yield f"**Blueprint Generated:**\n{outline_text}\n\n"
        
        # 2. Generate Chapters
        yield "**PHASE 2: Extrapolating Chapters...**\n\n"
        
        for i, chapter in enumerate(chapters):
            yield f"\n\n## {chapter}\n\n"
            
            chapter_prompt = f"""
            Write the full content for {chapter}.
            Context: This is a book about "{seed}".
            Style: {config.style}.
            Write at least 400 words. Focus on depth and clarity.
            """
            
            async for chunk in llm_backend.stream_generate(
                prompt=chapter_prompt,
                provider=provider,
                max_tokens=1000
            ):
                yield chunk
            
            yield "\n\n--- [Chapter Complete] ---\n"
            
        yield "\n\n**[Book Generation Complete]**"
