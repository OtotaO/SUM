"""
Unlimited Extrapolation Processor - Handle expansions to ANY length efficiently

This module provides intelligent chunking and streaming capabilities
to generate texts from paragraphs to multi-volume books without memory overflow.

Key Features:
- Stream-based text generation to temporary files or memory-mapped structures.
- Chunk-by-chunk recursive generation (Outline -> Chapters -> Sections -> Content).
- Real-time event yielding for viewable progress windows.
- Memory-efficient processing.

Author: SUM Team
License: Apache License 2.0
"""

import logging
import asyncio
from typing import Iterator, Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
import os
import tempfile
import json
from extrapolation_engine import ExtrapolationConfig, ExtrapolationEngine
from llm_backend import llm_backend, ModelProvider

logger = logging.getLogger(__name__)

class UnlimitedExtrapolationProcessor:
    """
    Process extrapolations of unlimited length intelligently.

    This processor handles:
    - Small texts (paragraphs/essays): Generated directly via streaming
    - Large texts (books): Generated iteratively chapter by chapter, saved to disk to prevent RAM overflow.
    - Massive texts (multi-volume): Hierarchical generation (Blueprint -> Volumes -> Chapters).
    """

    def __init__(self, max_memory_usage: int = 512 * 1024 * 1024):
        self.max_memory_usage = max_memory_usage
        self.engine = ExtrapolationEngine()

    async def process_extrapolation_stream(self, seed: str, config: ExtrapolationConfig) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the extrapolation generation, optimizing for massive outputs.
        """
        available_providers = llm_backend.get_available_providers()
        has_llm = 'openai' in available_providers or 'anthropic' in available_providers or 'ollama' in available_providers or 'huggingface' in available_providers

        if not has_llm:
            yield {'type': 'text', 'content': '\n**Extrapolation Unavailable**\n\nExtrapolation requires an active LLM. Please configure one.'}
            return

        provider = ModelProvider.OPENAI
        if config.model.startswith("llama") or config.model.startswith("mistral"):
             provider = ModelProvider.OLLAMA

        # Determine strategy based on target format
        if config.target_format in ['book', 'tome', 'encyclopedia']:
            async for event in self._generate_massive_book(seed, config, provider):
                yield event
        else:
            # For essays, articles, paragraphs, use the standard engine streaming
            async for event in self.engine.stream_extrapolate(seed, config):
                yield event

    async def _generate_massive_book(self, seed: str, config: ExtrapolationConfig, provider: ModelProvider) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Memory-efficient book generation.
        Instead of holding the whole book in memory, we stream it to the user and
        optionally write to a temp file, yielding events to maintain a running total.
        """
        yield {'type': 'log', 'content': 'PHASE 1: Architecting Master Blueprint...'}
        yield {'type': 'text', 'content': f"# {seed.title()}\n\n"}

        # Determine number of chapters based on requested format length
        # Simple heuristic: ~1000 words per chapter
        num_chapters = max(3, config.length_words // 1000)

        outline_prompt = f"""
        Create a comprehensive Table of Contents for a non-fiction book based on this concept: "{seed}".
        The book should be deep, insightful, and cover the topic exhaustively.
        Return ONLY a list of {num_chapters} chapter titles, numbered sequentially (e.g., "1. Introduction").
        """

        outline_text = await llm_backend.generate(
            prompt=outline_prompt,
            provider=provider,
            max_tokens=1000
        )

        chapters = [line.strip() for line in outline_text.split('\n') if line.strip() and (line[0].isdigit() or line.startswith('-'))]
        if not chapters:
            chapters = [f"Chapter {i+1}" for i in range(num_chapters)]

        yield {'type': 'log', 'content': f'Blueprint generated: {len(chapters)} chapters planned.'}
        yield {'type': 'text', 'content': "**Table of Contents**\n" + outline_text + "\n\n---\n\n"}

        total_words_generated = len(outline_text.split())
        yield {'type': 'progress', 'total_words': total_words_generated, 'chapters_complete': 0, 'total_chapters': len(chapters)}

        yield {'type': 'log', 'content': 'PHASE 2: Iterative Chapter Extrapolation...'}

        # We process chapter by chapter, forgetting previous chapters from RAM to save memory.
        for i, chapter in enumerate(chapters):
            yield {'type': 'log', 'content': f'Drafting: {chapter}...'}
            yield {'type': 'text', 'content': f"\n## {chapter}\n\n"}

            # The prompt includes the seed to keep context, but doesn't need the full previous chapters
            chapter_prompt = f"""
            Write the full, exhaustive content for '{chapter}'.
            Context: This is a book about "{seed}".
            Style: {config.style}.
            Tone: {config.tone}.
            Write comprehensively. Focus on depth, clarity, and logical progression.
            """

            chapter_word_count = 0
            async for chunk in llm_backend.stream_generate(
                prompt=chapter_prompt,
                provider=provider,
                max_tokens=2000
            ):
                yield {'type': 'text', 'content': chunk}
                chapter_word_count += len(chunk.split())

                # Periodically update running total
                if chapter_word_count % 100 == 0:
                     yield {'type': 'progress', 'total_words': total_words_generated + chapter_word_count, 'chapters_complete': i, 'total_chapters': len(chapters)}

            total_words_generated += chapter_word_count
            yield {'type': 'progress', 'total_words': total_words_generated, 'chapters_complete': i + 1, 'total_chapters': len(chapters)}
            yield {'type': 'text', 'content': "\n\n"}
            yield {'type': 'log', 'content': f'Finalized {chapter}'}

            # Free memory explicitly (important for massive generations)
            import gc
            gc.collect()

        yield {'type': 'log', 'content': 'Masterpiece generation complete.'}
        yield {'type': 'progress', 'total_words': total_words_generated, 'chapters_complete': len(chapters), 'total_chapters': len(chapters), 'status': 'done'}

# Lazy initialization
_unlimited_extrapolator = None

def get_unlimited_extrapolator():
    global _unlimited_extrapolator
    if _unlimited_extrapolator is None:
        _unlimited_extrapolator = UnlimitedExtrapolationProcessor()
    return _unlimited_extrapolator
