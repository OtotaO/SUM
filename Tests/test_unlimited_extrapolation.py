import pytest
import asyncio
from unittest.mock import patch, MagicMock
from unlimited_extrapolation_processor import UnlimitedExtrapolationProcessor
from extrapolation_engine import ExtrapolationConfig

@pytest.mark.asyncio
async def test_process_extrapolation_stream_delegates_to_book():
    processor = UnlimitedExtrapolationProcessor()

    config = ExtrapolationConfig(
        target_format="book",
        style="neutral",
        tone="neutral",
        length_words=2000,
        creativity=0.7,
        model="gpt-3.5-turbo"
    )

    with patch("unlimited_extrapolation_processor.llm_backend.get_available_providers", return_value=["openai"]):
        with patch.object(processor, "_generate_massive_book") as mock_generate_book:
            async def mock_generator(*args, **kwargs):
                yield {"type": "log", "content": "mocked event"}

            mock_generate_book.side_effect = mock_generator

            events = []
            async for event in processor.process_extrapolation_stream("test seed", config):
                events.append(event)

            assert len(events) == 1
            assert events[0]["content"] == "mocked event"

@pytest.mark.asyncio
async def test_generate_massive_book_yields_expected_events():
    processor = UnlimitedExtrapolationProcessor()

    config = ExtrapolationConfig(
        target_format="book",
        style="neutral",
        tone="neutral",
        length_words=1000,
        creativity=0.7,
        model="gpt-3.5-turbo"
    )

    with patch("unlimited_extrapolation_processor.llm_backend.get_available_providers", return_value=["openai"]):
        with patch("unlimited_extrapolation_processor.llm_backend.generate") as mock_generate:
            with patch("unlimited_extrapolation_processor.llm_backend.stream_generate") as mock_stream_generate:
                # Mock the outline generation
                mock_generate.return_value = "1. Introduction\n2. Body\n3. Conclusion"

                # Mock the chapter generation stream
                async def mock_stream(*args, **kwargs):
                    yield "chunk1 "
                    yield "chunk2 "

                mock_stream_generate.side_effect = mock_stream

                events = []
                async for event in processor._generate_massive_book("test seed", config, "openai"):
                    events.append(event)

                # Verify we got log events, progress events, and text events
                assert any(e.get("type") == "log" for e in events)
                assert any(e.get("type") == "progress" for e in events)
                assert any(e.get("type") == "text" for e in events)

                # Check that outline is yielded
                assert any(e.get("type") == "text" and "Table of Contents" in e.get("content", "") for e in events)

                # Check that chapters are yielded
                assert any(e.get("type") == "text" and "chunk1" in e.get("content", "") for e in events)
