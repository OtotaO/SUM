import pytest
import json
import asyncio
from flask import Flask
from api.web_compatibility import web_compat_bp
from unittest.mock import MagicMock, patch

# Minimal setup for testing
@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(web_compat_bp)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_stream_summarize_success(client):
    """Test successful streaming summarization."""

    # Mock data to stream
    mock_chunks = ["This ", "is ", "a ", "streamed ", "summary."]

    # Create an async generator for the mock
    async def mock_stream_generate(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    # Patch llm_backend in api.web_compatibility
    with patch('api.web_compatibility.llm_backend') as mock_backend:
        mock_backend.default_provider = "local" # Just a string or enum
        mock_backend.stream_generate = mock_stream_generate

        # Make request (synchronous client)
        response = client.post(
            '/api/stream/summarize',
            json={'text': 'Test text'}
        )

        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'

        # Read the streamed response
        content = response.data.decode('utf-8')

        # Check for SSE format
        lines = content.strip().split('\n\n')

        received_chunks = []
        completion_signal = False
        done_signal = False

        for line in lines:
            if not line.startswith('data: '):
                continue

            data_str = line[6:]
            if data_str == '[DONE]':
                done_signal = True
                continue

            try:
                data = json.loads(data_str)
                if 'chunk' in data:
                    received_chunks.append(data['chunk'])
                elif 'type' in data and data['type'] == 'complete':
                    completion_signal = True
                elif 'error' in data:
                    pytest.fail(f"Received error: {data['error']}")
            except json.JSONDecodeError:
                continue

        assert "".join(received_chunks) == "This is a streamed summary."
        assert completion_signal is True
        assert done_signal is True

def test_stream_summarize_missing_text(client):
    """Test streaming endpoint with missing text."""
    response = client.post(
        '/api/stream/summarize',
        json={}
    )

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_stream_summarize_error(client):
    """Test error handling during streaming."""

    async def mock_stream_error(*args, **kwargs):
        raise Exception("Streaming failed")
        yield "Should not be reached"

    with patch('api.web_compatibility.llm_backend') as mock_backend:
        mock_backend.default_provider = "local"
        mock_backend.stream_generate = mock_stream_error

        response = client.post(
            '/api/stream/summarize',
            json={'text': 'Test text'}
        )

        assert response.status_code == 200 # Error happens inside stream

        content = response.data.decode('utf-8')
        assert 'Streaming failed' in content
