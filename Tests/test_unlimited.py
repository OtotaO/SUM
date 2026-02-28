import pytest
import json
from flask import Flask
from api.summarization import summarization_bp
from unittest.mock import patch, MagicMock

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(summarization_bp, url_prefix='/api')
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_stream_unlimited_success(client):
    """Test successful streaming unlimited text processing."""

    mock_events = [
        {'type': 'log', 'content': 'Processing small text directly...'},
        {'type': 'result', 'data': {'summary': 'This is a test summary.', 'processing_method': 'direct', 'chunks_processed': 1}}
    ]

    def mock_process_text_stream(*args, **kwargs):
        for event in mock_events:
            yield event

    with patch('api.summarization.get_unlimited_processor') as mock_get_processor:
        mock_processor = MagicMock()
        mock_processor.process_text_stream = mock_process_text_stream
        mock_get_processor.return_value = mock_processor

        response = client.post(
            '/api/stream_unlimited',
            json={'text': 'Test text for streaming unlimited endpoint.'}
        )

        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'

        content = response.data.decode('utf-8')
        lines = content.strip().split('\n\n')

        received_events = []
        for line in lines:
            if not line.startswith('data: '):
                continue

            data_str = line[6:]
            try:
                data = json.loads(data_str)
                received_events.append(data)
            except json.JSONDecodeError:
                continue

        # Should receive:
        # 1. initializing status
        # 2. log event
        # 3. result event
        # 4. complete status

        assert len(received_events) >= 3
        assert received_events[0] == {'status': 'initializing'}
        assert received_events[1].get('type') == 'log'
        assert received_events[2].get('type') == 'result'
        assert received_events[-1] == {'status': 'complete'}

def test_stream_unlimited_missing_input(client):
    """Test streaming unlimited endpoint with missing text/file."""
    response = client.post(
        '/api/stream_unlimited',
        json={}
    )

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_stream_unlimited_error(client):
    """Test error handling during stream processing."""

    def mock_stream_error(*args, **kwargs):
        raise Exception("Stream processing failed")
        yield {}

    with patch('api.summarization.get_unlimited_processor') as mock_get_processor:
        mock_processor = MagicMock()
        mock_processor.process_text_stream = mock_stream_error
        mock_get_processor.return_value = mock_processor

        response = client.post(
            '/api/stream_unlimited',
            json={'text': 'Test text that will fail'}
        )

        assert response.status_code == 200  # Error inside stream
