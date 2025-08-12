"""
test_sum_simple.py - Tests for the simplified SUM API

Simple, effective tests that actually matter.
No mocking what doesn't need mocking.
No testing implementation details.
Just: does it work?
"""

import pytest
import json
import time
from flask import Flask
from flask.testing import FlaskClient
import redis
from unittest.mock import patch, MagicMock

# Import our simple app
import sys
sys.path.append('..')
from sum_simple import app, check_rate_limit, get_summary_from_cache, save_summary_to_cache


class TestSumSimple:
    """Test the core functionality of sum_simple.py"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis for testing"""
        with patch('sum_simple.r') as mock:
            yield mock
    
    @pytest.fixture
    def mock_summarizer(self):
        """Mock the transformer model"""
        with patch('sum_simple.summarizer') as mock:
            mock.return_value = [{'summary_text': 'This is a test summary.'}]
            yield mock
    
    def test_health_endpoint(self, client: FlaskClient):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] in ['healthy', 'degraded']
        assert 'model_loaded' in data
    
    def test_summarize_success(self, client: FlaskClient, mock_redis, mock_summarizer):
        """Test successful summarization"""
        # Mock Redis to simulate cache miss
        mock_redis.get.return_value = None
        
        # Make request
        response = client.post('/summarize',
                             json={'text': 'This is a long text that needs summarization.'},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'summary' in data
        assert data['summary'] == 'This is a test summary.'
        assert data['cached'] is False
    
    def test_summarize_cached(self, client: FlaskClient, mock_redis, mock_summarizer):
        """Test cached summarization"""
        # Mock Redis to return cached summary
        mock_redis.get.return_value = 'Cached summary'
        
        response = client.post('/summarize',
                             json={'text': 'Some text'},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['summary'] == 'Cached summary'
        assert data['cached'] is True
        
        # Summarizer should not be called
        mock_summarizer.assert_not_called()
    
    def test_summarize_missing_text(self, client: FlaskClient):
        """Test error when text is missing"""
        response = client.post('/summarize',
                             json={},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_summarize_empty_text(self, client: FlaskClient):
        """Test error when text is empty"""
        response = client.post('/summarize',
                             json={'text': ''},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_summarize_text_too_long(self, client: FlaskClient):
        """Test error when text exceeds limit"""
        long_text = 'a' * 100001  # Just over the limit
        response = client.post('/summarize',
                             json={'text': long_text},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert '100000' in data['error']
    
    def test_rate_limiting(self, mock_redis):
        """Test rate limiting logic"""
        # Test first request (should pass)
        mock_redis.incr.return_value = 1
        assert check_rate_limit('127.0.0.1') is True
        
        # Test rate limit exceeded
        mock_redis.incr.return_value = 61  # Over the limit
        assert check_rate_limit('127.0.0.1') is False
    
    def test_stats_endpoint(self, client: FlaskClient, mock_redis):
        """Test stats endpoint"""
        # Mock Redis info
        mock_redis.info.return_value = {
            'used_memory': 1048576,  # 1MB
            'uptime_in_seconds': 3600
        }
        mock_redis.dbsize.return_value = 42
        
        response = client.get('/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['cache_keys'] == 42
        assert data['memory_used_mb'] == 1.0
        assert data['uptime_seconds'] == 3600
    
    def test_summarizer_error_handling(self, client: FlaskClient, mock_redis, mock_summarizer):
        """Test handling of summarizer errors"""
        mock_redis.get.return_value = None
        mock_summarizer.side_effect = Exception("Model error")
        
        response = client.post('/summarize',
                             json={'text': 'Some text'},
                             content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestCacheFunctions:
    """Test caching functions"""
    
    def test_get_summary_from_cache(self, mock_redis):
        """Test getting summary from cache"""
        mock_redis.get.return_value = 'Cached summary'
        
        with patch('sum_simple.r', mock_redis):
            result = get_summary_from_cache('test_hash')
            assert result == 'Cached summary'
            mock_redis.get.assert_called_with('summary:test_hash')
    
    def test_save_summary_to_cache(self, mock_redis):
        """Test saving summary to cache"""
        with patch('sum_simple.r', mock_redis):
            save_summary_to_cache('test_hash', 'Test summary')
            mock_redis.setex.assert_called_with('summary:test_hash', 3600, 'Test summary')


class TestPerformance:
    """Performance tests to ensure simple version is fast"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_response_time(self, client: FlaskClient, mock_redis, mock_summarizer):
        """Test that cached responses are fast"""
        # Mock instant cache hit
        mock_redis.get.return_value = 'Cached summary'
        
        start_time = time.time()
        response = client.post('/summarize',
                             json={'text': 'Test text'},
                             content_type='application/json')
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should be very fast (< 100ms even in tests)
        response_time = end_time - start_time
        assert response_time < 0.1, f"Response too slow: {response_time}s"
    
    def test_health_check_performance(self, client: FlaskClient):
        """Test that health check is fast"""
        start_time = time.time()
        response = client.get('/health')
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Health check should be instant
        response_time = end_time - start_time
        assert response_time < 0.05, f"Health check too slow: {response_time}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])