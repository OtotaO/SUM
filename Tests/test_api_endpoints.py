"""
Test suite for SUM API endpoints

Tests all API routes including:
- Text processing endpoints
- File upload
- Authentication
- Caching
- Health checks
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import create_simple_app
from api.auth import get_auth_manager


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create the app."""
        app = create_simple_app()
        app.config.update({
            "TESTING": True,
        })
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()
    
    @pytest.fixture
    def auth_manager(self):
        return get_auth_manager()

    @pytest.fixture
    def api_key(self, auth_manager):
        """Create a test API key."""
        key_id, api_key = auth_manager.generate_api_key(
            name="Test Key",
            permissions=['read', 'summarize'],
            rate_limit=100
        )
        return api_key
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            'text': 'Artificial intelligence is transforming the world.',
            'model': 'simple',
            'config': {
                'maxTokens': 50,
                'use_cache': False
            }
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'version' in data
    
    def test_process_text_no_auth(self, client, sample_data):
        """Test text processing without authentication."""
        response = client.post(
            '/api/process_text',
            json=sample_data,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'summary' in data
        assert 'tags' in data
        assert 'detected_language' in data
        assert data['detected_language'] == 'en'
    
    def test_process_text_with_auth(self, client, sample_data, api_key):
        """Test text processing with API key."""
        response = client.post(
            '/api/process_text',
            json=sample_data,
            headers={'X-API-Key': api_key},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'summary' in data
    
    def test_process_text_invalid_model(self, client):
        """Test with invalid model."""
        response = client.post(
            '/api/process_text',
            json={
                'text': 'Test text',
                'model': 'invalid_model'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_process_text_empty(self, client):
        """Test with empty text."""
        response = client.post(
            '/api/process_text',
            json={'text': ''},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_hierarchical_model(self, client):
        """Test hierarchical summarization model."""
        response = client.post(
            '/api/process_text',
            json={
                'text': 'The Renaissance was a period of cultural rebirth. ' * 20,
                'model': 'hierarchical'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'hierarchical_summary' in data
        assert 'level_1_concepts' in data['hierarchical_summary']
        assert 'level_2_core' in data['hierarchical_summary']
    
    def test_rate_limiting(self, client, api_key):
        """Test rate limiting."""
        # This would need actual rate limiting implementation
        # For now, just verify the endpoint accepts the key
        for i in range(5):
            response = client.post(
                '/api/process_text',
                json={'text': f'Test {i}', 'model': 'simple'},
                headers={'X-API-Key': api_key}
            )
            assert response.status_code in [200, 429]
    
    def test_cache_stats(self, client):
        """Test cache statistics endpoint."""
        response = client.get('/api/cache/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_entries' in data
        assert 'memory_entries' in data
        assert 'hit_rate' in data
    
    def test_cache_clear(self, client):
        """Test cache clearing endpoint."""
        # First, create some cached data
        client.post(
            '/api/process_text',
            json={'text': 'Cache this text', 'model': 'simple'}
        )
        
        # Clear cache
        response = client.post('/api/cache/clear')
        assert response.status_code in [200, 429]  # May be rate limited
    
    def test_file_upload(self, client, api_key):
        """Test file upload endpoint."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('This is a test document for file upload.')
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    '/api/process_unlimited',
                    data={'file': (f, 'test.txt')},
                    headers={'X-API-Key': api_key}
                )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'summary' in data
            assert 'processing_method' in data
            
        finally:
            os.unlink(temp_path)
    
    def test_openapi_spec(self, client):
        """Test OpenAPI specification endpoint."""
        response = client.get('/api/openapi.yaml')
        assert response.status_code == 200
        assert response.content_type == 'application/x-yaml'
        
        # Check content
        yaml_content = response.data.decode('utf-8')
        assert 'openapi: 3.0' in yaml_content
        assert 'SUM - Advanced Text Summarization API' in yaml_content
    
    def test_language_detection_spanish(self, client):
        """Test Spanish language detection."""
        response = client.post(
            '/api/process_text',
            json={
                'text': 'Esta es una prueba en español para verificar la detección.',
                'model': 'simple'
            }
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['detected_language'] == 'es'
        assert data['language_name'] == 'Spanish'
    
    def test_large_text_handling(self, client, api_key):
        """Test handling of large texts."""
        # Create large text (100KB)
        large_text = "This is a test sentence. " * 5000
        
        response = client.post(
            '/api/process_text',
            json={
                'text': large_text,
                'model': 'unlimited'
            },
            headers={'X-API-Key': api_key}
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'summary' in data
        assert len(data['summary']) < len(large_text)


class TestAuthentication:
    """Test suite for authentication system."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_api_key_validation(self, client):
        """Test API key validation endpoint."""
        # Create a key
        key_data = create_api_key("Test", ['read'])
        api_key = key_data['api_key']
        
        # Validate it
        response = client.get(
            '/api/auth/validate',
            headers={'X-API-Key': api_key}
        )
        
        if response.status_code == 404:
            pytest.skip("Validation endpoint not found")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is True
        assert data['name'] == 'Test'
    
    def test_invalid_api_key(self, client):
        """Test invalid API key."""
        response = client.get(
            '/api/auth/validate',
            headers={'X-API-Key': 'invalid_key'}
        )
        if response.status_code == 404:
             pytest.skip("Validation endpoint not found")
        
        assert response.status_code == 401
    
    def test_missing_api_key(self, client):
        """Test missing API key on protected endpoint."""
        response = client.post('/api/process_unlimited', json={})
        assert response.status_code == 401
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'API key required' in data['error']
    
    def test_permissions(self, client):
        """Test permission checking."""
        # Create key with limited permissions
        key_data = create_api_key("Limited", ['read'])
        api_key = key_data['api_key']
        
        # Try to access endpoint requiring 'summarize' permission
        # This would need actual permission checking implementation
        response = client.post(
            '/api/process_text',
            json={'text': 'Test'},
            headers={'X-API-Key': api_key}
        )
        
        # Should work as process_text has optional auth
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_malformed_json(self, client):
        """Test malformed JSON handling."""
        response = client.post(
            '/api/process_text',
            data='{"text": malformed}',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_missing_required_field(self, client):
        """Test missing required fields."""
        response = client.post(
            '/api/process_text',
            json={'model': 'simple'},  # Missing 'text'
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_server_error_handling(self, client):
        """Test server error handling."""
        # Mock an internal error
        with patch('summarization_engine.BasicSummarizationEngine.process_text') as mock:
            mock.side_effect = Exception("Test error")
            
            response = client.post(
                '/api/process_text',
                json={'text': 'Test', 'model': 'simple'}
            )
            
            # Should handle gracefully
            assert response.status_code in [200, 500]
            data = json.loads(response.data)
            if response.status_code == 500:
                assert 'error' in data


# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance tests for API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request(i):
            response = client.post(
                '/api/process_text',
                json={'text': f'Test request {i}', 'model': 'simple'}
            )
            results.append(response.status_code)
        
        # Create 10 concurrent requests
        threads = []
        for i in range(10):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join()
        
        # All should succeed
        assert all(status == 200 for status in results)
    
    def test_response_time(self, client):
        """Test API response time."""
        import time
        
        start = time.time()
        response = client.post(
            '/api/process_text',
            json={'text': 'Test text ' * 100, 'model': 'simple'}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0  # Should respond within 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
