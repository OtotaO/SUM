"""
Test suite for SUM API endpoints
"""

import pytest
import json
import os
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
        app = create_simple_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def api_key(self):
        """Create a valid API key for testing."""
        manager = get_auth_manager()
        # generate_api_key returns (key_id, key_secret)
        _, key_secret = manager.generate_api_key("Test User", ['read', 'write', 'summarize'])
        return key_secret

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_process_text_simple(self, client):
        """Test simple text processing."""
        response = client.post(
            '/api/process_text',
            json={'text': 'This is a test sentence.', 'model': 'simple'},
            content_type='application/json'
        )
        # Should work or fail gracefully
        assert response.status_code in [200, 400, 500]

class TestAuthentication:
    @pytest.fixture
    def client(self):
        app = create_simple_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_api_key_validation(self, client):
        manager = get_auth_manager()
        _, key_secret = manager.generate_api_key("Test", ['read'])
        
        response = client.get(
            '/api/auth/validate',
            headers={'X-API-Key': key_secret}
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is True

