"""
Test suite for SUM API endpoints

Tests all API routes including:
- Text processing endpoints
- Authentication
- Health checks
"""

import pytest
import json
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import create_simple_app
from api.auth import get_auth_manager, APIAuthManager
import api.auth

# Shared fixtures
@pytest.fixture
def auth_db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def mock_auth_manager(auth_db_path):
    """Create an auth manager using the temp DB."""
    # Create manager with temp DB
    manager = APIAuthManager(db_path=auth_db_path)
    
    # Patch the global instance in api.auth module
    # This ensures any call to get_auth_manager() returns our instance
    original_manager = api.auth._auth_manager
    api.auth._auth_manager = manager
    
    yield manager
    
    # Restore
    api.auth._auth_manager = original_manager

@pytest.fixture
def client(mock_auth_manager):
    """Create a test client with mocked auth."""
    app = create_simple_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

@pytest.fixture
def api_key(mock_auth_manager):
    """Create a test API key."""
    key_id, key = mock_auth_manager.generate_api_key(
        name="Test Key",
        permissions=['read', 'summarize'],
        rate_limit=100
    )
    return key

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
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
            json={
                'text': 'This is a test sentence.',
                'model': 'simple'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'summary' in data
    
    def test_process_text_missing_text(self, client):
        """Test missing text field."""
        response = client.post(
            '/api/process_text',
            json={'model': 'simple'},
            content_type='application/json'
        )
        
        assert response.status_code in [400, 500]
    
    def test_openapi_spec(self, client):
        """Test OpenAPI specification endpoint."""
        response = client.get('/api/openapi.yaml')
        if response.status_code != 404:
            assert response.status_code == 200
            assert response.content_type in ['application/x-yaml', 'text/yaml']

class TestAuthentication:
    """Test suite for authentication system."""
    
    def test_api_key_validation(self, client, mock_auth_manager):
        """Test API key validation endpoint."""
        key_id, api_key = mock_auth_manager.generate_api_key("Test", ["read"])
        
        response = client.get(
            '/api/auth/validate',
            headers={'X-API-Key': key_secret}
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is True

