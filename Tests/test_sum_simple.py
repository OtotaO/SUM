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
from unittest.mock import patch, MagicMock

# Import our simple app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock redis since it's not installed in CI environment
sys.modules['redis'] = MagicMock()

# Import from main instead of missing sum_simple
from main import create_simple_app

@pytest.fixture
def app():
    return create_simple_app()

class TestSumSimple:
    """Test the core functionality"""

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client: FlaskClient):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
