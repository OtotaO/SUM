"""
routes.py - Web UI Routes

Clean web interface routes following Carmack's principles:
- Simple static file serving
- Fast template rendering
- Minimal overhead
- Clear separation from API logic

Author: ototao
License: Apache License 2.0
"""

from flask import Blueprint, render_template, send_from_directory


web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def index():
    """Render the main application page."""
    return send_from_directory('static', 'index.html')


@web_bp.route('/dashboard')
def dashboard():
    """Render the analytics dashboard."""
    return render_template('dashboard.html')


@web_bp.route('/collaborative')
def collaborative_intelligence():
    """Render the collaborative intelligence interface."""
    return send_from_directory('static', 'collaborative_intelligence.html')


@web_bp.route('/api/docs')
def api_docs():
    """Render the API documentation."""
    return render_template('api_docs.html')


@web_bp.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)