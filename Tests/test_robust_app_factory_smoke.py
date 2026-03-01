"""Smoke tests for robust app factory startup behavior."""

import importlib

from flask import Flask
from config import TestingConfig


def test_robust_app_factory_imports():
    """Importing the robust factory module should not fail on optional deps."""
    module = importlib.import_module('web.robust_app_factory')
    assert module is not None
    assert hasattr(module, 'create_robust_app')


def test_create_robust_app_smoke():
    """App creation should succeed even when optional modules are absent."""
    module = importlib.import_module('web.robust_app_factory')
    app = module.create_robust_app(TestingConfig)

    assert isinstance(app, Flask)
    assert app.config['TESTING'] is True
    assert 'health' in app.blueprints
