"""
Ollama Manager - Manages local AI models
"""
import requests
import logging
import socket
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, host="http://localhost:11434"):
        self.host = host
        self._is_running_cache = None
        self._last_check_time = 0
        self._cache_ttl = 2.0  # Cache result for 2 seconds

    def list_models(self):
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            if response.status_code == 200:
                return response.json().get('models', [])
            return []
        except Exception as e:
            logger.debug(f"Ollama connection failed: {e}")
            return []
            
    def is_running(self):
        """
        Check if Ollama service is running.
        Uses socket for efficiency and implements short-term caching to avoid blocking.
        """
        now = time.time()
        if self._is_running_cache is not None and (now - self._last_check_time) < self._cache_ttl:
            return self._is_running_cache

        try:
            host_to_parse = self.host
            if "://" not in host_to_parse:
                host_to_parse = f"http://{host_to_parse}"

            parsed = urlparse(host_to_parse)
            hostname = parsed.hostname or "localhost"
            port = parsed.port or 11434

            # Use socket for a much faster connection check
            # 0.1s timeout is sufficient for a local service
            with socket.create_connection((hostname, port), timeout=0.1):
                self._is_running_cache = True
        except (socket.timeout, ConnectionRefusedError, OSError):
            self._is_running_cache = False
        except Exception:
            self._is_running_cache = False

        self._last_check_time = now
        return self._is_running_cache

    def install_recommended_models(self, models=None):
        if models is None:
            models = ["llama2", "mistral"]
        # Logic to trigger pull would go here, usually via POST /api/pull
        pass
