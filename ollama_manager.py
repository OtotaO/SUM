"""
Ollama Manager - Manages local AI models
"""
import requests
import logging

logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, host="http://localhost:11434"):
        self.host = host

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
        try:
            requests.get(f"{self.host}", timeout=1)
            return True
        except:
            return False

    def install_recommended_models(self, models=None):
        if models is None:
            models = ["llama2", "mistral"]
        # Logic to trigger pull would go here, usually via POST /api/pull
        pass
