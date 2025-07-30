#!/usr/bin/env python3
"""
ai_models.py - AI Model Integration Layer

Provides seamless integration with OpenAI and Anthropic models for enhanced
summarization capabilities. Supports multiple models with fallback to traditional
NLP methods when API keys are not available.

Features:
- OpenAI GPT-4, GPT-3.5 integration
- Anthropic Claude integration
- Secure API key management
- Model comparison and selection
- Hybrid processing with traditional NLP
- Token counting and cost estimation
- Retry logic and error handling

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from functools import lru_cache
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Import SUM components
from SUM import HierarchicalDensificationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing AI libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not installed. Run: pip install anthropic")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("Tiktoken not installed. Token counting will be approximate.")


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    name: str
    provider: str
    max_tokens: int
    temperature: float = 0.3
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_streaming: bool = False
    context_window: int = 4096


# Model configurations
AVAILABLE_MODELS = {
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo-preview",
        provider="openai",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supports_streaming=True,
        context_window=128000
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo-16k",
        provider="openai",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        supports_streaming=True,
        context_window=16384
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus-20240229",
        provider="anthropic",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_streaming=True,
        context_window=200000
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet-20240229",
        provider="anthropic",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_streaming=True,
        context_window=200000
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku-20240307",
        provider="anthropic",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        supports_streaming=True,
        context_window=200000
    )
}


class SecureKeyManager:
    """Secure API key management with encryption."""
    
    def __init__(self, key_file: str = "api_keys.enc"):
        self.key_file = key_file
        self._master_key = self._get_or_create_master_key()
        self._cipher = Fernet(self._master_key)
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        master_key_file = ".master_key"
        
        if os.path.exists(master_key_file):
            with open(master_key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            password = os.urandom(32)
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            with open(master_key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(master_key_file, 0o600)
            
            return key
    
    def save_api_key(self, provider: str, api_key: str):
        """Save encrypted API key."""
        keys = self.load_api_keys()
        keys[provider] = self._cipher.encrypt(api_key.encode()).decode()
        
        with open(self.key_file, 'w') as f:
            json.dump(keys, f)
        
        # Set restrictive permissions
        os.chmod(self.key_file, 0o600)
    
    def load_api_keys(self) -> Dict[str, str]:
        """Load and decrypt API keys."""
        if not os.path.exists(self.key_file):
            return {}
        
        with open(self.key_file, 'r') as f:
            encrypted_keys = json.load(f)
        
        decrypted_keys = {}
        for provider, encrypted_key in encrypted_keys.items():
            try:
                decrypted_keys[provider] = self._cipher.decrypt(
                    encrypted_key.encode()
                ).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt key for {provider}: {e}")
        
        return decrypted_keys
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get decrypted API key for provider."""
        keys = self.load_api_keys()
        return keys.get(provider)
    
    def remove_api_key(self, provider: str):
        """Remove API key for provider."""
        keys = self.load_api_keys()
        if provider in keys:
            del keys[provider]
            with open(self.key_file, 'w') as f:
                json.dump(keys, f)


class AIModelBase(ABC):
    """Base class for AI model integration."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        self.api_key = api_key
        self.config = model_config
        
    @abstractmethod
    async def summarize(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary using AI model."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for processing."""
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost


class OpenAIModel(AIModelBase):
    """OpenAI model integration."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        super().__init__(api_key, model_config)
        if OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ImportError("OpenAI library not available")
    
    async def summarize(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary using OpenAI model."""
        try:
            # Prepare system prompt
            system_prompt = """You are an expert at hierarchical knowledge densification. 
Your task is to create a comprehensive summary following this structure:

1. **Key Concepts** (3-7 main concepts, single words or short phrases)
2. **Core Summary** (50-100 words capturing the essence)
3. **Expanded Context** (100-200 words with important details)
4. **Key Insights** (2-3 profound insights with types: TRUTH, WISDOM, PURPOSE, or INNOVATION)

Format your response as JSON with keys: concepts, core_summary, expanded_context, insights"""

            # Create completion
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please summarize this text:\n\n{text}"}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Calculate tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return {
                "hierarchical_summary": {
                    "level_1_concepts": result.get("concepts", []),
                    "level_2_core": result.get("core_summary", ""),
                    "level_3_expanded": result.get("expanded_context", "")
                },
                "key_insights": [
                    {
                        "type": insight.get("type", "WISDOM"),
                        "text": insight.get("text", ""),
                        "score": insight.get("score", 0.8)
                    }
                    for insight in result.get("insights", [])
                ],
                "ai_metadata": {
                    "model": self.config.name,
                    "provider": "openai",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "estimated_cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(self.config.name)
                return len(encoding.encode(text))
            except:
                # Fallback to approximate
                return len(text) // 4
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4


class AnthropicModel(AIModelBase):
    """Anthropic Claude model integration."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        super().__init__(api_key, model_config)
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ImportError("Anthropic library not available")
    
    async def summarize(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary using Claude model."""
        try:
            # Prepare prompt
            prompt = f"""You are an expert at hierarchical knowledge densification. 
Create a comprehensive summary of the following text with this exact JSON structure:

{{
    "concepts": ["concept1", "concept2", ...],  // 3-7 key concepts
    "core_summary": "50-100 word summary capturing the essence",
    "expanded_context": "100-200 word expanded summary with important details",
    "insights": [
        {{"type": "TRUTH", "text": "insight text", "score": 0.9}},
        {{"type": "WISDOM", "text": "insight text", "score": 0.85}}
    ]
}}

Text to summarize:
{text}

Respond only with valid JSON."""

            # Create completion
            response = self.client.messages.create(
                model=self.config.name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            result = json.loads(response.content[0].text)
            
            # Calculate tokens (approximate for Claude)
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response.content[0].text)
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            return {
                "hierarchical_summary": {
                    "level_1_concepts": result.get("concepts", []),
                    "level_2_core": result.get("core_summary", ""),
                    "level_3_expanded": result.get("expanded_context", "")
                },
                "key_insights": [
                    {
                        "type": insight.get("type", "WISDOM"),
                        "text": insight.get("text", ""),
                        "score": insight.get("score", 0.8)
                    }
                    for insight in result.get("insights", [])
                ],
                "ai_metadata": {
                    "model": self.config.name,
                    "provider": "anthropic",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "estimated_cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for Claude)."""
        # Claude uses a similar tokenization to GPT
        # Approximate: 1 token ≈ 4 characters
        return len(text) // 4


class HybridAIEngine:
    """Hybrid engine combining traditional NLP with AI models."""
    
    def __init__(self, key_manager: Optional[SecureKeyManager] = None):
        self.key_manager = key_manager or SecureKeyManager()
        self.traditional_engine = HierarchicalDensificationEngine()
        self._model_cache = {}
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models based on API keys."""
        available = []
        api_keys = self.key_manager.load_api_keys()
        
        for model_id, config in AVAILABLE_MODELS.items():
            is_available = (
                (config.provider == "openai" and "openai" in api_keys and OPENAI_AVAILABLE) or
                (config.provider == "anthropic" and "anthropic" in api_keys and ANTHROPIC_AVAILABLE)
            )
            
            available.append({
                "id": model_id,
                "name": config.name,
                "provider": config.provider,
                "available": is_available,
                "cost_per_1k_input": config.cost_per_1k_input,
                "cost_per_1k_output": config.cost_per_1k_output,
                "context_window": config.context_window,
                "supports_streaming": config.supports_streaming
            })
        
        # Always include traditional engine
        available.append({
            "id": "traditional",
            "name": "Hierarchical Densification Engine",
            "provider": "sum",
            "available": True,
            "cost_per_1k_input": 0,
            "cost_per_1k_output": 0,
            "context_window": float('inf'),
            "supports_streaming": True
        })
        
        return available
    
    def _get_model(self, model_id: str) -> Optional[AIModelBase]:
        """Get or create model instance."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        if model_id not in AVAILABLE_MODELS:
            return None
        
        config = AVAILABLE_MODELS[model_id]
        api_key = self.key_manager.get_api_key(config.provider)
        
        if not api_key:
            return None
        
        try:
            if config.provider == "openai":
                model = OpenAIModel(api_key, config)
            elif config.provider == "anthropic":
                model = AnthropicModel(api_key, config)
            else:
                return None
            
            self._model_cache[model_id] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_id}: {e}")
            return None
    
    async def process_text(
        self, 
        text: str, 
        model_id: str = "traditional",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process text using specified model or traditional engine."""
        config = config or {}
        
        # Use traditional engine if specified or AI not available
        if model_id == "traditional":
            return self.traditional_engine.process_text(text, config)
        
        # Try AI model
        model = self._get_model(model_id)
        if model:
            try:
                result = await model.summarize(text, config)
                result["processing_engine"] = model_id
                return result
            except Exception as e:
                logger.error(f"AI model failed, falling back to traditional: {e}")
        
        # Fallback to traditional engine
        result = self.traditional_engine.process_text(text, config)
        result["processing_engine"] = "traditional"
        result["fallback_reason"] = "AI model unavailable or failed"
        return result
    
    async def compare_models(
        self,
        text: str,
        model_ids: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compare outputs from multiple models."""
        results = {}
        
        for model_id in model_ids:
            try:
                result = await self.process_text(text, model_id, config)
                results[model_id] = result
            except Exception as e:
                logger.error(f"Error processing with {model_id}: {e}")
                results[model_id] = {"error": str(e)}
        
        return {
            "comparison_results": results,
            "models_compared": model_ids,
            "timestamp": time.time()
        }
    
    def estimate_processing_cost(self, text: str, model_id: str) -> Dict[str, Any]:
        """Estimate cost for processing text with specified model."""
        if model_id == "traditional":
            return {
                "model_id": model_id,
                "estimated_cost": 0,
                "input_tokens": 0,
                "estimated_output_tokens": 0,
                "free": True
            }
        
        model = self._get_model(model_id)
        if not model:
            return {"error": "Model not available"}
        
        input_tokens = model.count_tokens(text)
        # Estimate output tokens as ~20% of input
        estimated_output_tokens = int(input_tokens * 0.2)
        cost = model.estimate_cost(input_tokens, estimated_output_tokens)
        
        return {
            "model_id": model_id,
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost": cost,
            "cost_breakdown": {
                "input_cost": (input_tokens / 1000) * model.config.cost_per_1k_input,
                "output_cost": (estimated_output_tokens / 1000) * model.config.cost_per_1k_output
            }
        }


# Convenience functions for non-async usage
def process_with_ai(
    text: str,
    model_id: str = "traditional",
    config: Optional[Dict[str, Any]] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for AI processing."""
    key_manager = SecureKeyManager()
    
    # Save any provided API keys
    if api_keys:
        for provider, key in api_keys.items():
            key_manager.save_api_key(provider, key)
    
    engine = HybridAIEngine(key_manager)
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            engine.process_text(text, model_id, config)
        )
    finally:
        loop.close()


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence has transformed how we approach problem-solving across industries.
    Machine learning models can now understand natural language, recognize images, and make
    complex predictions. The key to success lies in choosing the right model for your use case
    and understanding the trade-offs between accuracy, speed, and cost.
    """
    
    # Initialize hybrid engine
    engine = HybridAIEngine()
    
    # Show available models
    print("Available Models:")
    for model in engine.get_available_models():
        status = "✅" if model["available"] else "❌"
        print(f"{status} {model['id']}: {model['name']} ({model['provider']})")
    
    # Process with traditional engine
    print("\nProcessing with traditional engine...")
    result = asyncio.run(engine.process_text(sample_text, "traditional"))
    print(f"Concepts: {result['hierarchical_summary']['level_1_concepts']}")