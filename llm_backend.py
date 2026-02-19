"""
LLM Backend - Universal interface for all language models
Supports OpenAI, Anthropic, Cohere, and local models
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    stream: bool = False


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def stream_generate(self, prompt: str, config: LLMConfig):
        """Stream text generation"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT models provider"""
    
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=config.api_key or os.getenv('OPENAI_API_KEY'))
            
            response = await client.chat.completions.create(
                model=config.model_name or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            return self._fallback_generate(prompt)
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return self._fallback_generate(prompt)
    
    async def stream_generate(self, prompt: str, config: LLMConfig):
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=config.api_key or os.getenv('OPENAI_API_KEY'))
            
            stream = await client.chat.completions.create(
                model=config.model_name or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except ImportError:
            yield self._fallback_generate(prompt)
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback to basic summarization"""
        try:
            from summarization_engine import SummarizationEngine
            engine = SummarizationEngine()
            # Use the correct method name
            return engine.process_text(prompt, {'method': 'advanced'}).get('summary', prompt[:200])
        except:
            # Ultimate fallback - simple extraction
            sentences = prompt.split('. ')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            return prompt[:200]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude models provider"""
    
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=config.api_key or os.getenv('ANTHROPIC_API_KEY')
            )
            
            response = await client.messages.create(
                model=config.model_name or "claude-3-haiku-20240307",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.warning("Anthropic library not installed. Install with: pip install anthropic")
            return self._fallback_generate(prompt)
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return self._fallback_generate(prompt)
    
    async def stream_generate(self, prompt: str, config: LLMConfig):
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=config.api_key or os.getenv('ANTHROPIC_API_KEY')
            )
            
            async with client.messages.stream(
                model=config.model_name or "claude-3-haiku-20240307",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except ImportError:
            yield self._fallback_generate(prompt)
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        from summarization_engine import SummarizationEngine
        engine = SummarizationEngine()
        return engine.process_text(prompt, {'method': 'advanced'}).get('summary', prompt[:200])


class OllamaProvider(BaseLLMProvider):
    """Ollama local models provider"""
    
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        try:
            import aiohttp
            
            endpoint = config.endpoint or "http://localhost:11434/api/generate"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={
                        "model": config.model_name or "llama2",
                        "prompt": prompt,
                        "temperature": config.temperature,
                        "stream": False
                    }
                ) as response:
                    result = await response.json()
                    return result.get("response", "")
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return self._fallback_generate(prompt)
    
    async def stream_generate(self, prompt: str, config: LLMConfig):
        try:
            import aiohttp
            
            endpoint = config.endpoint or "http://localhost:11434/api/generate"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={
                        "model": config.model_name or "llama2",
                        "prompt": prompt,
                        "temperature": config.temperature,
                        "stream": True
                    }
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        from summarization_engine import SummarizationEngine
        engine = SummarizationEngine()
        return engine.process_text(prompt, {'method': 'advanced'}).get('summary', prompt[:200])


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace models provider"""
    
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        try:
            from transformers import pipeline
            
            # Use a summarization model by default
            model_name = config.model_name or "facebook/bart-large-cnn"
            
            summarizer = pipeline("summarization", model=model_name)
            
            # Truncate if needed
            max_length = min(1024, len(prompt.split()))
            
            result = summarizer(
                prompt,
                max_length=int(max_length * 0.3),
                min_length=int(max_length * 0.1),
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except ImportError:
            logger.warning("Transformers library not installed. Install with: pip install transformers")
            return self._fallback_generate(prompt)
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            return self._fallback_generate(prompt)
    
    async def stream_generate(self, prompt: str, config: LLMConfig):
        # HuggingFace doesn't support streaming in the same way
        result = await self.generate(prompt, config)
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i:i+chunk_size]
            await asyncio.sleep(0.01)
    
    def _fallback_generate(self, prompt: str) -> str:
        from summarization_engine import SummarizationEngine
        engine = SummarizationEngine()
        return engine.process_text(prompt, {'method': 'advanced'}).get('summary', prompt[:200])


class LocalProvider(BaseLLMProvider):
    """Local NLTK-based provider (no external API needed)"""
    
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        """Use the existing summarization engine"""

        # Check if this is an expansion/generation request rather than summarization
        prompt_lower = prompt.lower()
        generation_keywords = ["expand", "write", "create", "chapter", "extrapolate"]
        # Simple heuristic: if prompt is asking to create content and is relatively short (instructional)
        if any(keyword in prompt_lower for keyword in generation_keywords) and len(prompt) < 2000:
             return "Text generation requires an active LLM (OpenAI, Anthropic, or Ollama). Please configure one."

        try:
            from summarization_engine import SummarizationEngine
            
            engine = SummarizationEngine()
            
            # Basic extractive summarization
            return engine.process_text(prompt, {'maxTokens': config.max_tokens}).get('summary', prompt[:config.max_tokens])
        except:
            # Fallback to simple extraction
            sentences = prompt.split('. ')
            max_sentences = max(1, int(config.max_tokens / 50))
            if len(sentences) > max_sentences:
                return '. '.join(sentences[:max_sentences]) + '.'
            return prompt[:config.max_tokens]
    
    async def stream_generate(self, prompt: str, config: LLMConfig):
        """Stream the local generation"""
        result = await self.generate(prompt, config)
        
        # Simulate streaming
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i:i+chunk_size]
            await asyncio.sleep(0.01)


class UniversalLLMBackend:
    """
    Universal backend that automatically selects and uses the best available LLM
    """
    
    def __init__(self):
        self.providers = {
            ModelProvider.OPENAI: OpenAIProvider(),
            ModelProvider.ANTHROPIC: AnthropicProvider(),
            ModelProvider.OLLAMA: OllamaProvider(),
            ModelProvider.HUGGINGFACE: HuggingFaceProvider(),
            ModelProvider.LOCAL: LocalProvider()
        }
        
        self.default_provider = self._detect_available_provider()
    
    def _detect_available_provider(self) -> ModelProvider:
        """Detect which provider is available"""
        
        # Check for API keys in environment
        if os.getenv('OPENAI_API_KEY'):
            return ModelProvider.OPENAI
        elif os.getenv('ANTHROPIC_API_KEY'):
            return ModelProvider.ANTHROPIC
        
        # Check for local Ollama
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            if response.status_code == 200:
                return ModelProvider.OLLAMA
        except:
            pass
        
        # Check for HuggingFace transformers
        try:
            import transformers
            return ModelProvider.HUGGINGFACE
        except ImportError:
            pass
        
        # Fallback to local NLTK
        return ModelProvider.LOCAL
    
    async def generate(self, 
                      prompt: str,
                      provider: Optional[ModelProvider] = None,
                      model_name: Optional[str] = None,
                      **kwargs) -> str:
        """
        Generate text using the specified or default provider
        """
        
        # Use specified provider or default
        provider = provider or self.default_provider
        
        # Create config
        config = LLMConfig(
            provider=provider,
            model_name=model_name,
            **kwargs
        )
        
        # Get provider implementation
        provider_impl = self.providers.get(provider)
        if not provider_impl:
            logger.error(f"Provider {provider} not found, using local")
            provider_impl = self.providers[ModelProvider.LOCAL]
        
        # Generate
        try:
            return await provider_impl.generate(prompt, config)
        except Exception as e:
            logger.error(f"Generation failed: {e}, falling back to local")
            return await self.providers[ModelProvider.LOCAL].generate(prompt, config)
    
    async def stream_generate(self,
                            prompt: str,
                            provider: Optional[ModelProvider] = None,
                            model_name: Optional[str] = None,
                            **kwargs):
        """
        Stream text generation
        """
        
        provider = provider or self.default_provider
        
        config = LLMConfig(
            provider=provider,
            model_name=model_name,
            stream=True,
            **kwargs
        )
        
        provider_impl = self.providers.get(provider)
        if not provider_impl:
            provider_impl = self.providers[ModelProvider.LOCAL]
        
        try:
            async for chunk in provider_impl.stream_generate(prompt, config):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed: {e}, falling back to local")
            async for chunk in self.providers[ModelProvider.LOCAL].stream_generate(prompt, config):
                yield chunk
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        
        if os.getenv('OPENAI_API_KEY'):
            available.append('openai')
        if os.getenv('ANTHROPIC_API_KEY'):
            available.append('anthropic')
        
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            if response.status_code == 200:
                available.append('ollama')
        except:
            pass
        
        try:
            import transformers
            available.append('huggingface')
        except ImportError:
            pass
        
        available.append('local')  # Always available
        
        return available
    
    def get_available_models(self, provider: ModelProvider) -> List[str]:
        """Get available models for a provider"""
        
        models = {
            ModelProvider.OPENAI: [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ],
            ModelProvider.ANTHROPIC: [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-instant-1.2"
            ],
            ModelProvider.OLLAMA: [
                "llama2",
                "mistral",
                "codellama",
                "phi",
                "neural-chat",
                "starling-lm",
                "orca-mini"
            ],
            ModelProvider.HUGGINGFACE: [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "t5-base",
                "t5-small"
            ],
            ModelProvider.LOCAL: [
                "nltk-simple",
                "nltk-advanced",
                "nltk-hierarchical"
            ]
        }
        
        return models.get(provider, [])


# Global instance for easy access
llm_backend = UniversalLLMBackend()