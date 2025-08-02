#!/usr/bin/env python3
"""
config_system.py - Advanced Configuration Management for SUM

Provides centralized configuration management with environment-aware settings,
validation, and hot-reloading capabilities.

Author: SUM Development Team  
License: Apache License 2.0
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class AIModelConfig:
    """Configuration for AI models."""
    default_model: str = "gpt-3.5-turbo"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class ServerConfig:
    """Configuration for server settings."""
    host: str = "0.0.0.0"
    port: int = 3000
    websocket_port: int = 8765
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds


@dataclass
class ProcessingConfig:
    """Configuration for content processing."""
    max_text_length: int = 100000
    chunk_size: int = 1000
    batch_size: int = 10
    parallel_workers: int = 4
    summary_lengths: Dict[str, int] = field(default_factory=lambda: {
        "brief": 1,
        "standard": 3,
        "detailed": 5
    })
    enable_progressive: bool = True
    enable_streaming: bool = True


@dataclass
class NotesConfig:
    """Configuration for notes system."""
    auto_tag_enabled: bool = True
    distillation_enabled: bool = True
    crystallization_enabled: bool = True
    default_policy: str = "general"
    storage_backend: str = "sqlite"  # sqlite, postgres, mongodb
    storage_path: str = "./notes_data"
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    auth_enabled: bool = False
    auth_provider: str = "local"  # local, oauth, jwt
    session_timeout: int = 3600
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256"
    api_key_required: bool = False
    allowed_origins: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "./logs/sum.log"
    file_rotation: str = "daily"
    file_retention: int = 30  # days
    console_enabled: bool = True
    structured_logging: bool = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    cache_enabled: bool = True
    cache_backend: str = "memory"  # memory, redis, memcached
    cache_size: int = 1000
    lazy_loading: bool = True
    preload_models: bool = False
    optimize_memory: bool = True
    profiling_enabled: bool = False


class ConfigManager:
    """
    Centralized configuration management for SUM.
    
    Features:
    - Environment-aware configuration
    - Configuration validation
    - Hot-reloading support
    - Override hierarchy: env vars > config file > defaults
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, staging, prod)
        """
        self.config_path = config_path or self._find_config_file()
        self.environment = self._determine_environment(environment)
        
        # Initialize configurations
        self.ai = AIModelConfig()
        self.server = ServerConfig()
        self.processing = ProcessingConfig()
        self.notes = NotesConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        
        # Load configuration
        self._load_configuration()
        self._apply_environment_overrides()
        self._apply_env_var_overrides()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {self.environment.value}")
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        search_paths = [
            "config.yaml",
            "config.json",
            "sum_config.yaml",
            "sum_config.json",
            ".sum/config.yaml",
            ".sum/config.json",
            os.path.expanduser("~/.sum/config.yaml"),
            os.path.expanduser("~/.sum/config.json"),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _determine_environment(self, environment: Optional[str]) -> Environment:
        """Determine the current environment."""
        if environment:
            return Environment(environment.lower())
        
        # Check environment variable
        env_var = os.getenv("SUM_ENV", "").lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                logger.warning(f"Invalid environment: {env_var}, using development")
        
        # Default to development
        return Environment.DEVELOPMENT
    
    def _load_configuration(self):
        """Load configuration from file."""
        if not self.config_path or not os.path.exists(self.config_path):
            logger.info("No configuration file found, using defaults")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Apply configuration sections
            if 'ai' in config_data:
                self._update_dataclass(self.ai, config_data['ai'])
            if 'server' in config_data:
                self._update_dataclass(self.server, config_data['server'])
            if 'processing' in config_data:
                self._update_dataclass(self.processing, config_data['processing'])
            if 'notes' in config_data:
                self._update_dataclass(self.notes, config_data['notes'])
            if 'security' in config_data:
                self._update_dataclass(self.security, config_data['security'])
            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])
            if 'performance' in config_data:
                self._update_dataclass(self.performance, config_data['performance'])
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific overrides."""
        if self.environment == Environment.PRODUCTION:
            self.server.debug = False
            self.security.auth_enabled = True
            self.logging.level = "WARNING"
            self.performance.profiling_enabled = False
        elif self.environment == Environment.DEVELOPMENT:
            self.server.debug = True
            self.logging.level = "DEBUG"
            self.performance.profiling_enabled = True
        elif self.environment == Environment.TEST:
            self.server.port = 5555
            self.notes.storage_backend = "memory"
            self.logging.file_enabled = False
    
    def _apply_env_var_overrides(self):
        """Apply environment variable overrides."""
        # Server overrides
        if os.getenv("SUM_HOST"):
            self.server.host = os.getenv("SUM_HOST")
        if os.getenv("SUM_PORT"):
            self.server.port = int(os.getenv("SUM_PORT"))
        
        # AI model overrides
        if os.getenv("SUM_AI_MODEL"):
            self.ai.default_model = os.getenv("SUM_AI_MODEL")
        if os.getenv("OPENAI_API_KEY"):
            # Note: Don't store the key in config, just note it exists
            logger.info("OpenAI API key detected in environment")
        
        # Logging overrides
        if os.getenv("SUM_LOG_LEVEL"):
            self.logging.level = os.getenv("SUM_LOG_LEVEL")
        
        # Feature flags
        if os.getenv("SUM_DISABLE_STREAMING"):
            self.processing.enable_streaming = False
        if os.getenv("SUM_DISABLE_CACHE"):
            self.performance.cache_enabled = False
    
    def _validate_configuration(self):
        """Validate configuration values."""
        # Validate ports
        if not 1 <= self.server.port <= 65535:
            raise ValueError(f"Invalid port: {self.server.port}")
        if not 1 <= self.server.websocket_port <= 65535:
            raise ValueError(f"Invalid websocket port: {self.server.websocket_port}")
        
        # Validate paths
        if self.logging.file_enabled:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        
        # Validate processing limits
        if self.processing.max_text_length < 100:
            raise ValueError("max_text_length must be at least 100")
        if self.processing.chunk_size > self.processing.max_text_length:
            raise ValueError("chunk_size cannot exceed max_text_length")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            'environment': self.environment.value,
            'ai': asdict(self.ai),
            'server': asdict(self.server),
            'processing': asdict(self.processing),
            'notes': asdict(self.notes),
            'security': asdict(self.security),
            'logging': asdict(self.logging),
            'performance': asdict(self.performance)
        }
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            save_path = "config.yaml"
        
        config_dict = self.get_config_dict()
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {save_path}")
    
    def reload(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_configuration()
        self._apply_environment_overrides()
        self._apply_env_var_overrides()
        self._validate_configuration()
    
    @classmethod
    def get_default_config(cls) -> 'ConfigManager':
        """Get configuration with all defaults."""
        return cls(config_path=None, environment="development")


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def initialize_config(config_path: Optional[str] = None, 
                     environment: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration."""
    global _config_instance
    _config_instance = ConfigManager(config_path, environment)
    return _config_instance


# Example configuration file generator
def generate_example_config(path: str = "config.example.yaml"):
    """Generate example configuration file."""
    example_config = {
        "ai": {
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 2000,
            "temperature": 0.7
        },
        "server": {
            "host": "0.0.0.0",
            "port": 3000,
            "debug": False
        },
        "processing": {
            "max_text_length": 100000,
            "chunk_size": 1000,
            "enable_streaming": True
        },
        "notes": {
            "auto_tag_enabled": True,
            "distillation_enabled": True,
            "default_policy": "general"
        },
        "security": {
            "auth_enabled": False,
            "encryption_enabled": True
        },
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "file_path": "./logs/sum.log"
        },
        "performance": {
            "cache_enabled": True,
            "optimize_memory": True
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print(f"Example configuration file generated: {path}")


if __name__ == "__main__":
    # Test configuration system
    print("Testing SUM Configuration System")
    print("=" * 50)
    
    # Generate example config
    generate_example_config()
    
    # Test loading configuration
    config = ConfigManager()
    
    # Display configuration
    print(f"\nEnvironment: {config.environment.value}")
    print(f"Server: {config.server.host}:{config.server.port}")
    print(f"AI Model: {config.ai.default_model}")
    print(f"Logging Level: {config.logging.level}")
    
    # Test environment variable override
    os.environ["SUM_PORT"] = "4000"
    config.reload()
    print(f"\nAfter env override - Port: {config.server.port}")
    
    # Save configuration
    config.save_config("test_config.yaml")
    print("\nConfiguration saved successfully!")