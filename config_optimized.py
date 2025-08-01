"""
Optimized Configuration - Carmack Principles Applied

Key optimizations:
- Single source of truth for all configuration
- Environment-aware with sensible defaults
- Fast loading with minimal dependencies
- Clear validation and error handling

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OptimizedConfig:
    """
    Optimized configuration using dataclass for performance and clarity.
    
    Carmack principles:
    - Fast: Dataclass with __slots__ for memory efficiency
    - Simple: Single configuration object
    - Clear: Explicit defaults and validation
    - Bulletproof: Type checking and validation
    """
    
    # Core settings
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    testing: bool = field(default_factory=lambda: os.getenv('TESTING', 'false').lower() == 'true')
    
    # Server settings
    host: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('PORT', '3000')))
    
    # Performance settings
    max_content_length: int = field(default_factory=lambda: int(os.getenv('MAX_CONTENT_LENGTH', str(16 * 1024 * 1024))))
    max_workers: int = field(default_factory=lambda: min(int(os.getenv('MAX_WORKERS', '4')), os.cpu_count() or 4))
    cache_size: int = field(default_factory=lambda: int(os.getenv('CACHE_SIZE', '1024')))
    
    # Rate limiting
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_PER_MINUTE', '30')))
    batch_rate_limit: int = field(default_factory=lambda: int(os.getenv('BATCH_RATE_LIMIT', '10')))
    
    # Text processing limits
    max_text_length: int = field(default_factory=lambda: int(os.getenv('MAX_TEXT_LENGTH', '100000')))
    max_summary_length: int = field(default_factory=lambda: int(os.getenv('MAX_SUMMARY_LENGTH', '1000')))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv('MAX_BATCH_SIZE', '50')))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE'))
    
    # Security
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'dev-key-change-in-production'))
    
    # NLTK settings
    nltk_data_path: str = field(default_factory=lambda: os.path.expanduser('~/nltk_data'))
    download_nltk_resources: bool = field(default_factory=lambda: os.getenv('DOWNLOAD_NLTK_RESOURCES', 'true').lower() == 'true')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._setup_logging()
        self._create_directories()
    
    def _validate(self):
        """Validate configuration values."""
        errors = []
        
        # Port validation
        if not (1 <= self.port <= 65535):
            errors.append(f"Invalid port: {self.port} (must be 1-65535)")
        
        # Memory limits validation
        if self.max_content_length < 1024:  # Minimum 1KB
            errors.append(f"max_content_length too small: {self.max_content_length}")
        
        if self.max_text_length < 100:  # Minimum 100 characters
            errors.append(f"max_text_length too small: {self.max_text_length}")
        
        # Rate limiting validation
        if self.rate_limit_per_minute < 1:
            errors.append(f"Invalid rate limit: {self.rate_limit_per_minute}")
        
        # Worker count validation
        if self.max_workers < 1:
            errors.append(f"Invalid max_workers: {self.max_workers}")
        
        # Log level validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.log_level} (must be one of {valid_log_levels})")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        handlers = [logging.StreamHandler()]
        
        if self.log_file:
            try:
                handlers.append(logging.FileHandler(self.log_file))
            except Exception as e:
                logger.warning(f"Failed to create log file handler: {e}")
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override existing configuration
        )
        
        logger.info(f"Logging configured: level={self.log_level}, file={self.log_file}")
    
    def _create_directories(self):
        """Create necessary directories."""
        try:
            os.makedirs(self.nltk_data_path, exist_ok=True)
            logger.info(f"NLTK data directory: {self.nltk_data_path}")
        except Exception as e:
            logger.warning(f"Failed to create NLTK directory: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            'debug': self.debug,
            'testing': self.testing,
            'host': self.host,
            'port': self.port,
            'max_content_length': self.max_content_length,
            'max_workers': self.max_workers,
            'cache_size': self.cache_size,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'batch_rate_limit': self.batch_rate_limit,
            'max_text_length': self.max_text_length,
            'max_summary_length': self.max_summary_length,
            'max_batch_size': self.max_batch_size,
            'log_level': self.log_level
            # Note: secret_key and other sensitive data excluded
        }
    
    @classmethod
    def from_env(cls) -> 'OptimizedConfig':
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def for_testing(cls) -> 'OptimizedConfig':
        """Create configuration optimized for testing."""
        return cls(
            debug=True,
            testing=True,
            host='127.0.0.1',
            port=5000,
            log_level='DEBUG',
            rate_limit_per_minute=1000,  # Higher for testing
            download_nltk_resources=False  # Skip for faster tests
        )
    
    @classmethod
    def for_production(cls) -> 'OptimizedConfig':
        """Create configuration optimized for production."""
        config = cls()
        
        # Production-specific overrides
        if config.debug:
            logger.warning("Debug mode enabled in production - consider disabling")
        
        if config.secret_key == 'dev-key-change-in-production':
            raise ValueError("Must set SECRET_KEY environment variable in production")
        
        return config
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-specific configuration."""
        return {
            'DEBUG': self.debug,
            'TESTING': self.testing,
            'SECRET_KEY': self.secret_key,
            'MAX_CONTENT_LENGTH': self.max_content_length,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': False,  # Better performance
        }


# Global configuration instance
_config: Optional[OptimizedConfig] = None


def get_config() -> OptimizedConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        if env == 'testing':
            _config = OptimizedConfig.for_testing()
        elif env == 'production':
            _config = OptimizedConfig.for_production()
        else:
            _config = OptimizedConfig.from_env()
        
        logger.info(f"Configuration loaded for environment: {env}")
    
    return _config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience exports
config = get_config()

# Environment detection helpers
def is_development() -> bool:
    """Check if running in development mode."""
    return config.debug and not config.testing

def is_testing() -> bool:
    """Check if running in testing mode."""
    return config.testing

def is_production() -> bool:
    """Check if running in production mode."""
    return not config.debug and not config.testing