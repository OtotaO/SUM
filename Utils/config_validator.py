"""
config_validator.py - Configuration Validation and Management

This module provides configuration validation to ensure:
- All required settings are present
- Values are within acceptable ranges
- Secure defaults are applied
- Environment-specific validation

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class ConfigRule:
    """Defines a validation rule for a configuration value."""
    key: str
    required: bool = True
    type: type = str
    default: Any = None
    validator: Optional[callable] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: str = ""


class ConfigValidator:
    """Validates and manages configuration settings."""
    
    def __init__(self):
        self.rules = self._define_rules()
        self.validated_config = {}
        self.validation_errors = []
        self.validation_warnings = []
    
    def _define_rules(self) -> List[ConfigRule]:
        """Define validation rules for all configuration values."""
        return [
            # Flask settings
            ConfigRule(
                key="SECRET_KEY",
                required=True,
                type=str,
                validator=lambda x: len(x) >= 32,
                description="Flask secret key (minimum 32 characters)"
            ),
            ConfigRule(
                key="PORT",
                type=int,
                default=3000,
                min_value=1024,
                max_value=65535,
                description="Server port number"
            ),
            ConfigRule(
                key="HOST",
                type=str,
                default="127.0.0.1",
                pattern=r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$|^localhost$|^0\.0\.0\.0$",
                description="Server host address"
            ),
            ConfigRule(
                key="DEBUG",
                type=bool,
                default=False,
                description="Debug mode (should be False in production)"
            ),
            
            # File upload settings
            ConfigRule(
                key="MAX_CONTENT_LENGTH",
                type=int,
                default=100 * 1024 * 1024,  # 100MB
                min_value=1024 * 1024,  # 1MB
                max_value=500 * 1024 * 1024,  # 500MB
                description="Maximum file upload size in bytes"
            ),
            ConfigRule(
                key="UNIVERSAL_FILE_SUPPORT",
                type=bool,
                default=True,
                description="Whether to accept all file extensions"
            ),
            
            # Summarization settings
            ConfigRule(
                key="MAX_SUMMARY_LENGTH",
                type=int,
                default=200,
                min_value=10,
                max_value=1000,
                description="Maximum summary length in words"
            ),
            ConfigRule(
                key="MIN_SUMMARY_LENGTH",
                type=int,
                default=50,
                min_value=10,
                max_value=500,
                description="Minimum summary length in words"
            ),
            ConfigRule(
                key="DEFAULT_THRESHOLD",
                type=float,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                description="Default similarity threshold"
            ),
            
            # Processing settings
            ConfigRule(
                key="BATCH_SIZE",
                type=int,
                default=100,
                min_value=1,
                max_value=1000,
                description="Batch processing size"
            ),
            ConfigRule(
                key="MAX_WORKERS",
                type=int,
                default=4,
                min_value=1,
                max_value=16,
                validator=lambda x: x <= (os.cpu_count() or 4) * 2,
                description="Maximum worker threads"
            ),
            ConfigRule(
                key="MAX_TEXT_LENGTH",
                type=int,
                default=1000000,  # 1M characters
                min_value=10000,
                max_value=10000000,
                description="Maximum text length for processing"
            ),
            
            # Memory settings
            ConfigRule(
                key="MAX_MEMORY_MB",
                type=int,
                default=500,
                min_value=100,
                max_value=4096,
                description="Maximum memory usage in MB"
            ),
            ConfigRule(
                key="CHUNK_SIZE",
                type=int,
                default=50000,
                min_value=1000,
                max_value=500000,
                description="Text chunk size for processing"
            ),
            
            # Semantic Memory settings
            ConfigRule(
                key="SEMANTIC_MEMORY_MODEL",
                type=str,
                default="all-MiniLM-L6-v2",
                allowed_values=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
                description="Sentence transformer model for embeddings"
            ),
            ConfigRule(
                key="SEMANTIC_MEMORY_PATH",
                type=str,
                default="./semantic_memory",
                description="Path for semantic memory storage"
            ),
            
            # Knowledge Graph settings
            ConfigRule(
                key="NEO4J_URI",
                type=str,
                required=False,
                pattern=r"^(bolt|neo4j)://",
                description="Neo4j database URI"
            ),
            ConfigRule(
                key="SPACY_MODEL",
                type=str,
                default="en_core_web_sm",
                allowed_values=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
                description="spaCy model for NLP"
            ),
            
            # API settings
            ConfigRule(
                key="API_RATE_LIMIT",
                type=int,
                default=100,
                min_value=10,
                max_value=1000,
                description="API rate limit per minute"
            ),
            ConfigRule(
                key="API_TIMEOUT",
                type=int,
                default=300,
                min_value=30,
                max_value=3600,
                description="API request timeout in seconds"
            ),
            
            # Logging settings
            ConfigRule(
                key="LOG_LEVEL",
                type=str,
                default="INFO",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                description="Logging level"
            ),
            ConfigRule(
                key="LOG_FILE",
                type=str,
                default="sum_service.log",
                pattern=r"^[\w\-\.]+\.log$",
                description="Log file name"
            ),
        ]
    
    def validate_config(self, config: Dict[str, Any], env: str = "development") -> bool:
        """
        Validate configuration values.
        
        Args:
            config: Configuration dictionary
            env: Environment name (development, production, testing)
            
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors = []
        self.validation_warnings = []
        self.validated_config = {}
        
        # Validate each rule
        for rule in self.rules:
            self._validate_rule(rule, config, env)
        
        # Environment-specific validation
        self._validate_environment(config, env)
        
        # Log results
        if self.validation_errors:
            for error in self.validation_errors:
                logger.error(f"Config validation error: {error}")
        
        if self.validation_warnings:
            for warning in self.validation_warnings:
                logger.warning(f"Config validation warning: {warning}")
        
        return len(self.validation_errors) == 0
    
    def _validate_rule(self, rule: ConfigRule, config: Dict[str, Any], env: str):
        """Validate a single configuration rule."""
        value = config.get(rule.key, os.getenv(f"SUM_{rule.key}", rule.default))
        
        # Check if required
        if rule.required and value is None:
            self.validation_errors.append(f"{rule.key} is required but not provided")
            return
        
        # Skip if not provided and not required
        if value is None:
            return
        
        # Type conversion
        try:
            if rule.type == bool:
                value = str(value).lower() in ('true', '1', 't', 'yes', 'on')
            else:
                value = rule.type(value)
        except (ValueError, TypeError) as e:
            self.validation_errors.append(f"{rule.key} type error: expected {rule.type.__name__}, got {type(value).__name__}")
            return
        
        # Min/max validation
        if rule.min_value is not None and value < rule.min_value:
            self.validation_errors.append(f"{rule.key} value {value} is below minimum {rule.min_value}")
            return
        
        if rule.max_value is not None and value > rule.max_value:
            self.validation_errors.append(f"{rule.key} value {value} is above maximum {rule.max_value}")
            return
        
        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            self.validation_errors.append(f"{rule.key} value '{value}' not in allowed values: {rule.allowed_values}")
            return
        
        # Pattern validation
        if rule.pattern and isinstance(value, str):
            if not re.match(rule.pattern, value):
                self.validation_errors.append(f"{rule.key} value '{value}' does not match required pattern")
                return
        
        # Custom validator
        if rule.validator:
            try:
                if not rule.validator(value):
                    self.validation_errors.append(f"{rule.key} failed custom validation")
                    return
            except Exception as e:
                self.validation_errors.append(f"{rule.key} validator error: {e}")
                return
        
        # Store validated value
        self.validated_config[rule.key] = value
    
    def _validate_environment(self, config: Dict[str, Any], env: str):
        """Perform environment-specific validation."""
        if env == "production":
            # Production-specific checks
            if self.validated_config.get("DEBUG", False):
                self.validation_errors.append("DEBUG must be False in production")
            
            if not self.validated_config.get("SECRET_KEY"):
                self.validation_errors.append("SECRET_KEY must be explicitly set in production")
            
            if self.validated_config.get("HOST") == "0.0.0.0":
                self.validation_warnings.append("Using 0.0.0.0 as HOST in production - ensure proper firewall rules")
            
            if self.validated_config.get("LOG_LEVEL") == "DEBUG":
                self.validation_warnings.append("DEBUG log level in production may expose sensitive information")
        
        elif env == "development":
            # Development-specific checks
            if not self.validated_config.get("DEBUG", True):
                self.validation_warnings.append("DEBUG is False in development environment")
    
    def get_config_with_defaults(self) -> Dict[str, Any]:
        """Get configuration with all defaults applied."""
        config = {}
        for rule in self.rules:
            if rule.key in self.validated_config:
                config[rule.key] = self.validated_config[rule.key]
            elif rule.default is not None:
                config[rule.key] = rule.default
        return config
    
    def generate_config_template(self, format: str = "env") -> str:
        """Generate a configuration template."""
        if format == "env":
            lines = ["# SUM Configuration Template", ""]
            for rule in self.rules:
                lines.append(f"# {rule.description}")
                if rule.default is not None:
                    lines.append(f"SUM_{rule.key}={rule.default}")
                else:
                    lines.append(f"# SUM_{rule.key}=")
                lines.append("")
            return "\n".join(lines)
        
        elif format == "json":
            config = {}
            for rule in self.rules:
                config[rule.key] = {
                    "value": rule.default,
                    "description": rule.description,
                    "type": rule.type.__name__,
                    "required": rule.required
                }
                if rule.allowed_values:
                    config[rule.key]["allowed_values"] = rule.allowed_values
                if rule.min_value is not None:
                    config[rule.key]["min"] = rule.min_value
                if rule.max_value is not None:
                    config[rule.key]["max"] = rule.max_value
            return json.dumps(config, indent=2)
    
    def validate_file_config(self, file_path: str, env: str = "development") -> bool:
        """Validate configuration from a file."""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    config = json.load(f)
            elif file_path.endswith('.env'):
                config = {}
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if key.startswith('SUM_'):
                                key = key[4:]  # Remove SUM_ prefix
                            config[key] = value
            else:
                self.validation_errors.append(f"Unsupported config file format: {file_path}")
                return False
            
            return self.validate_config(config, env)
            
        except Exception as e:
            self.validation_errors.append(f"Error reading config file: {e}")
            return False


# Singleton instance
_validator = None

def get_config_validator() -> ConfigValidator:
    """Get or create the config validator instance."""
    global _validator
    if _validator is None:
        _validator = ConfigValidator()
    return _validator