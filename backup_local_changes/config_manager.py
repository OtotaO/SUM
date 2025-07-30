"""
config_manager.py - Configuration management utilities

This module provides utilities for managing configuration settings across
the SUM platform. It allows for dynamic configuration loading, validation,
and access.

Design principles:
- Centralized configuration management
- Flexible validation
- Environment-specific settings
- Secure credential handling

Author: ototao
License: Apache License 2.0
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the SUM platform.
    
    This class provides methods for loading, validating, and accessing
    configuration settings from various sources (environment variables,
    JSON files, etc.).
    
    Attributes:
        config_dict (dict): The current configuration dictionary
        config_sources (list): Sources of configuration data
    """
    
    def __init__(self, base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            base_config: Optional base configuration dictionary
        """
        self.config_dict = base_config or {}
        self.config_sources = []
        
    def load_from_env(self, prefix: str = 'SUM_', lowercase: bool = True) -> 'ConfigManager':
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables to load
            lowercase: Whether to convert keys to lowercase
            
        Returns:
            Self for method chaining
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):]
                if lowercase:
                    config_key = config_key.lower()
                
                # Try to convert to appropriate type
                if value.lower() in ('true', 'yes', '1'):
                    env_config[config_key] = True
                elif value.lower() in ('false', 'no', '0'):
                    env_config[config_key] = False
                elif value.isdigit():
                    env_config[config_key] = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    env_config[config_key] = float(value)
                else:
                    env_config[config_key] = value
        
        self.config_dict.update(env_config)
        self.config_sources.append('environment')
        return self
    
    def load_from_json(self, file_path: Union[str, Path]) -> 'ConfigManager':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return self
            
        try:
            with open(file_path, 'r') as f:
                json_config = json.load(f)
                
            self.config_dict.update(json_config)
            self.config_sources.append(str(file_path))
            logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            
        return self
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Self for method chaining
        """
        self.config_dict.update(config_dict)
        self.config_sources.append('dictionary')
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config_dict.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_dict[key] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Dictionary with all configuration values
        """
        return dict(self.config_dict)
    
    def validate(self, schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Validate the configuration against a schema.
        
        Args:
            schema: Dictionary with validation rules
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for key, rules in schema.items():
            # Check if required
            if rules.get('required', False) and key not in self.config_dict:
                errors.append(f"Missing required configuration: {key}")
                continue
                
            # Skip validation if key not present and not required
            if key not in self.config_dict:
                continue
                
            value = self.config_dict[key]
            
            # Check type
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")
                
            # Check min/max for numeric values
            if isinstance(value, (int, float)):
                if 'min' in rules and value < rules['min']:
                    errors.append(f"Value for {key} is below minimum: {value} < {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"Value for {key} is above maximum: {value} > {rules['max']}")
                    
            # Check allowed values
            if 'allowed' in rules and value not in rules['allowed']:
                errors.append(f"Invalid value for {key}: {value} not in {rules['allowed']}")
                
        return errors
    
    def save_to_json(self, file_path: Union[str, Path], include_sensitive: bool = False) -> bool:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration
            include_sensitive: Whether to include sensitive values
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Filter sensitive values if needed
            config_to_save = dict(self.config_dict)
            if not include_sensitive:
                sensitive_keys = ['secret', 'password', 'token', 'key', 'credential']
                for key in list(config_to_save.keys()):
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        del config_to_save[key]
            
            with open(file_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
                
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            return False
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        if key not in self.config_dict:
            raise KeyError(f"Configuration key not found: {key}")
        return self.config_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary syntax."""
        self.config_dict[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self.config_dict
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(sources={self.config_sources}, keys={list(self.config_dict.keys())})"


# Example usage
if __name__ == "__main__":
    # Create a configuration manager with base settings
    config = ConfigManager({
        'app_name': 'SUM',
        'version': '1.0.0',
        'debug': False
    })
    
    # Load from environment variables
    config.load_from_env()
    
    # Load from a JSON file
    config.load_from_json('config.json')
    
    # Validate configuration
    schema = {
        'app_name': {'required': True, 'type': str},
        'version': {'required': True, 'type': str},
        'debug': {'type': bool},
        'port': {'type': int, 'min': 1024, 'max': 65535},
        'log_level': {'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
    }
    
    errors = config.validate(schema)
    if errors:
        for error in errors:
            print(f"Validation error: {error}")
    
    # Access configuration values
    print(f"App name: {config.get('app_name')}")
    print(f"Debug mode: {config.get('debug')}")
    
    # Save configuration
    config.save_to_json('config_export.json')
