"""
simple_config_example.py - Simple example of using the ConfigManager

This script demonstrates the basic usage of the ConfigManager for simple
configuration management tasks.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.config_manager import ConfigManager

def main():
    """Demonstrate basic ConfigManager usage."""
    print("SUM Simple Configuration Example")
    print("================================")
    
    # Create a configuration manager with default settings
    config = ConfigManager({
        'app_name': 'SUM',
        'version': '1.0.0',
        'debug': False
    })
    
    print("\nDefault configuration:")
    print(f"- App name: {config.get('app_name')}")
    print(f"- Version: {config.get('version')}")
    print(f"- Debug mode: {config.get('debug')}")
    
    # Set some configuration values
    print("\nSetting configuration values...")
    config.set('max_tokens', 100)
    config.set('language', 'en')
    config.set('theme', 'dark')
    
    # Dictionary-style access
    config['batch_size'] = 64
    
    print("\nUpdated configuration:")
    print(f"- App name: {config.get('app_name')}")
    print(f"- Max tokens: {config.get('max_tokens')}")
    print(f"- Language: {config.get('language')}")
    print(f"- Theme: {config.get('theme')}")
    print(f"- Batch size: {config.get('batch_size')}")
    
    # Check if keys exist
    print("\nChecking if keys exist:")
    print(f"- 'app_name' exists: {'app_name' in config}")
    print(f"- 'non_existent' exists: {'non_existent' in config}")
    
    # Get with default value
    print("\nGetting values with defaults:")
    print(f"- Existing key: {config.get('app_name', 'Default App')}")
    print(f"- Non-existent key: {config.get('non_existent', 'Default Value')}")
    
    # Get configuration as dictionary
    config_dict = config.as_dict()
    print("\nConfiguration as dictionary:")
    for key, value in config_dict.items():
        print(f"- {key}: {value}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
