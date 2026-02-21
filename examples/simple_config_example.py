"""simple_config_example.py - Simple example of using the ConfigManager

This example demonstrates the basic usage of the ConfigManager class
for loading, validating, and accessing configuration settings.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_manager import ConfigManager

def main():
    """Run the simple configuration example."""
    print("\n=== Simple Configuration Example ===")
    
    # Create a configuration manager with base settings
    print("\n1. Creating ConfigManager with base settings...")
    config = ConfigManager({
        'app_name': 'SUM',
        'version': '1.0.0',
        'debug': False
    })
    print(f"Initial config: {config}")
    
    # Access configuration values
    print("\n2. Accessing configuration values...")
    print(f"App name: {config.get('app_name')}")
    print(f"Version: {config.get('version')}")
    print(f"Debug mode: {config.get('debug')}")
    print(f"Non-existent value with default: {config.get('non_existent', 'default value')}")
    
    # Set configuration values
    print("\n3. Setting configuration values...")
    config.set('log_level', 'INFO')
    config['max_tokens'] = 100  # Dictionary-style access
    print(f"Updated config: {config}")
    
    # Create a temporary JSON configuration file
    print("\n4. Creating a temporary JSON configuration file...")
    temp_config = {
        'port': 3000,
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'sum_db'
        },
        'algorithms': ['lda', 'nmf', 'lsa']
    }
    
    temp_file = Path('temp_config.json')
    with open(temp_file, 'w') as f:
        json.dump(temp_config, f, indent=2)
    print(f"Created temporary config file: {temp_file}")
    
    # Load configuration from JSON file
    print("\n5. Loading configuration from JSON file...")
    config.load_from_json(temp_file)
    print(f"Config after loading from JSON: {config}")
    print(f"Port: {config.get('port')}")
    print(f"Database host: {config.get('database')['host']}")
    print(f"Algorithms: {config.get('algorithms')}")
    
    # Set environment variables for testing
    print("\n6. Setting environment variables...")
    os.environ['SUM_ENV_SETTING'] = 'value from environment'
    os.environ['SUM_DEBUG'] = 'true'
    os.environ['SUM_PORT'] = '5000'
    
    # Load configuration from environment variables
    print("\n7. Loading configuration from environment variables...")
    config.load_from_env()
    print(f"Config after loading from environment: {config}")
    print(f"Environment setting: {config.get('env_setting')}")
    print(f"Debug (from environment): {config.get('debug')}")
    print(f"Port (from environment): {config.get('port')}")
    
    # Validate configuration
    print("\n8. Validating configuration...")
    schema = {
        'app_name': {'required': True, 'type': str},
        'version': {'required': True, 'type': str},
        'debug': {'type': bool},
        'port': {'type': int, 'min': 1024, 'max': 65535},
        'log_level': {'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
    }
    
    errors = config.validate(schema)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Save configuration to a new file
    print("\n9. Saving configuration to a new file...")
    output_file = Path('output_config.json')
    config.save_to_json(output_file)
    print(f"Configuration saved to: {output_file}")
    
    # Clean up temporary files
    print("\n10. Cleaning up temporary files...")
    temp_file.unlink()
    output_file.unlink()
    print("Temporary files removed.")
    
    # Clean up environment variables
    del os.environ['SUM_ENV_SETTING']
    del os.environ['SUM_DEBUG']
    del os.environ['SUM_PORT']
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()