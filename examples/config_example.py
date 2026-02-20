"""
config_example.py - Example of using the ConfigManager

This script demonstrates how to use the ConfigManager to load, validate,
and access configuration settings from various sources.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_manager import ConfigManager

def main():
    """Demonstrate ConfigManager usage."""
    print("SUM Configuration Manager Example")
    print("=================================")
    
    # Create a configuration manager with base settings
    config = ConfigManager({
        'app_name': 'SUM',
        'version': '1.0.0',
        'debug': False
    })
    print(f"Initial config: {config}")
    
    # Load from environment variables
    config.load_from_env()
    print(f"After loading from environment: {config}")
    
    # Create a sample JSON config file
    sample_config = {
        'num_topics': 10,
        'max_summary_length': 300,
        'algorithms': ['lda', 'nmf', 'lsa'],
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'sum_db'
        }
    }
    
    # Save sample config to a temporary file
    temp_config_path = Path('temp_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    # Load from the JSON file
    config.load_from_json(temp_config_path)
    print(f"After loading from JSON: {config}")
    
    # Validate configuration
    schema = {
        'app_name': {'required': True, 'type': str},
        'version': {'required': True, 'type': str},
        'debug': {'type': bool},
        'num_topics': {'type': int, 'min': 1, 'max': 100},
        'max_summary_length': {'type': int, 'min': 50, 'max': 1000},
        'algorithms': {'type': list}
    }
    
    errors = config.validate(schema)
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nConfiguration is valid!")
    
    # Access configuration values
    print("\nConfiguration values:")
    print(f"- App name: {config.get('app_name')}")
    print(f"- Debug mode: {config.get('debug')}")
    print(f"- Number of topics: {config.get('num_topics')}")
    print(f"- Max summary length: {config.get('max_summary_length')}")
    print(f"- Algorithms: {config.get('algorithms')}")
    print(f"- Database: {config.get('database')}")
    
    # Dictionary-style access
    try:
        print(f"\nUsing dictionary-style access:")
        print(f"- App name: {config['app_name']}")
        print(f"- Non-existent key: {config['non_existent']}")
    except KeyError as e:
        print(f"- Error: {e}")
    
    # Save configuration to a new file
    export_path = Path('exported_config.json')
    config.save_to_json(export_path)
    print(f"\nConfiguration saved to {export_path}")
    
    # Clean up temporary files
    temp_config_path.unlink()
    export_path.unlink()
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
