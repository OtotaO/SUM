# SUM Examples

This directory contains example scripts demonstrating how to use various components of the SUM (Summarizer) platform.

## Overview

These examples are designed to help you understand how to use the SUM platform's components in different scenarios. They provide practical demonstrations of key features and best practices.

## Examples

### `simple_config_example.py`

Demonstrates the basic usage of the `ConfigManager` for simple configuration management tasks.

**Features demonstrated:**
- Creating a configuration manager with default settings
- Setting and getting configuration values
- Dictionary-style access
- Checking if keys exist
- Getting values with defaults
- Converting configuration to a dictionary

**How to run:**
```bash
python examples/simple_config_example.py
```

### `config_example.py`

Demonstrates how to use the `ConfigManager` class to load, validate, and access configuration settings from various sources.

**Features demonstrated:**
- Loading configuration from environment variables
- Loading configuration from JSON files
- Validating configuration against schemas
- Accessing configuration values
- Saving configuration to JSON files

**How to run:**
```bash
python examples/config_example.py
```

### `integrated_config_example.py`

Shows how to integrate the `ConfigManager` with SUM components like `SimpleSUM`, `MagnumOpusSUM`, and `TopicModeler`.

**Features demonstrated:**
- Creating a configuration file
- Loading configuration from multiple sources
- Using configuration to initialize SUM components
- Processing text with configured components
- Displaying results

**How to run:**
```bash
python examples/integrated_config_example.py
```

### `config_integration_example.py`

Demonstrates a more advanced integration of the `ConfigManager` with SUM components in a real application structure.

**Features demonstrated:**
- Creating a default configuration file
- Validating configuration against a schema
- Initializing components based on configuration
- Processing text with configured components
- Saving modified configuration
- Creating a configuration summary

**How to run:**
```bash
python examples/config_integration_example.py
```

## Best Practices

These examples demonstrate several best practices for using the SUM platform:

1. **Centralized configuration management**
   - Use the `ConfigManager` to manage configuration settings
   - Load configuration from multiple sources
   - Validate configuration against schemas

2. **Component initialization**
   - Initialize components with configuration settings
   - Use sensible defaults for missing configuration

3. **Error handling**
   - Validate input data
   - Handle exceptions gracefully
   - Provide meaningful error messages

4. **Resource cleanup**
   - Clean up temporary files
   - Release resources when done

## Creating Your Own Examples

Feel free to use these examples as a starting point for your own scripts. Here's a template to get you started:

```python
"""
my_example.py - Description of your example

Author: Your Name
License: Apache License 2.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SUM components
from SUM import SimpleSUM
from Utils.config_manager import ConfigManager

def main():
    """Main function for the example."""
    # Initialize configuration
    config = ConfigManager()
    config.load_from_env()
    
    # Initialize SUM components
    summarizer = SimpleSUM()
    
    # Your example code here
    
    print("Example complete!")

if __name__ == "__main__":
    main()
```

## Contributing

If you create a useful example, consider contributing it back to the project. This helps others learn how to use the SUM platform effectively.
