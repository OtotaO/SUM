# SUM Project Coding Standards

This document outlines the coding standards and conventions for the SUM project to ensure consistency, readability, and maintainability across the codebase.

## File Naming Conventions

### Python Files

- Use **snake_case** for all Python files (lowercase with underscores)
- Example: `text_preprocessing.py`, `config_manager.py`, `topic_modeling.py`
- Exception: Main implementation files that define core classes may use PascalCase if they match the primary class name (e.g., `SUM.py` containing the `SUM` class)

### Directories

- Use **snake_case** for directory names
- Example: `utils`, `models`, `tests`, `examples`

### Web Files

- HTML: Use **snake_case** - `index.html`, `user_profile.html`
- CSS: Use **snake_case** - `main_style.css`, `dark_theme.css`
- JavaScript: Use **snake_case** - `main_script.js`, `data_visualization.js`

### Configuration and Data Files

- Use **snake_case** for configuration files - `config.json`, `default_settings.yaml`
- Use **UPPER_SNAKE_CASE** for constants files - `CONSTANTS.py`, `ERROR_CODES.py`
- Use **kebab-case** for documentation files - `api-documentation.md`, `user-guide.md`

## Code Style Conventions

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 100 characters
- Use docstrings for all modules, classes, and functions
- Use type hints for function parameters and return values

### Class Naming and Structure

- Use **PascalCase** for class names - `TextPreprocessor`, `ConfigManager`
- Use **snake_case** for method and function names - `process_text()`, `load_config()`
- Use **snake_case** for variable names - `word_count`, `max_tokens`
- Use **UPPER_SNAKE_CASE** for constants - `MAX_TOKENS`, `DEFAULT_THRESHOLD`

### Imports

- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Within each group, imports should be in alphabetical order
- Use absolute imports rather than relative imports

Example:
```python
# Standard library imports
import os
import sys
import logging
from typing import Dict, List, Optional

# Third-party imports
import nltk
import numpy as np
from flask import Flask, request, jsonify

# Local imports
from utils.config_manager import ConfigManager
from Models.topic_modeling import TopicModeler
```

### Comments and Documentation

- Use docstrings for all modules, classes, and functions
- Follow the Google docstring style
- Include the following in module docstrings:
  - Brief description
  - Extended description (if needed)
  - Design principles
  - Author
  - License

Example:
```python
"""
module_name.py - Brief description

Extended description of the module's purpose and functionality.

Design principles:
- Principle 1
- Principle 2

Author: ototao
License: Apache License 2.0
"""
```

- Include the following in function/method docstrings:
  - Brief description
  - Args
  - Returns
  - Raises (if applicable)

Example:
```python
def function_name(param1: str, param2: int = 0) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2, default is 0
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
    """
```

## Project Structure

The SUM project follows this directory structure:

```
SUM/
├── SUM.py                  # Core implementation
├── main.py                 # Main entry point
├── config.py               # Configuration system
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── CHANGELOG.md            # Change tracking
├── CODING_STANDARDS.md     # This file
├── Models/                 # Model implementations
│   ├── summarizer.py
│   └── topic_modeling.py
├── utils/                  # Utility modules
│   ├── config_manager.py
│   ├── data_loader.py
│   └── text_preprocessing.py
├── Tests/                  # Test suite
│   ├── test_sum.py
│   └── test_config_manager.py
├── examples/               # Example scripts
│   └── config_example.py
├── static/                 # Web static files
│   ├── css/
│   └── js/
└── templates/              # Web templates
    └── index.html
```

## Error Handling

- Use explicit exception handling with specific exception types
- Provide meaningful error messages
- Log exceptions with appropriate log levels
- Use a centralized error handling approach where possible

Example:
```python
try:
    result = process_data(data)
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise
except IOError as e:
    logger.error(f"I/O error during processing: {e}")
    raise
except Exception as e:
    logger.critical(f"Unexpected error: {e}", exc_info=True)
    raise
```

## Testing

- Write unit tests for all modules
- Use pytest for testing
- Aim for high test coverage
- Include both positive and negative test cases
- Mock external dependencies

## Version Control

- Use descriptive commit messages
- Reference issue numbers in commit messages when applicable
- Keep commits focused on a single change
- Use feature branches for new features
- Use pull requests for code review

## Changelog

- Update the CHANGELOG.md file for all significant changes
- Categorize changes as:
  - Added (new features)
  - Changed (changes in existing functionality)
  - Fixed (bug fixes)
  - Refactored (code improvements without functionality changes)
  - Removed (removed features)

## Conclusion

Following these standards will help maintain a consistent, readable, and maintainable codebase. These standards may evolve over time as the project grows and new best practices emerge.