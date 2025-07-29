# SUM Utilities

This directory contains utility modules that provide common functionality used across the SUM platform. These utilities are designed to be reusable, maintainable, and follow consistent patterns.

## Available Utilities

### config_manager.py

The `ConfigManager` class provides centralized configuration management for the SUM platform:

- Load configuration from multiple sources (environment variables, JSON files, dictionaries)
- Validate configuration against schemas
- Access configuration with type safety
- Secure handling of sensitive configuration values
- Method chaining for fluent API

**Example usage:**

```python
from Utils.config_manager import ConfigManager

# Create a configuration manager with base settings
config = ConfigManager({
    'app_name': 'SUM',
    'version': '1.0.0'
})

# Load from environment variables and JSON file
config.load_from_env().load_from_json('config.json')

# Access configuration values
app_name = config.get('app_name')
port = config.get('port', 3000)  # With default value
```

See the examples directory for more detailed usage examples.

### error_handling.py

Provides standardized error handling utilities for consistent error management across the platform:

- Custom exception hierarchy for SUM-specific errors
- Decorators for standardized exception handling
- Input validation utilities
- Safe execution wrappers
- Standardized error response formatting

**Example usage:**

```python
from Utils.error_handling import handle_exceptions, ValidationError, safe_execute

# Use the handle_exceptions decorator for standardized error handling
@handle_exceptions(logger_instance=logger)
def process_data(data):
    if not data:
        raise ValidationError("Data cannot be empty")
    # Process data...
    return result

# Use safe_execute for operations that might fail
result = safe_execute(
    process_data,
    input_data,
    default_return={'error': True, 'message': 'Processing failed'}
)
```

### nltk_utils.py

Provides centralized utilities for managing NLTK resources:

- Thread-safe initialization of NLTK resources
- Cached access to common NLTK components (stopwords, lemmatizer)
- Robust error handling for resource loading
- Configurable download directory

**Example usage:**

```python
from Utils.nltk_utils import initialize_nltk, get_stopwords, get_lemmatizer

# Initialize NLTK resources
initialize_nltk()

# Get stopwords and lemmatizer
stopwords = get_stopwords()
lemmatizer = get_lemmatizer()

# Use in text processing
if word.lower() not in stopwords:
    lemmatized_word = lemmatizer.lemmatize(word)
```

### text_preprocessing.py

Provides standardized text preprocessing functions:

- Text cleaning (URLs, emails, special characters)
- Tokenization (sentences, words)
- N-gram extraction
- Word frequency calculation
- String safety validation

**Example usage:**

```python
from Utils.text_preprocessing import preprocess_text, tokenize_sentences, tokenize_words

# Preprocess text with various options
processed_text = preprocess_text(
    text,
    lowercase=True,
    remove_stopwords=True,
    remove_urls=True,
    remove_special_chars=False
)

# Tokenize text
sentences = tokenize_sentences(text)
words = tokenize_words(text)
```

### data_loader.py

Provides utilities for loading data from various sources:

- File loading (text, JSON, CSV)
- Web content fetching
- Data validation
- Format conversion

## Best Practices

When using or extending these utilities, follow these best practices:

1. **Centralized Usage**: Use these utilities instead of reimplementing similar functionality in different parts of the codebase.

2. **Error Handling**: Always handle exceptions that might be raised by these utilities, especially when dealing with external resources or user input.

3. **Configuration**: Use the `ConfigManager` for all configuration needs to ensure consistent configuration handling across the platform.

4. **Thread Safety**: Be aware of thread safety considerations, especially when using shared resources like NLTK components.

5. **Testing**: Write tests for any new utilities or extensions to existing utilities.

## Adding New Utilities

When adding new utility modules:

1. Follow the project's coding standards (see `CODING_STANDARDS.md`)
2. Include comprehensive docstrings
3. Add appropriate error handling
4. Write unit tests in the `Tests` directory
5. Update this README.md with documentation for the new utility
6. Add an entry to the `CHANGELOG.md` file

## Utility Design Principles

The SUM utilities follow these design principles:

- **Single Responsibility**: Each utility module has a clear, focused purpose
- **Reusability**: Utilities are designed to be reused across the platform
- **Robustness**: Utilities include proper error handling and validation
- **Configurability**: Utilities can be configured for different use cases
- **Testability**: Utilities are designed to be easily testable
