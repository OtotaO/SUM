# SUM Examples

This directory contains example scripts demonstrating various features and usage patterns of the SUM platform.

## Configuration Examples

The following examples demonstrate how to use the `ConfigManager` class for configuration management in SUM applications:

### simple_config_example.py

A basic example demonstrating the fundamental features of the `ConfigManager` class:

- Creating a configuration manager with base settings
- Loading configuration from environment variables
- Loading configuration from JSON files
- Accessing and modifying configuration values
- Validating configuration against a schema
- Saving configuration to a file

Run this example to get a quick overview of the configuration system:

```bash
python examples/simple_config_example.py
```

### config_integration_example.py

A more advanced example showing how to integrate the `ConfigManager` with SUM components in a real-world application structure:

- Defining a configuration schema for validation
- Creating service classes that use configuration
- Building an application that integrates multiple services
- Handling configuration errors gracefully
- Processing documents with configurable parameters

Run this example to see how configuration can be used in a structured application:

```bash
python examples/config_integration_example.py
```

### integrated_config_example.py

A comprehensive example demonstrating the full integration of the `ConfigManager` with core SUM functionality:

- Configuring summarization with different algorithms (extractive, abstractive, hybrid)
- Configuring topic modeling with different algorithms (LDA, NMF, LSA)
- Setting up preprocessing options for text processing
- Using safe execution and error handling with configuration
- Building a complete application with multiple integrated components

Run this example to see a complete SUM application with configurable components:

```bash
python examples/integrated_config_example.py
```

## NLTK Resource Management

### download_nltk_resources.py

A utility script for downloading and managing NLTK resources required by the SUM platform:

- Downloads essential NLTK resources (punkt, stopwords, wordnet, etc.)
- Configurable download directory
- Progress reporting
- Error handling for failed downloads

Run this script to ensure all required NLTK resources are available:

```bash
python examples/download_nltk_resources.py
```

## Additional Examples

### config_example.py

An example demonstrating advanced configuration features:

- Loading configuration from multiple sources with priority
- Using environment variables for sensitive settings
- Validating complex configuration structures
- Handling configuration errors

Run this example to explore advanced configuration techniques:

```bash
python examples/config_example.py
```

## Running the Examples

All examples can be run directly from the command line. Make sure you are in the root directory of the SUM project when running the examples.

For example:

```bash
# From the SUM project root directory
python examples/simple_config_example.py
```

Some examples may create temporary files during execution, but they will clean up after themselves when they complete.

## Creating Your Own Examples

Feel free to modify these examples or create your own to explore different aspects of the SUM platform. The examples are designed to be educational and demonstrate best practices for using the SUM components.

When creating new examples, consider following these guidelines:

1. Include clear documentation at the top of the file explaining the purpose of the example
2. Use meaningful variable and function names
3. Add comments to explain complex or important sections
4. Clean up any temporary files or resources created during execution
5. Follow the coding standards outlined in the project's `CODING_STANDARDS.md` file
