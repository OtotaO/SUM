# Contributing to SUM

Thank you for your interest in contributing to SUM! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Clear description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error messages and logs

### Suggesting Features

1. **Open a discussion** first for major features
2. **Provide use cases** and examples
3. **Consider implementation complexity**
4. **Be open to feedback** and alternative approaches

### Code Contributions

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/SUM.git
cd SUM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run unit tests
   pytest tests/

   # Run linting
   flake8 .
   black --check .

   # Run type checking
   mypy .
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature X"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes
   - `chore:` Maintenance tasks

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

1. **PR Title**: Use conventional commit format
2. **Description**: 
   - Explain what changes you made
   - Why these changes are needed
   - Any breaking changes
   - Related issues

3. **Requirements**:
   - All tests must pass
   - Code must be formatted with `black`
   - No linting errors
   - Documentation updated if needed
   - Signed commits preferred

### Code Style

- **Python**: Follow PEP 8
- **Formatting**: Use `black` with default settings
- **Imports**: Sort with `isort`
- **Type hints**: Use where beneficial
- **Docstrings**: Google style for functions/classes

Example:
```python
def process_text(text: str, max_length: int = 1000) -> Dict[str, Any]:
    """
    Process input text for summarization.
    
    Args:
        text: Input text to process
        max_length: Maximum allowed text length
        
    Returns:
        Dictionary containing processed text and metadata
        
    Raises:
        ValueError: If text exceeds max_length
    """
    if len(text) > max_length:
        raise ValueError(f"Text length {len(text)} exceeds maximum {max_length}")
    
    # Processing logic here
    return {"processed": text, "length": len(text)}
```

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use `pytest` for testing
- Mock external dependencies

Example test:
```python
def test_process_text():
    """Test text processing function."""
    result = process_text("Hello world")
    assert result["processed"] == "Hello world"
    assert result["length"] == 11
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update API documentation for endpoint changes
- Include examples where helpful

## Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers
3. **Address feedback** promptly
4. **Squash commits** if requested
5. **Merge** once approved

## Release Process

1. Maintainers handle releases
2. Semantic versioning is used
3. Changelog is updated
4. Git tags mark releases

## Getting Help

- **Discord**: [Join our community](https://discord.gg/sum-community)
- **Discussions**: Use GitHub Discussions
- **Email**: dev@sum-project.org

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Project website

Thank you for contributing to SUM! 🎉