# SUM Project Naming Convention Standard

## Philosophy: Carmack-Inspired Clear Naming

> "Code should be self-documenting through clear naming" - John Carmack

This document establishes a comprehensive naming convention standard for the SUM project, inspired by John Carmack's philosophy of crystal-clear, intent-revealing names that make code self-documenting.

## Core Principles

### 1. Clarity Over Brevity
- Names should clearly communicate purpose and intent
- No abbreviations unless universally understood (e.g., `url`, `html`, `json`)
- Prefer descriptive names over short cryptic ones

### 2. Consistency Above All
- Follow Python PEP 8 conventions strictly
- Maintain consistent patterns throughout the codebase
- Use the same naming pattern for similar concepts

### 3. Intent-Revealing Names
- Names should answer: what it does, why it exists, how it's used
- Avoid mental mapping - the name should be the concept
- No misleading names or names that require comments to understand

### 4. Searchability
- Use complete words that are easily searchable
- Avoid single-letter variables except for short loop counters
- Use consistent terminology across the project

## Naming Conventions by Category

### File Names
**Standard: snake_case for all Python files**

✅ **Correct:**
- `summarization_engine.py`
- `text_preprocessing.py`
- `config_manager.py`
- `streaming_engine.py`

❌ **Incorrect:**
- `advanced-summarization-engine.py` (hyphenated)
- `StreamingEngine.py` (PascalCase)
- `sum-cli-interface.py` (mixed hyphenated)

### Class Names
**Standard: PascalCase (CapWords)**

✅ **Correct:**
- `class SummarizationEngine:`
- `class TextPreprocessor:`
- `class ConfigManager:`
- `class StreamingHierarchicalEngine:`

❌ **Incorrect:**
- `class summarization_engine:` (snake_case)
- `class SUM:` (acronym without expansion)
- `class AI_Enhanced_Interface:` (mixed case with underscores)

### Function and Method Names
**Standard: snake_case**

✅ **Correct:**
- `def process_text()`
- `def extract_key_concepts()`
- `def initialize_nltk_resources()`
- `def calculate_semantic_similarity()`

❌ **Incorrect:**
- `def processText()` (camelCase)
- `def process-text()` (hyphenated - invalid syntax)
- `def ProcessText()` (PascalCase)

### Variable Names
**Standard: snake_case**

✅ **Correct:**
- `text_content = "..."`
- `similarity_threshold = 0.85`
- `processed_chunks = []`
- `semantic_embeddings = np.array(...)`

❌ **Incorrect:**
- `textContent = "..."` (camelCase)
- `Text_Content = "..."` (mixed PascalCase with underscores)
- `tc = "..."` (abbreviated, unclear)

### Constants
**Standard: SCREAMING_SNAKE_CASE**

✅ **Correct:**
- `DEFAULT_CHUNK_SIZE = 1000`
- `MAXIMUM_FILE_SIZE_BYTES = 10_000_000`
- `NLTK_DOWNLOAD_DIRECTORY = '~/nltk_data'`
- `SUPPORTED_FILE_EXTENSIONS = ['.txt', '.md', '.pdf']`

❌ **Incorrect:**
- `MaxFileSize = 10000000` (PascalCase)
- `max_file_size = 10000000` (snake_case for constant)

### Module and Package Names
**Standard: snake_case, short, all lowercase**

✅ **Correct:**
- `utils/`
- `models/`
- `processors/`
- `engines/`

❌ **Incorrect:**
- `Utils/` (capitalized)
- `AI_Models/` (mixed case)
- `Text-Processing/` (hyphenated)

### Acronyms and Initialisms
**Standard: Treat as regular words in context**

✅ **Correct:**
- `class HtmlProcessor:` (not HTMLProcessor)
- `class JsonLoader:` (not JSONLoader)  
- `class ApiClient:` (not APIClient)
- `class SummarizationEngine:` (expand SUM to Summarization)

❌ **Incorrect:**
- `class HTMLProcessor:`
- `class XMLParser:`
- `class AIModel:` (too vague)

### Exception Classes
**Standard: PascalCase ending with 'Error' or 'Exception'**

✅ **Correct:**
- `class SummarizationError(Exception):`
- `class ConfigurationError(Exception):`
- `class TextProcessingError(Exception):`

❌ **Incorrect:**
- `class SumError(Exception):` (abbreviated)
- `class processing_error(Exception):` (snake_case)

### Private Methods and Variables
**Standard: Leading underscore for private**

✅ **Correct:**
- `def _initialize_nltk_resources(self):`
- `self._cached_embeddings = {}`
- `def _calculate_word_frequencies(self, text):`

❌ **Incorrect:**
- `def initNltk(self):` (public when should be private)
- `def __calculate_frequencies(self):` (double underscore for non-special methods)

## Specific Naming Patterns for SUM Project

### Core Components
- `SummarizationEngine` (not `SUM` or `Summarizer`)
- `TextPreprocessor` (not `TextProcessor`)
- `SemanticAnalyzer` (not `SemanticEngine`)
- `ContentExtractor` (not `Extractor`)

### Processing Functions
- `process_text_content()` (not `processText()`)
- `extract_key_concepts()` (not `getKeys()`)
- `calculate_similarity()` (not `calcSim()`)
- `generate_summary()` (not `summarize()`)

### Data Structures
- `processed_documents` (not `docs`)
- `similarity_matrix` (not `sim_matrix`)
- `concept_embeddings` (not `embeddings`)
- `configuration_settings` (not `config`)

### File Organization Patterns
```
utils/
├── text_preprocessing.py
├── configuration_manager.py
├── error_handling.py
└── nltk_utilities.py

models/
├── summarization_model.py
├── topic_modeling.py
└── similarity_calculator.py

engines/
├── streaming_engine.py
├── hierarchical_engine.py
└── compression_engine.py
```

## Documentation and Comments

### Function Docstrings
```python
def extract_key_concepts(text_content: str, concept_threshold: float = 0.8) -> List[str]:
    """
    Extract key concepts from text using semantic analysis.
    
    Args:
        text_content: The input text to analyze
        concept_threshold: Minimum relevance score for concept inclusion
        
    Returns:
        List of extracted key concepts sorted by relevance
        
    Raises:
        TextProcessingError: If text cannot be processed
    """
```

### Variable Comments (when needed)
```python
# Similarity threshold for semantic clustering (empirically determined)
SEMANTIC_CLUSTERING_THRESHOLD = 0.85

# Maximum number of processing threads (based on CPU cores)
maximum_worker_threads = min(8, os.cpu_count())
```

## Testing Conventions

### Test Class Names
```python
class TestSummarizationEngine(unittest.TestCase):
class TestTextPreprocessor(unittest.TestCase):
class TestConfigurationManager(unittest.TestCase):
```

### Test Method Names
```python
def test_extract_key_concepts_with_valid_text(self):
def test_process_empty_text_raises_error(self):
def test_configuration_loading_from_file(self):
```

## Migration Guidelines

### Phase 1: Critical Inconsistencies
1. Rename hyphenated files to snake_case
2. Fix mixed PascalCase/snake_case in class names
3. Standardize function names to snake_case

### Phase 2: Clarity Improvements  
1. Expand abbreviated names
2. Rename vague/misleading names
3. Ensure consistent terminology

### Phase 3: Documentation Alignment
1. Update all docstrings to match new names
2. Update configuration files
3. Update README and documentation

## Tools and Automation

### Naming Validation Script
A script should be created to automatically check for naming convention violations and suggest corrections.

### IDE Configuration
Configure IDEs to enforce these conventions:
- Enable PEP 8 checking
- Set up naming convention warnings
- Configure auto-formatting rules

## Enforcement

### Code Review Checklist
- [ ] All file names follow snake_case convention
- [ ] Class names use PascalCase
- [ ] Function/method names use snake_case
- [ ] Variables use snake_case
- [ ] Constants use SCREAMING_SNAKE_CASE
- [ ] Names are descriptive and intent-revealing
- [ ] No abbreviations except universally understood ones
- [ ] Consistent terminology throughout

### Automated Checks
- Pre-commit hooks to validate naming conventions
- CI/CD pipeline checks for naming violations
- Regular audits of naming consistency

---

**Remember:** The goal is not perfection, but consistency and clarity. When in doubt, choose the name that makes the code most readable and maintainable for future developers (including yourself).