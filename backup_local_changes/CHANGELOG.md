# SUM Project Changelog

## [Unreleased]

### Added
- Created this changelog file to track project changes
- Added `Utils/nltk_utils.py` for centralized NLTK resource management
- Added `Utils/text_preprocessing.py` for standardized text preprocessing
- Added `CODING_STANDARDS.md` with project coding conventions and file naming standards
- Added `Utils/error_handling.py` for centralized error management
- Added `Utils/config_manager.py` for centralized configuration management
- Added test files for new utilities:
  - `Tests/test_nltk_utils.py`
  - `Tests/test_text_preprocessing.py`
  - `Tests/test_error_handling.py`
  - `Tests/test_config_manager.py`
- Added configuration examples:
  - `examples/simple_config_example.py`
  - `examples/config_example.py`
  - `examples/integrated_config_example.py`
  - `examples/config_integration_example.py`

### Changed
- Updated `Utils/README.md` with documentation for new utility modules
- Updated `examples/README.md` with documentation for configuration examples
- Improved integration between summarization and topic modeling in `Models/summarizer.py`

### Fixed

### Refactored
- Updated `examples/download_nltk_resources.py` to use centralized NLTK utilities
- Clarified inheritance structure between summarization classes in `SUM.py`
- Standardized error handling across all components using `Utils/error_handling.py`
- Unified configuration management using `ConfigManager` in `Utils/config_manager.py`
