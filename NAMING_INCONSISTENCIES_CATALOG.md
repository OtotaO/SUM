# SUM Project Naming Inconsistencies Catalog

This document catalogs all naming inconsistencies found in the SUM project codebase, organized by severity and type.

## 🔴 Critical Issues (Must Fix First)

### File Naming Inconsistencies

#### Hyphenated Files (Should be snake_case)
| Current Name | Correct Name | Location |
|-------------|--------------|----------|
| `advanced-summarization-engine.py` | `advanced_summarization_engine.py` | `/SUM/` |
| `advanced-topic-modeler.py` | `advanced_topic_modeler.py` | `/SUM/` |
| `comprehensive-test-suite.py` | `comprehensive_test_suite.py` | `/SUM/` |
| `knowledge-graph-visualizer.py` | `knowledge_graph_visualizer.py` | `/SUM/` |
| `knowledge-graph-web-interface.py` | `knowledge_graph_web_interface.py` | `/SUM/` |
| `temporal-knowledge-analysis.py` | `temporal_knowledge_analysis.py` | `/SUM/` |
| `sum-cli-interface.py` | `sum_cli_interface.py` | `/SUM/` |
| `documentation-generator.py` | `documentation_generator.py` | `/SUM/` |
| `enhanced-data-loader.py` | `enhanced_data_loader.py` | `/SUM/` |

#### PascalCase Files (Should be snake_case)
| Current Name | Correct Name | Location |
|-------------|--------------|----------|
| `StreamingEngine.py` | `streaming_engine.py` | `/SUM/` |
| `SUM.py` | `summarization_engine.py` | `/SUM/` |

### Class Naming Issues

#### Vague/Abbreviated Class Names
| Current Name | Suggested Name | File Location | Reason |
|-------------|----------------|---------------|---------|
| `SUM` | `SummarizationEngine` | `SUM.py` | Acronym without expansion |
| `SimpleSUM` | `BasicSummarizationEngine` | `SUM.py` | More descriptive |
| `MagnumOpusSUM` | `AdvancedSummarizationEngine` | `SUM.py` | Clearer purpose |
| `AdvancedSUM` | `SemanticSummarizationEngine` | `advanced-summarization-engine.py` | More specific |

#### Mixed Case Issues
| Current Name | Correct Name | Location |
|-------------|--------------|----------|
| `AIEnhancedInterface` | `AiEnhancedInterface` | Multiple files | Acronym should be treated as word |

## 🟡 Medium Priority Issues

### Function Naming Inconsistencies

#### camelCase Functions (Should be snake_case)
| Current Name | Correct Name | File Location |
|-------------|--------------|---------------|
| Various Flask routes | Need snake_case conversion | `main_enhanced.py` |

#### Vague Function Names
| Current Name | Suggested Name | File Location | Reason |
|-------------|----------------|---------------|---------|
| `process()` | `process_text_content()` | Multiple files | More specific |
| `get_data()` | `extract_text_data()` | Multiple files | Intent unclear |
| `init()` | `initialize_components()` | Multiple files | What is being initialized? |

### Variable Naming Issues

#### Abbreviated Variables
| Current Name | Suggested Name | File Location | Context |
|-------------|----------------|---------------|---------|
| `cfg` | `configuration` | Multiple files | Configuration object |
| `proc` | `processor` | Multiple files | Processing object |
| `sim` | `similarity` | Multiple files | Similarity values |
| `emb` | `embeddings` | Multiple files | Vector embeddings |
| `doc` | `document` | Multiple files | Document object |
| `res` | `result` | Multiple files | Processing result |

#### Single Letter Variables (Outside Loops)
| Current Name | Suggested Name | File Location | Context |
|-------------|----------------|---------------|---------|
| `e` | `exception` | Multiple files | Exception handling |
| `f` | `file_handle` | Multiple files | File operations |
| `d` | `data_item` | Multiple files | Data processing |

## 🟢 Low Priority (Style Improvements)

### Constants Not in SCREAMING_SNAKE_CASE

#### Configuration Values
| Current Name | Correct Name | File Location |
|-------------|--------------|---------------|
| `default_chunk_size` | `DEFAULT_CHUNK_SIZE` | Multiple files |
| `max_file_size` | `MAXIMUM_FILE_SIZE_BYTES` | Multiple files |
| `supported_formats` | `SUPPORTED_FILE_FORMATS` | Multiple files |

### Inconsistent Terminology

#### Mixed Terms for Same Concept
| Term 1 | Term 2 | Standardize To | Files Affected |
|--------|--------|----------------|----------------|
| "summarize" | "compress" | "summarize" | Multiple |
| "process" | "analyze" | "process" (for data flow) | Multiple |
| "extract" | "get" | "extract" (for content) | Multiple |
| "config" | "configuration" | "configuration" | Multiple |

## 📊 Statistics Summary

### File Naming Issues
- **Hyphenated files:** 9 files
- **PascalCase files:** 2 files
- **Total file renames needed:** 11 files

### Class Naming Issues
- **Vague/abbreviated names:** 4 classes
- **Mixed case issues:** Multiple instances
- **Total class renames needed:** ~10 classes

### Function/Variable Issues
- **Abbreviated variables:** ~50+ instances
- **Single letter variables:** ~30+ instances
- **Vague function names:** ~20+ instances

## 🎯 Impact Analysis

### High Impact Changes
1. **File renames** - Will require import statement updates
2. **Main class renames** - Will require extensive refactoring
3. **Public API changes** - May break external integrations

### Medium Impact Changes
1. **Private method renames** - Internal refactoring only
2. **Variable renames** - Localized changes
3. **Function renames** - May affect tests

### Low Impact Changes
1. **Comment updates** - Documentation only
2. **Constant renames** - Usually localized
3. **Type hint improvements** - Clarity enhancement

## 🔄 Dependencies and Risk Assessment

### High Risk Renames
| Item | Risk Level | Reason | Mitigation |
|------|------------|--------|------------|
| `SUM.py` → `summarization_engine.py` | High | Core module name change | Update all imports systematically |
| `StreamingEngine.py` → `streaming_engine.py` | Medium | May break imports | Search and replace all references |
| Class `SUM` → `SummarizationEngine` | High | Base class change | Inheritance chain updates needed |

### Circular Dependencies
- No circular dependencies found in naming changes
- Most renames are independent of each other

## 📋 Recommended Rename Order

### Phase 1: File Names (Low Risk)
1. Rename hyphenated files to snake_case
2. Update corresponding imports
3. Test basic functionality

### Phase 2: Class Names (Medium Risk)  
1. Rename vague class names
2. Update inheritance relationships
3. Update instantiation code
4. Run comprehensive tests

### Phase 3: Function/Variable Names (Low Risk)
1. Rename abbreviated variables
2. Rename vague function names
3. Update documentation
4. Final testing

## 🛠️ Tools Needed for Rename

### Automated Tools
- **Rope** (Python refactoring library)
- **sed/awk** for bulk text replacement
- **ripgrep** for finding all references
- **Custom Python scripts** for complex renames

### Manual Review Needed
- Import statements
- Configuration files
- Documentation files
- Test files
- External API references

---

**Next Steps:** Use this catalog to create a systematic renaming plan and automated scripts for implementing these changes safely.