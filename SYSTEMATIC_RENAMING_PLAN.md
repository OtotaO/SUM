# Systematic Renaming Plan for SUM Project

This document provides a step-by-step plan for implementing the naming convention changes identified in the SUM project. The plan prioritizes safety and minimizes breaking changes.

## 🎯 Goals

1. **Consistency**: Apply uniform naming conventions across the entire codebase
2. **Clarity**: Make code self-documenting through clear names
3. **Safety**: Minimize disruption to existing functionality
4. **Maintainability**: Create a sustainable naming system for future development

## 📋 Pre-Execution Checklist

- [ ] **Backup**: Create complete backup of the project
- [ ] **Git**: Ensure all changes are committed before starting
- [ ] **Tests**: Run full test suite to establish baseline
- [ ] **Dependencies**: Document all external dependencies that might be affected
- [ ] **Team Notification**: Inform team members of upcoming changes

## 🔄 Execution Phases

### Phase 1: Foundation Preparation (1-2 hours)

#### Step 1.1: Create Working Branch
```bash
git checkout -b naming-convention-updates
git push -u origin naming-convention-updates
```

#### Step 1.2: Run Baseline Tests
```bash
python -m pytest Tests/ -v
python test_SUM.py
python benchmark.py
```

#### Step 1.3: Create Backup
```bash
cp -r /Users/ototao/Github\ Projects/SUM/SUM /Users/ototao/Github\ Projects/SUM/SUM_BACKUP_$(date +%Y%m%d_%H%M%S)
```

### Phase 2: Low-Risk File Renames (2-3 hours)

**Priority**: Start with hyphenated files as they're safest to rename

#### Step 2.1: Rename Hyphenated Files
```bash
# Create rename mapping
RENAMES=(
    "advanced-summarization-engine.py:advanced_summarization_engine.py"
    "advanced-topic-modeler.py:advanced_topic_modeler.py"
    "comprehensive-test-suite.py:comprehensive_test_suite.py"
    "knowledge-graph-visualizer.py:knowledge_graph_visualizer.py"
    "knowledge-graph-web-interface.py:knowledge_graph_web_interface.py"
    "temporal-knowledge-analysis.py:temporal_knowledge_analysis.py"
    "sum-cli-interface.py:sum_cli_interface.py"
    "documentation-generator.py:documentation_generator.py"
    "enhanced-data-loader.py:enhanced_data_loader.py"
)
```

#### Step 2.2: Update Import Statements
For each renamed file, find and update all import references:
```bash
# Example for advanced-summarization-engine.py
grep -r "advanced-summarization-engine" . --include="*.py"
grep -r "from advanced-summarization-engine" . --include="*.py"
```

#### Step 2.3: Test After Each Rename
```bash
python -c "import sys; sys.path.append('.'); import advanced_summarization_engine"
```

### Phase 3: Core Module Renames (3-4 hours)

**Priority**: High-impact changes that affect multiple files

#### Step 3.1: Rename Core Files
| Current | New | Impact Level |
|---------|-----|--------------|
| `SUM.py` | `summarization_engine.py` | HIGH |
| `StreamingEngine.py` | `streaming_engine.py` | MEDIUM |

#### Step 3.2: Update Import Dependencies
Create mapping of all files that import these modules:
```bash
grep -r "from SUM import" . --include="*.py" > sum_imports.txt
grep -r "import SUM" . --include="*.py" >> sum_imports.txt
grep -r "from StreamingEngine import" . --include="*.py" > streaming_imports.txt
```

#### Step 3.3: Systematic Import Updates
For each importing file:
1. Update import statement
2. Update any direct class references
3. Test import works
4. Run related tests

### Phase 4: Class Name Changes (4-5 hours)

**Priority**: Update class names for clarity

#### Step 4.1: Core Class Renames
| Current Class | New Class | Files Affected | Breaking Change Level |
|---------------|-----------|----------------|----------------------|
| `SUM` | `SummarizationEngine` | Multiple | HIGH |
| `SimpleSUM` | `BasicSummarizationEngine` | Multiple | HIGH |
| `MagnumOpusSUM` | `AdvancedSummarizationEngine` | Multiple | HIGH |
| `AdvancedSUM` | `SemanticSummarizationEngine` | 1-2 files | MEDIUM |

#### Step 4.2: Update Class References
For each class rename:
1. **Find all references**:
   ```bash
   grep -r "class SUM" . --include="*.py"
   grep -r "SUM(" . --include="*.py"
   grep -r "isinstance.*SUM" . --include="*.py"
   ```

2. **Update inheritance chains**:
   ```bash
   grep -r "class.*SUM)" . --include="*.py"
   ```

3. **Update instantiation**:
   ```bash
   grep -r "= SUM(" . --include="*.py"
   grep -r "= SimpleSUM(" . --include="*.py"
   ```

#### Step 4.3: Update Tests
- Rename test classes to match new naming
- Update test instantiation code
- Update assertion messages

### Phase 5: Function and Variable Renames (2-3 hours)

**Priority**: Improve code clarity with better naming

#### Step 5.1: Function Renames (High Priority)
Focus on most commonly used functions:

| Current | New | Impact |
|---------|-----|--------|
| `process()` | `process_text_content()` | HIGH |
| `get_data()` | `extract_text_data()` | MEDIUM |
| `init()` | `initialize_components()` | MEDIUM |

#### Step 5.2: Variable Renames (Medium Priority)  
Target most confusing abbreviations:

| Current | New | Context |
|---------|-----|---------|
| `cfg` | `configuration` | Configuration objects |
| `proc` | `processor` | Processing objects |
| `sim` | `similarity` | Similarity calculations |
| `emb` | `embeddings` | Vector embeddings |

#### Step 5.3: Constants Updates
Convert to SCREAMING_SNAKE_CASE:
```python
# Before
default_chunk_size = 1000
max_file_size = 10000000

# After  
DEFAULT_CHUNK_SIZE = 1000
MAXIMUM_FILE_SIZE_BYTES = 10_000_000
```

### Phase 6: Documentation and Configuration Updates (1-2 hours)

#### Step 6.1: Update Documentation Files
- `README.md`
- `INSTALLATION.md`
- `AI_FEATURES.md`
- All markdown files with code examples

#### Step 6.2: Update Configuration Files
- `config.py`
- `requirements.txt` (if any module names changed)
- `setup.py` (if it exists)

#### Step 6.3: Update Comments and Docstrings
```bash
# Find files with old class/function references in comments
grep -r "#.*SUM\|""".*SUM" . --include="*.py"
```

## 🧪 Testing Strategy

### After Each Phase
```bash
# Run basic import tests
python -c "import sys; sys.path.append('.'); import summarization_engine"

# Run unit tests
python -m pytest Tests/ -v

# Run integration tests
python test_trinity_api.py
python test_streaming_api.py

# Run benchmarks
python benchmark.py
```

### Full Test Suite After All Changes
```bash
# Comprehensive testing
python -m pytest Tests/ -v --tb=short
python test_SUM.py -v
python test_streaming_direct.py
python test_streaming_api.py
python test_trinity_api.py
python test_multimodal_system.py
python test_adaptive_system.py
python benchmark.py
```

## 🚨 Rollback Plan

### If Critical Issues Arise

1. **Immediate Rollback**:
   ```bash
   git checkout main
   git branch -D naming-convention-updates
   ```

2. **Restore from Backup**:
   ```bash
   rm -rf /Users/ototao/Github\ Projects/SUM/SUM
   cp -r /Users/ototao/Github\ Projects/SUM/SUM_BACKUP_* /Users/ototao/Github\ Projects/SUM/SUM
   ```

3. **Identify Issues**:
   - Run tests to identify what broke
   - Check import statements
   - Verify configuration files

### Progressive Rollback
If only some changes cause issues:
```bash
git log --oneline naming-convention-updates
git revert <commit-hash>  # Revert specific problematic commits
```

## 📊 Progress Tracking

### Completion Checklist

#### Phase 2: File Renames
- [ ] `advanced-summarization-engine.py` → `advanced_summarization_engine.py`
- [ ] `advanced-topic-modeler.py` → `advanced_topic_modeler.py`
- [ ] `comprehensive-test-suite.py` → `comprehensive_test_suite.py`
- [ ] `knowledge-graph-visualizer.py` → `knowledge_graph_visualizer.py`
- [ ] `knowledge-graph-web-interface.py` → `knowledge_graph_web_interface.py`
- [ ] `temporal-knowledge-analysis.py` → `temporal_knowledge_analysis.py`
- [ ] `sum-cli-interface.py` → `sum_cli_interface.py`
- [ ] `documentation-generator.py` → `documentation_generator.py`
- [ ] `enhanced-data-loader.py` → `enhanced_data_loader.py`

#### Phase 3: Core Module Renames
- [ ] `SUM.py` → `summarization_engine.py`
- [ ] `StreamingEngine.py` → `streaming_engine.py`
- [ ] Update all import statements
- [ ] Test all imports work

#### Phase 4: Class Renames
- [ ] `SUM` → `SummarizationEngine`
- [ ] `SimpleSUM` → `BasicSummarizationEngine`
- [ ] `MagnumOpusSUM` → `AdvancedSummarizationEngine`
- [ ] `AdvancedSUM` → `SemanticSummarizationEngine`
- [ ] Update all references
- [ ] Update tests

#### Phase 5: Function/Variable Renames
- [ ] Update critical function names
- [ ] Update abbreviated variables
- [ ] Update constants to SCREAMING_SNAKE_CASE
- [ ] Update private method names

#### Phase 6: Documentation
- [ ] Update README.md
- [ ] Update all .md files
- [ ] Update docstrings
- [ ] Update comments

## ⏱️ Time Estimates

| Phase | Estimated Time | Dependencies |
|-------|----------------|--------------|
| Phase 1: Preparation | 1-2 hours | None |
| Phase 2: File Renames | 2-3 hours | Phase 1 |
| Phase 3: Core Modules | 3-4 hours | Phase 2 |
| Phase 4: Class Names | 4-5 hours | Phase 3 |
| Phase 5: Functions/Variables | 2-3 hours | Phase 4 |
| Phase 6: Documentation | 1-2 hours | Phase 5 |
| **Total** | **13-19 hours** | Sequential |

## 🤝 Team Coordination

### Communication Plan
1. **Start**: Notify team of renaming project start
2. **Milestones**: Update after each phase completion  
3. **Issues**: Immediate notification if rollback needed
4. **Completion**: Full summary of changes made

### Code Review Process
1. Create pull request after Phase 4 (major changes)
2. Get team review before final phases
3. Final review before merging to main

---

**Remember**: Take it slow, test frequently, and don't hesitate to rollback if something breaks. The goal is improvement, not perfection in one attempt.