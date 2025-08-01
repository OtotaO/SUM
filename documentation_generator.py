#!/usr/bin/env python
"""
generate_docs.py - Comprehensive documentation generator for SUM platform

This script scans the SUM codebase and generates detailed, structured documentation
in Markdown, HTML, and PDF formats, with API references, usage examples, and
architectural diagrams.

Design principles:
- Comprehensive coverage (Stroustrup approach)
- Clear organization (Torvalds/van Rossum style)
- Cross-referencing (Knuth hypertext linking)
- Adaptable output formats (Fowler flexibility)
- Automatic discovery (Dijkstra automation)

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import re
import inspect
import importlib
import pkgutil
import logging
import argparse
import time
import json
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('docs_generator')


class DocsGenerator:
    """
    Comprehensive documentation generator for SUM platform.
    
    This class scans Python modules and packages to generate detailed
    documentation with class hierarchies, method signatures, examples,
    and cross-references.
    """
    
    def __init__(self, 
                output_dir: str = 'docs',
                src_dirs: List[str] = None,
                include_private: bool = False,
                generate_diagrams: bool = True,
                output_formats: List[str] = None):
        """
        Initialize the documentation generator.
        
        Args:
            output_dir: Directory to save generated documentation
            src_dirs: List of source code directories to scan
            include_private: Whether to include private methods (starting with _)
            generate_diagrams: Whether to generate architectural diagrams
            output_formats: List of output formats (markdown, html, pdf)
        """
        self.output_dir = output_dir
        self.src_dirs = src_dirs or ['.']
        self.include_private = include_private
        self.generate_diagrams = generate_diagrams
        self.output_formats = output_formats or ['markdown', 'html']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Internal state
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.dependencies = {}
        self.examples = {}
        
        # Module blacklist (don't process these)
        self.module_blacklist = set(['__pycache__', 'setup', 'tests', 'test_'])
        
        logger.info(f"DocsGenerator initialized: {', '.join(self.src_dirs)} -> {output_dir}")
    
    def scan_codebase(self) -> None:
        """Scan the codebase to discover all modules, classes, and functions."""
        logger.info("Scanning codebase...")
        start_time = time.time()
        
        # Add src_dirs to Python path
        for src_dir in self.src_dirs:
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
        
        # Scan each directory
        for src_dir in self.src_dirs:
            self._scan_directory(src_dir)
            
        logger.info(f"Scanning completed in {time.time() - start_time:.2f}s")
        logger.info(f"Found {len(self.modules)} modules, {len(self.classes)} classes, {len(self.functions)} functions")
    
    def _scan_directory(self, directory: str) -> None:
        """
        Scan a directory recursively for Python modules.
        
        Args:
            directory: Directory path to scan
        """
        for root, dirs, files in os.walk(directory):
            # Skip blacklisted directories
            dirs[:] = [d for d in dirs if d not in self.module_blacklist and not d.startswith('.')]
            
            # Process Python files
            for file in files:
                if file.endswith('.py') and not any(file.startswith(bl) for bl in self.module_blacklist):
                    # Get module path
                    file_path = os.path.join(root, file)
                    module_path = self._get_module_path(file_path)
                    
                    if module_path:
                        # Process module
                        self._process_module(module_path)
    
    def _get_module_path(self, file_path: str) -> Optional[str]:
        """
        Convert file path to module import path.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Module import path or None if not importable
        """
        # Get relative path from src_dir
        rel_path = None
        for src_dir in self.src_dirs:
            if file_path.startswith(src_dir):
                rel_path = os.path.relpath(file_path, src_dir)
                break
                
        if not rel_path:
            return None
            
        # Convert to module path
        module_path = rel_path.replace(os.path.sep, '.')
        
        # Remove .py extension
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
            
        return module_path
    
    def _process_module(self, module_path: str) -> None:
        """
        Process a Python module to extract documentation.
        
        Args:
            module_path: Module import path
        """
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Extract module documentation
            module_doc = {
                'name': module_path,
                'docstring': inspect.getdoc(module) or "",
                'classes': {},
                'functions': {},
                'variables': {}
            }
            
            # Process module members
            for name, obj in inspect.getmembers(module):
                # Skip private members
                if name.startswith('_') and not self.include_private:
                    continue
                    
                if inspect.isclass(obj) and self._is_defined_in_module(obj, module):
                    # Process class
                    class_doc = self._process_class(obj)
                    module_doc['classes'][name] = class_doc
                    self.classes[f"{module_path}.{name}"] = class_doc
                    
                elif inspect.isfunction(obj) and self._is_defined_in_module(obj, module):
                    # Process function
                    func_doc = self._process_function(obj)
                    module_doc['functions'][name] = func_doc
                    self.functions[f"{module_path}.{name}"] = func_doc
                    
                elif (
                    not inspect.ismodule(obj) and 
                    not inspect.isbuiltin(obj) and
                    not name.startswith('__')
                ):
                    # Process variable
                    module_doc['variables'][name] = {
                        'value': str(obj),
                        'type': type(obj).__name__
                    }
            
            # Store module documentation
            self.modules[module_path] = module_doc
            
            # Extract dependencies
            self._extract_dependencies(module, module_path)
            
            # Look for examples in docstrings
            self._extract_examples(module_doc, module_path)
            
        except ImportError as e:
            logger.error(f"Error importing module {module_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing module {module_path}: {e}")
    
    def _is_defined_in_module(self, obj: Any, module: Any) -> bool:
        """
        Check if an object is defined in the given module.
        
        Args:
            obj: Object to check
            module: Module to check against
            
        Returns:
            True if the object is defined in the module
        """
        try:
            return obj.__module__ == module.__name__
        except AttributeError:
            return False
    
    def _process_class(self, cls: Any) -> Dict:
        """
        Process a class to extract documentation.
        
        Args:
            cls: Class to process
            
        Returns:
            Class documentation dictionary
        """
        # Extract class documentation
        class_doc = {
            'name': cls.__name__,
            'docstring': inspect.getdoc(cls) or "",
            'bases': [base.__name__ for base in cls.__bases__ if base.__name__ != 'object'],
            'methods': {},
            'properties': {},
            'attributes': {}
        }
        
        # Process class members
        for name, obj in inspect.getmembers(cls):
            # Skip private members
            if name.startswith('_') and not self.include_private and name != '__init__':
                continue
                
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                # Process method
                method_doc = self._process_function(obj)
                class_doc['methods'][name] = method_doc
                
            elif isinstance(obj, property):
                # Process property
                class_doc['properties'][name] = {
                    'docstring': inspect.getdoc(obj) or "",
                    'type': self._get_type_hints(obj)
                }
        
        # Extract class attributes from __init__ method if available
        if hasattr(cls, '__init__'):
            init_method = cls.__init__
            init_src = inspect.getsource(init_method)
            
            # Find self.attr assignments
            attr_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            for match in re.finditer(attr_pattern, init_src):
                attr_name = match.group(1)
                class_doc['attributes'][attr_name] = {
                    'docstring': "",  # No easy way to get attribute docstrings
                    'type': 'Unknown'
                }
        
        return class_doc
    
    def _process_function(self, func: Any) -> Dict:
        """
        Process a function to extract documentation.
        
        Args:
            func: Function to process
            
        Returns:
            Function documentation dictionary
        """
        # Get source code
        try:
            source = inspect.getsource(func)
        except (OSError, IOError):
            source = "Source code not available"
            
        # Get signature
        try:
            signature = str(inspect.signature(func))
        except (ValueError, TypeError):
            signature = "(unknown)"
            
        # Extract function documentation
        func_doc = {
            'name': func.__name__,
            'docstring': inspect.getdoc(func) or "",
            'signature': signature,
            'source': source,
            'parameters': {},
            'returns': {},
            'raises': {}
        }
        
        # Extract parameter and return type hints
        type_hints = self._get_type_hints(func)
        func_doc['type_hints'] = type_hints
        
        # Extract information from docstring
        if func_doc['docstring']:
            # Parse parameters
            param_pattern = r'Args:\s+([^:]+):\s+([^\n]+)'
            for match in re.finditer(param_pattern, func_doc['docstring']):
                param_name = match.group(1).strip()
                param_desc = match.group(2).strip()
                func_doc['parameters'][param_name] = {
                    'description': param_desc,
                    'type': type_hints.get(param_name, 'Unknown')
                }
                
            # Parse returns
            returns_pattern = r'Returns:\s+([^\n]+)'
            returns_match = re.search(returns_pattern, func_doc['docstring'])
            if returns_match:
                func_doc['returns'] = {
                    'description': returns_match.group(1).strip(),
                    'type': type_hints.get('return', 'Unknown')
                }
                
            # Parse raises
            raises_pattern = r'Raises:\s+([^:]+):\s+([^\n]+)'
            for match in re.finditer(raises_pattern, func_doc['docstring']):
                exception_type = match.group(1).strip()
                exception_desc = match.group(2).strip()
                func_doc['raises'][exception_type] = exception_desc
        
        return func_doc
    
    def _get_type_hints(self, obj: Any) -> Dict:
        """
        Get type hints for a function or class.
        
        Args:
            obj: Object to get type hints for
            
        Returns:
            Dictionary of parameter/attribute names to type hints
        """
        try:
            return {
                name: str(hint)
                for name, hint in inspect.get_annotations(obj).items()
            }
        except (ValueError, TypeError, AttributeError):
            return {}
    
    def _extract_dependencies(self, module: Any, module_path: str) -> None:
        """
        Extract module dependencies from import statements.
        
        Args:
            module: Module object
            module_path: Module import path
        """
        try:
            # Get module source
            source = inspect.getsource(module)
            
            # Find import statements
            import_patterns = [
                r'^\s*import\s+([^\n]+)',  # import x, y, z
                r'^\s*from\s+([^\s]+)\s+import\s+([^\n]+)'  # from x import y, z
            ]
            
            dependencies = set()
            
            for pattern in import_patterns:
                for match in re.finditer(pattern, source, re.MULTILINE):
                    if pattern.startswith(r'^\s*from'):
                        # from x import y, z
                        from_module = match.group(1)
                        dependencies.add(from_module)
                    else:
                        # import x, y, z
                        modules = match.group(1).split(',')
                        for mod in modules:
                            # Handle 'import x as y'
                            base_module = mod.split('as')[0].strip()
                            dependencies.add(base_module)
            
            # Store dependencies
            self.dependencies[module_path] = list(dependencies)
            
        except (OSError, IOError, TypeError):
            pass
    
    def _extract_examples(self, module_doc: Dict, module_path: str) -> None:
        """
        Extract code examples from docstrings.
        
        Args:
            module_doc: Module documentation dictionary
            module_path: Module import path
        """
        examples = []
        
        # Check module docstring
        module_examples = self._find_examples_in_docstring(module_doc['docstring'], 'module')
        examples.extend(module_examples)
        
        # Check class docstrings
        for class_name, class_doc in module_doc['classes'].items():
            class_examples = self._find_examples_in_docstring(
                class_doc['docstring'], 
                f"class {class_name}"
            )
            examples.extend(class_examples)
            
            # Check method docstrings
            for method_name, method_doc in class_doc['methods'].items():
                method_examples = self._find_examples_in_docstring(
                    method_doc['docstring'],
                    f"method {class_name}.{method_name}"
                )
                examples.extend(method_examples)
        
        # Check function docstrings
        for func_name, func_doc in module_doc['functions'].items():
            func_examples = self._find_examples_in_docstring(
                func_doc['docstring'],
                f"function {func_name}"
            )
            examples.extend(func_examples)
        
        # Store examples
        if examples:
            self.examples[module_path] = examples
    
    def _find_examples_in_docstring(self, docstring: str, context: str) -> List[Dict]:
        """
        Find code examples in a docstring.
        
        Args:
            docstring: Docstring to search
            context: Context for the example (module, class, method, function)
            
        Returns:
            List of example dictionaries
        """
        if not docstring:
            return []
            
        examples = []
        
        # Find examples in code blocks
        code_blocks = []
        in_block = False
        current_block = []
        block_language = None
        
        for line in docstring.split('\n'):
            if line.strip().startswith('```'):
                if in_block:
                    # End of block
                    code_blocks.append((block_language, '\n'.join(current_block)))
                    in_block = False
                    current_block = []
                    block_language = None
                else:
                    # Start of block
                    in_block = True
                    current_block = []
                    # Try to get language from ```python
                    block_language = line.strip()[3:].strip() or None
            elif in_block:
                current_block.append(line)
        
        # Process found code blocks
        for language, code in code_blocks:
            if language and language.lower() in ['python', 'py']:
                examples.append({
                    'context': context,
                    'code': code,
                    'language': 'python'
                })
        
        return examples
    
    def generate_documentation(self) -> None:
        """Generate documentation in all requested formats."""
        logger.info("Generating documentation...")
        start_time = time.time()
        
        # Ensure we have scanned the codebase
        if not self.modules:
            self.scan_codebase()
            
        # Generate documentation in each format
        for output_format in self.output_formats:
            if output_format == 'markdown':
                self._generate_markdown_docs()
            elif output_format == 'html':
                self._generate_html_docs()
            elif output_format == 'pdf':
                self._generate_pdf_docs()
                
        logger.info(f"Documentation generation completed in {time.time() - start_time:.2f}s")
    
    def _generate_markdown_docs(self) -> None:
        """Generate Markdown documentation."""
        logger.info("Generating Markdown documentation...")
        
        # Create output directory
        md_dir = os.path.join(self.output_dir, 'markdown')
        os.makedirs(md_dir, exist_ok=True)
        
        # Generate README
        self._generate_markdown_readme(md_dir)
        
        # Generate module documentation
        self._generate_markdown_modules(md_dir)
        
        # Generate class index
        self._generate_markdown_class_index(md_dir)
        
        # Generate function index
        self._generate_markdown_function_index(md_dir)
        
        # Generate examples
        self._generate_markdown_examples(md_dir)
        
        # Generate dependency graph
        if self.generate_diagrams:
            self._generate_markdown_dependency_graph(md_dir)
            
        logger.info(f"Markdown documentation saved to {md_dir}")
    
    def _generate_markdown_readme(self, output_dir: str) -> None:
        """
        Generate the main README in Markdown.
        
        Args:
            output_dir: Output directory
        """
        content = "# SUM - The Ultimate Knowledge Distillation Platform\n\n"
        
        # Add description
        content += "## Overview\n\n"
        content += "SUM is a knowledge distillation platform that harnesses the power of AI, NLP, and ML to extract, analyze, and present insights from vast datasets in a structured, concise, and engaging manner.\n\n"
        
        # Add key features
        content += "## Key Features\n\n"
        content += "- Multi-level summarization (tags, sentences, paragraphs)\n"
        content += "- Interactive analysis with user feedback\n"
        content += "- Temporal analysis for tracking concept and sentiment changes\n"
        content += "- Topic modeling for cross-document analysis\n"
        content += "- Knowledge Graph construction and visualization\n"
        content += "- Multi-lingual support with language detection and translation\n\n"
        
        # Add module index
        content += "## Module Index\n\n"
        
        for module_path in sorted(self.modules.keys()):
            module_doc = self.modules[module_path]
            module_name = module_path.split('.')[-1]
            
            # Create clean module description
            description = module_doc['docstring'].split('\n\n')[0] if module_doc['docstring'] else "No description available"
            
            content += f"- [{module_name}](modules/{module_path.replace('.', '_')}.md): {description}\n"
        
        # Add class index preview
        content += "\n## Key Classes\n\n"
        
        important_classes = ["SimpleSUM", "AdvancedSUM", "TopicModeler", "KnowledgeGraph", "DataLoader"]
        
        for class_name in important_classes:
            found = False
            for class_path, class_doc in self.classes.items():
                if class_doc['name'] == class_name:
                    module_path = class_path.rsplit('.', 1)[0]
                    module_file = f"modules/{module_path.replace('.', '_')}.md"
                    
                    # Create clean class description
                    description = class_doc['docstring'].split('\n\n')[0] if class_doc['docstring'] else "No description available"
                    
                    content += f"- [{class_name}]({module_file}#{class_name.lower()}): {description}\n"
                    found = True
                    break
                    
            if not found:
                content += f"- {class_name}: Class not found in codebase\n"
        
        # Add usage examples preview
        content += "\n## Usage Examples\n\n"
        content += "See the [Examples](examples.md) page for code examples.\n\n"
        
        # Add documentation structure
        content += "## Documentation Structure\n\n"
        content += "- [Module Index](module_index.md): Index of all modules\n"
        content += "- [Class Index](class_index.md): Index of all classes\n"
        content += "- [Function Index](function_index.md): Index of all functions\n"
        content += "- [Examples](examples.md): Code examples\n"
        
        if self.generate_diagrams:
            content += "- [Dependency Graph](dependency_graph.md): Module dependency graph\n"
        
        # Write to file
        with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_markdown_modules(self, output_dir: str) -> None:
        """
        Generate Markdown documentation for each module.
        
        Args:
            output_dir: Output directory
        """
        # Create modules directory
        modules_dir = os.path.join(output_dir, 'modules')
        os.makedirs(modules_dir, exist_ok=True)
        
        # Generate module index
        module_index = "# Module Index\n\n"
        
        # Process each module
        for module_path in sorted(self.modules.keys()):
            module_doc = self.modules[module_path]
            module_file = f"{module_path.replace('.', '_')}.md"
            
            # Add to index
            description = module_doc['docstring'].split('\n\n')[0] if module_doc['docstring'] else "No description available"
            module_index += f"- [{module_path}]({module_file}): {description}\n"
            
            # Generate module documentation
            self._generate_markdown_module(module_path, module_doc, modules_dir)
        
        # Write module index
        with open(os.path.join(output_dir, 'module_index.md'), 'w', encoding='utf-8') as f:
            f.write(module_index)
    
    def _generate_markdown_module(self, module_path: str, module_doc: Dict, output_dir: str) -> None:
        """
        Generate Markdown documentation for a single module.
        
        Args:
            module_path: Module import path
            module_doc: Module documentation dictionary
            output_dir: Output directory
        """
        content = f"# Module `{module_path}`\n\n"
        
        # Add description
        if module_doc['docstring']:
            content += f"{module_doc['docstring']}\n\n"
        else:
            content += "No module description available.\n\n"
        
        # Add dependencies
        if module_path in self.dependencies and self.dependencies[module_path]:
            content += "## Dependencies\n\n"
            for dependency in sorted(self.dependencies[module_path]):
                if dependency in self.modules:
                    content += f"- [`{dependency}`]({dependency.replace('.', '_')}.md)\n"
                else:
                    content += f"- `{dependency}`\n"
            content += "\n"
        
        # Add classes
        if module_doc['classes']:
            content += "## Classes\n\n"
            for class_name, class_doc in sorted(module_doc['classes'].items()):
                description = class_doc['docstring'].split('\n\n')[0] if class_doc['docstring'] else "No description available"
                content += f"- [`{class_name}`](#{class_name.lower()}): {description}\n"
            content += "\n"
            
            # Add detailed class documentation
            for class_name, class_doc in sorted(module_doc['classes'].items()):
                content += self._generate_markdown_class(class_name, class_doc)
        
        # Add functions
        if module_doc['functions']:
            content += "## Functions\n\n"
            for func_name, func_doc in sorted(module_doc['functions'].items()):
                description = func_doc['docstring'].split('\n\n')[0] if func_doc['docstring'] else "No description available"
                content += f"- [`{func_name}`](#{func_name.lower()}): {description}\n"
            content += "\n"
            
            # Add detailed function documentation
            for func_name, func_doc in sorted(module_doc['functions'].items()):
                content += self._generate_markdown_function(func_name, func_doc)
        
        # Add variables
        if module_doc['variables']:
            content += "## Variables\n\n"
            content += "| Name | Type | Value |\n"
            content += "|------|------|-------|\n"
            
            for var_name, var_info in sorted(module_doc['variables'].items()):
                content += f"| `{var_name}` | `{var_info['type']}` | `{var_info['value']}` |\n"
            
            content += "\n"
        
        # Add examples
        if module_path in self.examples:
            content += "## Examples\n\n"
            for i, example in enumerate(self.examples[module_path]):
                content += f"### Example {i+1}: {example['context']}\n\n"
                content += f"```{example['language']}\n{example['code']}\n```\n\n"
        
        # Write to file
        with open(os.path.join(output_dir, f"{module_path.replace('.', '_')}.md"), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_markdown_class(self, class_name: str, class_doc: Dict) -> str:
        """
        Generate Markdown documentation for a class.
        
        Args:
            class_name: Class name
            class_doc: Class documentation dictionary
            
        Returns:
            Markdown content for the class
        """
        content = f"### Class `{class_name}`\n\n"
        
        # Add inheritance
        if class_doc['bases']:
            content += "**Inheritance:** "
            content += ", ".join([f"`{base}`" for base in class_doc['bases']])
            content += "\n\n"
        
        # Add description
        if class_doc['docstring']:
            content += f"{class_doc['docstring']}\n\n"
        else:
            content += "No class description available.\n\n"
        
        # Add attributes
        if class_doc['attributes']:
            content += "#### Attributes\n\n"
            content += "| Name | Type | Description |\n"
            content += "|------|------|-------------|\n"
            
            for attr_name, attr_info in sorted(class_doc['attributes'].items()):
                attr_type = attr_info['type']
                attr_desc = attr_info['docstring'] or "No description available"
                content += f"| `{attr_name}` | `{attr_type}` | {attr_desc} |\n"
            
            content += "\n"
        
        # Add properties
        if class_doc['properties']:
            content += "#### Properties\n\n"
            content += "| Name | Type | Description |\n"
            content += "|------|------|-------------|\n"
            
            for prop_name, prop_info in sorted(class_doc['properties'].items()):
                prop_type = prop_info['type'].get('return', 'Unknown')
                prop_desc = prop_info['docstring'] or "No description available"
                content += f"| `{prop_name}` | `{prop_type}` | {prop_desc} |\n"
            
            content += "\n"
        
        # Add methods
        if class_doc['methods']:
            content += "#### Methods\n\n"
            for method_name, method_doc in sorted(class_doc['methods'].items()):
                description = method_doc['docstring'].split('\n\n')[0] if method_doc['docstring'] else "No description available"
                content += f"- [`{method_name}{method_doc['signature']}`](#{class_name.lower()}-{method_name.lower()}): {description}\n"
            content += "\n"
            
            # Add detailed method documentation
            for method_name, method_doc in sorted(class_doc['methods'].items()):
                content += self._generate_markdown_method(class_name, method_name, method_doc)
        
        return content
    
    def _generate_markdown_method(self, class_name: str, method_name: str, method_doc: Dict) -> str:
        """
        Generate Markdown documentation for a method.
        
        Args:
            class_name: Class name
            method_name: Method name
            method_doc: Method documentation dictionary
            
        Returns:
            Markdown content for the method
        """
        content = f"##### `{class_name}.{method_name}{method_doc['signature']}`\n\n"
        
        # Add description
        if method_doc['docstring']:
            content += f"{method_doc['docstring']}\n\n"
        else:
            content += "No method description available.\n\n"
        
        # Add parameters
        if method_doc['parameters']:
            content += "**Parameters:**\n\n"
            content += "| Name | Type | Description |\n"
            content += "|------|------|-------------|\n"
            
            for param_name, param_info in sorted(method_doc['parameters'].items()):
                param_type = param_info['type']
                param_desc = param_info['description']
                content += f"| `{param_name}` | `{param_type}` | {param_desc} |\n"
            
            content += "\n"
        
        # Add returns
        if method_doc['returns']:
            content += "**Returns:**\n\n"
            content += f"- Type: `{method_doc['returns']['type']}`\n"
            content += f"- Description: {method_doc['returns']['description']}\n\n"
        
        # Add raises
        if method_doc['raises']:
            content += "**Raises:**\n\n"
            for exception_type, exception_desc in sorted(method_doc['raises'].items()):
                content += f"- `{exception_type}`: {exception_desc}\n"
            content += "\n"
        
        return content
    
    def _generate_markdown_function(self, func_name: str, func_doc: Dict) -> str:
        """
        Generate Markdown documentation for a function.
        
        Args:
            func_name: Function name
            func_doc: Function documentation dictionary
            
        Returns:
            Markdown content for the function
        """
        content = f"### Function `{func_name}{func_doc['signature']}`\n\n"
        
        # Add description
        if func_doc['docstring']:
            content += f"{func_doc['docstring']}\n\n"
        else:
            content += "No function description available.\n\n"
        
        # Add parameters
        if func_doc['parameters']:
            content += "**Parameters:**\n\n"
            content += "| Name | Type | Description |\n"
            content += "|------|------|-------------|\n"
            
            for param_name, param_info in sorted(func_doc['parameters'].items()):
                param_type = param_info['type']
                param_desc = param_info['description']
                content += f"| `{param_name}` | `{param_type}` | {param_desc} |\n"
            
            content += "\n"
        
        # Add returns
        if func_doc['returns']:
            content += "**Returns:**\n\n"
            content += f"- Type: `{func_doc['returns']['type']}`\n"
            content += f"- Description: {func_doc['returns']['description']}\n\n"
        
        # Add raises
        if func_doc['raises']:
            content += "**Raises:**\n\n"
            for exception_type, exception_desc in sorted(func_doc['raises'].items()):
                content += f"- `{exception_type}`: {exception_desc}\n"
            content += "\n"
        
        return content
    
    def _generate_markdown_class_index(self, output_dir: str) -> None:
        """
        Generate a Markdown index of all classes.
        
        Args:
            output_dir: Output directory
        """
        content = "# Class Index\n\n"
        
        # Group classes by module
        module_classes = {}
        
        for class_path, class_doc in self.classes.items():
            module_path = class_path.rsplit('.', 1)[0]
            class_name = class_doc['name']
            
            if module_path not in module_classes:
                module_classes[module_path] = []
                
            module_classes[module_path].append((class_name, class_doc))
        
        # Generate index
        for module_path in sorted(module_classes.keys()):
            content += f"## Module: `{module_path}`\n\n"
            
            for class_name, class_doc in sorted(module_classes[module_path]):
                module_file = f"modules/{module_path.replace('.', '_')}.md"
                description = class_doc['docstring'].split('\n\n')[0] if class_doc['docstring'] else "No description available"
                
                content += f"- [`{class_name}`]({module_file}#{class_name.lower()}): {description}\n"
            
            content += "\n"
        
        # Write to file
        with open(os.path.join(output_dir, 'class_index.md'), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_markdown_function_index(self, output_dir: str) -> None:
        """
        Generate a Markdown index of all functions.
        
        Args:
            output_dir: Output directory
        """
        content = "# Function Index\n\n"
        
        # Group functions by module
        module_functions = {}
        
        for func_path, func_doc in self.functions.items():
            module_path = func_path.rsplit('.', 1)[0]
            func_name = func_doc['name']
            
            if module_path not in module_functions:
                module_functions[module_path] = []
                
            module_functions[module_path].append((func_name, func_doc))
        
        # Generate index
        for module_path in sorted(module_functions.keys()):
            content += f"## Module: `{module_path}`\n\n"
            
            for func_name, func_doc in sorted(module_functions[module_path]):
                module_file = f"modules/{module_path.replace('.', '_')}.md"
                description = func_doc['docstring'].split('\n\n')[0] if func_doc['docstring'] else "No description available"
                
                content += f"- [`{func_name}`]({module_file}#{func_name.lower()}): {description}\n"
            
            content += "\n"
        
        # Write to file
        with open(os.path.join(output_dir, 'function_index.md'), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_markdown_examples(self, output_dir: str) -> None:
        """
        Generate a Markdown page with all code examples.
        
        Args:
            output_dir: Output directory
        """
        content = "# Code Examples\n\n"
        
        if not self.examples:
            content += "No code examples found in the codebase.\n"
        else:
            # Group examples by module
            for module_path in sorted(self.examples.keys()):
                content += f"## Module: `{module_path}`\n\n"
                
                for i, example in enumerate(self.examples[module_path]):
                    content += f"### Example {i+1}: {example['context']}\n\n"
                    content += f"```{example['language']}\n{example['code']}\n```\n\n"
        
        # Write to file
        with open(os.path.join(output_dir, 'examples.md'), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_markdown_dependency_graph(self, output_dir: str) -> None:
        """
        Generate a Markdown page with module dependency graph.
        
        Args:
            output_dir: Output directory
        """
        content = "# Module Dependency Graph\n\n"
        
        try:
            # Try to generate graph using Mermaid
            content += "```mermaid\ngraph TD\n"
            
            # Add nodes
            for module_path in sorted(self.modules.keys()):
                node_id = module_path.replace('.', '_')
                module_name = module_path.split('.')[-1]
                content += f"    {node_id}[{module_name}]\n"
            
            # Add edges
            for module_path, dependencies in self.dependencies.items():
                if not dependencies:
                    continue
                    
                source_id = module_path.replace('.', '_')
                
                for dependency in dependencies:
                    # Only include dependencies that are in our modules
                    if dependency in self.modules:
                        target_id = dependency.replace('.', '_')
                        content += f"    {source_id} --> {target_id}\n"
            
            content += "```\n\n"
            
            # Add module list
            content += "## Module List\n\n"
            content += "| Module | Description |\n"
            content += "|--------|-------------|\n"
            
            for module_path in sorted(self.modules.keys()):
                module_doc = self.modules[module_path]
                description = module_doc['docstring'].split('\n\n')[0] if module_doc['docstring'] else "No description available"
                
                module_file = f"modules/{module_path.replace('.', '_')}.md"
                content += f"| [`{module_path}`]({module_file}) | {description} |\n"
            
        except Exception as e:
            content += f"Error generating dependency graph: {e}\n"
        
        # Write to file
        with open(os.path.join(output_dir, 'dependency_graph.md'), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_html_docs(self) -> None:
        """Generate HTML documentation."""
        logger.info("Generating HTML documentation...")
        
        try:
            # Check if we can use markdown2html converter
            import markdown
            
            # Create output directory
            html_dir = os.path.join(self.output_dir, 'html')
            os.makedirs(html_dir, exist_ok=True)
            
            # Create CSS file
            self._create_html_css(html_dir)
            
            # Convert Markdown to HTML
            md_dir = os.path.join(self.output_dir, 'markdown')
            
            # Ensure Markdown docs exist
            if not os.path.exists(md_dir):
                self._generate_markdown_docs()
            
            # Create index.html from README.md
            readme_path = os.path.join(md_dir, 'README.md')
            if os.path.exists(readme_path):
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Convert relative links to HTML
                readme_content = re.sub(
                    r'\]\(([^)]+)\.md\)',
                    r'](\1.html)',
                    readme_content
                )
                
                # Convert to HTML
                html_content = markdown.markdown(
                    readme_content,
                    extensions=['tables', 'fenced_code', 'codehilite']
                )
                
                # Wrap in HTML template
                html_content = self._wrap_html_template(html_content, "SUM Documentation")
                
                # Write to file
                with open(os.path.join(html_dir, 'index.html'), 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            # Convert all Markdown files
            for root, _, files in os.walk(md_dir):
                for file in files:
                    if file.endswith('.md') and file != 'README.md':
                        # Get relative path
                        rel_path = os.path.relpath(os.path.join(root, file), md_dir)
                        
                        # Read Markdown content
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            md_content = f.read()
                        
                        # Convert relative links to HTML
                        md_content = re.sub(
                            r'\]\(([^)]+)\.md\)',
                            r'](\1.html)',
                            md_content
                        )
                        
                        # Convert to HTML
                        html_content = markdown.markdown(
                            md_content,
                            extensions=['tables', 'fenced_code', 'codehilite']
                        )
                        
                        # Wrap in HTML template
                        title = file[:-3].replace('_', '.')
                        html_content = self._wrap_html_template(html_content, f"SUM: {title}")
                        
                        # Create output directory
                        output_subdir = os.path.dirname(os.path.join(html_dir, rel_path))
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        # Write to file
                        output_file = os.path.join(
                            html_dir,
                            os.path.splitext(rel_path)[0] + '.html'
                        )
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
            
            logger.info(f"HTML documentation saved to {html_dir}")
            
        except ImportError:
            logger.error("Could not generate HTML documentation: markdown module not available")
    
    def _create_html_css(self, html_dir: str) -> None:
        """
        Create CSS file for HTML documentation.
        
        Args:
            html_dir: HTML output directory
        """
        css_content = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1100px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        a {
            color: #0366d6;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        pre, code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            background-color: #f6f8fa;
            border-radius: 3px;
        }
        
        pre {
            padding: 16px;
            overflow: auto;
            border: 1px solid #e1e4e8;
        }
        
        code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
        }
        
        pre > code {
            padding: 0;
            background-color: transparent;
        }
        
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #24292e;
        }
        
        h1 {
            font-size: 2em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eaecef;
        }
        
        h2 {
            font-size: 1.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eaecef;
        }
        
        h3 {
            font-size: 1.25em;
        }
        
        h4 {
            font-size: 1em;
        }
        
        h5, h6 {
            font-size: 0.875em;
        }
        
        table {
            border-collapse: collapse;
            margin: 1em 0;
            width: 100%;
            overflow: auto;
        }
        
        table th, table td {
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }
        
        table th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
        
        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }
        
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 260px;
            height: 100%;
            overflow: auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-right: 1px solid #e1e4e8;
        }
        
        .sidebar h3 {
            margin-top: 0;
        }
        
        .sidebar ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .sidebar li {
            margin-bottom: 8px;
        }
        
        .content {
            margin-left: 300px;
            padding: 20px;
        }
        
        @media (max-width: 1100px) {
            .sidebar {
                display: none;
            }
            
            .content {
                margin-left: 0;
            }
        }
        
        .highlight .k { color: #0000aa; } /* Keyword */
        .highlight .s { color: #aa5500; } /* String */
        .highlight .c { color: #888888; font-style: italic; } /* Comment */
        .highlight .o { color: #555555; } /* Operator */
        .highlight .n { color: #000000; } /* Name */
        .highlight .p { color: #555555; } /* Punctuation */
        """
        
        with open(os.path.join(html_dir, 'style.css'), 'w', encoding='utf-8') as f:
            f.write(css_content)
    
    def _wrap_html_template(self, content: str, title: str) -> str:
        """
        Wrap HTML content in a template.
        
        Args:
            content: HTML content
            title: Page title
            
        Returns:
            Complete HTML page
        """
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <link rel="stylesheet" href="{self._get_relative_path(title)}style.css">
        </head>
        <body>
            <div class="content">
                {content}
                <hr>
                <footer>
                    <p>Generated by SUM Documentation Generator</p>
                </footer>
            </div>
        </body>
        </html>
        """
    
    def _get_relative_path(self, title: str) -> str:
        """
        Calculate relative path to root for CSS.
        
        Args:
            title: Page title
            
        Returns:
            Relative path to root directory
        """
        # Count directories in title
        depth = title.count('.')
        
        if 'Module' in title or 'Class' in title or 'Function' in title:
            depth += 1
            
        if depth == 0:
            return ""
        else:
            return "../" * depth
    
    def _generate_pdf_docs(self) -> None:
        """Generate PDF documentation."""
        logger.info("Generating PDF documentation...")
        
        try:
            # Check if we can use markdown2pdf converter
            from weasyprint import HTML
            
            # Create output directory
            pdf_dir = os.path.join(self.output_dir, 'pdf')
            os.makedirs(pdf_dir, exist_ok=True)
            
            # Ensure HTML docs exist
            html_dir = os.path.join(self.output_dir, 'html')
            if not os.path.exists(html_dir):
                self._generate_html_docs()
            
            # Convert all HTML files to PDF
            for root, _, files in os.walk(html_dir):
                for file in files:
                    if file.endswith('.html'):
                        # Get relative path
                        rel_path = os.path.relpath(os.path.join(root, file), html_dir)
                        
                        # Create output directory
                        output_subdir = os.path.dirname(os.path.join(pdf_dir, rel_path))
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        # Convert to PDF
                        html_path = os.path.join(root, file)
                        pdf_path = os.path.join(
                            pdf_dir,
                            os.path.splitext(rel_path)[0] + '.pdf'
                        )
                        
                        HTML(filename=html_path).write_pdf(pdf_path)
            
            logger.info(f"PDF documentation saved to {pdf_dir}")
            
        except ImportError:
            logger.error("Could not generate PDF documentation: weasyprint module not available")
    
    def export_json(self, output_path: str = None) -> None:
        """
        Export documentation data as JSON.
        
        Args:
            output_path: Output file path (default: docs/sum_docs.json)
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, 'sum_docs.json')
            
        # Ensure we have scanned the codebase
        if not self.modules:
            self.scan_codebase()
            
        # Prepare data for export
        export_data = {
            'modules': self.modules,
            'dependencies': self.dependencies,
            'examples': self.examples,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Documentation data exported to {output_path}")


def main():
    """Main function to run the documentation generator."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SUM Documentation Generator')
    
    parser.add_argument(
        '--src',
        nargs='+',
        default=['.'],
        help='Source directories to scan (default: current directory)'
    )
    
    parser.add_argument(
        '--output',
        default='docs',
        help='Output directory for documentation (default: docs)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['markdown', 'html', 'pdf'],
        default=['markdown', 'html'],
        help='Output formats (default: markdown, html)'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Include private methods and attributes (starting with _)'
    )
    
    parser.add_argument(
        '--no-diagrams',
        action='store_true',
        help='Disable generation of architectural diagrams'
    )
    
    parser.add_argument(
        '--json',
        help='Export documentation data as JSON to specified file'
    )
    
    args = parser.parse_args()
    
    # Create documentation generator
    generator = DocsGenerator(
        output_dir=args.output,
        src_dirs=args.src,
        include_private=args.private,
        generate_diagrams=not args.no_diagrams,
        output_formats=args.formats
    )
    
    # Generate documentation
    generator.scan_codebase()
    generator.generate_documentation()
    
    # Export JSON if requested
    if args.json:
        generator.export_json(args.json)


if __name__ == '__main__':
    main()
