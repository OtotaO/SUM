#!/usr/bin/env python3
"""
Naming Convention Validator for SUM Project
===========================================

This script validates that the SUM project follows the established naming conventions
and reports any violations found.

Author: Claude
License: Apache License 2.0
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NamingConventionValidator:
    """
    Validates naming conventions across the SUM project.
    
    Checks for:
    - File naming conventions (snake_case)
    - Class naming conventions (PascalCase)
    - Function/method naming conventions (snake_case)
    - Variable naming conventions (snake_case)
    - Constant naming conventions (SCREAMING_SNAKE_CASE)
    """
    
    def __init__(self, project_root: str):
        """
        Initialize validator with project root.
        
        Args:
            project_root: Path to the SUM project root directory
        """
        self.project_root = Path(project_root).resolve()
        self.violations = []
        
        # Naming patterns
        self.snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        self.pascal_case_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.screaming_snake_case_pattern = re.compile(r'^[A-Z][A-Z0-9_]*$')
        self.private_method_pattern = re.compile(r'^_[a-z][a-z0-9_]*$')
        
        # Exceptions - these names are allowed to violate conventions
        self.allowed_exceptions = {
            '__init__', '__str__', '__repr__', '__len__', '__iter__',
            '__next__', '__enter__', '__exit__', '__call__', '__getitem__',
            '__setitem__', '__delitem__', '__contains__', '__eq__', '__ne__',
            '__lt__', '__le__', '__gt__', '__ge__', '__hash__', '__bool__',
            'setUp', 'tearDown', 'setUpClass', 'tearDownClass'  # unittest methods
        }
        
        # Known abbreviations that are acceptable
        self.acceptable_abbreviations = {
            'url', 'uri', 'html', 'xml', 'json', 'csv', 'pdf', 'api',
            'http', 'https', 'ftp', 'ssh', 'ssl', 'tls', 'sql', 'id',
            'uid', 'uuid', 'os', 'io', 'ui', 'cli', 'gui'
        }
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        return file_path.name.startswith('test_') or 'test' in file_path.parts
    
    def validate_file_names(self) -> List[Dict[str, Any]]:
        """
        Validate file naming conventions.
        
        Returns:
            List of file naming violations
        """
        violations = []
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            file_name = file_path.stem  # filename without extension
            
            # Skip special files
            if file_name in ['__init__', 'setup', 'conftest']:
                continue
            
            # Check for snake_case
            if not self.snake_case_pattern.match(file_name):
                # Check for specific problematic patterns
                if '-' in file_name:
                    violation_type = "hyphenated_filename"
                    suggestion = file_name.replace('-', '_')
                elif any(c.isupper() for c in file_name):
                    violation_type = "non_snake_case_filename"
                    suggestion = self.to_snake_case(file_name)
                else:
                    violation_type = "invalid_filename"
                    suggestion = file_name.lower()
                
                violations.append({
                    'type': violation_type,
                    'file': str(file_path.relative_to(self.project_root)),
                    'current_name': file_name,
                    'suggested_name': suggestion,
                    'severity': 'high'
                })
        
        return violations
    
    def validate_python_code(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Validate Python code naming conventions.
        
        Args:
            file_path: Path to Python file to validate
            
        Returns:
            List of code naming violations
        """
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Walk through AST nodes
            for node in ast.walk(tree):
                violations.extend(self.validate_node(node, file_path))
                
        except SyntaxError as e:
            violations.append({
                'type': 'syntax_error',
                'file': str(file_path.relative_to(self.project_root)),
                'message': f"Cannot parse file: {e}",
                'severity': 'high'
            })
        except Exception as e:
            violations.append({
                'type': 'file_error',
                'file': str(file_path.relative_to(self.project_root)),
                'message': f"Error reading file: {e}",
                'severity': 'medium'
            })
        
        return violations
    
    def validate_node(self, node: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """
        Validate a single AST node for naming conventions.
        
        Args:
            node: AST node to validate
            file_path: Path to the file being validated
            
        Returns:
            List of violations found in this node
        """
        violations = []
        relative_path = str(file_path.relative_to(self.project_root))
        
        # Class definitions
        if isinstance(node, ast.ClassDef):
            if not self.pascal_case_pattern.match(node.name):
                violations.append({
                    'type': 'class_naming',
                    'file': relative_path,
                    'line': node.lineno,
                    'current_name': node.name,
                    'suggested_name': self.to_pascal_case(node.name),
                    'severity': 'high'
                })
        
        # Function definitions
        elif isinstance(node, ast.FunctionDef):
            if node.name not in self.allowed_exceptions:
                # Check for private methods
                if node.name.startswith('_') and not node.name.startswith('__'):
                    if not self.private_method_pattern.match(node.name):
                        violations.append({
                            'type': 'private_method_naming',
                            'file': relative_path,
                            'line': node.lineno,
                            'current_name': node.name,
                            'suggested_name': self.to_snake_case(node.name),
                            'severity': 'medium'
                        })
                # Regular methods/functions
                elif not self.snake_case_pattern.match(node.name):
                    violations.append({
                        'type': 'function_naming',
                        'file': relative_path,
                        'line': node.lineno,
                        'current_name': node.name,
                        'suggested_name': self.to_snake_case(node.name),
                        'severity': 'high'
                    })
        
        # Variable assignments
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    
                    # Skip special variables
                    if name in ['__all__', '__version__', '__author__']:
                        continue
                    
                    # Check if it's a constant (all uppercase)
                    if name.isupper():
                        if not self.screaming_snake_case_pattern.match(name):
                            violations.append({
                                'type': 'constant_naming',
                                'file': relative_path,
                                'line': node.lineno,
                                'current_name': name,
                                'suggested_name': self.to_screaming_snake_case(name),
                                'severity': 'medium'
                            })
                    # Regular variables
                    elif not self.snake_case_pattern.match(name):
                        violations.append({
                            'type': 'variable_naming',
                            'file': relative_path,
                            'line': node.lineno,
                            'current_name': name,
                            'suggested_name': self.to_snake_case(name),
                            'severity': 'medium'
                        })
        
        return violations
    
    def to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Handle camelCase and PascalCase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1)
        
        # Handle hyphens
        s3 = s2.replace('-', '_')
        
        # Convert to lowercase and clean up multiple underscores
        result = re.sub('_+', '_', s3.lower())
        
        # Remove leading/trailing underscores (unless it was intentionally private)
        if not name.startswith('_'):
            result = result.strip('_')
        
        return result
    
    def to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        # Split on underscores and hyphens
        parts = re.split(r'[_-]', name.lower())
        return ''.join(word.capitalize() for word in parts if word)
    
    def to_screaming_snake_case(self, name: str) -> str:
        """Convert name to SCREAMING_SNAKE_CASE."""
        return self.to_snake_case(name).upper()
    
    def check_for_abbreviations(self) -> List[Dict[str, Any]]:
        """
        Check for potentially problematic abbreviations.
        
        Returns:
            List of abbreviation violations
        """
        violations = []
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for short variable names that might be abbreviations
                short_names = re.findall(r'\\b[a-z]{1,3}\\b(?=\\s*=)', content)
                
                for name in set(short_names):
                    if (len(name) <= 3 and 
                        name not in self.acceptable_abbreviations and
                        name not in ['i', 'j', 'k', 'x', 'y', 'z']):  # Common loop vars
                        
                        violations.append({
                            'type': 'potential_abbreviation',
                            'file': str(file_path.relative_to(self.project_root)),
                            'name': name,
                            'message': f"Short variable name '{name}' might be an abbreviation",
                            'severity': 'low'
                        })
                        
            except Exception:
                continue  # Skip files that can't be read
        
        return violations
    
    def validate_project(self) -> Dict[str, Any]:
        """
        Validate the entire project for naming convention violations.
        
        Returns:
            Dictionary containing all violations organized by type
        """
        logger.info(f"Validating naming conventions in: {self.project_root}")
        
        all_violations = {
            'file_naming': [],
            'code_naming': [],
            'abbreviations': [],
            'summary': {}
        }
        
        # Validate file names
        logger.info("Checking file naming conventions...")
        all_violations['file_naming'] = self.validate_file_names()
        
        # Validate Python code
        logger.info("Checking Python code naming conventions...")
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            violations = self.validate_python_code(file_path)
            all_violations['code_naming'].extend(violations)
        
        # Check for abbreviations
        logger.info("Checking for potential abbreviations...")
        all_violations['abbreviations'] = self.check_for_abbreviations()
        
        # Generate summary
        all_violations['summary'] = self.generate_summary(all_violations)
        
        return all_violations
    
    def generate_summary(self, violations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of all violations.
        
        Args:
            violations: Dictionary of all violations
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_violations': 0,
            'by_severity': {'high': 0, 'medium': 0, 'low': 0},
            'by_type': {},
            'files_with_violations': set(),
            'most_common_violations': []
        }
        
        # Count violations
        for category, violation_list in violations.items():
            if category == 'summary':
                continue
                
            for violation in violation_list:
                summary['total_violations'] += 1
                
                # Count by severity
                severity = violation.get('severity', 'medium')
                summary['by_severity'][severity] += 1
                
                # Count by type
                violation_type = violation['type']
                summary['by_type'][violation_type] = summary['by_type'].get(violation_type, 0) + 1
                
                # Track files with violations
                if 'file' in violation:
                    summary['files_with_violations'].add(violation['file'])
        
        # Convert set to list for JSON serialization
        summary['files_with_violations'] = list(summary['files_with_violations'])
        
        # Find most common violations
        summary['most_common_violations'] = sorted(
            summary['by_type'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return summary
    
    def print_report(self, violations: Dict[str, Any]):
        """
        Print a formatted report of violations.
        
        Args:
            violations: Dictionary of all violations
        """
        summary = violations['summary']
        
        print("\\n" + "="*60)
        print("SUM PROJECT NAMING CONVENTION VALIDATION REPORT")
        print("="*60)
        
        print(f"\\nSUMMARY:")
        print(f"  Total violations: {summary['total_violations']}")
        print(f"  Files with violations: {len(summary['files_with_violations'])}")
        print(f"  High severity: {summary['by_severity']['high']}")
        print(f"  Medium severity: {summary['by_severity']['medium']}")
        print(f"  Low severity: {summary['by_severity']['low']}")
        
        if summary['most_common_violations']:
            print(f"\\nMOST COMMON VIOLATIONS:")
            for violation_type, count in summary['most_common_violations']:
                print(f"  {violation_type}: {count}")
        
        # Print detailed violations by category
        for category, violation_list in violations.items():
            if category == 'summary' or not violation_list:
                continue
            
            print(f"\\n{category.upper().replace('_', ' ')} VIOLATIONS:")
            print("-" * 40)
            
            for violation in violation_list[:10]:  # Show first 10 of each type
                print(f"  • {violation['type']}: {violation.get('current_name', violation.get('name', 'N/A'))}")
                print(f"    File: {violation.get('file', 'N/A')}")
                if 'line' in violation:
                    print(f"    Line: {violation['line']}")
                if 'suggested_name' in violation:
                    print(f"    Suggested: {violation['suggested_name']}")
                if 'message' in violation:
                    print(f"    Message: {violation['message']}")
                print()
            
            if len(violation_list) > 10:
                print(f"    ... and {len(violation_list) - 10} more")
                print()
        
        print("\\nRECOMMENDATIONS:")
        if summary['total_violations'] == 0:
            print("  ✅ All naming conventions are properly followed!")
        else:
            print("  1. Address high-severity violations first")
            print("  2. Use the automated renaming script for bulk changes")
            print("  3. Update documentation after renaming")
            print("  4. Run tests after each renaming phase")

def main():
    """Main entry point for the validation script."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python naming_convention_validator.py <project_root>")
        print("Example: python naming_convention_validator.py /Users/ototao/Github\\ Projects/SUM/SUM")
        sys.exit(1)
    
    project_root = sys.argv[1]
    
    if not os.path.exists(project_root):
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)
    
    validator = NamingConventionValidator(project_root)
    violations = validator.validate_project()
    validator.print_report(violations)
    
    # Exit with error code if violations found
    if violations['summary']['total_violations'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()