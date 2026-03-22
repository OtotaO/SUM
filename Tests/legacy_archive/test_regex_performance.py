import re
import pytest
from summarization_engine import BasicSummarizationEngine, HierarchicalDensificationEngine

class TestRegexPerformance:
    def test_unsafe_patterns_do_not_contain_catastrophic_backtracking_prefix(self):
        """
        Verify that unsafe_patterns do not use the dangerous [\s\S]* prefix.
        This is a static analysis of the source code to prevent regression.
        """
        with open('summarization_engine.py', 'r') as f:
            content = f.read()

        # Check for the pattern r'[\s\S]* inside unsafe_patterns lists
        # We search for lines that look like a regex definition inside the list

        # This is a simple heuristic: check if r'[\s\S]* appears in the file
        # associated with the unsafe_patterns list definition.

        # Actually, let's just check that r'[\s\S]* is NOT present in the file at all
        # where it was previously used.

        # Previous risky patterns were:
        # r'[\s\S]*exec\s*\(', r'[\s\S]*eval\s*\(', r'[\s\S]*\bimport\b',
        # r'[\s\S]*__[a-zA-Z]+__', r'[\s\S]*\bopen\s*\('

        risky_markers = [
            r"r'[\s\S]*exec",
            r"r'[\s\S]*eval",
            r"r'[\s\S]*\bimport",
            r"r'[\s\S]*__",
            r"r'[\s\S]*\bopen"
        ]

        for marker in risky_markers:
            assert marker not in content, f"Found risky regex pattern in source code: {marker}"

    def test_is_safe_string_functional(self):
        """
        Verify that _is_safe_string still correctly identifies unsafe strings
        (within the 100 char limit).
        """
        engine = BasicSummarizationEngine()

        # Safe strings
        assert engine._is_safe_string("This is a safe string.") is True
        assert engine._is_safe_string("executing tasks is fun") is True # 'exec' but not 'exec ('

        # Unsafe strings (short enough to be checked)
        assert engine._is_safe_string("exec ('rm -rf')") is False
        assert engine._is_safe_string("eval (code)") is False
        assert engine._is_safe_string("import os") is False
        assert engine._is_safe_string("__import__") is False
        assert engine._is_safe_string("open ('/etc/passwd')") is False

        # Boundary checks
        assert engine._is_safe_string("a" * 101) is False # Too long

        # Hierarchical engine
        h_engine = HierarchicalDensificationEngine()
        assert h_engine._is_safe_string("exec ('rm -rf')") is False
        assert h_engine._is_safe_string("This is safe") is True
