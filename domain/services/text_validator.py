"""
Text validation service - single responsibility for input validation.
Fast, focused, no external dependencies.
"""

import re
from typing import Set


class TextValidator:
    """Validates text input for safety and quality."""
    
    @staticmethod
    def is_safe_string(text: str) -> bool:
        """Verify string doesn't contain potentially unsafe patterns."""
        if len(text) > 100:  # Unusually long for a stopword
            return False
        unsafe_patterns = [
            r'[\s\S]*exec\s*\(', r'[\s\S]*eval\s*\(', r'[\s\S]*\bimport\b',
            r'[\s\S]*__[a-zA-Z]+__', r'[\s\S]*\bopen\s*\('
        ]
        return not any(re.search(pattern, text) for pattern in unsafe_patterns)
    
    @staticmethod
    def validate_input(text: str) -> bool:
        """Validate input text for processing."""
        return bool(text and isinstance(text, str) and text.strip())