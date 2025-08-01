"""
Base interface for summarization engines.
Simple, clean abstraction following Carmack's minimalism.
"""

from typing import Dict, Any, Optional


class SummarizationEngine:
    """Base interface for all summarization algorithms."""
    
    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text using the implemented algorithm."""
        raise NotImplementedError("Subclasses must implement process_text")