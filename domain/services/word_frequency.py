"""
Word frequency calculation service.
Pure computation, no side effects.
"""

from collections import Counter
from typing import Dict, List, Set


class WordFrequencyCalculator:
    """Calculates word frequencies for text analysis."""
    
    def __init__(self, stop_words: Set[str]):
        self.stop_words = stop_words
        
    def calculate_frequencies(self, words: List[str]) -> Dict[str, int]:
        """Calculate word frequencies excluding stop words."""
        return Counter(
            word for word in words
            if word.isalnum() and word not in self.stop_words
        )
    
    def calculate_normalized_frequency(self, word: str, word_freq: Dict[str, int], max_freq: int) -> float:
        """Calculate normalized word frequency."""
        return word_freq.get(word, 0) / max_freq if max_freq > 0 else 0