"""
Sentence scoring service - pure algorithm implementation.
No external dependencies, just math and logic.
"""

from typing import Dict, List, Set
from nltk.tokenize import word_tokenize


class SentenceScorer:
    """Scores sentences based on various criteria."""
    
    def __init__(self, stop_words: Set[str]):
        self.stop_words = stop_words
    
    def score_by_frequency(
        self, 
        sentences: List[str], 
        word_freq: Dict[str, int],
        max_freq: int
    ) -> Dict[str, float]:
        """Score sentences based on word frequency."""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            sent_words = word_tokenize(sentence.lower())
            
            if len(sent_words) < 3:
                sentence_scores[sentence] = 0
                continue
                
            score = sum(
                word_freq.get(word, 0) / max_freq 
                for word in sent_words 
                if word.isalnum() and word not in self.stop_words
            ) if max_freq > 0 else 0
                
            position_weight = 1.0 - (0.1 * (i / max(1, len(sentences))))
            sentence_scores[sentence] = (score / max(1, len(sent_words))) * position_weight
            
        return sentence_scores
    
    def score_by_importance(self, sentence: str) -> float:
        """Score sentence based on importance markers."""
        importance_markers = [
            'essential', 'fundamental', 'key', 'important', 'crucial', 'vital',
            'core', 'central', 'primary', 'main', 'principal', 'significant',
            'critical', 'necessary', 'major', 'basic', 'relevant', 'notable'
        ]
        
        sentence_lower = sentence.lower()
        boost = sum(0.1 for marker in importance_markers if marker in sentence_lower)
        
        # Additional boost for definitive statements
        if any(pattern in sentence_lower for pattern in ['is ', 'are ', 'means ', 'represents ']):
            boost += 0.05
        
        return min(boost, 0.8)