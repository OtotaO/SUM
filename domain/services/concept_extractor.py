"""
Concept extraction service - extracts key concepts from text.
Single responsibility, focused on concept identification.
"""

from collections import Counter
from typing import Dict, List, Set
from nltk.tokenize import word_tokenize, sent_tokenize


class ConceptExtractor:
    """Extracts key concepts and thematic keywords from text."""
    
    def __init__(self, stop_words: Set[str], concept_weights: Dict[str, float] = None):
        self.stop_words = stop_words
        self.concept_weights = concept_weights or self._default_concept_weights()
    
    def _default_concept_weights(self) -> Dict[str, float]:
        """Default concept weights for importance scoring."""
        return {
            'important': 0.9, 'essential': 0.9, 'key': 0.85, 'critical': 0.85, 
            'fundamental': 0.8, 'core': 0.8, 'primary': 0.75, 'main': 0.75, 
            'significant': 0.7, 'central': 0.7, 'major': 0.65, 'basic': 0.6, 
            'principal': 0.65, 'necessary': 0.7, 'vital': 0.75, 'crucial': 0.85
        }
    
    def extract_concepts(self, text: str, max_concepts: int = 5, min_weight: float = 0.3) -> List[str]:
        """Extract key concepts from text."""
        words = word_tokenize(text.lower())
        word_freq = Counter(words)
        
        # Score words by conceptual importance
        concept_scores = {}
        
        for word in word_freq:
            if len(word) < 3 or word in self.stop_words:
                continue
                
            base_score = word_freq[word] / len(words)  # Frequency score
            concept_boost = self.concept_weights.get(word, 0)
            context_boost = self._calculate_context_importance(word, text)
            
            concept_scores[word] = base_score + (concept_boost * 0.5) + (context_boost * 0.3)
        
        # Extract top concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        concepts = [word for word, score in sorted_concepts[:max_concepts] 
                   if score >= min_weight]
        
        return concepts if concepts else [sorted_concepts[0][0]] if sorted_concepts else []
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords based on frequency."""
        words = word_tokenize(text.lower())
        word_freq = Counter(
            word for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 2
        )
        return [word for word, _ in word_freq.most_common(num_keywords)]
    
    def _calculate_context_importance(self, word: str, text: str) -> float:
        """Calculate boost based on contextual importance."""
        sentences = sent_tokenize(text)
        boost = 0.0
        
        for sentence in sentences:
            if word in sentence.lower():
                # Check for importance markers in the same sentence
                importance_markers = [
                    'important', 'essential', 'key', 'critical', 'fundamental',
                    'crucial', 'vital', 'primary', 'main', 'significant'
                ]
                
                for marker in importance_markers:
                    if marker in sentence.lower() and marker != word:
                        boost += 0.1
                        
                # Boost for technical/analytical concepts
                if any(tech in sentence.lower() for tech in 
                      ['concept', 'framework', 'principle', 'methodology', 'approach']):
                    boost += 0.05
        
        return min(boost, 0.5)