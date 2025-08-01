"""
Adaptive expander - expands summaries based on content complexity.
Smart expansion only when needed.
"""

import logging
from typing import Optional, Set
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class AdaptiveExpander:
    """Adaptive expansion based on content complexity and information gaps."""
    
    def expand_if_needed(
        self, 
        text: str, 
        core_summary: str,
        complexity_threshold: float = 0.7,
        max_expansion_ratio: float = 2.0
    ) -> Optional[str]:
        """Expand the summary adaptively if complexity analysis indicates it's necessary."""
        # Analyze if expansion is needed
        complexity_score = self._analyze_complexity(text, core_summary)
        
        if complexity_score < complexity_threshold:
            return None  # No expansion needed
        
        # Calculate expansion needs
        expansion_factor = min(complexity_score * max_expansion_ratio, max_expansion_ratio)
        target_length = int(len(core_summary.split()) * expansion_factor)
        
        # Generate contextual expansion
        expanded_summary = self._generate_contextual_expansion(text, core_summary, target_length)
        
        return expanded_summary
    
    def _analyze_complexity(self, text: str, core_summary: str) -> float:
        """Analyze whether the core summary captures the full complexity."""
        # Compression ratio
        compression_ratio = len(core_summary.split()) / len(text.split())
        compression_penalty = max(0, (0.05 - compression_ratio) * 10)
        
        # Concept diversity loss
        text_concepts = set(word_tokenize(text.lower()))
        summary_concepts = set(word_tokenize(core_summary.lower()))
        concept_retention = len(summary_concepts.intersection(text_concepts)) / len(text_concepts)
        concept_penalty = max(0, (0.3 - concept_retention) * 2)
        
        # Structural complexity
        text_sentences = len(sent_tokenize(text))
        structural_complexity = min(text_sentences / 10, 0.5)
        
        # Technical content
        technical_markers = ['concept', 'principle', 'theory', 'method', 'approach', 'framework']
        technical_score = sum(0.1 for marker in technical_markers if marker in text.lower())
        
        complexity_score = compression_penalty + concept_penalty + structural_complexity + technical_score
        return min(complexity_score, 1.0)
    
    def _generate_contextual_expansion(self, text: str, core_summary: str, target_length: int) -> str:
        """Generate intelligent contextual expansion."""
        sentences = sent_tokenize(text)
        summary_concepts = set(word_tokenize(core_summary.lower()))
        
        # Find sentences that add context without redundancy
        contextual_sentences = []
        
        for sentence in sentences:
            sentence_concepts = set(word_tokenize(sentence.lower()))
            
            # Skip if sentence is already well-represented
            overlap_ratio = len(sentence_concepts.intersection(summary_concepts)) / len(sentence_concepts)
            if overlap_ratio > 0.7:
                continue
            
            # Add sentences that provide complementary information
            contextual_sentences.append(sentence)
        
        # Build expanded summary
        current_length = len(core_summary.split())
        expansion_parts = [core_summary]
        
        for sentence in contextual_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= target_length:
                expansion_parts.append(sentence)
                current_length += sentence_length
            else:
                break
        
        return ' '.join(expansion_parts)