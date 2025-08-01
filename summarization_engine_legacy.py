"""
Legacy wrapper for backward compatibility.
Imports all engines from the new modular structure.
This file can replace the original summarization_engine.py
"""

# Import everything from the new modular structure
from sum_engines import *

# Import base class for compatibility
from domain.interfaces.summarization_engine import SummarizationEngine

# Re-export everything that was in the original file
__all__ = [
    'SummarizationEngine',
    'BasicSummarizationEngine',
    'AdvancedSummarizationEngine',
    'HierarchicalDensificationEngine',
    # Legacy aliases
    'SimpleSUM',
    'BasicSUM', 
    'MagnumOpusSUM',
    'AdvancedSUM',
    'TrinityEngine'
]

# For complete backward compatibility, also import the sub-components
# that were exposed in the original file
from domain.services.concept_extractor import ConceptExtractor
from application.core_summarizer import CoreSummarizer
from application.adaptive_expander import AdaptiveExpander
from domain.services.insight_extractor import InsightExtractor

# Note: SemanticCompressionEngine and CompletenessValidator were placeholder
# implementations in the original, so we'll create minimal versions here
class SemanticCompressionEngine:
    """Placeholder for backward compatibility."""
    def __init__(self):
        self.compression_strategies = ['hierarchical_clustering', 'semantic_similarity', 'concept_extraction']
    
    def compress_semantically(self, text: str, target_ratio: float) -> str:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        target_sentences = max(1, int(len(sentences) * target_ratio))
        return ' '.join(sentences[:target_sentences])


class CompletenessValidator:
    """Placeholder for backward compatibility."""
    def validate_and_refine(self, original: str, compressed: str, target_density: float) -> str:
        return compressed
    
    def _calculate_completeness(self, original: str, compressed: str) -> float:
        from nltk.tokenize import word_tokenize
        original_concepts = set(word_tokenize(original.lower()))
        compressed_concepts = set(word_tokenize(compressed.lower()))
        if not original_concepts:
            return 1.0
        return len(compressed_concepts.intersection(original_concepts)) / len(original_concepts)
    
    def _refine_for_completeness(self, original: str, compressed: str, target_density: float) -> str:
        return compressed