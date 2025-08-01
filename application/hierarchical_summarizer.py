"""
Hierarchical densification engine - multi-level knowledge densification.
Orchestrates three levels of abstraction for comprehensive summarization.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from nltk.tokenize import sent_tokenize, word_tokenize

from domain.interfaces.summarization_engine import SummarizationEngine
from domain.services.text_validator import TextValidator
from domain.services.concept_extractor import ConceptExtractor
from domain.services.insight_extractor import InsightExtractor
from application.core_summarizer import CoreSummarizer
from application.adaptive_expander import AdaptiveExpander
from infrastructure.nltk_manager import NLTKResourceManager

logger = logging.getLogger(__name__)


class HierarchicalDensificationEngine(SummarizationEngine):
    """Multi-level knowledge densification with three hierarchical abstraction levels."""
    
    def __init__(self, concept_database_path: Optional[str] = None):
        """Initialize the hierarchical densification engine."""
        self.nltk_manager = NLTKResourceManager()
        self.nltk_manager.initialize_resources([
            'punkt', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon'
        ])
        
        # Get stopwords with philosophical filtering
        raw_stopwords = self.nltk_manager.get_english_stopwords()
        wisdom_words = {'being', 'truth', 'wisdom', 'knowledge', 'virtue', 'beauty', 'justice', 'love'}
        self.stop_words = raw_stopwords - wisdom_words
        
        # Initialize services
        self.text_validator = TextValidator()
        self.concept_extractor = ConceptExtractor(self.stop_words)
        self.core_summarizer = CoreSummarizer(self.stop_words)
        self.adaptive_expander = AdaptiveExpander()
        self.insight_extractor = InsightExtractor()
        
        logger.info("Hierarchical Densification Engine initialized successfully")
    
    def process_text(self, text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text through three hierarchical levels of abstraction."""
        if not self.text_validator.validate_input(text):
            return {'error': 'Empty or invalid text provided'}
        
        try:
            config = config or {}
            start_time = time.time()
            
            # Level 1: Extract key concepts
            concepts = self.concept_extractor.extract_concepts(
                text, 
                max_concepts=config.get('max_concepts', 5),
                min_weight=config.get('min_concept_weight', 0.3)
            )
            
            # Level 2: Generate core summary
            core_summary = self.core_summarizer.generate_summary(
                text,
                target_density=config.get('target_density', 0.15),
                max_tokens=config.get('max_summary_tokens', 50),
                ensure_completeness=config.get('ensure_completeness', True)
            )
            
            # Level 3: Adaptive expansion (only if complexity demands it)
            expanded_summary = self.adaptive_expander.expand_if_needed(
                text, 
                core_summary,
                complexity_threshold=config.get('complexity_threshold', 0.7),
                max_expansion_ratio=config.get('max_expansion_ratio', 2.0)
            )
            
            # Extract key insights and quotes
            insights = self.insight_extractor.extract_insights(
                text,
                max_insights=config.get('max_insights', 3),
                min_score=config.get('min_insight_score', 0.6)
            )
            
            # Calculate metadata
            processing_time = time.time() - start_time
            compression_ratio = len(core_summary.split()) / len(text.split()) if text else 1.0
            concept_density = len(concepts) / len(text.split()) if text else 0.0
            
            # Compile hierarchical result
            result = {
                'hierarchical_summary': {
                    'level_1_concepts': concepts,
                    'level_2_core': core_summary,
                    'level_3_expanded': expanded_summary
                },
                'key_insights': insights,
                'metadata': {
                    'processing_time': processing_time,
                    'compression_ratio': compression_ratio,
                    'concept_density': concept_density,
                    'insight_count': len(insights)
                }
            }
            
            # Backward compatibility
            result['summary'] = core_summary
            result['tags'] = concepts
            result['sum'] = core_summary
            
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical Engine processing failed: {str(e)}", exc_info=True)
            return {'error': f'Hierarchical Engine processing failed: {str(e)}'}