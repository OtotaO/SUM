"""
NLTK resource management - handles external NLTK dependencies.
Isolates external concerns from business logic.
"""

import nltk
import os
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class NLTKResourceManager:
    """Manages NLTK resource downloads and initialization."""
    
    def __init__(self):
        self.nltk_data_dir = os.path.expanduser('~/nltk_data')
        
    def initialize_resources(self, resources: List[str]) -> None:
        """Initialize required NLTK resources."""
        os.makedirs(self.nltk_data_dir, exist_ok=True)
        
        for resource in resources:
            try:
                nltk.download(resource, download_dir=self.nltk_data_dir, quiet=True)
            except Exception as e:
                logger.error(f"Error downloading {resource}: {str(e)}")
                raise RuntimeError(f"Failed to initialize NLTK resource: {resource}")
    
    def get_english_stopwords(self) -> Set[str]:
        """Get English stopwords with safe defaults."""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error loading stopwords: {str(e)}")
            # Return minimal default set
            return {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with"}
    
    def get_sentiment_analyzer(self):
        """Get VADER sentiment analyzer."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Error loading sentiment analyzer: {str(e)}")
            raise RuntimeError("Failed to initialize sentiment analyzer")