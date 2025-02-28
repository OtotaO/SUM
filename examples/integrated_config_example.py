"""integrated_config_example.py - Example of integrating ConfigManager with SUM components

This example demonstrates how to integrate the ConfigManager with the core SUM
components, showing how to configure and use the summarization and topic modeling
functionality with different configuration settings.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.config_manager import ConfigManager
from Utils.error_handling import handle_exceptions, ValidationError, safe_execute
from Utils.text_preprocessing import preprocess_text, tokenize_sentences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryCoreService:
    """Core summarization service that integrates with configuration."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the summarization service with configuration."""
        self.config = config
        
        # Load summarization settings
        summarizer_config = config.get('summarizer', {})
        self.max_tokens = summarizer_config.get('max_tokens', 100)
        self.min_sentence_length = summarizer_config.get('min_sentence_length', 5)
        self.algorithm = summarizer_config.get('algorithm', 'extractive')
        self.include_metadata = summarizer_config.get('include_metadata', True)
        
        # Load preprocessing settings
        preprocessing_config = config.get('preprocessing', {})
        self.lowercase = preprocessing_config.get('lowercase', True)
        self.remove_stopwords = preprocessing_config.get('remove_stopwords', True)
        self.remove_urls = preprocessing_config.get('remove_urls', True)
        self.remove_special_chars = preprocessing_config.get('remove_special_chars', False)
        
        logger.info(f"Initialized SummaryCoreService with algorithm={self.algorithm}, max_tokens={self.max_tokens}")
    
    @handle_exceptions(logger_instance=logger)
    def summarize(self, text: str) -> Dict[str, Any]:
        """Summarize the given text based on configuration settings."""
        if not text:
            raise ValidationError("Text cannot be empty")
        
        # Preprocess the text
        processed_text = preprocess_text(
            text,
            lowercase=self.lowercase,
            remove_stopwords=self.remove_stopwords,
            remove_urls=self.remove_urls,
            remove_special_chars=self.remove_special_chars
        )
        
        # Tokenize into sentences
        sentences = tokenize_sentences(text)
        
        # Filter sentences by length
        valid_sentences = [s for s in sentences if len(s.split()) >= self.min_sentence_length]
        
        # Generate summary based on algorithm
        if self.algorithm == 'extractive':
            summary = self._extractive_summarization(valid_sentences)
        elif self.algorithm == 'abstractive':
            summary = self._abstractive_summarization(processed_text)
        else:  # hybrid
            summary = self._hybrid_summarization(valid_sentences, processed_text)
        
        # Prepare result
        result = {'summary': summary}
        
        # Include metadata if requested
        if self.include_metadata:
            result['metadata'] = {
                'original_length': len(text),
                'sentence_count': len(sentences),
                'algorithm': self.algorithm,
                'preprocessing': {
                    'lowercase': self.lowercase,
                    'remove_stopwords': self.remove_stopwords,
                    'remove_urls': self.remove_urls,
                    'remove_special_chars': self.remove_special_chars
                }
            }
        
        return result
    
    def _extractive_summarization(self, sentences: List[str]) -> str:
        """Perform extractive summarization by selecting top sentences."""
        # Simple implementation: take first few sentences up to max_tokens
        word_count = 0
        selected_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) <= self.max_tokens:
                selected_sentences.append(sentence)
                word_count += len(words)
            else:
                break
        
        return ' '.join(selected_sentences)
    
    def _abstractive_summarization(self, text: str) -> str:
        """Perform abstractive summarization (simplified simulation)."""
        # Simplified simulation: take first max_tokens words
        words = text.split()
        summary_words = words[:self.max_tokens]
        return ' '.join(summary_words)
    
    def _hybrid_summarization(self, sentences: List[str], processed_text: str) -> str:
        """Perform hybrid summarization (combination of extractive and abstractive)."""
        # Simplified simulation: take half from extractive, half from abstractive
        half_tokens = self.max_tokens // 2
        
        # Get extractive part (first few sentences)
        extractive_part = self._extractive_summarization(sentences[:half_tokens])
        
        # Get abstractive part (first few words of processed text)
        words = processed_text.split()
        abstractive_part = ' '.join(words[:half_tokens])
        
        return f"{extractive_part} {abstractive_part}"

class TopicModelingService:
    """Topic modeling service that integrates with configuration."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the topic modeling service with configuration."""
        self.config = config
        
        # Load topic modeling settings
        topic_config = config.get('topic_modeling', {})
        self.num_topics = topic_config.get('num_topics', 5)
        self.algorithm = topic_config.get('algorithm', 'lda')
        self.min_topic_coherence = topic_config.get('min_topic_coherence', 0.3)
        self.include_word_weights = topic_config.get('include_word_weights', True)
        
        # Load preprocessing settings
        preprocessing_config = config.get('preprocessing', {})
        self.lowercase = preprocessing_config.get('lowercase', True)
        self.remove_stopwords = preprocessing_config.get('remove_stopwords', True)
        
        logger.info(f"Initialized TopicModelingService with algorithm={self.algorithm}, num_topics={self.num_topics}")
    
    @handle_exceptions(logger_instance=logger)
    def extract_topics(self, documents: List[str]) -> Dict[str, Any]:
        """Extract topics from the given documents based on configuration settings."""
        if not documents:
            raise ValidationError("Documents list cannot be empty")
        
        # Preprocess documents
        processed_docs = [
            preprocess_text(
                doc,
                lowercase=self.lowercase,
                remove_stopwords=self.remove_stopwords
            ) for doc in documents
        ]
        
        # Extract topics based on algorithm
        if self.algorithm == 'lda':
            topics = self._lda_topic_extraction(processed_docs)
        elif self.algorithm == 'nmf':
            topics = self._nmf_topic_extraction(processed_docs)
        else:  # lsa
            topics = self._lsa_topic_extraction(processed_docs)
        
        # Filter topics by coherence
        filtered_topics = [t for t in topics if t['coherence'] >= self.min_topic_coherence]
        
        # Prepare result
        result = {
            'num_documents': len(documents),
            'num_topics': len(filtered_topics),
            'algorithm': self.algorithm,
            'topics': filtered_topics
        }
        
        return result
    
    def _lda_topic_extraction(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Perform LDA topic extraction (simplified simulation)."""
        return self._simulate_topic_extraction(documents, 'lda')
    
    def _nmf_topic_extraction(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Perform NMF topic extraction (simplified simulation)."""
        return self._simulate_topic_extraction(documents, 'nmf')
    
    def _lsa_topic_extraction(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Perform LSA topic extraction (simplified simulation)."""
        return self._simulate_topic_extraction(documents, 'lsa')
    
    def _simulate_topic_extraction(self, documents: List[str], method: str) -> List[Dict[str, Any]]:
        """Simulate topic extraction for demonstration purposes."""
        topics = []
        
        # Create simulated topics
        for i in range(min(self.num_topics, len(documents) + 2)):
            # Get words from documents or generate placeholder
            if i < len(documents):
                doc_words = documents[i].split()[:10]
                words = doc_words[:5] if doc_words else [f"topic{i}_word{j}" for j in range(5)]
            else:
                words = [f"topic{i}_word{j}" for j in range(5)]
            
            # Create topic with words and weights
            topic = {
                'id': i,
                'words': words,
                'coherence': 0.8 - (i * 0.1),  # Decreasing coherence for demonstration
            }
            
            # Add word weights if configured
            if self.include_word_weights:
                topic['word_weights'] = [
                    {'word': word, 'weight': 0.9 - (j * 0.1)} 
                    for j, word in enumerate(words)
                ]
            
            topics.append(topic)
        
        return topics

class SUMIntegratedApplication:
    """Integrated SUM application that uses ConfigManager for all components."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the SUM application with optional config file."""
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize services
        self.summarizer = SummaryCoreService(self.config)
        self.topic_modeler = TopicModelingService(self.config)
        
        # Set application mode
        self.debug_mode = self.config.get('debug', False)
        if self.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Initialized SUMIntegratedApplication with config sources: {self.config.config_sources}")
    
    def _load_configuration(self, config_file: Optional[str] = None) -> ConfigManager:
        """Load configuration from various sources."""
        # Create base configuration with defaults
        config = ConfigManager({
            'app_name': 'SUM',
            'version': '1.0.0',
            'debug': False,
            'summarizer': {
                'max_tokens': 100,
                'min_sentence_length': 5,
                'algorithm': 'extractive',
                'include_metadata': True
            },
            'topic_modeling': {
                'num_topics': 5,
                'algorithm': 'lda',
                'min_topic_coherence': 0.3,
                'include_word_weights': True
            },
            'preprocessing': {
                'lowercase': True,
                'remove_stopwords': True,
                'remove_urls': True,
                'remove_special_chars': False
            }
        })
        
        # Load from specified JSON file if provided
        if config_file and Path(config_file).exists():
            config.load_from_json(config_file)
            logger.info(f"Loaded configuration from {config_file}")
        
        # Load from default config file if it exists
        default_config_file = Path('sum_config.json')
        if default_config_file.exists():
            config.load_from_json(default_config_file)
            logger.info(f"Loaded configuration from {default_config_file}")
        
        # Load from environment variables
        config.load_from_env()
        
        return config
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process a document with summarization and topic modeling."""
        # Summarize the document
        summary_result = safe_execute(
            self.summarizer.summarize,
            text,
            default_return={'summary': 'Error generating summary', 'error': True}
        )
        
        # Extract topics from the document
        topic_result = safe_execute(
            self.topic_modeler.extract_topics,
            [text],
            default_return={'topics': [], 'error': True}
        )
        
        return {
            'summary': summary_result,
            'topics': topic_result,
            'config': {
                'debug_mode': self.debug_mode,
                'summarizer_algorithm': self.summarizer.algorithm,
                'topic_modeling_algorithm': self.topic_modeler.algorithm
            }
        }

def main():
    """Run the integrated configuration example."""
    print("\n=== Integrated Configuration Example ===")
    
    # Create a temporary configuration file
    config_data = {
        'debug': True,
        'summarizer': {
            'max_tokens': 50,
            'algorithm': 'hybrid',
            'include_metadata': True
        },
        'topic_modeling': {
            'num_topics': 3,
            'algorithm': 'nmf',
            'include_word_weights': True
        },
        'preprocessing': {
            'lowercase': True,
            'remove_stopwords': True,
            'remove_urls': True
        }
    }
    
    config_file = Path('sum_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Created temporary config file: {config_file}")
    
    try:
        # Initialize the application
        print("\nInitializing SUMIntegratedApplication...")
        app = SUMIntegratedApplication()
        
        # Process a sample document
        print("\nProcessing sample document...")
        sample_text = """
        The SUM platform provides advanced text summarization and topic modeling capabilities.
        It can process documents of various lengths and extract key insights.
        The platform uses state-of-the-art natural language processing techniques.
        Users can configure the summarization algorithm and the number of topics to extract.
        The platform is designed to be flexible and extensible.
        Integration with various data sources is supported through a plugin architecture.
        The configuration system allows for fine-tuning of all components.
        Performance optimization ensures efficient processing of large documents.
        """
        
        result = app.process_document(sample_text)
        
        # Display the results
        print("\nApplication Configuration:")
        print(f"  Debug Mode: {result['config']['debug_mode']}")
        print(f"  Summarizer Algorithm: {result['config']['summarizer_algorithm']}")
        print(f"  Topic Modeling Algorithm: {result['config']['topic_modeling_algorithm']}")
        
        print("\nSummarization Result:")
        if 'metadata' in result['summary']:
            print(f"  Original Length: {result['summary']['metadata']['original_length']} characters")
            print(f"  Sentence Count: {result['summary']['metadata']['sentence_count']}")
            print(f"  Algorithm: {result['summary']['metadata']['algorithm']}")
        print(f"  Summary: {result['summary']['summary']}")
        
        print("\nTopic Modeling Result:")
        print(f"  Algorithm: {result['topics']['algorithm']}")
        print(f"  Number of Topics: {result['topics']['num_topics']}")
        print("  Topics:")
        for topic in result['topics']['topics']:
            print(f"    Topic {topic['id']}: {', '.join(topic['words'][:3])}... (coherence: {topic['coherence']:.2f})")
            if 'word_weights' in topic and topic['word_weights']:
                top_word = topic['word_weights'][0]
                print(f"      Top word: {top_word['word']} (weight: {top_word['weight']:.2f})")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up temporary file
        if config_file.exists():
            config_file.unlink()
            print(f"\nRemoved temporary config file: {config_file}")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()