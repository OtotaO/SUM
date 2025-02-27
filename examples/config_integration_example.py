"""config_integration_example.py - Example of integrating ConfigManager with SUM components

This example demonstrates how to integrate the ConfigManager with various
SUM components, showing a real-world application structure.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.config_manager import ConfigManager
from Utils.error_handling import handle_exceptions, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define configuration schema
CONFIG_SCHEMA = {
    'app_name': {'required': True, 'type': str},
    'version': {'required': True, 'type': str},
    'debug': {'type': bool},
    'summarizer': {
        'required': True,
        'type': dict
    },
    'summarizer.max_tokens': {
        'type': int,
        'min': 10,
        'max': 1000
    },
    'summarizer.algorithm': {
        'allowed': ['extractive', 'abstractive', 'hybrid']
    },
    'topic_modeling': {
        'required': True,
        'type': dict
    },
    'topic_modeling.num_topics': {
        'type': int,
        'min': 1,
        'max': 100
    },
    'topic_modeling.algorithm': {
        'allowed': ['lda', 'nmf', 'lsa']
    }
}

class SummarizerService:
    """Example summarizer service that uses configuration."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the summarizer service with configuration."""
        self.config = config
        self.max_tokens = config.get('summarizer', {}).get('max_tokens', 100)
        self.algorithm = config.get('summarizer', {}).get('algorithm', 'extractive')
        logger.info(f"Initialized SummarizerService with algorithm={self.algorithm}, max_tokens={self.max_tokens}")
    
    @handle_exceptions(logger_instance=logger)
    def summarize(self, text: str) -> Dict[str, Any]:
        """Summarize the given text."""
        if not text:
            raise ValidationError("Text cannot be empty")
            
        # Simulate summarization
        words = text.split()
        num_words = len(words)
        
        if num_words <= self.max_tokens:
            summary = text
        else:
            summary = ' '.join(words[:self.max_tokens]) + '...'
        
        return {
            'original_length': num_words,
            'summary_length': min(num_words, self.max_tokens),
            'algorithm': self.algorithm,
            'summary': summary
        }

class TopicModelingService:
    """Example topic modeling service that uses configuration."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the topic modeling service with configuration."""
        self.config = config
        self.num_topics = config.get('topic_modeling', {}).get('num_topics', 5)
        self.algorithm = config.get('topic_modeling', {}).get('algorithm', 'lda')
        logger.info(f"Initialized TopicModelingService with algorithm={self.algorithm}, num_topics={self.num_topics}")
    
    @handle_exceptions(logger_instance=logger)
    def extract_topics(self, documents: List[str]) -> Dict[str, Any]:
        """Extract topics from the given documents."""
        if not documents:
            raise ValidationError("Documents list cannot be empty")
            
        # Simulate topic extraction
        topics = []
        for i in range(min(self.num_topics, len(documents))):
            # Create a simple topic with words from the document
            words = documents[i].split()[:5] if i < len(documents) else [f"topic{i}"]
            topics.append({
                'id': i,
                'words': words,
                'weight': 1.0 - (i * 0.1)
            })
        
        return {
            'num_documents': len(documents),
            'num_topics': len(topics),
            'algorithm': self.algorithm,
            'topics': topics
        }

class SUMApplication:
    """Example SUM application that integrates configuration with services."""
    
    def __init__(self):
        """Initialize the SUM application."""
        # Load configuration
        self.config = self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize services
        self.summarizer = SummarizerService(self.config)
        self.topic_modeler = TopicModelingService(self.config)
        
        logger.info(f"Initialized SUMApplication with config sources: {self.config.config_sources}")
    
    def _load_configuration(self) -> ConfigManager:
        """Load configuration from various sources."""
        # Create base configuration
        config = ConfigManager({
            'app_name': 'SUM',
            'version': '1.0.0',
            'debug': False,
            'summarizer': {
                'max_tokens': 100,
                'algorithm': 'extractive'
            },
            'topic_modeling': {
                'num_topics': 5,
                'algorithm': 'lda'
            }
        })
        
        # Load from JSON file if it exists
        config_file = Path('config.json')
        if config_file.exists():
            config.load_from_json(config_file)
        
        # Load from environment variables
        config.load_from_env()
        
        return config
    
    def _validate_configuration(self) -> None:
        """Validate the configuration against the schema."""
        errors = self.config.validate(CONFIG_SCHEMA)
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValidationError(f"Invalid configuration: {len(errors)} errors found")
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process a document with summarization and topic modeling."""
        # Summarize the document
        summary_result = self.summarizer.summarize(text)
        
        # Extract topics from the document
        topic_result = self.topic_modeler.extract_topics([text])
        
        return {
            'summary': summary_result,
            'topics': topic_result
        }

def main():
    """Run the configuration integration example."""
    print("\n=== Configuration Integration Example ===")
    
    # Create a temporary configuration file
    config_data = {
        'debug': True,
        'summarizer': {
            'max_tokens': 50,
            'algorithm': 'hybrid'
        },
        'topic_modeling': {
            'num_topics': 3,
            'algorithm': 'nmf'
        }
    }
    
    config_file = Path('config.json')
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Created temporary config file: {config_file}")
    
    try:
        # Initialize the application
        print("\nInitializing SUMApplication...")
        app = SUMApplication()
        
        # Process a sample document
        print("\nProcessing sample document...")
        sample_text = """
        The SUM platform provides advanced text summarization and topic modeling capabilities.
        It can process documents of various lengths and extract key insights.
        The platform uses state-of-the-art natural language processing techniques.
        Users can configure the summarization algorithm and the number of topics to extract.
        The platform is designed to be flexible and extensible.
        """
        
        result = app.process_document(sample_text)
        
        # Display the results
        print("\nSummarization Result:")
        print(f"  Algorithm: {result['summary']['algorithm']}")
        print(f"  Original Length: {result['summary']['original_length']} words")
        print(f"  Summary Length: {result['summary']['summary_length']} words")
        print(f"  Summary: {result['summary']['summary']}")
        
        print("\nTopic Modeling Result:")
        print(f"  Algorithm: {result['topics']['algorithm']}")
        print(f"  Number of Topics: {result['topics']['num_topics']}")
        print("  Topics:")
        for topic in result['topics']['topics']:
            print(f"    Topic {topic['id']}: {', '.join(topic['words'])} (weight: {topic['weight']:.2f})")
        
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