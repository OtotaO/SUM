"""
integrated_config_example.py - Example of integrating ConfigManager with SUM components

This script demonstrates how to use the ConfigManager to configure and use
the SUM components in an integrated way.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.config_manager import ConfigManager
from summarization_engine import SimpleSUM, MagnumOpusSUM
from Models.topic_modeling import TopicModeler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_config():
    """Create a sample configuration file for the example."""
    config_path = Path('sum_config.json')
    
    config_data = {
        "summarization": {
            "default_model": "simple",
            "max_tokens": 150,
            "threshold": 0.3,
            "include_analysis": True
        },
        "topic_modeling": {
            "num_topics": 5,
            "algorithm": "lda",
            "top_n_words": 10
        },
        "processing": {
            "batch_size": 100,
            "max_workers": 4
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
        
    return config_path

def main():
    """Demonstrate integrated configuration management."""
    print("SUM Integrated Configuration Example")
    print("====================================")
    
    # Create a sample configuration file
    config_path = create_sample_config()
    print(f"Created sample configuration file: {config_path}")
    
    # Initialize configuration manager
    config = ConfigManager()
    
    # Load configuration from environment and file
    config.load_from_env()
    config.load_from_json(config_path)
    
    # Print loaded configuration
    print("\nLoaded configuration:")
    print(f"- Summarization settings: {config.get('summarization', {})}")
    print(f"- Topic modeling settings: {config.get('topic_modeling', {})}")
    print(f"- Processing settings: {config.get('processing', {})}")
    
    # Sample text for demonstration
    sample_text = """
    Machine learning has seen rapid advancements in recent years. From image recognition to
    natural language processing, AI systems are becoming increasingly sophisticated. Deep learning
    models, in particular, have shown remarkable capabilities in handling complex tasks. However,
    challenges remain in areas such as explainability and bias mitigation. As the field continues
    to evolve, researchers are developing new approaches to address these limitations and expand
    the applications of machine learning across various domains.
    """
    
    # Initialize SUM components with configuration
    summarization_config = config.get('summarization', {})
    topic_modeling_config = config.get('topic_modeling', {})
    
    # Choose summarizer based on configuration
    if summarization_config.get('default_model') == 'advanced':
        print("\nInitializing MagnumOpusSUM...")
        summarizer = MagnumOpusSUM()
    else:
        print("\nInitializing SimpleSUM...")
        summarizer = SimpleSUM()
    
    # Initialize topic modeler with configuration
    print("Initializing TopicModeler...")
    topic_modeler = TopicModeler(
        n_topics=topic_modeling_config.get('num_topics', 5),
        algorithm=topic_modeling_config.get('algorithm', 'lda'),
        n_top_words=topic_modeling_config.get('top_n_words', 10)
    )
    
    # Process text with configured summarizer
    print("\nProcessing text...")
    model_config = {
        'maxTokens': summarization_config.get('max_tokens', 100),
        'threshold': summarization_config.get('threshold', 0.3),
        'include_analysis': summarization_config.get('include_analysis', False)
    }
    
    result = summarizer.process_text(sample_text, model_config)
    
    # Print summarization results
    print("\nSummarization Results:")
    print(f"- Summary: {result.get('summary', '')}")
    print(f"- Condensed: {result.get('sum', '')}")
    print(f"- Tags: {result.get('tags', [])}")
    
    if 'sentiment' in result:
        print(f"- Sentiment: {result.get('sentiment', '')}")
    
    # Process text with topic modeler
    print("\nPerforming topic modeling...")
    documents = [sample_text]
    topic_modeler.fit(documents)
    topics = topic_modeler.get_topics()
    
    # Print topic modeling results
    print("\nTopic Modeling Results:")
    for i, topic in enumerate(topics):
        print(f"- Topic {i+1}: {', '.join(topic)}")
    
    # Clean up
    config_path.unlink()
    print("\nExample complete!")

if __name__ == "__main__":
    main()
