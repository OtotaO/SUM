"""
config_integration_example.py - Example of integrating ConfigManager with SUM components

This script demonstrates how to use the ConfigManager to configure and integrate
with various SUM components, showing best practices for configuration management.

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
from SUM import SimpleSUM, MagnumOpusSUM
from Models.topic_modeling import TopicModeler
from Utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_config():
    """Create a default configuration file for the example."""
    config_path = Path('sum_default_config.json')
    
    # Using flattened keys for compatibility with validation schema
    config_data = {
        "app.name": "SUM Platform",
        "app.version": "1.2.0",
        "app.debug": False,
        "app.log_level": "INFO",
        
        "summarization.default_model": "simple",
        "summarization.max_tokens": 150,
        "summarization.threshold": 0.3,
        "summarization.include_analysis": True,
        
        "topic_modeling.num_topics": 5,
        "topic_modeling.algorithm": "lda",
        "topic_modeling.top_n_words": 10,
        
        "data_loading.batch_size": 100,
        "data_loading.max_workers": 4,
        "data_loading.cache_enabled": True,
        "data_loading.cache_ttl": 3600,
        
        "output.format": "json",
        "output.pretty_print": True,
        "output.include_metadata": True
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
        
    return config_path

class SUMApplication:
    """
    Example application that integrates ConfigManager with SUM components.
    
    This class demonstrates how to use ConfigManager to configure and
    coordinate multiple SUM components in a real application.
    """
    
    def __init__(self, config_path=None, env_prefix='SUM_'):
        """
        Initialize the application with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
            env_prefix: Prefix for environment variables (default: 'SUM_')
        """
        # Initialize configuration manager with flattened keys
        self.config = ConfigManager({
            "app.name": "SUM Platform",
            "app.version": "1.0.0",
            "app.debug": False
        })
        
        # Load configuration from environment variables
        self.config.load_from_env(prefix=env_prefix)
        
        # Load configuration from file if provided
        if config_path:
            self.config.load_from_json(config_path)
            
        # Validate configuration
        self._validate_config()
        
        # Initialize components based on configuration
        self._init_components()
        
        logger.info(f"Initialized {self.config.get('app.name')} v{self.config.get('app.version')}")
        
    def _validate_config(self):
        """Validate the application configuration."""
        schema = {
            "app.name": {"required": True, "type": str},
            "app.version": {"required": True, "type": str},
            "app.debug": {"type": bool},
            "app.log_level": {"allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
            "summarization.default_model": {"allowed": ["simple", "advanced"]},
            "summarization.max_tokens": {"type": int, "min": 10, "max": 1000},
            "topic_modeling.num_topics": {"type": int, "min": 1, "max": 100},
            "topic_modeling.algorithm": {"allowed": ["lda", "nmf", "lsa"]}
        }
        
        errors = self.config.validate(schema)
        if errors:
            logger.warning("Configuration validation errors:")
            for error in errors:
                logger.warning(f"  - {error}")
                
        # Set log level from configuration
        log_level = self.config.get("app.log_level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level))
        
    def _init_components(self):
        """Initialize SUM components based on configuration."""
        # Initialize summarizer
        summarizer_type = self.config.get("summarization.default_model", "simple")
        if summarizer_type == "advanced":
            self.summarizer = MagnumOpusSUM()
            logger.info("Initialized MagnumOpusSUM summarizer")
        else:
            self.summarizer = SimpleSUM()
            logger.info("Initialized SimpleSUM summarizer")
            
        # Initialize topic modeler
        topic_config = self.config.get("topic_modeling", {})
        self.topic_modeler = TopicModeler(
            n_topics=topic_config.get("num_topics", 5),
            algorithm=topic_config.get("algorithm", "lda"),
            n_top_words=topic_config.get("top_n_words", 10)
        )
        logger.info(f"Initialized TopicModeler with {topic_config.get('algorithm', 'lda')} algorithm")
        
        # Initialize data loader - store configuration for later use
        self.data_config = self.config.get("data_loading", {})
        # DataLoader will be initialized when needed with specific data
        self.data_loader = None
        logger.info("Data loading configuration prepared")
        
    def process_text(self, text):
        """
        Process text using configured components.
        
        Args:
            text: Input text to process
            
        Returns:
            dict: Processing results
        """
        if not text:
            return {"error": "Empty text provided"}
            
        logger.info(f"Processing text ({len(text)} chars)")
        
        # Configure summarizer
        summarization_config = self.config.get("summarization", {})
        model_config = {
            "maxTokens": summarization_config.get("max_tokens", 150),
            "threshold": summarization_config.get("threshold", 0.3),
            "include_analysis": summarization_config.get("include_analysis", True)
        }
        
        # Process with summarizer
        try:
            summary_result = self.summarizer.process_text(text, model_config)
        except Exception as e:
            logger.error(f"Error in summarizer: {e}")
            summary_result = {
                "summary": "Error generating summary",
                "sum": "Error generating summary",
                "tags": []
            }
        
        # Process with topic modeler if text is long enough
        topic_result = {}
        if len(text.split()) > 20:
            try:
                self.topic_modeler.fit([text])
                topics = self.topic_modeler.get_topics()
                topic_result = {
                    "topics": topics,
                    "algorithm": self.config.get("topic_modeling.algorithm", "lda")
                }
            except Exception as e:
                logger.error(f"Error in topic modeling: {e}")
                topic_result = {
                    "topics": [],
                    "algorithm": self.config.get("topic_modeling.algorithm", "lda"),
                    "error": str(e)
                }
        
        # Combine results
        result = {
            "summary": summary_result.get("summary", ""),
            "condensed": summary_result.get("sum", ""),
            "tags": summary_result.get("tags", []),
            "topics": topic_result.get("topics", [])
        }
        
        # Add additional analysis if available and configured
        if summarization_config.get("include_analysis", True):
            if "sentiment" in summary_result:
                result["sentiment"] = summary_result["sentiment"]
            if "entities" in summary_result:
                result["entities"] = summary_result["entities"]
                
        # Add metadata if configured
        if self.config.get("output.include_metadata", True):
            result["metadata"] = {
                "app_version": self.config.get("app.version"),
                "model": self.config.get("summarization.default_model"),
                "compression_ratio": summary_result.get("compression_ratio", 0)
            }
            
        return result
        
    def save_config(self, path):
        """
        Save current configuration to a file.
        
        Args:
            path: Path to save configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.config.save_to_json(path)
        
    def get_config_summary(self):
        """
        Get a summary of the current configuration.
        
        Returns:
            dict: Configuration summary
        """
        # Create a structured summary from flattened keys
        return {
            "app": {
                "name": self.config.get("app.name", "SUM Platform"),
                "version": self.config.get("app.version", "1.0.0"),
                "debug": self.config.get("app.debug", False),
                "log_level": self.config.get("app.log_level", "INFO")
            },
            "components": {
                "summarizer": self.config.get("summarization.default_model", "simple"),
                "topic_modeler": self.config.get("topic_modeling.algorithm", "lda"),
                "data_loader": {
                    "batch_size": self.config.get("data_loading.batch_size", 100),
                    "workers": self.config.get("data_loading.max_workers", 4)
                }
            },
            "sources": self.config.config_sources
        }


def main():
    """Demonstrate integrated configuration management."""
    print("SUM Configuration Integration Example")
    print("=====================================")
    
    # Create a default configuration file
    config_path = create_default_config()
    print(f"Created default configuration file: {config_path}")
    
    # Initialize application with configuration
    app = SUMApplication(config_path)
    
    # Display configuration summary
    config_summary = app.get_config_summary()
    print("\nConfiguration Summary:")
    print(f"- App: {config_summary['app']['name']} v{config_summary['app']['version']}")
    print(f"- Summarizer: {config_summary['components']['summarizer']}")
    print(f"- Topic Modeler: {config_summary['components']['topic_modeler']}")
    print(f"- Data Loader: batch_size={config_summary['components']['data_loader']['batch_size']}, workers={config_summary['components']['data_loader']['workers']}")
    print(f"- Config Sources: {', '.join(config_summary['sources'])}")
    
    # Sample text for demonstration
    sample_text = """
    Artificial intelligence (AI) is revolutionizing industries across the globe. From healthcare to finance, 
    transportation to entertainment, AI technologies are being integrated into various systems and processes. 
    Machine learning, a subset of AI, enables computers to learn from data and improve their performance over time 
    without explicit programming. Deep learning, which uses neural networks with many layers, has shown remarkable 
    results in tasks such as image recognition, natural language processing, and game playing. However, the rapid 
    advancement of AI also raises important ethical considerations regarding privacy, bias, job displacement, and 
    autonomous decision-making. As AI continues to evolve, it is crucial for researchers, policymakers, and society 
    to address these challenges while harnessing the potential benefits of this transformative technology.
    """
    
    # Process text with the application
    print("\nProcessing sample text...")
    result = app.process_text(sample_text)
    
    # Print processing results
    print("\nProcessing Results:")
    print(f"- Summary: {result['summary']}")
    print(f"- Condensed: {result['condensed']}")
    print(f"- Tags: {result['tags']}")
    
    if "topics" in result and result["topics"]:
        print("\nIdentified Topics:")
        for i, topic in enumerate(result["topics"]):
            print(f"- Topic {i+1}: {', '.join(topic)}")
    
    if "sentiment" in result:
        print(f"\nSentiment: {result['sentiment']}")
        
    if "entities" in result:
        print("\nEntities:")
        for entity, entity_type in result["entities"]:
            print(f"- {entity} ({entity_type})")
    
    # Save modified configuration
    app.config.set("summarization.max_tokens", 200)
    app.config.set("topic_modeling.num_topics", 3)
    modified_config_path = Path("modified_config.json")
    app.save_config(modified_config_path)
    print(f"\nSaved modified configuration to {modified_config_path}")
    
    # Clean up
    config_path.unlink()
    modified_config_path.unlink()
    print("\nExample complete!")

if __name__ == "__main__":
    main()
