"""
config.py - Configuration management for SUM platform

This module provides a centralized configuration system for the SUM platform,
supporting different environments (development, production, testing) and
flexible configuration options.

Design principles:
- Environment-specific configuration
- Centralized settings management
- Flexible validation
- Secure credential handling

Author: ototao
License: Apache License 2.0
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class Config:
    """Base configuration class with common settings."""
    
    # Base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Directory paths
    DATA_DIR = os.path.join(BASE_DIR, 'Data')
    MODELS_DIR = os.path.join(BASE_DIR, 'Models')
    UTILS_DIR = os.path.join(BASE_DIR, 'Utils')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
    UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
    # Data paths
    KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
    DATA_SOURCES = [
        os.path.join(DATA_DIR, 'data_source1.json'),
        os.path.join(DATA_DIR, 'data_source2.json')
    ]

    # Model paths
    LEMMATIZER_MODEL = os.path.join(MODELS_DIR, 'lemmatizer_model.pkl')
    VECTORIZER_MODEL = os.path.join(MODELS_DIR, 'vectorizer_model.pkl')
    NAIVE_BAYES_MODEL = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')

    # Utils paths
    STOP_WORDS_FILE = os.path.join(UTILS_DIR, 'stop_words.txt')

    # Output paths
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'progress.json')
    TOPIC_MODEL_VISUALIZATION = os.path.join(OUTPUT_DIR, 'topic_model_visualization.png')

    # Model parameters
    NUM_TOPICS = int(os.getenv('SUM_NUM_TOPICS', '5'))
    DEFAULT_ALGORITHM = os.getenv('SUM_DEFAULT_ALGORITHM', 'lda')
    
    # Summarization settings
    MAX_SUMMARY_LENGTH = int(os.getenv('SUM_MAX_SUMMARY_LENGTH', '200'))
    MIN_SUMMARY_LENGTH = int(os.getenv('SUM_MIN_SUMMARY_LENGTH', '50'))
    DEFAULT_THRESHOLD = float(os.getenv('SUM_DEFAULT_THRESHOLD', '0.3'))

    # Processing settings
    BATCH_SIZE = int(os.getenv('SUM_BATCH_SIZE', '100'))
    MAX_WORKERS = int(os.getenv('SUM_MAX_WORKERS', '4'))

    # API keys
    API_KEY = os.getenv('SUM_API_KEY', '')

    # Flask settings
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    PORT = int(os.getenv('FLASK_PORT', '3000'))
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('SUM_MAX_CONTENT_LENGTH', str(100 * 1024 * 1024)))  # 100MB for larger files
    # Universal file support - accept any file extension
    ALLOWED_EXTENSIONS = {'txt', 'json', 'csv', 'md', 'pdf', 'docx', 'doc', 'html', 'htm', 'xml', 
                         'rtf', 'odt', 'epub', 'log', 'ini', 'cfg', 'conf', 'yaml', 'yml',
                         'toml', 'sql', 'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp', 'cs',
                         'rb', 'go', 'rs', 'swift', 'kt', 'scala', 'r', 'php', 'pl', 'sh',
                         'bat', 'ps1', 'tsx', 'jsx', 'vue', 'svelte', 'astro', 'lua'}
    # Set to None to accept ALL file extensions
    UNIVERSAL_FILE_SUPPORT = os.getenv('SUM_UNIVERSAL_FILE_SUPPORT', 'True').lower() in ('true', '1', 't')

    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'sum_service.log')
    
    # Semantic Memory settings
    SEMANTIC_MEMORY_MODEL = os.getenv('SUM_SEMANTIC_MEMORY_MODEL', 'all-MiniLM-L6-v2')
    SEMANTIC_MEMORY_PATH = os.getenv('SUM_SEMANTIC_MEMORY_PATH', './semantic_memory')
    USE_GPU_FOR_EMBEDDINGS = os.getenv('SUM_USE_GPU', 'False').lower() in ('true', '1', 't')
    
    # Knowledge Graph settings
    NEO4J_URI = os.getenv('SUM_NEO4J_URI', None)
    NEO4J_USER = os.getenv('SUM_NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('SUM_NEO4J_PASSWORD', None)
    KNOWLEDGE_GRAPH_PATH = os.getenv('SUM_KNOWLEDGE_GRAPH_PATH', './knowledge_graph')
    SPACY_MODEL = os.getenv('SUM_SPACY_MODEL', 'en_core_web_sm')

    @classmethod
    def init_app(cls):
        """Initialize application with this configuration."""
        # Ensure directories exist
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.UTILS_DIR, 
                         cls.OUTPUT_DIR, cls.UPLOADS_DIR, cls.TEMP_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Check for critical files and warn if missing
        critical_paths = [cls.KNOWLEDGE_BASE_PATH, cls.STOP_WORDS_FILE]
        for path in critical_paths:
            if not os.path.exists(path):
                logger.warning(f"Critical file not found: {path}")
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as a dictionary (for API exposure)."""
        # Only include non-sensitive configuration
        safe_config = {
            'num_topics': cls.NUM_TOPICS,
            'default_algorithm': cls.DEFAULT_ALGORITHM,
            'max_summary_length': cls.MAX_SUMMARY_LENGTH,
            'min_summary_length': cls.MIN_SUMMARY_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'max_workers': cls.MAX_WORKERS,
            'allowed_extensions': list(cls.ALLOWED_EXTENSIONS)
        }
        return safe_config
    
    @classmethod
    def from_json(cls, config_file: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Config instance with values from JSON file
        """
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            return cls
            
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Create a new config instance
            new_config = cls()
            
            # Update attributes from JSON
            for key, value in config_data.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
                    
            return new_config
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
            return cls


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    
    # More conservative settings for production
    MAX_WORKERS = min(os.cpu_count() or 4, 8)  # Limit to available CPUs or 8
    
    # More secure settings
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')  # Must be provided in production
    
    @classmethod
    def init_app(cls):
        super().init_app()
        
        # Additional production checks
        if not cls.SECRET_KEY:
            logger.warning("No SECRET_KEY provided for production environment")
            
        if cls.DEBUG:
            logger.warning("DEBUG mode should not be enabled in production")


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = False
    
    # Use in-memory/temporary storage for tests
    DATA_DIR = os.path.join(Config.BASE_DIR, 'Tests', 'test_data')
    OUTPUT_DIR = os.path.join(Config.BASE_DIR, 'Tests', 'test_output')
    
    @classmethod
    def init_app(cls):
        super().init_app()
        # Ensure test directories exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Set the active configuration
env = os.getenv('FLASK_ENV', 'default')
active_config = config.get(env, config['default'])
logger.info(f"Using {env} configuration")

# Initialize the active configuration
active_config.init_app()
