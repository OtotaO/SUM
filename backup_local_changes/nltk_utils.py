"""
nltk_utils.py - Centralized NLTK resource management

This module provides centralized utilities for managing NLTK resources,
ensuring consistent initialization and resource availability across the SUM platform.

Design principles:
- Centralized resource management
- Efficient initialization
- Robust error handling
- Thread safety

Author: ototao
License: Apache License 2.0
"""

import os
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Set, Optional, List
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Thread safety for resource initialization
_nltk_lock = threading.Lock()
_initialized = False
_stopwords_cache = None
_lemmatizer_cache = None

# Define required NLTK resources
REQUIRED_RESOURCES = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'vader_lexicon'
]

def initialize_nltk(resources: Optional[List[str]] = None, 
                   download_dir: Optional[str] = None,
                   quiet: bool = True) -> bool:
    """
    Initialize NLTK resources in a centralized, thread-safe manner.
    
    Args:
        resources: List of NLTK resources to initialize (None = use default list)
        download_dir: Directory to download resources to (None = use default)
        quiet: Whether to suppress download messages
        
    Returns:
        True if initialization was successful, False otherwise
    """
    global _initialized, _stopwords_cache, _lemmatizer_cache
    
    # Use default resources if none provided
    resources_to_download = resources or REQUIRED_RESOURCES
    
    # Set download directory
    if download_dir:
        nltk_data_dir = download_dir
    else:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    # Ensure download directory exists
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Use lock to ensure thread safety
    with _nltk_lock:
        if _initialized:
            return True
            
        try:
            # Download each resource
            for resource in resources_to_download:
                try:
                    nltk.download(resource, download_dir=nltk_data_dir, quiet=quiet)
                    logger.debug(f"Successfully downloaded NLTK resource: {resource}")
                except Exception as e:
                    logger.error(f"Error downloading NLTK resource {resource}: {str(e)}")
                    return False
            
            # Initialize stopwords
            try:
                _stopwords_cache = set(stopwords.words('english'))
                logger.debug("Initialized stopwords cache")
            except Exception as e:
                logger.error(f"Error initializing stopwords: {str(e)}")
                _stopwords_cache = {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with"}
                logger.warning("Using fallback stopwords")
            
            # Initialize lemmatizer
            try:
                _lemmatizer_cache = WordNetLemmatizer()
                logger.debug("Initialized lemmatizer cache")
            except Exception as e:
                logger.error(f"Error initializing lemmatizer: {str(e)}")
                _lemmatizer_cache = None
            
            _initialized = True
            logger.info("NLTK resources initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during NLTK initialization: {str(e)}")
            return False

def get_stopwords() -> Set[str]:
    """
    Get the set of English stopwords.
    
    Returns:
        Set of stopwords
    """
    global _stopwords_cache
    
    # Initialize if not already done
    if not _initialized:
        initialize_nltk()
    
    return _stopwords_cache

def get_lemmatizer() -> Optional[WordNetLemmatizer]:
    """
    Get the WordNet lemmatizer.
    
    Returns:
        WordNetLemmatizer instance or None if initialization failed
    """
    global _lemmatizer_cache
    
    # Initialize if not already done
    if not _initialized:
        initialize_nltk()
    
    return _lemmatizer_cache

def is_initialized() -> bool:
    """
    Check if NLTK resources have been initialized.
    
    Returns:
        True if initialized, False otherwise
    """
    return _initialized

def download_specific_resource(resource: str, 
                             download_dir: Optional[str] = None,
                             quiet: bool = True) -> bool:
    """
    Download a specific NLTK resource.
    
    Args:
        resource: Name of the resource to download
        download_dir: Directory to download to (None = use default)
        quiet: Whether to suppress download messages
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        # Set download directory
        if download_dir:
            nltk_data_dir = download_dir
        else:
            nltk_data_dir = os.path.expanduser('~/nltk_data')
        
        # Ensure download directory exists
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download the resource
        nltk.download(resource, download_dir=nltk_data_dir, quiet=quiet)
        logger.info(f"Successfully downloaded NLTK resource: {resource}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading NLTK resource {resource}: {str(e)}")
        return False
