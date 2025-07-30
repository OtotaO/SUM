"""
data_loader.py - Robust data acquisition and preprocessing module

This module handles loading, validating, and preprocessing data from various sources
for the SUM platform. It implements a secure and efficient pipeline for text processing.

Design principles:
- Clean, readable code (Torvalds/van Rossum style)
- Efficient memory usage and performance (Knuth approach)
- Well-documented interfaces (Stroustrup documentation)
- Secure data handling (Schneier principles)
- Extensible design (Fowler architecture)

Author: ototao
License: Apache License 2.0
"""

import json
import os
import logging
import re
import time
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and preprocess text data from various sources with robust validation
    and security measures.
    
    This class handles different data sources (JSON, CSV, text files) and 
    provides comprehensive preprocessing capabilities for NLP tasks.
    
    Attributes:
        data (Union[Dict, List]): The loaded data after preprocessing
        cache_dir (Path): Directory for caching preprocessed data
        nlp (Optional[spacy.Language]): spaCy NLP model for advanced processing
    """
    
    # Secure file extensions whitelist
    ALLOWED_EXTENSIONS = {'.json', '.txt', '.csv', '.md'}
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, 
                 data_file: Optional[str] = None,
                 data: Optional[Union[Dict, List]] = None,
                 cache_dir: Optional[str] = None,
                 use_spacy: bool = False):
        """
        Initialize the DataLoader with optional data source and configuration.
        
        Args:
            data_file: Path to the data file (if loading from file)
            data: Pre-loaded data to use (if not loading from file)
            cache_dir: Directory to cache preprocessed data for performance
            use_spacy: Whether to use spaCy for advanced NLP processing
            
        Raises:
            ValueError: If both data_file and data are provided or neither is provided
            SecurityError: If the file path or data fails validation
        """
        # Input validation
        if data_file is None and data is None:
            raise ValueError("Please provide either data_file or data argument.")
            
        if data_file is not None and data is not None:
            raise ValueError("Cannot provide both data_file and data argument.")
        
        # Initialize cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path('~/.sum_cache').expanduser()
        self._setup_cache()
        
        # Initialize NLP components
        self._initialize_nlp(use_spacy)
        
        # Load data
        self.data = data
        if data_file:
            self._validate_file_path(data_file)
            self.data = self.load_data(data_file)
            
        # Track performance metrics
        self.processing_time = 0.0
            
    def _setup_cache(self) -> None:
        """Set up cache directory securely."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Secure cache directory permissions
            os.chmod(self.cache_dir, 0o700)  # Only owner can access
            logger.debug(f"Cache directory set up at {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Could not set up cache directory: {e}")
            self.cache_dir = None
    
    def _initialize_nlp(self, use_spacy: bool) -> None:
        """Initialize NLP components securely."""
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            # Fallback to basic stopwords
            self.stop_words = {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with"}
            self.lemmatizer = None
            self.stemmer = None
            
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                # Start with a small model by default
                self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
                logger.info("Loaded spaCy model successfully")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}. Falling back to NLTK.")
    
    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate a file path for security and accessibility.
        
        Args:
            file_path: Path to validate
            
        Raises:
            SecurityError: If the file path is invalid or insecure
            FileNotFoundError: If the file doesn't exist
        """
        # Convert to Path object for safer handling
        path = Path(file_path).resolve()
        
        # Validate file existence
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Validate file size
        if path.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {path.stat().st_size} bytes (max {self.MAX_FILE_SIZE})")
            
        # Validate file extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
            
        # Additional path traversal protection
        if "/../" in str(path) or str(path).startswith("../"):
            raise ValueError(f"Insecure file path detected: {file_path}")
    
    def load_data(self, data_file: str) -> Union[Dict, List]:
        """
        Load data from a file with secure handling.
        
        Supports different file formats based on extension with robust error handling.
        
        Args:
            data_file: Path to the data file
            
        Returns:
            The loaded data structure
            
        Raises:
            ValueError: For invalid format or content
            SecurityError: If the content fails validation
        """
        # Check cache first if enabled
        if self.cache_dir:
            cached_data = self._check_cache(data_file)
            if cached_data:
                return cached_data
        
        # Get file extension to determine loading method
        file_path = Path(data_file)
        extension = file_path.suffix.lower()
        
        start_time = time.time()
        
        try:
            # Load based on file extension
            if extension == '.json':
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._validate_json_structure(data)
            elif extension == '.txt':
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = f.read()
                    # For text files, structure as a list of paragraphs
                    data = [p.strip() for p in data.split('\n\n') if p.strip()]
            elif extension == '.csv':
                data = self._load_csv(data_file)
            elif extension == '.md':
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = f.read()
                    # Process markdown into sections
                    data = self._process_markdown(data)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
                
            self.processing_time = time.time() - start_time
            logger.info(f"Loaded data from {data_file} in {self.processing_time:.2f}s")
            
            # Cache the loaded data if caching is enabled
            if self.cache_dir:
                self._cache_data(data_file, data)
                
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {data_file}: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Error loading data from {data_file}: {e}")
            raise
    
    def _validate_json_structure(self, data: Union[Dict, List]) -> None:
        """
        Validate JSON data structure for security and consistency.
        
        Args:
            data: The loaded JSON data to validate
            
        Raises:
            ValueError: If the data structure is invalid or potentially unsafe
        """
        # Check for basic structure
        if not isinstance(data, (dict, list)):
            raise ValueError("JSON data must be a dictionary or list")
            
        # Validate dictionary structure
        if isinstance(data, dict):
            # Check for expected keys if it's a SUM format
            if 'entries' in data:
                if not isinstance(data['entries'], list):
                    raise ValueError("'entries' must be a list")
                # Validate each entry
                for i, entry in enumerate(data['entries']):
                    if not isinstance(entry, dict):
                        raise ValueError(f"Entry {i} must be a dictionary")
                    # Check for required fields
                    for field in ['title', 'content']:
                        if field not in entry:
                            raise ValueError(f"Entry {i} missing required field: {field}")
        
        # For list data, check consistency
        if isinstance(data, list) and data:
            # If list of dictionaries, check they have consistent keys
            if all(isinstance(item, dict) for item in data):
                key_sets = [set(item.keys()) for item in data]
                if len(set.intersection(*key_sets)) == 0:
                    logger.warning("List items have no common keys")
    
    def _load_csv(self, data_file: str) -> List[Dict]:
        """
        Securely load and parse CSV data.
        
        Args:
            data_file: Path to CSV file
            
        Returns:
            List of dictionaries representing CSV rows
        """
        try:
            import csv
            
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                # Use CSV DictReader for structured parsing
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert to dict and validate
                    entry = dict(row)
                    # Check for potential security issues in keys
                    for key in entry.keys():
                        if len(key) > 100 or not re.match(r'^[a-zA-Z0-9_\- ]+$', key):
                            logger.warning(f"Potentially unsafe CSV header: {key}")
                            # Replace with sanitized version
                            entry[re.sub(r'[^a-zA-Z0-9_\- ]', '', key[:100])] = entry.pop(key)
                    data.append(entry)
            return data
            
        except Exception as e:
            logger.error(f"Error parsing CSV {data_file}: {e}")
            raise ValueError(f"Could not parse CSV: {e}")
    
    def _process_markdown(self, content: str) -> List[Dict]:
        """
        Process markdown content into structured sections.
        
        Args:
            content: Markdown content
            
        Returns:
            List of dictionaries with structured content
        """
        # Extract sections based on headings
        sections = []
        current_section = {"title": "Introduction", "content": ""}
        
        lines = content.split('\n')
        for line in lines:
            # Check for headings
            if line.startswith('#'):
                # Save previous section if not empty
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                heading_level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {
                    "title": title,
                    "level": heading_level,
                    "content": ""
                }
            else:
                # Add to current section
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
            
        return sections
    
    def _check_cache(self, data_file: str) -> Optional[Union[Dict, List]]:
        """
        Check if data is cached and still valid.
        
        Args:
            data_file: Original data file path
            
        Returns:
            Cached data if valid, None otherwise
        """
        if not self.cache_dir:
            return None
            
        # Create cache key based on file path and modification time
        file_path = Path(data_file)
        if not file_path.exists():
            return None
            
        # Include file modification time in cache key for freshness
        mod_time = file_path.stat().st_mtime
        cache_key = f"{file_path.name}_{mod_time}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.debug(f"Loaded data from cache: {cache_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                
        return None
    
    def _cache_data(self, data_file: str, data: Union[Dict, List]) -> None:
        """
        Cache processed data for future use.
        
        Args:
            data_file: Original data file path
            data: Processed data to cache
        """
        if not self.cache_dir:
            return
            
        try:
            # Create cache key
            file_path = Path(data_file)
            mod_time = file_path.stat().st_mtime
            cache_key = f"{file_path.name}_{mod_time}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            cache_path = self.cache_dir / f"{cache_key}.json"
            
            # Cache the data securely
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            # Secure file permissions
            os.chmod(cache_path, 0o600)  # Owner read/write only
            
            logger.debug(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def preprocess_data(self, 
                        lowercase: bool = True, 
                        remove_stopwords: bool = True,
                        stem: bool = False, 
                        lemmatize: bool = True) -> List:
        """
        Preprocess text data for analysis with configurable options.
        
        This function handles tokenization, stopword removal, and normalization
        with focus on performance for large datasets.
        
        Args:
            lowercase: Convert tokens to lowercase
            remove_stopwords: Remove common stopwords
            stem: Apply stemming to tokens
            lemmatize: Apply lemmatization (overrides stem if both True)
            
        Returns:
            Preprocessed data as a list of tokenized sentences
            
        Raises:
            TypeError: If data is not in expected format
            ValueError: For invalid preprocessing options
        """
        # Validate input
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded to preprocess")
            
        # Extract raw text based on data structure
        texts = self._extract_texts()
        
        start_time = time.time()
        
        # Preprocess each text
        preprocessed_data = []
        
        for text in texts:
            # Tokenize into sentences then words
            sentences = sent_tokenize(text)
            tokenized_sentences = []
            
            for sentence in sentences:
                # Use spaCy if available for better tokenization
                if self.nlp:
                    doc = self.nlp(sentence)
                    tokens = [token.text for token in doc]
                else:
                    tokens = word_tokenize(sentence)
                
                # Apply preprocessing steps
                if lowercase:
                    tokens = [token.lower() for token in tokens]
                
                if remove_stopwords:
                    tokens = [token for token in tokens if token.lower() not in self.stop_words]
                
                # Apply normalization
                if lemmatize and self.lemmatizer:
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                elif stem and self.stemmer:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                
                # Only include non-empty sentences
                if tokens:
                    tokenized_sentences.append(tokens)
            
            preprocessed_data.append(tokenized_sentences)
        
        self.processing_time = time.time() - start_time
        logger.info(f"Preprocessed {len(texts)} texts in {self.processing_time:.2f}s")
        
        return preprocessed_data
    
    def _extract_texts(self) -> List[str]:
        """
        Extract raw text content from the loaded data structure.
        
        Returns:
            List of text strings for processing
        """
        texts = []
        
        if isinstance(self.data, list):
            # Handle list of dictionaries
            if all(isinstance(item, dict) for item in self.data):
                # Look for content field
                for item in self.data:
                    for field in ['content', 'text', 'body']:
                        if field in item and isinstance(item[field], str):
                            texts.append(item[field])
                            break
                    else:
                        # If no recognized field, use concatenated values
                        text = ' '.join(str(v) for v in item.values() if isinstance(v, str))
                        if text:
                            texts.append(text)
            # Handle list of strings
            elif all(isinstance(item, str) for item in self.data):
                texts.extend(self.data)
            # Handle list of lists (assume tokenized data)
            elif all(isinstance(item, list) for item in self.data):
                # Join nested lists into strings
                for item in self.data:
                    if all(isinstance(subitem, str) for subitem in item):
                        texts.append(' '.join(item))
                    elif all(isinstance(subitem, list) for subitem in item):
                        # Handle nested sentence structure
                        texts.append(' '.join(' '.join(sent) for sent in item if sent))
        
        # Handle dictionary format
        elif isinstance(self.data, dict):
            # Handle SUM-specific format
            if 'entries' in self.data and isinstance(self.data['entries'], list):
                for entry in self.data['entries']:
                    if isinstance(entry, dict) and 'content' in entry:
                        texts.append(entry['content'])
            # Try common field names
            else:
                for field in ['content', 'text', 'body']:
                    if field in self.data and isinstance(self.data[field], str):
                        texts.append(self.data[field])
                        break
        
        # Handle plain string
        elif isinstance(self.data, str):
            texts.append(self.data)
            
        return texts

    def get_metadata(self) -> Dict:
        """
        Extract metadata about the loaded dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        texts = self._extract_texts()
        
        # Calculate basic statistics
        num_texts = len(texts)
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(word_tokenize(text)) for text in texts)
        total_sentences = sum(len(sent_tokenize(text)) for text in texts)
        
        metadata = {
            "num_documents": num_texts,
            "total_characters": total_chars,
            "total_words": total_words,
            "total_sentences": total_sentences,
            "avg_chars_per_doc": total_chars / num_texts if num_texts else 0,
            "avg_words_per_doc": total_words / num_texts if num_texts else 0,
            "avg_sentences_per_doc": total_sentences / num_texts if num_texts else 0,
            "loading_time": self.processing_time
        }
        
        return metadata


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with sample data
    sample_data = {
        "entries": [
            {
                "title": "Sample Article",
                "content": "This is a sample article that demonstrates the data loader functionality."
            }
        ]
    }
    
    # Create loader with in-memory data
    loader = DataLoader(data=sample_data, use_spacy=False)
    
    # Preprocess the data
    processed_data = loader.preprocess_data()
    
    # Get metadata
    metadata = loader.get_metadata()
    
    # Print results
    print("Metadata:", json.dumps(metadata, indent=2))
    print("Processed data sample:", processed_data[0][0])
