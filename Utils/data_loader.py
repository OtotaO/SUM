"""
data_loader.py - Efficient data loading and preprocessing

This module provides utilities for loading and preprocessing data from various sources.
It ensures robust handling of different file formats and secure processing of inputs.

Design principles:
- Robust error handling (Stroustrup)
- Memory efficiency (Knuth)
- Clean, readable code (Torvalds/van Rossum)
- Security-first design (Schneier)
- Extensible architecture (Fowler)

Author: ototao
License: Apache License 2.0
"""

import json
import csv
import os
import logging
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Any, Union, Optional, Tuple
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and preprocess text data from various sources.

    This class handles loading data from different file formats,
    preprocessing text for analysis, and extracting metadata.
    
    Attributes:
        data_file (str, optional): Path to the data file
        data (Any, optional): Pre-loaded data to be used
        stop_words (set): Set of stopwords for preprocessing
        lemmatizer (WordNetLemmatizer): Lemmatizer for normalization
    """

    def __init__(self, data_file: Optional[str] = None, data: Any = None):
        """
        Initialize the DataLoader.

        Args:
            data_file (str, optional): Path to the data file. Defaults to None.
            data (Any, optional): Pre-loaded data to be used. Defaults to None.

        Raises:
            ValueError: If both data_file and data are provided or neither is provided.
        """
        if data_file is None and data is None:
            raise ValueError("Please provide either data_file or data argument.")

        if data_file is not None and data is not None:
            raise ValueError("Cannot provide both data_file and data argument.")

        self.data_file = data_file
        self.data = data
        
        # Initialize NLP components
        self._init_nltk()

    def _init_nltk(self):
        """Initialize and validate NLTK resources."""
        try:
            # Ensure required NLTK resources are available
            for resource in ['punkt', 'stopwords', 'wordnet']:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            # Load and validate stopwords
            raw_stopwords = stopwords.words('english')
            self.stop_words = set(self._validate_words(raw_stopwords))
            
            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            logger.debug("NLTK resources initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK resources: {e}")
            # Fallback to basic stopwords
            self.stop_words = {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with"}
            self.lemmatizer = None

    def _validate_words(self, words: List[str]) -> List[str]:
        """
        Validate words to prevent security issues.
        
        Args:
            words: List of words to validate
            
        Returns:
            List of validated words
        """
        valid_words = []
        unsafe_patterns = [
            r'[\s\S]*exec\s*\(', r'[\s\S]*eval\s*\(', r'[\s\S]*\bimport\b',
            r'[\s\S]*__[a-zA-Z]+__', r'[\s\S]*\bopen\s*\('
        ]
        
        for word in words:
            # Skip unusually long "words"
            if len(word) > 30:
                continue
                
            # Check for unsafe patterns
            if any(re.search(pattern, word) for pattern in unsafe_patterns):
                continue
                
            valid_words.append(word)
            
        return valid_words

    def load_data(self) -> Any:
        """
        Load the data from the specified file or return pre-loaded data.

        Returns:
            The loaded data in an appropriate format.
            
        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the data file doesn't exist.
            Exception: For other loading errors.
        """
        if self.data is not None:
            return self.data

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
        try:
            file_ext = os.path.splitext(self.data_file)[1].lower()
            
            if file_ext == '.json':
                return self._load_json()
            elif file_ext == '.txt':
                return self._load_txt()
            elif file_ext == '.csv':
                return self._load_csv()
            elif file_ext == '.md':
                return self._load_txt()  # Treat markdown as text
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading data from {self.data_file}: {e}")
            raise

    def _load_json(self) -> Any:
        """
        Load data from a JSON file.

        Returns:
            dict or list: The parsed JSON data.
            
        Raises:
            json.JSONDecodeError: If the JSON is invalid.
        """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_txt(self) -> str:
        """
        Load data from a text file.

        Returns:
            str: The loaded text.
        """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_csv(self) -> List[List[str]]:
        """
        Load data from a CSV file.

        Returns:
            list: A list of rows, each row is a list of values.
        """
        rows = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        return rows

    def preprocess_data(self, lowercase: bool = True, lemmatize: bool = True) -> List[List[List[str]]]:
        """
        Preprocess the loaded data for analysis.
        
        This method tokenizes text into sentences and words, removes stopwords,
        and optionally applies lemmatization.

        Args:
            lowercase (bool, optional): Convert tokens to lowercase. Defaults to True.
            lemmatize (bool, optional): Apply lemmatization to tokens. Defaults to True.

        Returns:
            List of documents, each containing a list of sentences, each containing tokens.
        """
        # Load data if not already loaded
        if self.data is None:
            self.data = self.load_data()
            
        # Extract text content based on data type
        texts = self._extract_texts(self.data)
        
        # Process each text
        processed_texts = []
        for text in texts:
            processed_text = []
            
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Tokenize sentence into words
                tokens = word_tokenize(sentence)
                
                # Apply preprocessing steps
                processed_tokens = []
                for token in tokens:
                    # Skip non-alphanumeric tokens and stopwords
                    if not token.isalnum() or token.lower() in self.stop_words:
                        continue
                        
                    # Apply lowercase if needed
                    if lowercase:
                        token = token.lower()
                        
                    # Apply lemmatization if needed
                    if lemmatize and self.lemmatizer:
                        try:
                            token = self.lemmatizer.lemmatize(token)
                        except Exception as e:
                            logger.warning(f"Lemmatization error for token '{token}': {e}")
                    
                    processed_tokens.append(token)
                
                processed_text.append(processed_tokens)
            
            processed_texts.append(processed_text)
            
        return processed_texts

    def _extract_texts(self, data: Any) -> List[str]:
        """
        Extract text content from various data structures.
        
        Args:
            data: Input data in various formats
            
        Returns:
            List of text strings
        """
        texts = []
        
        if isinstance(data, str):
            # Plain text
            texts.append(data)
        elif isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # List of strings
                texts.extend(data)
            elif all(isinstance(item, list) for item in data):
                # CSV-like data
                texts.append('\n'.join([', '.join(map(str, row)) for row in data]))
            elif all(isinstance(item, dict) for item in data):
                # List of objects
                for item in data:
                    if 'content' in item:
                        texts.append(item['content'])
                    elif 'text' in item:
                        texts.append(item['text'])
                    else:
                        texts.append(json.dumps(item))
        elif isinstance(data, dict):
            if 'entries' in data and isinstance(data['entries'], list):
                # Structure like data.json
                for entry in data['entries']:
                    if isinstance(entry, dict) and 'content' in entry:
                        texts.append(entry['content'])
            else:
                # Generic JSON object
                texts.append(json.dumps(data))
        else:
            # Fallback: convert to string
            texts.append(str(data))
            
        return texts

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata about the loaded data.
        
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'source': self.data_file if self.data_file else "memory",
            'type': self._determine_data_type(),
            'size': self._get_data_size()
        }
        
        # Add format-specific metadata
        if self.data_file:
            file_ext = os.path.splitext(self.data_file)[1].lower()
            
            if file_ext == '.json':
                metadata.update(self._get_json_metadata())
            elif file_ext == '.csv':
                metadata.update(self._get_csv_metadata())
                
        return metadata

    def _determine_data_type(self) -> str:
        """Determine the type of the loaded data."""
        if self.data_file:
            ext = os.path.splitext(self.data_file)[1].lower()
            return ext.lstrip('.')
        elif isinstance(self.data, str):
            return "text"
        elif isinstance(self.data, list):
            return "array"
        elif isinstance(self.data, dict):
            return "object"
        else:
            return "unknown"

    def _get_data_size(self) -> int:
        """Get the size of the data (in bytes for files, elements for collections)."""
        if self.data_file:
            return os.path.getsize(self.data_file)
        elif isinstance(self.data, (list, str)):
            return len(self.data)
        elif isinstance(self.data, dict):
            return len(self.data)
        else:
            return 0

    def _get_json_metadata(self) -> Dict[str, Any]:
        """Extract metadata from JSON data."""
        metadata = {}
        
        if isinstance(self.data, dict):
            metadata['keys'] = list(self.data.keys())
            if 'entries' in self.data and isinstance(self.data['entries'], list):
                metadata['entry_count'] = len(self.data['entries'])
        elif isinstance(self.data, list):
            metadata['item_count'] = len(self.data)
            if self.data and isinstance(self.data[0], dict):
                # Sample keys from first item
                metadata['sample_keys'] = list(self.data[0].keys())
                
        return metadata

    def _get_csv_metadata(self) -> Dict[str, Any]:
        """Extract metadata from CSV data."""
        metadata = {}
        
        if isinstance(self.data, list):
            metadata['row_count'] = len(self.data)
            if self.data:
                metadata['column_count'] = len(self.data[0])
                if len(self.data) > 1:
                    # Assume first row is header
                    metadata['headers'] = self.data[0]
                    
        return metadata
