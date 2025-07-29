"""
topic_modeling.py - Advanced topic modeling engine

This module provides topic modeling capabilities for the SUM platform,
supporting various algorithms including LDA, NMF, and LSA.

Design principles:
- Algorithm efficiency (Knuth)
- Code readability (Torvalds/van Rossum)
- Extensible architecture (Fowler)
- Secure implementation (Schneier)
- Scientific rigor (Dijkstra)

Author: ototao
License: Apache License 2.0
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from nltk.corpus import stopwords
import re
import time
from concurrent.futures import ThreadPoolExecutor
import os

# Configure logging
logger = logging.getLogger(__name__)

class TopicModeler:
    """
    Topic modeling using various algorithms (LDA, NMF, LSA).
    
    Provides methods for extracting topics from text data with support
    for different algorithms and visualization options.
    """
    
    SUPPORTED_ALGORITHMS = ['lda', 'nmf', 'lsa']
    
    def __init__(self, n_topics: int = 5, algorithm: str = 'lda', n_top_words: int = 10):
        """
        Initialize the topic modeler.
        
        Args:
            n_topics: Number of topics to extract (default: 5)
            algorithm: Algorithm to use - 'lda', 'nmf', or 'lsa' (default: 'lda')
            n_top_words: Number of top words to include per topic (default: 10)
        """
        self.n_topics = max(1, min(n_topics, 100))  # Bound between 1 and 100
        self.algorithm = algorithm.lower()
        self.n_top_words = max(1, min(n_top_words, 50))  # Bound between 1 and 50
        
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Initialize other attributes
        self.vectorizer = None
        self.model = None
        self.features = []
        self.coherence_scores = []
        self.dtm = None
        self.topic_word_matrix = None
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize the vectorizer and model based on the selected algorithm."""
        # Get stopwords - use sklearn's built-in 'english' stopwords string
        # rather than a set to avoid compatibility issues with CountVectorizer
        try:
            # Just check if we can access stopwords, but use the string 'english'
            # for compatibility with sklearn's vectorizers
            _ = stopwords.words('english')
            stop_words = 'english'
        except:
            stop_words = 'english'  # Fall back to sklearn's built-in stopwords
        
        # Initialize vectorizer with parameters that work for both small and large document sets
        if self.algorithm == 'lda':
            # LDA typically works better with raw term counts
            self.vectorizer = CountVectorizer(
                max_df=1.0,          # Don't filter out any terms based on document frequency for small datasets
                min_df=1,            # Include terms that appear in at least 1 document
                stop_words=stop_words,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Only words with at least 2 letters
                max_features=10000   # Limit vocab size for efficiency
            )
        else:
            # NMF and LSA typically work better with TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_df=1.0,
                min_df=1,
                stop_words=stop_words,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
                max_features=10000,
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True    # Apply sublinear tf scaling (1 + log(tf))
            )
        
        # Initialize model
        if self.algorithm == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=20,         # Maximum number of iterations
                learning_method='online',
                learning_offset=50.0,
                random_state=42,      # For reproducibility
                n_jobs=-1            # Use all processors
            )
        elif self.algorithm == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                alpha=0.1,           # Sparseness controller
                l1_ratio=0.5,        # Balance between L1 and L2 regularization
                max_iter=200,
                init='nndsvd'        # Better initialization for sparse data
            )
        elif self.algorithm == 'lsa':
            self.model = TruncatedSVD(
                n_components=self.n_topics,
                algorithm='randomized',
                n_iter=5,            # Number of iterations for randomized SVD
                random_state=42
            )
    
    def fit(self, texts: List[str]) -> 'TopicModeler':
        """
        Fit the topic model to the data.
        
        Args:
            texts: List of text documents to analyze
            
        Returns:
            Self for method chaining
        """
        if not texts:
            raise ValueError("Empty text list provided")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements must be strings")
        
        start_time = time.time()
        logger.info(f"Fitting {self.algorithm.upper()} topic model to {len(texts)} documents")
        
        try:
            # For large datasets, use parallel processing
            if len(texts) > 100:
                self._fit_parallel(texts)
            else:
                # Transform texts to document-term matrix
                self.dtm = self.vectorizer.fit_transform(texts)
                self.features = self.vectorizer.get_feature_names_out()
                
                # Fit the model
                self.model.fit(self.dtm)
            
            # Extract the topic-word matrix
            self.topic_word_matrix = self.model.components_
                
            # Calculate coherence scores
            self._calculate_coherence_scores()
            
            logger.info(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            raise
    
    def _fit_parallel(self, texts: List[str]) -> None:
        """
        Fit the model using parallel processing for large datasets.
        
        Args:
            texts: List of text documents to analyze
        """
        # For very large datasets, preprocess in chunks
        chunk_size = max(100, len(texts) // (os.cpu_count() or 4))
        
        def preprocess_chunk(chunk_texts):
            return [self._preprocess_text(text) for text in chunk_texts]
        
        # Process texts in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                futures.append(executor.submit(preprocess_chunk, chunk))
            
            # Collect processed texts
            processed_texts = []
            for future in futures:
                processed_texts.extend(future.result())
        
        # Now fit on processed texts
        self.dtm = self.vectorizer.fit_transform(processed_texts)
        self.features = self.vectorizer.get_feature_names_out()
        self.model.fit(self.dtm)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing - in production, you might want more sophisticated steps
        text = text.lower()
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            texts: List of text documents to transform
            
        Returns:
            Document-topic matrix
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform texts to DTM
        dtm = self.vectorizer.transform(texts)
        
        # Transform to topic space
        return self.model.transform(dtm)
    
    def get_topics(self) -> List[List[str]]:
        """
        Get the top words for each topic.
        
        Returns:
            List of topics, each containing top words
        """
        if self.topic_word_matrix is None or not self.features:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topics = []
        
        for topic_idx, topic in enumerate(self.topic_word_matrix):
            # Get top word indices for this topic
            top_words_idx = topic.argsort()[:-self.n_top_words-1:-1]
            
            # Get the actual words
            top_words = [self.features[i] for i in top_words_idx]
            topics.append(top_words)
        
        return topics
    
    def get_topics_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the topics.
        
        Returns:
            Dictionary with topic information
        """
        if self.topic_word_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get top words for each topic
        topics = self.get_topics()
        
        # Prepare the summary
        topics_summary = {
            'algorithm': self.algorithm,
            'num_topics': self.n_topics,
            'topics': []
        }
        
        # Add details for each topic
        for idx, topic_words in enumerate(topics):
            topic_info = {
                'id': idx,
                'top_words': topic_words,
                'coherence': self.coherence_scores[idx] if idx < len(self.coherence_scores) else None
            }
            topics_summary['topics'].append(topic_info)
        
        return topics_summary
    
    def _calculate_coherence_scores(self) -> None:
        """Calculate coherence scores for each topic."""
        # For small datasets, use a simplified coherence calculation
        # to avoid sparse matrix issues
        coherence_scores = []
        
        # If we have very few documents, just assign neutral coherence scores
        if self.dtm.shape[0] <= 2:
            self.coherence_scores = [0.0] * self.n_topics
            return
            
        try:
            # Get co-occurrence counts from the DTM
            dtm_csc = self.dtm.tocsc()
            doc_counts = np.zeros(len(self.features), dtype=np.int32)
            
            # Count documents containing each word
            for i in range(len(self.features)):
                doc_counts[i] = dtm_csc[:, i].nnz
            
            for topic_idx, topic in enumerate(self.topic_word_matrix):
                # Get top word indices for this topic
                top_word_indices = topic.argsort()[:-self.n_top_words-1:-1]
                
                # Calculate coherence score for this topic
                score = 0.0
                pairs = 0
                
                for i in range(1, len(top_word_indices)):
                    word_i = top_word_indices[i]
                    
                    for j in range(0, i):
                        word_j = top_word_indices[j]
                        
                        # Count documents with both words - safely access sparse matrix
                        co_docs = 0
                        for doc_idx in range(self.dtm.shape[0]):
                            # Safely get values from sparse matrix
                            val_i = dtm_csc[doc_idx, word_i]
                            val_j = dtm_csc[doc_idx, word_j]
                            
                            # Check if both words exist in this document
                            if (isinstance(val_i, np.ndarray) and val_i.size > 0 and val_i[0] > 0) or \
                               (not isinstance(val_i, np.ndarray) and val_i > 0):
                                if (isinstance(val_j, np.ndarray) and val_j.size > 0 and val_j[0] > 0) or \
                                   (not isinstance(val_j, np.ndarray) and val_j > 0):
                                    co_docs += 1
                        
                        # Add to score using UMass approach: log(P(w_i, w_j) / P(w_j))
                        if co_docs > 0 and doc_counts[word_j] > 0:
                            pair_score = np.log((co_docs + 1) / (doc_counts[word_j] + 1))
                            score += pair_score
                            pairs += 1
                
                # Average coherence
                coherence = score / max(1, pairs)
                coherence_scores.append(coherence)
        except Exception as e:
            logger.warning(f"Error calculating coherence scores: {e}")
            # Fallback to neutral coherence scores
            coherence_scores = [0.0] * self.n_topics
        
        self.coherence_scores = coherence_scores
    
    def visualize_topics(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate visualization data for topics.
        
        Args:
            output_file: Optional file path to save visualization data
            
        Returns:
            Dictionary with visualization data
        """
        if self.topic_word_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topics = self.get_topics()
        
        # Create visualization data
        viz_data = {
            'topics': topics,
            'coherence_scores': self.coherence_scores,
            'algorithm': self.algorithm,
            'num_topics': self.n_topics
        }
        
        # Save to file if requested
        if output_file:
            try:
                import json
                with open(output_file, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                logger.info(f"Visualization data saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving visualization data: {e}")
        
        return viz_data
