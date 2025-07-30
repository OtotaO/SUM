"""
topic_modeling.py - Advanced topic modeling and knowledge extraction engine

This module implements sophisticated algorithms for discovering latent themes and
concepts in text data, with optimizations for performance and accuracy.

Design principles:
- Algorithmic elegance (Dijkstra approach)
- Performance optimization (Knuth principles)
- Clean, maintainable code (Torvalds/van Rossum style)
- Comprehensive documentation (Stroustrup methodology)
- Secure implementation (Schneier security focus)

Author: ototao
License: Apache License 2.0
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional, Any
import time
import os
from functools import lru_cache
from threading import Lock
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Configure logging
logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Advanced topic modeling engine with multiple algorithm options and optimizations.
    
    This class provides a unified interface to various topic modeling techniques
    with smart parameter selection, parallelization, and result interpretation.
    
    Attributes:
        n_topics (int): Number of topics to extract
        algorithm (str): Algorithm used for topic modeling
        max_features (int): Maximum number of terms in the vocabulary
        model: The underlying topic model instance
        vectorizer: The text vectorization component
        topic_words (List[List[str]]): Top words for each topic
        topic_word_scores (List[List[float]]): Word importance scores
        topic_viz_data (Dict): Visualization data for topics
    """
    
    # Supported algorithms
    ALGORITHMS = ['lda', 'nmf', 'lsa']
    
    # Hyperparameter ranges for auto-tuning
    PARAM_RANGES = {
        'lda': {
            'learning_decay': [0.5, 0.7, 0.9],
            'max_iter': [10, 20, 50],
            'doc_topic_prior': [None, 0.1, 0.5, 1.0, 5.0],
            'topic_word_prior': [None, 0.1, 0.5, 1.0, 5.0]
        },
        'nmf': {
            'alpha': [0.0, 0.1, 0.5],
            'l1_ratio': [0.0, 0.5, 1.0],
            'max_iter': [100, 200, 500]
        },
        'lsa': {
            'algorithm': ['arpack', 'randomized'],
            'n_iter': [5, 10, 20]
        }
    }
    
    def __init__(self, 
                 n_topics: int = 5, 
                 algorithm: str = 'lda',
                 max_features: int = 5000,
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95,
                 n_top_words: int = 10,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 use_tfidf: bool = False,
                 auto_optimize: bool = False):
        """
        Initialize the topic modeler with the specified settings.
        
        Args:
            n_topics: Number of topics to extract
            algorithm: Topic modeling algorithm ('lda', 'nmf', or 'lsa')
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            n_top_words: Number of top words to extract per topic
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            use_tfidf: Whether to use TF-IDF weighting (vs. raw counts)
            auto_optimize: Whether to automatically optimize hyperparameters
            
        Raises:
            ValueError: For invalid configuration parameters
        """
        # Validate inputs
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.ALGORITHMS}")
            
        if n_topics < 1:
            raise ValueError("Number of topics must be at least 1")
            
        if n_top_words < 1:
            raise ValueError("Number of top words must be at least 1")
            
        # Store configuration
        self.n_topics = n_topics
        self.algorithm = algorithm.lower()
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.n_top_words = n_top_words
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() or 1
        self.use_tfidf = use_tfidf
        self.auto_optimize = auto_optimize
        
        # Initialize internal state
        self.model = None
        self.vectorizer = None
        self.vocabulary = None
        self.feature_names = None
        self.topic_words = []
        self.topic_word_scores = []
        self.topic_viz_data = {}
        self.coherence_scores = []
        self.is_fitted = False
        
        # Thread safety for parallel operations
        self._lock = Lock()
        
        # Initialize vectorizer and model
        self._initialize_components()
        
        logger.info(f"Initialized TopicModeler with algorithm={algorithm}, n_topics={n_topics}")
    
    def _initialize_components(self) -> None:
        """Set up the vectorizer and topic model with optimal configuration."""
        # Choose vectorizer based on configuration
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        
        # Initialize model based on selected algorithm
        if self.algorithm == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=20,
                learning_method='online',
                learning_offset=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.algorithm == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                max_iter=200,
                random_state=self.random_state,
                alpha=0.1
            )
        elif self.algorithm == 'lsa':
            self.model = TruncatedSVD(
                n_components=self.n_topics,
                random_state=self.random_state,
                algorithm='randomized'
            )
    
    def fit(self, documents: List[str], auto_interpret: bool = True) -> 'TopicModeler':
        """
        Fit the topic model to the provided documents.
        
        Args:
            documents: List of text documents to analyze
            auto_interpret: Whether to automatically extract topic interpretations
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If the input data is invalid
        """
        start_time = time.time()
        logger.info(f"Fitting topic model to {len(documents)} documents")
        
        # Validate input
        if not documents or not all(isinstance(doc, str) for doc in documents):
            raise ValueError("Documents must be a non-empty list of strings")
        
        # Preprocess the documents (remove very short docs)
        valid_docs = [doc for doc in documents if len(doc.strip().split()) >= 3]
        if len(valid_docs) < len(documents):
            logger.warning(f"Filtered out {len(documents) - len(valid_docs)} very short documents")
        
        if not valid_docs:
            raise ValueError("No valid documents after filtering")
        
        # Transform the documents to matrix representation
        document_term_matrix = self.vectorizer.fit_transform(valid_docs)
        self.vocabulary = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Document-term matrix shape: {document_term_matrix.shape}")
        
        # Auto-optimize if requested
        if self.auto_optimize:
            self._optimize_hyperparameters(document_term_matrix)
        
        # Fit the model
        self.model.fit(document_term_matrix)
        
        # Extract topic interpretations
        if auto_interpret:
            self._extract_topic_words()
            self._calculate_coherence(document_term_matrix)
        
        self.is_fitted = True
        
        fitting_time = time.time() - start_time
        logger.info(f"Topic model fitting completed in {fitting_time:.2f}s")
        
        return self
    
    def _optimize_hyperparameters(self, document_term_matrix: Any) -> None:
        """
        Automatically tune model hyperparameters for optimal results.
        
        Args:
            document_term_matrix: Vectorized document data
        """
        logger.info("Performing hyperparameter optimization")
        
        best_score = float('-inf')
        best_params = {}
        
        # Get parameter ranges for current algorithm
        param_ranges = self.PARAM_RANGES.get(self.algorithm, {})
        if not param_ranges:
            logger.warning(f"No parameter ranges defined for {self.algorithm}, skipping optimization")
            return
        
        # Generate parameter combinations (simplified grid search)
        from itertools import product
        
        # Basic subset of parameters to avoid combinatorial explosion
        basic_params = {}
        for k, v in list(param_ranges.items())[:2]:  # Take first two parameters only
            basic_params[k] = v
        
        param_names = list(basic_params.keys())
        param_values = list(basic_params.values())
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # Create and fit model with these parameters
            try:
                model_copy = self._create_model_with_params(params)
                model_copy.fit(document_term_matrix)
                
                # Score the model (use different metrics based on algorithm)
                if self.algorithm == 'lda':
                    score = model_copy.score(document_term_matrix)
                elif self.algorithm in ['nmf', 'lsa']:
                    # For NMF/LSA, use reconstruction error
                    transformed = model_copy.transform(document_term_matrix)
                    reconstructed = transformed @ model_copy.components_
                    score = -np.mean((document_term_matrix - reconstructed) ** 2)
                
                logger.debug(f"Parameters {params}, score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.warning(f"Error fitting model with parameters {params}: {e}")
        
        # Update model with best parameters
        if best_params:
            logger.info(f"Best parameters found: {best_params} with score {best_score}")
            self.model = self._create_model_with_params(best_params)
        else:
            logger.warning("Optimization failed to find better parameters")
    
    def _create_model_with_params(self, params: Dict) -> Any:
        """
        Create a model instance with the specified parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Configured model instance
        """
        # Start with base parameters
        base_params = {
            'n_components': self.n_topics,
            'random_state': self.random_state
        }
        
        # Add algorithm-specific base parameters
        if self.algorithm == 'lda':
            base_params.update({
                'learning_method': 'online',
                'n_jobs': self.n_jobs
            })
        
        # Update with optimization parameters
        base_params.update(params)
        
        # Create appropriate model
        if self.algorithm == 'lda':
            return LatentDirichletAllocation(**base_params)
        elif self.algorithm == 'nmf':
            return NMF(**base_params)
        elif self.algorithm == 'lsa':
            return TruncatedSVD(**base_params)
    
    def _extract_topic_words(self) -> None:
        """Extract the top words for each topic from the fitted model."""
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot extract topic words")
            return
        
        # Get the feature names from the vectorizer
        feature_names = self.feature_names
        
        # Extract the weight of each term for each topic
        if self.algorithm in ['lda', 'nmf']:
            topic_word_dist = self.model.components_
        elif self.algorithm == 'lsa':
            # For LSA, we need to handle differently since components might have negative values
            topic_word_dist = np.abs(self.model.components_)
        
        # Extract top words for each topic
        self.topic_words = []
        self.topic_word_scores = []
        
        for topic_idx, topic in enumerate(topic_word_dist):
            # Sort terms by weight
            sorted_indices = topic.argsort()[::-1]
            top_indices = sorted_indices[:self.n_top_words]
            
            # Get top words and their weights
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [topic[i] for i in top_indices]
            
            self.topic_words.append(top_words)
            self.topic_word_scores.append(top_scores)
            
            # Log topic for debugging
            logger.debug(f"Topic {topic_idx}: " + ", ".join(
                f"{word} ({score:.3f})" for word, score in zip(top_words, top_scores)
            ))
    
    def _calculate_coherence(self, document_term_matrix: Any) -> None:
        """
        Calculate topic coherence scores to evaluate model quality.
        
        Args:
            document_term_matrix: Vectorized document data
        """
        # UMass coherence metric (simplified)
        self.coherence_scores = []
        
        # Get document-topic distribution
        if hasattr(self.model, 'transform'):
            doc_topic_dist = self.model.transform(document_term_matrix)
        elif hasattr(self.model, 'fit_transform'):
            # Some models don't separate fit and transform
            doc_topic_dist = self.model.fit_transform(document_term_matrix)
        
        # Feature co-occurrence matrix (approximation for speed)
        term_freq = document_term_matrix.sum(axis=0).A1
        cooccurrence = (document_term_matrix.T @ document_term_matrix).toarray()
        np.fill_diagonal(cooccurrence, 0)  # Exclude self-cooccurrence
        
        # Calculate coherence for each topic
        for topic_idx, topic_words_indices in enumerate(
            [np.argsort(topic)[::-1][:self.n_top_words] for topic in self.model.components_]
        ):
            coherence = 0.0
            count = 0
            
            # Calculate pairwise coherence for top words
            for i in range(len(topic_words_indices)):
                for j in range(i+1, len(topic_words_indices)):
                    term_i, term_j = topic_words_indices[i], topic_words_indices[j]
                    
                    # Skip if either term is very rare
                    if term_freq[term_i] < 2 or term_freq[term_j] < 2:
                        continue
                    
                    # Calculate co-occurrence based coherence
                    joint_prob = cooccurrence[term_i, term_j] / cooccurrence.sum()
                    term_i_prob = term_freq[term_i] / term_freq.sum()
                    term_j_prob = term_freq[term_j] / term_freq.sum()
                    
                    # Add smoothing to avoid log(0)
                    score = np.log((joint_prob + 0.01) / (term_i_prob * term_j_prob + 0.01))
                    coherence += score
                    count += 1
            
            # Average coherence
            if count > 0:
                coherence /= count
            
            self.coherence_scores.append(coherence)
            
        logger.info(f"Topic coherence scores: {[round(score, 3) for score in self.coherence_scores]}")
        logger.info(f"Mean coherence: {np.mean(self.coherence_scores):.3f}")
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to topic space.
        
        Args:
            documents: List of text documents to transform
            
        Returns:
            Document-topic matrix (one row per document, one column per topic)
            
        Raises:
            ValueError: If the model is not fitted or input is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
            
        if not documents:
            raise ValueError("Documents list cannot be empty")
            
        # Vectorize the documents
        X = self.vectorizer.transform(documents)
        
        # Transform to topic space
        return self.model.transform(X)
    
    def get_topic_terms(self, topic_idx: int, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get the top terms for a specific topic with their weights.
        
        Args:
            topic_idx: Index of the topic (0-based)
            top_n: Number of top terms to return (default: use n_top_words)
            
        Returns:
            List of (term, weight) tuples for the topic
            
        Raises:
            ValueError: If topic_idx is out of range or model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise ValueError(f"Topic index must be between 0 and {self.n_topics-1}")
            
        n = top_n or self.n_top_words
        return list(zip(self.topic_words[topic_idx][:n], self.topic_word_scores[topic_idx][:n]))
    
    def get_topics_summary(self) -> Dict[int, Dict]:
        """
        Get a comprehensive summary of all topics.
        
        Returns:
            Dictionary mapping topic IDs to topic metadata
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        topics_summary = {}
        
        for topic_idx in range(self.n_topics):
            topic_terms = self.get_topic_terms(topic_idx)
            
            # Generate a short name/label for the topic based on top 3 words
            topic_label = " + ".join(term for term, _ in topic_terms[:3])
            
            topics_summary[topic_idx] = {
                'id': topic_idx,
                'label': topic_label,
                'terms': dict(topic_terms),
                'coherence': self.coherence_scores[topic_idx] if self.coherence_scores else None
            }
            
        return topics_summary
    
    def predict_topic(self, text: str) -> Tuple[int, float]:
        """
        Predict the most likely topic for a text document.
        
        Args:
            text: The text to classify
            
        Returns:
            Tuple of (topic_id, confidence)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Vectorize the document
        X = self.vectorizer.transform([text])
        
        # Get topic distribution
        topic_dist = self.model.transform(X)[0]
        
        # Find the dominant topic
        dominant_topic = topic_dist.argmax()
        confidence = topic_dist[dominant_topic]
        
        return dominant_topic, confidence
    
    def visualize_topics(self, output_file: Optional[str] = None, 
                         width: int = 1200, height: int = 800,
                         show_plot: bool = False) -> Dict:
        """
        Generate a visualization of the topics.
        
        Args:
            output_file: Path to save the visualization (optional)
            width: Width of the plot in pixels
            height: Height of the plot in pixels
            show_plot: Whether to display the plot interactively
            
        Returns:
            Dictionary with visualization data
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            plt.figure(figsize=(width/100, height/100), dpi=100)
            
            # Create a grid for the word clouds
            grid_size = int(np.ceil(np.sqrt(self.n_topics)))
            
            for topic_idx in range(self.n_topics):
                plt.subplot(grid_size, grid_size, topic_idx + 1)
                
                # Get the top words for this topic with their weights
                top_words = self.topic_words[topic_idx]
                top_weights = self.topic_word_scores[topic_idx]
                
                # Scale weights for better visualization
                min_weight = min(top_weights)
                max_weight = max(top_weights)
                scaled_weights = [(w - min_weight) / (max_weight - min_weight + 1e-6) * 100 + 10 
                                 for w in top_weights]
                
                # Create a dictionary of word frequencies for the wordcloud
                word_freq = {word: weight for word, weight in zip(top_words, scaled_weights)}
                
                # Generate word cloud
                wordcloud = WordCloud(width=400, height=400, background_color='white', 
                                      max_words=self.n_top_words, prefer_horizontal=1.0,
                                      color_func=lambda *args, **kwargs: 'black',
                                      normalize_plurals=False)
                wordcloud.generate_from_frequencies(word_freq)
                
                # Display the word cloud
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # Add title with coherence score if available
                title = f"Topic {topic_idx+1}"
                if self.coherence_scores:
                    title += f" (C: {self.coherence_scores[topic_idx]:.2f})"
                plt.title(title)
            
            plt.tight_layout()
            
            # Save the visualization
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                logger.info(f"Topic visualization saved to {output_file}")
            
            # Show the plot if requested
            if show_plot:
                plt.show()
            
            # Store visualization data
            self.topic_viz_data = {
                'topics': self.get_topics_summary(),
                'coherence': self.coherence_scores,
                'mean_coherence': np.mean(self.coherence_scores) if self.coherence_scores else None
            }
            
            return self.topic_viz_data
            
        except Exception as e:
            logger.error(f"Error generating topic visualization: {e}")
            raise
        finally:
            plt.close()
    
    def cluster_documents(self, documents: List[str], n_clusters: Optional[int] = None,
                       method: str = 'kmeans') -> Tuple[List[int], Dict]:
        """
        Cluster documents based on their topic distributions.
        
        Args:
            documents: List of text documents to cluster
            n_clusters: Number of clusters (default: n_topics)
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Tuple of (cluster_labels, cluster_metadata)
            
        Raises:
            ValueError: If model is not fitted or method is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Default to number of topics if not specified
        n_clusters = n_clusters or self.n_topics
        
        # Transform documents to topic space
        doc_topic_dist = self.transform(documents)
        
        # Apply clustering
        labels = None
        method = method.lower()
        
        if method == 'kmeans':
            # Use KMeans for clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(doc_topic_dist)
            centroids = kmeans.cluster_centers_
            
            # Calculate distances to centroids
            cluster_distances = []
            for i, centroid in enumerate(centroids):
                # Get documents in this cluster
                cluster_docs = doc_topic_dist[labels == i]
                if len(cluster_docs) > 0:
                    # Calculate average distance to centroid
                    distances = np.sqrt(((cluster_docs - centroid) ** 2).sum(axis=1))
                    avg_distance = distances.mean()
                    cluster_distances.append(avg_distance)
                else:
                    cluster_distances.append(0)
                    
            # Calculate silhouette score if there are enough clusters and documents
            silhouette = 0.0
            if n_clusters > 1 and len(documents) > n_clusters:
                from sklearn.metrics import silhouette_score
                try:
                    silhouette = silhouette_score(doc_topic_dist, labels)
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score: {e}")
            
            metadata = {
                'centroids': centroids.tolist(),
                'cluster_sizes': [(labels == i).sum() for i in range(n_clusters)],
                'cluster_distances': cluster_distances,
                'silhouette_score': silhouette
            }
            
        elif method == 'dbscan':
            # Use DBSCAN for density-based clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(doc_topic_dist)
            
            # Count documents in each cluster
            unique_labels = set(labels)
            cluster_sizes = [(labels == i).sum() for i in unique_labels]
            
            metadata = {
                'n_clusters': len(unique_labels) - (1 if -1 in labels else 0),
                'cluster_sizes': cluster_sizes,
                'noise_points': (labels == -1).sum()
            }
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        return labels.tolist(), metadata
    
    def detect_outliers(self, documents: List[str], threshold: float = 2.0) -> List[bool]:
        """
        Detect outlier documents based on their topic distribution.
        
        Args:
            documents: List of text documents to analyze
            threshold: Standard deviation threshold for outlier detection
            
        Returns:
            List of boolean flags (True for outliers)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Transform documents to topic space
        doc_topic_dist = self.transform(documents)
        
        # Calculate average topic vector
        mean_vector = doc_topic_dist.mean(axis=0)
        
        # Calculate distances from mean
        distances = np.sqrt(((doc_topic_dist - mean_vector) ** 2).sum(axis=1))
        
        # Find statistical thresholds for outliers
        mean_dist = distances.mean()
        std_dist = distances.std()
        outlier_threshold = mean_dist + threshold * std_dist
        
        # Identify outliers
        is_outlier = distances > outlier_threshold
        
        logger.info(f"Detected {is_outlier.sum()} outliers out of {len(documents)} documents")
        
        return is_outlier.tolist()
    
    def generate_topic_labels(self, custom_descriptions: Optional[Dict[int, str]] = None) -> Dict[int, str]:
        """
        Generate descriptive labels for topics.
        
        Args:
            custom_descriptions: Optional custom descriptions for topics
            
        Returns:
            Dictionary mapping topic IDs to descriptive labels
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        labels = {}
        
        # Use custom descriptions if provided
        if custom_descriptions:
            labels.update(custom_descriptions)
            
        # Generate automatic labels for remaining topics
        for topic_idx in range(self.n_topics):
            if topic_idx in labels:
                continue
                
            # Get top words for this topic
            top_words = self.topic_words[topic_idx][:5]
            
            # Clean words and create a grammatically correct label
            clean_words = [word.strip() for word in top_words if re.match(r'^[a-zA-Z]+$', word)]
            
            if clean_words:
                # Use top 3 most representative words
                top_three = clean_words[:3]
                labels[topic_idx] = " + ".join(top_three)
            else:
                labels[topic_idx] = f"Topic {topic_idx+1}"
                
        return labels
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        import pickle
        
        try:
            # Create a state dictionary with all necessary components
            state = {
                'algorithm': self.algorithm,
                'n_topics': self.n_topics,
                'model': self.model,
                'vectorizer': self.vectorizer,
                'vocabulary': self.vocabulary,
                'feature_names': self.feature_names,
                'topic_words': self.topic_words,
                'topic_word_scores': self.topic_word_scores,
                'coherence_scores': self.coherence_scores,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TopicModeler':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded TopicModeler instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid model
        """
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Verify this is a valid model file
            required_keys = ['algorithm', 'n_topics', 'model', 'vectorizer', 'is_fitted']
            if not all(key in state for key in required_keys):
                raise ValueError("Invalid model file format")
                
            # Create a new instance
            instance = cls(n_topics=state['n_topics'], algorithm=state['algorithm'])
            
            # Restore state
            instance.model = state['model']
            instance.vectorizer = state['vectorizer']
            instance.vocabulary = state.get('vocabulary')
            instance.feature_names = state.get('feature_names')
            instance.topic_words = state.get('topic_words', [])
            instance.topic_word_scores = state.get('topic_word_scores', [])
            instance.coherence_scores = state.get('coherence_scores', [])
            instance.is_fitted = state.get('is_fitted', False)
            
            logger.info(f"Model loaded from {filepath}")
            
            return instance
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Sample documents
    docs = [
        "Machine learning involves computers learning from data to perform tasks.",
        "Deep learning models use neural networks with many layers.",
        "Neural networks are inspired by the human brain's structure.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns without labeled data.",
        "Reinforcement learning involves agents learning from environment feedback.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Data preprocessing is crucial for effective machine learning.",
        "Overfitting happens when models learn noise in training data."
    ]
    
    # Create and fit topic modeler
    modeler = TopicModeler(n_topics=3, algorithm='lda')
    modeler.fit(docs)
    
    # Get topics summary
    topics = modeler.get_topics_summary()
    print("\nTopics Summary:")
    for topic_id, topic_data in topics.items():
        print(f"Topic {topic_id}: {topic_data['label']}")
        for term, weight in list(topic_data['terms'].items())[:5]:
            print(f"  - {term}: {weight:.3f}")
            
    # Predict topic for a new document
    new_doc = "Neural networks have revolutionized artificial intelligence."
    topic_id, confidence = modeler.predict_topic(new_doc)
    print(f"\nPredicted topic for new document: {topic_id} (confidence: {confidence:.3f})")
    
    # Cluster documents
    clusters, metadata = modeler.cluster_documents(docs)
    print("\nDocument clusters:", clusters)
    print(f"Cluster sizes: {metadata['cluster_sizes']}")
    
    # Generate topic labels
    labels = modeler.generate_topic_labels()
    print("\nTopic labels:", labels)
