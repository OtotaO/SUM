"""
advanced_sum.py - Advanced knowledge distillation engine

This module implements a sophisticated summarization engine that combines
extractive and abstractive methods with semantic understanding to generate
high-quality summaries of complex text data.

Design principles:
- Algorithmic elegance (Dijkstra approach)
- Performance optimization (Knuth principles)
- Clean, maintainable code (Torvalds/van Rossum style)
- Comprehensive documentation (Stroustrup methodology)
- Security by design (Schneier focus)

Author: ototao
License: Apache License 2.0
"""

import re
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from threading import Lock
from functools import lru_cache
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import base SUM class
from summarization_engine import summarization_engine, SimpleSUM

# Configure logging
logger = logging.getLogger(__name__)

class SemanticSummarizationEngine(SUM):
    """
    Advanced summarization engine combining multiple NLP techniques.
    
    This class implements sophisticated summarization algorithms that leverage
    semantic understanding, topic extraction, and graph-based ranking to
    generate high-quality summaries tailored to different knowledge domains.
    
    Features:
    - TextRank algorithm for sentence importance
    - Latent semantic analysis for concept extraction
    - Entity recognition and relationship mapping
    - Multi-level summarization (tag, sentence, paragraph)
    - Domain-specific customization
    - Coherence optimization
    """
    
    def __init__(self, 
                use_semantic: bool = True,
                use_entities: bool = True,
                language: str = 'english',
                coherence_weight: float = 0.3,
                diversity_weight: float = 0.2,
                max_workers: int = None):
        """
        Initialize the advanced summarization engine.
        
        Args:
            use_semantic: Whether to use semantic analysis
            use_entities: Whether to use entity recognition
            language: Language of the text to be summarized
            coherence_weight: Weight for coherence in summary generation
            diversity_weight: Weight for diversity in summary generation
            max_workers: Maximum number of worker threads (None = auto)
        """
        super().__init__()
        
        # Configuration
        self.use_semantic = use_semantic
        self.use_entities = use_entities
        self.language = language
        self.coherence_weight = coherence_weight
        self.diversity_weight = diversity_weight
        self.max_workers = max_workers or (os.cpu_count() or 4)
        
        # Initialize NLTK resources
        self._initialize_nlp()
        
        # Cache for expensive operations
        self._cache_lock = Lock()
        self.sentence_embeddings_cache = {}
        self.semantic_similarity_cache = {}
        
        logger.info("AdvancedSUM initialized successfully")
    
    def _initialize_nlp(self):
        """Initialize NLP components."""
        try:
            # Download required NLTK resources if not already present
            nltk_resources = ['punkt', 'stopwords', 'wordnet']
            for resource in nltk_resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download {resource}: {e}")
            
            # Initialize components
            self.stop_words = set(stopwords.words(self.language))
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize spaCy for entity recognition if available
            if self.use_entities:
                try:
                    import spacy
                    # Use a lighter model for better performance
                    self.nlp = spacy.load('en_core_web_sm')
                    logger.info("Loaded spaCy for entity recognition")
                except ImportError:
                    logger.warning("spaCy not available, entity recognition disabled")
                    self.use_entities = False
                except Exception as e:
                    logger.warning(f"Error loading spaCy: {e}")
                    self.use_entities = False
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            raise RuntimeError(f"Failed to initialize NLP components: {e}")
    
    def process_text(self, text: str, model_config: Optional[Dict] = None) -> Dict:
        """
        Process and summarize the input text using advanced techniques.
        
        Args:
            text: Text to summarize
            model_config: Configuration parameters including:
                - max_tokens: Maximum number of tokens in summary
                - min_tokens: Minimum number of tokens in summary
                - summary_levels: Levels of summary to generate ('tag', 'sentence', 'paragraph')
                - domain: Optional domain for customized summarization
                
        Returns:
            Dictionary containing summaries and metadata:
                - tags: List of key tags/topics
                - sum: One-sentence summary
                - summary: Detailed summary
                - entities: Recognized entities (if enabled)
                - topics: Main topics (if topic modeling enabled)
                - metadata: Processing metadata
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided")
            return {'error': 'Empty or invalid text provided'}
            
        if not text.strip():
            logger.warning("Empty text provided")
            return {'error': 'Empty text provided'}
            
        try:
            start_time = time.time()
            
            # Process configuration
            config = model_config or {}
            max_tokens = max(10, config.get('max_tokens', 150))
            min_tokens = min(max_tokens - 10, config.get('min_tokens', 50))
            summary_levels = config.get('summary_levels', ['tag', 'sentence', 'paragraph'])
            domain = config.get('domain', None)
            
            logger.info(f"Processing text with max_tokens={max_tokens}, min_tokens={min_tokens}")
            
            # Preprocess text
            sentences, preprocessed_sentences, word_frequencies = self._preprocess_text(text)
            
            # Handle very short texts
            if len(sentences) <= 2:
                logger.info("Text too short for advanced summarization, returning as-is")
                return {
                    'tags': self._extract_keywords(word_frequencies, 5),
                    'sum': text,
                    'summary': text,
                    'metadata': {
                        'sentences': len(sentences),
                        'words': len(word_tokenize(text)),
                        'chars': len(text),
                        'processing_time': time.time() - start_time
                    }
                }
                
            # Calculate sentence importance using TextRank
            sentence_scores = self._rank_sentences(sentences, preprocessed_sentences)
            
            # Generate tag summary if requested
            tags = []
            if 'tag' in summary_levels:
                tags = self._generate_tag_summary(word_frequencies, text, max_n=10)
                
            # Generate one-sentence summary if requested
            one_sentence_summary = ""
            if 'sentence' in summary_levels:
                one_sentence_summary = self._generate_sentence_summary(
                    sentences, sentence_scores, word_frequencies, max_length=60
                )
                
            # Generate paragraph summary if requested
            paragraph_summary = ""
            if 'paragraph' in summary_levels:
                paragraph_summary = self._generate_paragraph_summary(
                    sentences, sentence_scores, min_tokens, max_tokens
                )
                
            # Extract entities if enabled
            entities = []
            if self.use_entities:
                entities = self._extract_entities(text)
                
            # Identify main topics if topic modeling enabled
            topics = []
            if config.get('extract_topics', False):
                topics = self._extract_topics(text, num_topics=config.get('num_topics', 3))
                
            # Prepare response
            processing_time = time.time() - start_time
            
            result = {
                'tags': tags,
                'sum': one_sentence_summary or self._select_best_sentence(sentences, sentence_scores),
                'summary': paragraph_summary,
                'metadata': {
                    'sentences': len(sentences),
                    'words': len(word_tokenize(text)),
                    'chars': len(text),
                    'compression_ratio': len(word_tokenize(paragraph_summary)) / max(1, len(word_tokenize(text))),
                    'processing_time': processing_time
                }
            }
            
            # Add entities and topics if available
            if entities:
                result['entities'] = entities
                
            if topics:
                result['topics'] = topics
                
            logger.info(f"Text processed in {processing_time:.2f}s, compression ratio: {result['metadata']['compression_ratio']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            return {'error': f"Error during advanced text processing: {str(e)}"}
    
    def _preprocess_text(self, text: str) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
        """
        Preprocess text for summarization.
        
        Args:
            text: Raw text to process
            
        Returns:
            Tuple of (original sentences, preprocessed sentence tokens, word frequencies)
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Process each sentence
        preprocessed_sentences = []
        word_freq = defaultdict(int)
        
        for sentence in sentences:
            # Tokenize and normalize
            tokens = word_tokenize(sentence.lower())
            
            # Remove stopwords and punctuation
            filtered_tokens = []
            for token in tokens:
                if token.isalnum() and token not in self.stop_words:
                    # Lemmatize
                    lemma = self.lemmatizer.lemmatize(token)
                    filtered_tokens.append(lemma)
                    word_freq[lemma] += 1
            
            if filtered_tokens:
                preprocessed_sentences.append(filtered_tokens)
            else:
                # Keep even empty sentences to maintain alignment with original
                preprocessed_sentences.append([])
        
        return sentences, preprocessed_sentences, dict(word_freq)
    
    def _rank_sentences(self, 
                       original_sentences: List[str], 
                       preprocessed_sentences: List[List[str]]) -> Dict[str, float]:
        """
        Rank sentences by importance using TextRank algorithm.
        
        Args:
            original_sentences: List of original sentences
            preprocessed_sentences: List of preprocessed sentence tokens
            
        Returns:
            Dictionary mapping sentences to importance scores
        """
        # Create sentence vectors
        if self.use_semantic:
            # Use TF-IDF for better semantic representation
            vectorizer = TfidfVectorizer(stop_words=self.language)
            sentence_vectors = vectorizer.fit_transform(
                [' '.join(sent) for sent in preprocessed_sentences]
            )
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(sentence_vectors)
            
        else:
            # Simpler word overlap similarity
            similarity_matrix = np.zeros((len(original_sentences), len(original_sentences)))
            
            for i in range(len(original_sentences)):
                for j in range(len(original_sentences)):
                    if i != j:
                        similarity_matrix[i][j] = self._sentence_similarity(
                            preprocessed_sentences[i], 
                            preprocessed_sentences[j]
                        )
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=100)
        
        # Map scores to original sentences
        sentence_scores = {}
        for i, sentence in enumerate(original_sentences):
            # Apply position bias (beginning and end sentences often more important)
            position_factor = 1.0
            if i < len(original_sentences) // 10:  # First 10%
                position_factor = 1.15
            elif i > len(original_sentences) * 0.9:  # Last 10%
                position_factor = 1.1
                
            # Apply length penalty to avoid very short sentences
            length_factor = min(1.0, len(preprocessed_sentences[i]) / 8)
            
            sentence_scores[sentence] = scores[i] * position_factor * length_factor
        
        return sentence_scores
    
    def _sentence_similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """
        Calculate similarity between two sentences based on word overlap.
        
        Args:
            sent1: First sentence tokens
            sent2: Second sentence tokens
            
        Returns:
            Similarity score between 0 and 1
        """
        if not sent1 or not sent2:
            return 0.0
            
        # Create sets of words
        set1 = set(sent1)
        set2 = set(sent2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / max(1, union)
    
    @lru_cache(maxsize=1024)
    def _semantic_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate semantic similarity between sentences using embeddings.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Semantic similarity score
        """
        # This would implement a more sophisticated similarity measure
        # using word or sentence embeddings, but for simplicity we'll
        # just use a cached version of the basic similarity
        
        tokens1 = word_tokenize(sent1.lower())
        tokens2 = word_tokenize(sent2.lower())
        
        filtered1 = [t for t in tokens1 if t.isalnum() and t not in self.stop_words]
        filtered2 = [t for t in tokens2 if t.isalnum() and t not in self.stop_words]
        
        return self._sentence_similarity(filtered1, filtered2)
    
    def _extract_keywords(self, word_freq: Dict[str, int], top_n: int = 10) -> List[str]:
        """
        Extract top keywords from word frequencies.
        
        Args:
            word_freq: Dictionary of word frequencies
            top_n: Number of top keywords to extract
            
        Returns:
            List of top keywords
        """
        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top words, skipping very short ones
        keywords = []
        for word, freq in sorted_words:
            if len(word) > 2:  # Skip very short words
                keywords.append(word)
                if len(keywords) >= top_n:
                    break
                    
        return keywords
    
    def _generate_tag_summary(self, 
                            word_freq: Dict[str, int], 
                            text: str,
                            max_n: int = 10) -> List[str]:
        """
        Generate a tag/keyword summary.
        
        Args:
            word_freq: Dictionary of word frequencies
            text: Original text
            max_n: Maximum number of tags
            
        Returns:
            List of tags/keywords
        """
        # Basic approach: extract top keywords by frequency
        keywords = self._extract_keywords(word_freq, top_n=max_n)
        
        # If entity recognition is enabled, include top entities
        if self.use_entities:
            entities = self._extract_entities(text)
            entity_names = [e[0] for e in entities[:max_n//2]]
            
            # Merge keywords and entities, avoiding duplicates
            seen = set(keywords)
            for entity in entity_names:
                if entity.lower() not in seen and len(seen) < max_n:
                    seen.add(entity.lower())
                    keywords.append(entity)
        
        return keywords[:max_n]
    
    def _generate_sentence_summary(self, 
                                 sentences: List[str],
                                 sentence_scores: Dict[str, float],
                                 word_freq: Dict[str, int],
                                 max_length: int = 60) -> str:
        """
        Generate a one-sentence summary.
        
        Args:
            sentences: Original sentences
            sentence_scores: Importance scores for sentences
            word_freq: Word frequency dictionary
            max_length: Maximum target length in words
            
        Returns:
            Single sentence summary
        """
        # Strategy 1: Return the highest scoring sentence if it's short enough
        best_sentence = self._select_best_sentence(sentences, sentence_scores)
        if len(word_tokenize(best_sentence)) <= max_length:
            return best_sentence
            
        # Strategy 2: Try to compress the best sentence
        compressed = self._compress_sentence(best_sentence, max_length)
        if compressed:
            return compressed
            
        # Strategy 3: Generate a new summary sentence from key phrases
        # (Implementation would be more complex, using extractive or abstractive methods)
        
        # Fallback: Return truncated best sentence
        tokens = word_tokenize(best_sentence)[:max_length-1]
        return ' '.join(tokens) + '...'
    
    def _select_best_sentence(self, 
                            sentences: List[str],
                            sentence_scores: Dict[str, float]) -> str:
        """
        Select the best representative sentence.
        
        Args:
            sentences: List of sentences
            sentence_scores: Importance scores for sentences
            
        Returns:
            Best representative sentence
        """
        # Filter out very short sentences
        valid_sentences = [s for s in sentences if len(word_tokenize(s)) >= 5]
        
        if not valid_sentences:
            return sentences[0] if sentences else ""
            
        # Find highest scoring sentence among valid ones
        best_sentence = max(valid_sentences, key=lambda s: sentence_scores.get(s, 0))
        return best_sentence
    
    def _compress_sentence(self, sentence: str, max_length: int) -> Optional[str]:
        """
        Compress a sentence to fit within max_length words.
        
        Args:
            sentence: Sentence to compress
            max_length: Maximum length in words
            
        Returns:
            Compressed sentence or None if compression failed
        """
        tokens = word_tokenize(sentence)
        if len(tokens) <= max_length:
            return sentence
            
        # Simple compression: remove adverbs, adjectives, and parenthetical expressions
        if self.use_entities and hasattr(self, 'nlp'):
            doc = self.nlp(sentence)
            
            # Identify less important words
            to_keep = []
            for token in doc:
                # Keep nouns, verbs, and parts of named entities
                if (token.pos_ in ('NOUN', 'PROPN', 'VERB') or 
                    token.ent_type_ or 
                    token.is_stop):
                    to_keep.append(token.text)
                    
            # If we managed to compress enough, reconstruct sentence
            if len(to_keep) <= max_length:
                return ' '.join(to_keep)
        
        # If sophisticated compression failed, try removing text in parentheses
        no_parentheses = re.sub(r'\([^)]*\)', '', sentence).strip()
        tokens = word_tokenize(no_parentheses)
        if len(tokens) <= max_length:
            return no_parentheses
            
        return None
    
    def _generate_paragraph_summary(self, 
                                   sentences: List[str],
                                   sentence_scores: Dict[str, float],
                                   min_tokens: int,
                                   max_tokens: int) -> str:
        """
        Generate a paragraph summary from ranked sentences.
        
        Args:
            sentences: Original sentences
            sentence_scores: Importance scores for sentences
            min_tokens: Minimum number of tokens
            max_tokens: Maximum number of tokens
            
        Returns:
            Paragraph summary
        """
        # Sort sentences by score
        ranked_sentences = sorted(
            [(i, s, sentence_scores.get(s, 0)) for i, s in enumerate(sentences)],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Estimate token count for each sentence
        sentence_tokens = {s: len(word_tokenize(s)) for s in sentences}
        
        # Select sentences, respecting token limits and coherence
        selected_sentences = []
        current_tokens = 0
        
        for _, sentence, _ in ranked_sentences:
            if current_tokens + sentence_tokens[sentence] <= max_tokens:
                selected_sentences.append((sentences.index(sentence), sentence))
                current_tokens += sentence_tokens[sentence]
                
                # If we've reached the minimum, check coherence
                if current_tokens >= min_tokens:
                    # If coherence is good enough, we can stop
                    if self._evaluate_coherence(selected_sentences) > 0.7:
                        break
            
            # Stop if we've reached the maximum
            if current_tokens >= max_tokens:
                break
                
        # Make sure we have at least min_tokens
        if current_tokens < min_tokens and len(ranked_sentences) > len(selected_sentences):
            # Add more sentences until we reach minimum
            for _, sentence, _ in ranked_sentences[len(selected_sentences):]:
                if sentence not in [s for _, s in selected_sentences]:
                    selected_sentences.append((sentences.index(sentence), sentence))
                    current_tokens += sentence_tokens[sentence]
                    if current_tokens >= min_tokens:
                        break
        
        # Sort selected sentences by original position
        selected_sentences.sort(key=lambda x: x[0])
        
        # Combine into paragraph
        summary = ' '.join(sentence for _, sentence in selected_sentences)
        
        return summary
    
    def _evaluate_coherence(self, selected_sentences: List[Tuple[int, str]]) -> float:
        """
        Evaluate the coherence of selected sentences.
        
        Args:
            selected_sentences: List of (position, sentence) tuples
            
        Returns:
            Coherence score between 0 and 1
        """
        if len(selected_sentences) <= 1:
            return 1.0
            
        # Sort by position
        sorted_sentences = sorted(selected_sentences, key=lambda x: x[0])
        sentences = [s for _, s in sorted_sentences]
        
        # Calculate average similarity between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._semantic_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
            
        return sum(similarities) / len(similarities)
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Extract named entities from text.
        
        Args:
            text: Source text
            
        Returns:
            List of (entity_text, entity_type, count) tuples
        """
        if not self.use_entities or not hasattr(self, 'nlp'):
            return []
            
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entity_counts = defaultdict(int)
            entity_types = {}
            
            for ent in doc.ents:
                # Normalize entity text
                entity_text = ent.text.strip()
                if entity_text:
                    entity_counts[entity_text] += 1
                    entity_types[entity_text] = ent.label_
            
            # Sort by frequency
            entities = [
                (entity, entity_types[entity], count)
                for entity, count in sorted(
                    entity_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            ]
            
            return entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []
    
    def _extract_topics(self, text: str, num_topics: int = 3) -> List[Dict]:
        """
        Extract main topics from text.
        
        Args:
            text: Source text
            num_topics: Number of topics to extract
            
        Returns:
            List of topic dictionaries with keywords and weights
        """
        try:
            # For proper topic modeling, we would use a dedicated module
            # like the TopicModeler class. This is a simplified version.
            
            # Split into sentences and preprocess
            sentences = sent_tokenize(text)
            preprocessed = []
            
            for sentence in sentences:
                tokens = word_tokenize(sentence.lower())
                filtered = [
                    self.lemmatizer.lemmatize(t) 
                    for t in tokens 
                    if t.isalnum() and t not in self.stop_words and len(t) > 2
                ]
                if filtered:
                    preprocessed.append(' '.join(filtered))
            
            if not preprocessed:
                return []
                
            # Use TF-IDF to identify important terms
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(preprocessed)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF score for each term
            avg_tfidf = np.array(tfidf_matrix.mean(axis=0))[0]
            
            # Sort terms by score
            sorted_indices = avg_tfidf.argsort()[::-1]
            
            # Group into topics (simplified approach)
            topics = []
            for i in range(min(num_topics, 10)):
                # Take a set of top terms as a "topic"
                start_idx = i * 5
                end_idx = start_idx + 5
                if start_idx >= len(sorted_indices):
                    break
                    
                topic_terms = []
                for idx in sorted_indices[start_idx:end_idx]:
                    term = feature_names[idx]
                    weight = avg_tfidf[idx]
                    topic_terms.append({"term": term, "weight": float(weight)})
                
                topics.append({
                    "id": i,
                    "label": f"Topic {i+1}",
                    "terms": topic_terms
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"Error extracting topics: {e}")
            return []

    def generate_knowledge_graph(self, 
                               text: str, 
                               max_nodes: int = 30,
                               min_weight: float = 0.1) -> Dict:
        """
        Generate a knowledge graph from the text.
        
        Args:
            text: Source text
            max_nodes: Maximum number of nodes in the graph
            min_weight: Minimum weight for edges
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if not self.use_entities or not hasattr(self, 'nlp'):
            return {"nodes": [], "edges": []}
            
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = {}
            for ent in doc.ents:
                if ent.text.strip() and len(ent.text.split()) <= 4:
                    if ent.text not in entities:
                        entities[ent.text] = {
                            "id": len(entities),
                            "text": ent.text,
                            "type": ent.label_,
                            "count": 1
                        }
                    else:
                        entities[ent.text]["count"] += 1
            
            # Limit to most frequent entities
            top_entities = sorted(
                entities.values(), 
                key=lambda x: x["count"], 
                reverse=True
            )[:max_nodes]
            
            # Create nodes
            nodes = []
            entity_to_node = {}
            
            for entity in top_entities:
                node = {
                    "id": len(nodes),
                    "label": entity["text"],
                    "type": entity["type"],
                    "weight": entity["count"]
                }
                nodes.append(node)
                entity_to_node[entity["text"]] = node["id"]
            
            # Create edges based on co-occurrence in sentences
            edge_weights = defaultdict(float)
            
            for sent in doc.sents:
                # Find entities in this sentence
                sent_entities = []
                for ent in doc.ents:
                    if ent.text in entity_to_node and sent.start <= ent.start < ent.end <= sent.end:
                        sent_entities.append(ent.text)
                
                # Create edges between all pairs
                for i in range(len(sent_entities)):
                    for j in range(i+1, len(sent_entities)):
                        edge = (
                            entity_to_node[sent_entities[i]],
                            entity_to_node[sent_entities[j]]
                        )
                        edge_weights[edge] += 1
            
            # Normalize edge weights
            max_weight = max(edge_weights.values()) if edge_weights else 1
            normalized_weights = {
                edge: weight / max_weight
                for edge, weight in edge_weights.items()
            }
            
            # Create edge list, filtering by minimum weight
            edges = []
            for (source, target), weight in normalized_weights.items():
                if weight >= min_weight:
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": weight
                    })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.warning(f"Error generating knowledge graph: {e}")
            return {"nodes": [], "edges": []}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    summarizer = AdvancedSUM()
    
    # Sample text
    sample_text = """
    Machine learning has seen rapid advancements in recent years. From image recognition to
    natural language processing, AI systems are becoming increasingly sophisticated. Deep learning
    models, in particular, have shown remarkable capabilities in handling complex tasks. However,
    challenges remain in areas such as explainability and bias mitigation. As the field continues
    to evolve, researchers are developing new approaches to address these limitations and expand
    the applications of machine learning across various domains including healthcare, finance,
    transportation, and environmental science. One key area of progress has been in reinforcement
    learning, where algorithms learn optimal behavior through interaction with environments.
    Another promising direction is federated learning, which enables model training across
    decentralized devices while preserving data privacy.
    """
    
    # Process text with default configuration
    result = summarizer.process_text(sample_text)
    
    # Print results
    print("=== Advanced SUM Results ===")
    print(f"Tags: {', '.join(result['tags'])}")
    print(f"\nOne-sentence summary:\n{result['sum']}")
    print(f"\nParagraph summary:\n{result['summary']}")
    
    if 'entities' in result:
        print("\nTop entities:")
        for entity, entity_type, count in result['entities'][:5]:
            print(f"- {entity} ({entity_type}): {count}")
    
    print(f"\nProcessing time: {result['metadata']['processing_time']:.2f}s")
    print(f"Compression ratio: {result['metadata']['compression_ratio']:.2f}")
    
    # Generate knowledge graph
    graph = summarizer.generate_knowledge_graph(sample_text)
    print(f"\nKnowledge graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
