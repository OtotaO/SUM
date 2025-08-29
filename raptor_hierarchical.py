"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
Implements hierarchical tree-based summarization with multi-level abstraction
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import hashlib
import asyncio
import logging
from llm_backend import llm_backend

logger = logging.getLogger(__name__)

# Safe imports with fallbacks
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Using simple clustering.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using simple embeddings.")


@dataclass
class TextChunk:
    """Represents a chunk of text at any level of the tree"""
    id: str
    text: str
    level: int
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class RAPTORNode:
    """Node in the RAPTOR tree"""
    chunk: TextChunk
    summary: str
    children: List['RAPTORNode'] = None
    similarity_score: float = 0.0


@dataclass
class RAPTORTree:
    """Complete RAPTOR tree structure"""
    root: RAPTORNode
    levels: Dict[int, List[RAPTORNode]]
    embeddings: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class RAPTORBuilder:
    """Builds RAPTOR hierarchical trees for text summarization"""
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 min_cluster_size: int = 2,
                 max_levels: int = 5,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        self.max_chunk_size = max_chunk_size
        self.min_cluster_size = min_cluster_size
        self.max_levels = max_levels
        self.embedder = SentenceTransformer(embedding_model)
        self.summarizer = HierarchicalSummarizer()
    
    def build_tree(self, text: str) -> RAPTORTree:
        """
        Build RAPTOR tree from input text
        
        Args:
            text: Input text to process
            
        Returns:
            RAPTORTree with hierarchical summaries
        """
        # Initial chunking
        chunks = self._create_initial_chunks(text)
        
        # Build tree recursively
        tree_levels = {}
        current_level_chunks = chunks
        level = 0
        
        while len(current_level_chunks) > 1 and level < self.max_levels:
            # Store current level
            tree_levels[level] = current_level_chunks
            
            # Embed chunks
            embeddings = self._embed_chunks(current_level_chunks)
            
            # Cluster chunks
            clusters = self._cluster_chunks(current_level_chunks, embeddings)
            
            # Summarize clusters to create next level
            next_level_chunks = []
            for cluster_chunks in clusters:
                summary_chunk = self._summarize_cluster(cluster_chunks, level + 1)
                next_level_chunks.append(summary_chunk)
            
            current_level_chunks = next_level_chunks
            level += 1
        
        # Store final level
        tree_levels[level] = current_level_chunks
        
        # Build tree structure
        tree = self._construct_tree(tree_levels)
        
        return tree
    
    def _create_initial_chunks(self, text: str) -> List[TextChunk]:
        """Create initial text chunks"""
        chunks = []
        
        # Split by sentences first
        sentences = text.split('.')
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunk_id = self._generate_chunk_id(chunk_text)
                
                chunks.append(TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    level=0,
                    child_ids=[],
                    metadata={'sentence_count': len(current_chunk)}
                ))
                
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunk_id = self._generate_chunk_id(chunk_text)
            
            chunks.append(TextChunk(
                id=chunk_id,
                text=chunk_text,
                level=0,
                child_ids=[],
                metadata={'sentence_count': len(current_chunk)}
            ))
        
        return chunks
    
    def _embed_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        """Generate embeddings for chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return embeddings
    
    def _cluster_chunks(self, 
                       chunks: List[TextChunk],
                       embeddings: np.ndarray) -> List[List[TextChunk]]:
        """Cluster chunks based on semantic similarity"""
        n_chunks = len(chunks)
        
        if n_chunks <= self.min_cluster_size:
            return [chunks]
        
        # Determine optimal number of clusters
        n_clusters = min(n_chunks // self.min_cluster_size, int(np.sqrt(n_chunks)))
        n_clusters = max(2, n_clusters)
        
        # Perform clustering
        if n_chunks < 10:
            # Use hierarchical clustering for small sets
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
        else:
            # Use K-means for larger sets
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group chunks by cluster
        clusters = defaultdict(list)
        for chunk, label in zip(chunks, labels):
            clusters[label].append(chunk)
        
        return list(clusters.values())
    
    def _summarize_cluster(self, 
                          chunks: List[TextChunk],
                          level: int) -> TextChunk:
        """Summarize a cluster of chunks into a higher-level chunk"""
        # Combine texts
        combined_text = ' '.join([chunk.text for chunk in chunks])
        
        # Generate summary
        summary = self.summarizer.summarize(combined_text, level)
        
        # Create new chunk
        chunk_id = self._generate_chunk_id(summary)
        child_ids = [chunk.id for chunk in chunks]
        
        # Set parent references
        for chunk in chunks:
            chunk.parent_id = chunk_id
        
        return TextChunk(
            id=chunk_id,
            text=summary,
            level=level,
            child_ids=child_ids,
            metadata={
                'source_chunks': len(chunks),
                'compression_ratio': len(summary) / len(combined_text)
            }
        )
    
    def _construct_tree(self, levels: Dict[int, List[TextChunk]]) -> RAPTORTree:
        """Construct tree structure from levels"""
        # Build nodes
        nodes_by_id = {}
        levels_nodes = defaultdict(list)
        
        for level, chunks in levels.items():
            for chunk in chunks:
                node = RAPTORNode(
                    chunk=chunk,
                    summary=self._generate_summary(chunk),
                    children=[]
                )
                nodes_by_id[chunk.id] = node
                levels_nodes[level].append(node)
        
        # Link parent-child relationships
        for node_id, node in nodes_by_id.items():
            if node.chunk.child_ids:
                for child_id in node.chunk.child_ids:
                    if child_id in nodes_by_id:
                        node.children.append(nodes_by_id[child_id])
        
        # Find root (highest level node)
        max_level = max(levels.keys())
        root_candidates = levels[max_level]
        
        if len(root_candidates) == 1:
            root = nodes_by_id[root_candidates[0].id]
        else:
            # Create synthetic root
            combined_text = ' '.join([c.text for c in root_candidates])
            root_chunk = TextChunk(
                id='root',
                text=self.summarizer.summarize(combined_text, max_level + 1),
                level=max_level + 1,
                child_ids=[c.id for c in root_candidates]
            )
            root = RAPTORNode(
                chunk=root_chunk,
                summary=root_chunk.text,
                children=[nodes_by_id[c.id] for c in root_candidates]
            )
        
        # Collect all embeddings
        embeddings = {}
        for node_id, node in nodes_by_id.items():
            if node.chunk.embedding is not None:
                embeddings[node_id] = node.chunk.embedding
        
        return RAPTORTree(
            root=root,
            levels=dict(levels_nodes),
            embeddings=embeddings,
            metadata={
                'total_levels': len(levels),
                'total_nodes': len(nodes_by_id)
            }
        )
    
    def _generate_chunk_id(self, text: str) -> str:
        """Generate unique ID for chunk"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _generate_summary(self, chunk: TextChunk) -> str:
        """Generate summary for a chunk"""
        # For leaf nodes, extract key sentences
        if chunk.level == 0:
            sentences = chunk.text.split('.')[:2]
            return '. '.join(sentences) + '.' if sentences else chunk.text[:200]
        else:
            # Higher level chunks are already summaries
            return chunk.text


class HierarchicalSummarizer:
    """Generates summaries at different hierarchical levels"""
    
    def summarize(self, text: str, level: int) -> str:
        """
        Generate summary appropriate for hierarchical level
        
        Args:
            text: Text to summarize
            level: Hierarchical level (0 = leaf, higher = more abstract)
            
        Returns:
            Summary text
        """
        # Adjust compression based on level
        compression_ratios = {
            0: 1.0,    # No compression at leaf level
            1: 0.5,    # 50% compression
            2: 0.3,    # 70% compression
            3: 0.2,    # 80% compression
            4: 0.1,    # 90% compression
            5: 0.05    # 95% compression
        }
        
        ratio = compression_ratios.get(level, 0.05)
        target_length = int(len(text.split()) * ratio)
        target_length = max(10, target_length)  # Minimum 10 words
        
        # Simple extractive summarization (would use LLM in production)
        sentences = text.split('.')
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            if current_length + len(words) <= target_length:
                summary_sentences.append(sentence)
                current_length += len(words)
            elif current_length < target_length / 2:
                # Include partial sentence if we're too short
                summary_sentences.append(sentence)
                break
        
        if not summary_sentences and sentences:
            summary_sentences = [sentences[0]]
        
        return '. '.join(summary_sentences) + '.'


class RAPTORQueryEngine:
    """Query engine for RAPTOR trees"""
    
    def __init__(self, embedder: SentenceTransformer = None):
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
    
    def query(self, 
             tree: RAPTORTree,
             query: str,
             top_k: int = 5,
             level: Optional[int] = None) -> List[Tuple[RAPTORNode, float]]:
        """
        Query the RAPTOR tree
        
        Args:
            tree: RAPTOR tree to query
            query: Query text
            top_k: Number of results to return
            level: Specific level to query (None = all levels)
            
        Returns:
            List of (node, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedder.encode([query])[0]
        
        # Collect nodes to search
        if level is not None:
            nodes = tree.levels.get(level, [])
        else:
            nodes = []
            for level_nodes in tree.levels.values():
                nodes.extend(level_nodes)
        
        # Calculate similarities
        similarities = []
        for node in nodes:
            if node.chunk.embedding is not None:
                similarity = self._cosine_similarity(
                    query_embedding,
                    node.chunk.embedding
                )
                similarities.append((node, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def traverse_to_leaves(self, 
                          node: RAPTORNode,
                          max_depth: int = -1) -> List[RAPTORNode]:
        """
        Traverse from node to leaf nodes
        
        Args:
            node: Starting node
            max_depth: Maximum depth to traverse (-1 = unlimited)
            
        Returns:
            List of leaf nodes
        """
        if not node.children or max_depth == 0:
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self.traverse_to_leaves(child, max_depth - 1))
        
        return leaves
    
    def get_context_window(self,
                          tree: RAPTORTree,
                          query: str,
                          max_tokens: int = 2000) -> str:
        """
        Get optimal context window for query
        
        Args:
            tree: RAPTOR tree
            query: Query text
            max_tokens: Maximum tokens in context
            
        Returns:
            Context text
        """
        # Query tree at multiple levels
        results = self.query(tree, query, top_k=10)
        
        context_parts = []
        current_tokens = 0
        
        for node, score in results:
            text = node.summary
            tokens = len(text.split())
            
            if current_tokens + tokens <= max_tokens:
                context_parts.append(text)
                current_tokens += tokens
            else:
                break
        
        return ' '.join(context_parts)


# Example usage
if __name__ == "__main__":
    # Sample text
    text = """
    Machine learning has revolutionized many industries. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers. 
    These networks can learn complex patterns from data. Transformers are 
    a type of neural network architecture that has shown remarkable success 
    in natural language processing. Models like GPT and BERT use transformer 
    architecture. They can understand context and generate human-like text. 
    The attention mechanism is key to transformer success. It allows models 
    to focus on relevant parts of the input. Self-attention enables the model 
    to understand relationships between words. Multi-head attention provides 
    multiple representation subspaces. Training these models requires vast 
    amounts of data and computational resources. Pre-training on large corpora 
    followed by fine-tuning has become the standard approach. Transfer learning 
    allows models to apply knowledge from one task to another. This has made 
    AI more accessible and practical for various applications.
    """
    
    # Build RAPTOR tree
    builder = RAPTORBuilder(max_chunk_size=50)
    tree = builder.build_tree(text)
    
    # Print tree structure
    print(f"Tree built with {tree.metadata['total_levels']} levels")
    print(f"Total nodes: {tree.metadata['total_nodes']}")
    print(f"\nRoot summary: {tree.root.summary[:200]}...")
    
    # Query the tree
    query_engine = RAPTORQueryEngine()
    query = "What is the attention mechanism?"
    results = query_engine.query(tree, query, top_k=3)
    
    print(f"\nQuery: {query}")
    print("Top results:")
    for node, score in results:
        print(f"  Score: {score:.3f} | Level: {node.chunk.level}")
        print(f"  Text: {node.summary[:100]}...")
    
    # Get context window
    context = query_engine.get_context_window(tree, query, max_tokens=100)
    print(f"\nContext window: {context[:200]}...")