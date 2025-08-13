"""
semantic_memory.py - Semantic Memory System for Knowledge Crystallization

This module implements a semantic memory layer that enables:
- Vector-based semantic search
- Knowledge persistence and retrieval
- Similarity-based concept linking
- Temporal knowledge evolution

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging

# Try to import vector DB clients
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    from Utils.cache import cache_embedding, get_memory_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from Utils.circuit_breaker import circuit_breaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    # Dummy decorator if circuit breaker not available
    def circuit_breaker(**kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the semantic memory system."""
    id: str
    text: str
    summary: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: Optional[float] = None
    relationships: List[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []


class SemanticMemoryEngine:
    """
    Core semantic memory engine for knowledge crystallization.
    Provides vector-based storage and retrieval of knowledge.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 storage_path: str = "./semantic_memory",
                 use_gpu: bool = False):
        """
        Initialize the semantic memory engine.
        
        Args:
            model_name: Sentence transformer model name
            storage_path: Path for persistent storage
            use_gpu: Whether to use GPU for embeddings
        """
        self.model_name = model_name
        self.storage_path = storage_path
        self.use_gpu = use_gpu
        
        # Initialize storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_store()
        self._init_metadata_store()
        
        # Memory statistics
        self.stats = {
            'total_memories': 0,
            'total_queries': 0,
            'avg_query_time': 0.0,
            'last_update': time.time()
        }
        
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                device = 'cuda' if self.use_gpu else 'cpu'
                self.embedding_model = SentenceTransformer(self.model_name, device=device)
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Initialized embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self._fallback_embedding()
        else:
            logger.warning("Sentence transformers not available, using fallback")
            self._fallback_embedding()
    
    def _fallback_embedding(self):
        """Fallback embedding method using simple hashing."""
        self.embedding_model = None
        self.embedding_dimension = 384  # Standard dimension
        logger.info("Using fallback embedding method")
    
    def _init_vector_store(self):
        """Initialize the vector storage backend."""
        if CHROMA_AVAILABLE:
            try:
                self.vector_store = chromadb.PersistentClient(
                    path=os.path.join(self.storage_path, "chroma"),
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.vector_store.get_or_create_collection(
                    name="semantic_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                self.vector_backend = "chroma"
                logger.info("Initialized ChromaDB vector store")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self._init_faiss_store()
        else:
            self._init_faiss_store()
    
    def _init_faiss_store(self):
        """Initialize FAISS as fallback vector store."""
        if FAISS_AVAILABLE:
            try:
                index_path = os.path.join(self.storage_path, "faiss_index.bin")
                if os.path.exists(index_path):
                    self.faiss_index = faiss.read_index(index_path)
                else:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
                self.vector_backend = "faiss"
                self.faiss_id_map = {}  # Map internal IDs to our IDs
                logger.info("Initialized FAISS vector store")
            except Exception as e:
                logger.error(f"Failed to initialize FAISS: {e}")
                self._init_numpy_store()
        else:
            self._init_numpy_store()
    
    def _init_numpy_store(self):
        """Initialize numpy-based vector store as final fallback."""
        self.vector_backend = "numpy"
        self.numpy_vectors = []
        self.numpy_ids = []
        vectors_path = os.path.join(self.storage_path, "vectors.npy")
        ids_path = os.path.join(self.storage_path, "ids.json")
        
        if os.path.exists(vectors_path) and os.path.exists(ids_path):
            self.numpy_vectors = np.load(vectors_path).tolist()
            with open(ids_path, 'r') as f:
                self.numpy_ids = json.load(f)
        
        logger.info("Initialized numpy-based vector store")
    
    def _init_metadata_store(self):
        """Initialize metadata storage."""
        self.metadata_path = os.path.join(self.storage_path, "metadata.json")
        self.metadata = {}
        
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.stats['total_memories'] = len(self.metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Use cache if available
        if CACHE_AVAILABLE:
            cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
            cache = get_memory_cache()
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                self.stats['cache_hits'] = self.stats.get('cache_hits', 0) + 1
                return cached_embedding
        
        # Generate embedding
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                embedding_list = embedding.tolist()
                
                # Cache the result
                if CACHE_AVAILABLE:
                    cache.set(cache_key, embedding_list, ttl=3600)
                
                return embedding_list
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return self._fallback_hash_embedding(text)
        else:
            return self._fallback_hash_embedding(text)
    
    def _fallback_hash_embedding(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding using hashing."""
        # Create multiple hash values for better distribution
        embeddings = []
        for i in range(self.embedding_dimension):
            hash_obj = hashlib.sha256(f"{text}_{i}".encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # Normalize to [-1, 1]
            normalized = (hash_int % 2000 - 1000) / 1000.0
            embeddings.append(normalized)
        return embeddings
    
    def store_memory(self, 
                    text: str,
                    summary: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    relationships: Optional[List[str]] = None) -> str:
        """
        Store a new memory entry.
        
        Args:
            text: Original text
            summary: Summarized version
            metadata: Additional metadata
            relationships: Related memory IDs
            
        Returns:
            Memory entry ID
        """
        # Generate ID
        memory_id = hashlib.sha256(f"{text}_{time.time()}".encode()).hexdigest()[:16]
        
        # Generate embedding
        embedding = self.generate_embedding(text)
        
        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            text=text,
            summary=summary,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time(),
            relationships=relationships or []
        )
        
        # Store in vector DB
        self._store_vector(memory_id, embedding, text)
        
        # Store metadata
        self.metadata[memory_id] = asdict(entry)
        self._save_metadata()
        
        # Update stats
        self.stats['total_memories'] += 1
        self.stats['last_update'] = time.time()
        
        logger.info(f"Stored memory: {memory_id}")
        return memory_id
    
    def _store_vector(self, memory_id: str, embedding: List[float], text: str):
        """Store vector in the appropriate backend."""
        if self.vector_backend == "chroma":
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[memory_id]
            )
        elif self.vector_backend == "faiss":
            self.faiss_index.add(np.array([embedding], dtype=np.float32))
            self.faiss_id_map[len(self.faiss_id_map)] = memory_id
            self._save_faiss_index()
        else:  # numpy
            self.numpy_vectors.append(embedding)
            self.numpy_ids.append(memory_id)
            self._save_numpy_vectors()
    
    def search_memories(self, 
                       query: str,
                       top_k: int = 5,
                       threshold: float = 0.7) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in vector store
        results = self._search_vectors(query_embedding, top_k * 2)  # Get more for filtering
        
        # Filter and prepare results
        memory_results = []
        for memory_id, similarity in results:
            if similarity >= threshold and memory_id in self.metadata:
                entry_data = self.metadata[memory_id]
                entry = MemoryEntry(**entry_data)
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.metadata[memory_id] = asdict(entry)
                
                memory_results.append((entry, similarity))
                
                if len(memory_results) >= top_k:
                    break
        
        # Update query statistics
        query_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_query_time'] = (
            (self.stats['avg_query_time'] * (self.stats['total_queries'] - 1) + query_time) /
            self.stats['total_queries']
        )
        
        # Save updated metadata
        self._save_metadata()
        
        return memory_results
    
    def _search_vectors(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """Search vectors in the appropriate backend."""
        if self.vector_backend == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            ids = results['ids'][0]
            distances = results['distances'][0]
            # Convert distances to similarities (cosine)
            similarities = [1 - d for d in distances]
            return list(zip(ids, similarities))
            
        elif self.vector_backend == "faiss":
            query_vec = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_vec, top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx in self.faiss_id_map:
                    # Convert L2 distance to similarity
                    similarity = 1 / (1 + dist)
                    results.append((self.faiss_id_map[idx], similarity))
            return results
            
        else:  # numpy
            if not self.numpy_vectors:
                return []
            
            # Compute cosine similarities
            query_vec = np.array(query_embedding)
            vectors = np.array(self.numpy_vectors)
            
            # Normalize vectors
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            
            # Compute similarities
            similarities = np.dot(vectors_norm, query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append((self.numpy_ids[idx], float(similarities[idx])))
            
            return results
    
    def get_related_memories(self, memory_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """
        Get memories related to a given memory.
        
        Args:
            memory_id: Starting memory ID
            max_depth: Maximum relationship depth to explore
            
        Returns:
            Dictionary mapping depth to list of memory IDs
        """
        if memory_id not in self.metadata:
            return {}
        
        related = {0: [memory_id]}
        visited = {memory_id}
        
        for depth in range(1, max_depth + 1):
            current_level = []
            
            for parent_id in related[depth - 1]:
                if parent_id in self.metadata:
                    parent_entry = self.metadata[parent_id]
                    
                    # Get direct relationships
                    for related_id in parent_entry.get('relationships', []):
                        if related_id not in visited and related_id in self.metadata:
                            current_level.append(related_id)
                            visited.add(related_id)
                    
                    # Also find similar memories
                    if 'embedding' in parent_entry and parent_entry['embedding']:
                        similar = self._search_vectors(parent_entry['embedding'], 5)
                        for sim_id, _ in similar:
                            if sim_id not in visited and sim_id != parent_id:
                                current_level.append(sim_id)
                                visited.add(sim_id)
            
            if current_level:
                related[depth] = current_level
            else:
                break
        
        return related
    
    def synthesize_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        Synthesize insights from multiple memories.
        
        Args:
            memory_ids: List of memory IDs to synthesize
            
        Returns:
            Synthesized knowledge representation
        """
        memories = []
        for memory_id in memory_ids:
            if memory_id in self.metadata:
                memories.append(MemoryEntry(**self.metadata[memory_id]))
        
        if not memories:
            return {}
        
        # Extract common themes
        all_text = " ".join([m.summary for m in memories])
        
        # Calculate centroid embedding
        embeddings = [m.embedding for m in memories if m.embedding]
        if embeddings:
            centroid = np.mean(embeddings, axis=0).tolist()
        else:
            centroid = None
        
        # Find contradictions (memories with very different embeddings)
        contradictions = []
        if len(embeddings) > 1:
            embedding_matrix = np.array(embeddings)
            similarities = np.dot(embedding_matrix, embedding_matrix.T)
            
            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    if similarities[i, j] < 0.3:  # Low similarity indicates contradiction
                        contradictions.append({
                            'memory1': memories[i].id,
                            'memory2': memories[j].id,
                            'similarity': float(similarities[i, j])
                        })
        
        return {
            'synthesized_from': memory_ids,
            'common_summary': all_text[:500] + "..." if len(all_text) > 500 else all_text,
            'centroid_embedding': centroid,
            'contradictions': contradictions,
            'timestamp': time.time(),
            'memory_count': len(memories)
        }
    
    def forget_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the system.
        
        Args:
            memory_id: Memory ID to remove
            
        Returns:
            Success status
        """
        if memory_id not in self.metadata:
            return False
        
        # Remove from metadata
        del self.metadata[memory_id]
        
        # Remove from vector store
        if self.vector_backend == "chroma":
            self.collection.delete(ids=[memory_id])
        elif self.vector_backend == "faiss":
            # FAISS doesn't support deletion, need to rebuild
            logger.warning("FAISS doesn't support deletion, index rebuild needed")
        else:  # numpy
            if memory_id in self.numpy_ids:
                idx = self.numpy_ids.index(memory_id)
                self.numpy_ids.pop(idx)
                self.numpy_vectors.pop(idx)
                self._save_numpy_vectors()
        
        # Update stats
        self.stats['total_memories'] -= 1
        self._save_metadata()
        
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.stats,
            'vector_backend': self.vector_backend,
            'embedding_model': self.model_name if self.embedding_model else 'fallback',
            'embedding_dimension': self.embedding_dimension,
            'storage_size_mb': self._calculate_storage_size()
        }
    
    def _calculate_storage_size(self) -> float:
        """Calculate total storage size in MB."""
        total_size = 0
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        if self.vector_backend == "faiss":
            index_path = os.path.join(self.storage_path, "faiss_index.bin")
            id_map_path = os.path.join(self.storage_path, "faiss_id_map.json")
            
            faiss.write_index(self.faiss_index, index_path)
            with open(id_map_path, 'w') as f:
                json.dump(self.faiss_id_map, f)
    
    def _save_numpy_vectors(self):
        """Save numpy vectors to disk."""
        if self.vector_backend == "numpy":
            vectors_path = os.path.join(self.storage_path, "vectors.npy")
            ids_path = os.path.join(self.storage_path, "ids.json")
            
            if self.numpy_vectors:
                np.save(vectors_path, np.array(self.numpy_vectors))
                with open(ids_path, 'w') as f:
                    json.dump(self.numpy_ids, f)


# Singleton instance
_semantic_memory_engine = None


def get_semantic_memory_engine() -> SemanticMemoryEngine:
    """Get or create the global semantic memory engine."""
    global _semantic_memory_engine
    if _semantic_memory_engine is None:
        _semantic_memory_engine = SemanticMemoryEngine()
    return _semantic_memory_engine


if __name__ == "__main__":
    # Example usage
    engine = get_semantic_memory_engine()
    
    # Store a memory
    memory_id = engine.store_memory(
        text="The mitochondria is the powerhouse of the cell.",
        summary="Mitochondria provide cellular energy.",
        metadata={"source": "biology textbook", "chapter": 3}
    )
    
    print(f"Stored memory: {memory_id}")
    
    # Search for related memories
    results = engine.search_memories("cellular energy production", top_k=3)
    
    for memory, score in results:
        print(f"Found: {memory.summary} (score: {score:.2f})")
    
    # Get statistics
    stats = engine.get_memory_stats()
    print(f"Memory stats: {stats}")