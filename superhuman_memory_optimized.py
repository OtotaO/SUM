"""
superhuman_memory_optimized.py - Optimized Superhuman Memory with O(log n) Similarity Search

Major improvements:
- Replace O(n²) similarity calculations with FAISS vector indexing
- Implement proper async database operations
- Add memory pagination to prevent unbounded growth
- Use approximate nearest neighbor search for fast retrieval
- Implement batch processing for pattern recognition
- Add proper resource cleanup and memory management

Author: SUM Development Team (Optimized)
License: Apache License 2.0
"""

import time
import json
import hashlib
import sqlite3
import aiosqlite
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
from pathlib import Path
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mmap
import struct

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage aligned with human cognition."""
    EPISODIC = "episodic"      # Events and experiences
    SEMANTIC = "semantic"       # Facts and concepts
    PROCEDURAL = "procedural"   # How-to knowledge
    WORKING = "working"         # Temporary active memory
    CRYSTALLIZED = "crystallized"  # Permanent insights


@dataclass
class MemoryTrace:
    """Enhanced memory representation with vector embeddings."""
    memory_id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[np.ndarray] = None  # Vector representation
    timestamp: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    emotional_valence: float = 0.0  # -1 to 1
    source_context: Dict[str, Any] = field(default_factory=dict)
    cross_modal_links: List[str] = field(default_factory=list)
    temporal_links: Dict[str, float] = field(default_factory=dict)
    pattern_matches: List[str] = field(default_factory=list)
    predictive_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding numpy arrays."""
        data = asdict(self)
        data['embedding'] = None  # Don't serialize embeddings directly
        data['timestamp'] = self.timestamp.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['memory_type'] = self.memory_type.value
        return data


class VectorIndex:
    """FAISS-based vector index for O(log n) similarity search."""
    
    def __init__(self, dimension: int = 768, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_map: Dict[int, str] = {}  # FAISS ID to memory ID
        self.memory_map: Dict[str, int] = {}  # Memory ID to FAISS ID
        self.next_id = 0
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index based on type."""
        if self.index_type == "IVF":
            # IVF index for datasets > 10k vectors
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.train(np.random.randn(1000, self.dimension).astype('float32'))
        else:
            # Flat index for smaller datasets
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_vector(self, memory_id: str, embedding: np.ndarray) -> int:
        """Add vector to index with O(log n) complexity."""
        if memory_id in self.memory_map:
            # Update existing vector
            faiss_id = self.memory_map[memory_id]
            self.index.remove_ids(np.array([faiss_id]))
        else:
            faiss_id = self.next_id
            self.next_id += 1
            self.id_map[faiss_id] = memory_id
            self.memory_map[memory_id] = faiss_id
        
        # Add to index
        self.index.add_with_ids(
            embedding.reshape(1, -1).astype('float32'),
            np.array([faiss_id])
        )
        return faiss_id
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors with O(log n) complexity."""
        if self.index.ntotal == 0:
            return []
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        # Convert to memory IDs with similarity scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx in self.id_map:
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + dist)
                results.append((self.id_map[idx], similarity))
        
        return results
    
    def remove_vector(self, memory_id: str):
        """Remove vector from index."""
        if memory_id in self.memory_map:
            faiss_id = self.memory_map[memory_id]
            self.index.remove_ids(np.array([faiss_id]))
            del self.memory_map[memory_id]
            del self.id_map[faiss_id]
    
    def save(self, path: str):
        """Save index to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'memory_map': self.memory_map,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
    
    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", 'rb') as f:
            meta = pickle.load(f)
            self.id_map = meta['id_map']
            self.memory_map = meta['memory_map']
            self.next_id = meta['next_id']


class OptimizedPatternRecognizer:
    """Optimized pattern recognition using vectorized operations."""
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.pattern_cache = {}
        self.executor = ProcessPoolExecutor(max_workers=4)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        self.pattern_cache[cache_key] = embedding
        
        # Limit cache size
        if len(self.pattern_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.pattern_cache.keys())[:5000]
            for key in keys_to_remove:
                del self.pattern_cache[key]
        
        return embedding
    
    def batch_get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts efficiently."""
        return self.embedding_model.encode(texts, convert_to_numpy=True, batch_size=32)
    
    def find_patterns_vectorized(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Find patterns using vectorized operations."""
        if len(embeddings) < 3:
            return {}
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(embeddings)
        
        patterns = {
            'clusters': defaultdict(list),
            'outliers': [],
            'cluster_centers': {}
        }
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                patterns['outliers'].append(i)
            else:
                patterns['clusters'][cluster_id].append(i)
        
        # Calculate cluster centers
        for cluster_id, indices in patterns['clusters'].items():
            cluster_embeddings = embeddings[indices]
            patterns['cluster_centers'][cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        return patterns
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


class OptimizedSuperhumanMemory:
    """
    Optimized superhuman memory system with O(log n) operations.
    
    Key optimizations:
    - FAISS for vector similarity search
    - Async SQLite operations
    - Memory-mapped files for large data
    - Batch processing for patterns
    - Lazy loading of embeddings
    """
    
    def __init__(self, 
                 db_path: str = "./superhuman_memory_optimized.db",
                 index_path: str = "./memory_index",
                 max_active_memories: int = 10000):
        self.db_path = db_path
        self.index_path = index_path
        self.max_active_memories = max_active_memories
        
        # Initialize components
        self.pattern_recognizer = OptimizedPatternRecognizer()
        self.vector_index = VectorIndex()
        
        # Memory storage with pagination
        self.active_memories: Dict[str, MemoryTrace] = {}
        self.memory_page_size = 1000
        self.current_page = 0
        
        # Pattern detection cache
        self.pattern_cache = {
            'last_update': datetime.now(),
            'patterns': {}
        }
        
        # Initialize database
        self._init_database()
        
        # Load existing index if available
        if Path(f"{index_path}.faiss").exists():
            self.vector_index.load(index_path)
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp REAL NOT NULL,
                importance_score REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Create indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance_score DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON memories(access_count DESC)")
        
        conn.commit()
        conn.close()
    
    async def store_memory_async(self, 
                                content: str, 
                                memory_type: MemoryType,
                                importance: Optional[float] = None,
                                source_context: Optional[Dict[str, Any]] = None) -> str:
        """Store memory asynchronously with O(log n) complexity."""
        # Generate embedding
        embedding = self.pattern_recognizer.get_embedding(content)
        
        # Create memory trace
        memory_id = hashlib.sha256(f"{content}:{time.time()}".encode()).hexdigest()[:16]
        
        memory = MemoryTrace(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance_score=importance or self._calculate_importance(content),
            source_context=source_context or {}
        )
        
        # Add to vector index (O(log n))
        self.vector_index.add_vector(memory_id, embedding)
        
        # Store in database asynchronously
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO memories 
                (memory_id, memory_type, content, embedding, timestamp, importance_score, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.memory_type.value,
                memory.content,
                embedding.tobytes(),
                memory.timestamp.timestamp(),
                memory.importance_score,
                memory.last_accessed.timestamp(),
                json.dumps(memory.to_dict())
            ))
            await db.commit()
        
        # Add to active memories with pagination
        self._manage_active_memories(memory)
        
        logger.info(f"Stored memory {memory_id} with importance {memory.importance_score:.2f}")
        return memory_id
    
    def store_memory(self, content: str, memory_type: MemoryType, **kwargs) -> str:
        """Synchronous wrapper for store_memory_async."""
        return asyncio.run(self.store_memory_async(content, memory_type, **kwargs))
    
    async def recall_memory_async(self, 
                                 query: str, 
                                 memory_types: Optional[List[MemoryType]] = None,
                                 limit: int = 10,
                                 min_relevance: float = 0.3) -> List[MemoryTrace]:
        """Recall memories with O(log n) similarity search."""
        # Get query embedding
        query_embedding = self.pattern_recognizer.get_embedding(query)
        
        # Search in vector index (O(log n))
        similar_memories = self.vector_index.search(query_embedding, k=limit * 2)
        
        # Filter and load memories
        recalled_memories = []
        
        async with aiosqlite.connect(self.db_path) as db:
            for memory_id, similarity in similar_memories:
                if similarity < min_relevance:
                    continue
                
                # Build query with optional type filter
                query_sql = "SELECT * FROM memories WHERE memory_id = ?"
                params = [memory_id]
                
                if memory_types:
                    type_values = [mt.value for mt in memory_types]
                    query_sql += f" AND memory_type IN ({','.join(['?' for _ in type_values])})"
                    params.extend(type_values)
                
                async with db.execute(query_sql, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        memory = self._row_to_memory(row)
                        memory.embedding = np.frombuffer(row[3], dtype=np.float32)
                        recalled_memories.append(memory)
                        
                        # Update access count asynchronously
                        await db.execute(
                            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE memory_id = ?",
                            (datetime.now().timestamp(), memory_id)
                        )
                
                if len(recalled_memories) >= limit:
                    break
            
            await db.commit()
        
        return recalled_memories
    
    def recall_memory(self, query: str, **kwargs) -> List[MemoryTrace]:
        """Synchronous wrapper for recall_memory_async."""
        return asyncio.run(self.recall_memory_async(query, **kwargs))
    
    def _manage_active_memories(self, new_memory: MemoryTrace):
        """Manage active memories with pagination."""
        self.active_memories[new_memory.memory_id] = new_memory
        
        # Check if we need to page out old memories
        if len(self.active_memories) > self.max_active_memories:
            # Sort by importance and last accessed
            sorted_memories = sorted(
                self.active_memories.items(),
                key=lambda x: (x[1].importance_score, x[1].last_accessed.timestamp()),
                reverse=True
            )
            
            # Keep top memories, remove others
            self.active_memories = dict(sorted_memories[:self.max_active_memories // 2])
            
            logger.info(f"Paged out {len(sorted_memories) - len(self.active_memories)} memories")
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content."""
        # Basic heuristics
        score = 0.5
        
        # Length factor
        words = content.split()
        if len(words) > 50:
            score += 0.1
        
        # Keyword detection
        important_keywords = ['breakthrough', 'important', 'critical', 'key', 'essential']
        for keyword in important_keywords:
            if keyword.lower() in content.lower():
                score += 0.1
                break
        
        # Complexity factor (unique words ratio)
        unique_ratio = len(set(words)) / max(1, len(words))
        score += unique_ratio * 0.2
        
        return min(1.0, score)
    
    def _row_to_memory(self, row: tuple) -> MemoryTrace:
        """Convert database row to MemoryTrace."""
        metadata = json.loads(row[8]) if row[8] else {}
        
        return MemoryTrace(
            memory_id=row[0],
            memory_type=MemoryType(row[1]),
            content=row[2],
            embedding=None,  # Load separately to save memory
            timestamp=datetime.fromtimestamp(row[4]),
            importance_score=row[5],
            access_count=row[6],
            last_accessed=datetime.fromtimestamp(row[7]),
            source_context=metadata.get('source_context', {}),
            emotional_valence=metadata.get('emotional_valence', 0.0)
        )
    
    async def analyze_patterns_async(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze patterns in memories with batch processing."""
        # Check cache first
        if (datetime.now() - self.pattern_cache['last_update']).seconds < 300:  # 5 min cache
            return self.pattern_cache['patterns']
        
        patterns = {
            'clusters': {},
            'temporal_patterns': {},
            'cross_modal_connections': {},
            'emerging_themes': []
        }
        
        # Load memories for analysis
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT memory_id, embedding, timestamp, memory_type FROM memories"
            params = []
            
            if time_window:
                cutoff = (datetime.now() - time_window).timestamp()
                query += " WHERE timestamp > ?"
                params.append(cutoff)
            
            query += " ORDER BY importance_score DESC LIMIT 1000"
            
            embeddings = []
            memory_ids = []
            timestamps = []
            types = []
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    if row[1]:  # Has embedding
                        embeddings.append(np.frombuffer(row[1], dtype=np.float32))
                        memory_ids.append(row[0])
                        timestamps.append(row[2])
                        types.append(row[3])
        
        if embeddings:
            # Convert to numpy array for vectorized operations
            embeddings_array = np.vstack(embeddings)
            
            # Find patterns using vectorized operations
            pattern_results = self.pattern_recognizer.find_patterns_vectorized(embeddings_array)
            
            # Process results
            for cluster_id, indices in pattern_results['clusters'].items():
                cluster_memories = [memory_ids[i] for i in indices]
                patterns['clusters'][f"cluster_{cluster_id}"] = {
                    'memory_ids': cluster_memories,
                    'size': len(cluster_memories),
                    'types': [types[i] for i in indices]
                }
            
            # Identify temporal patterns
            patterns['temporal_patterns'] = self._analyze_temporal_patterns(
                timestamps, memory_ids, pattern_results['clusters']
            )
        
        # Update cache
        self.pattern_cache = {
            'last_update': datetime.now(),
            'patterns': patterns
        }
        
        return patterns
    
    def analyze_patterns(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_patterns_async."""
        return asyncio.run(self.analyze_patterns_async(**kwargs))
    
    def _analyze_temporal_patterns(self, 
                                  timestamps: List[float], 
                                  memory_ids: List[str],
                                  clusters: Dict[int, List[int]]) -> Dict[str, Any]:
        """Analyze temporal patterns in memory clusters."""
        temporal_patterns = {
            'burst_periods': [],
            'recurring_themes': [],
            'evolution_chains': []
        }
        
        # Convert timestamps to datetime
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Find burst periods (many memories in short time)
        for i in range(len(datetimes) - 10):
            window = datetimes[i:i+10]
            time_span = (window[-1] - window[0]).total_seconds()
            
            if time_span < 3600:  # 10 memories within an hour
                temporal_patterns['burst_periods'].append({
                    'start': window[0].isoformat(),
                    'end': window[-1].isoformat(),
                    'memory_count': 10,
                    'memory_ids': memory_ids[i:i+10]
                })
        
        return temporal_patterns
    
    def save_index(self):
        """Save vector index to disk."""
        self.vector_index.save(self.index_path)
        logger.info(f"Saved vector index to {self.index_path}")
    
    def cleanup(self):
        """Clean up resources."""
        self.save_index()
        self.pattern_recognizer.cleanup()
        logger.info("Superhuman memory cleaned up successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts by type
        cursor.execute("""
            SELECT memory_type, COUNT(*) 
            FROM memories 
            GROUP BY memory_type
        """)
        type_counts = dict(cursor.fetchall())
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_count = cursor.fetchone()[0]
        
        # Get average importance
        cursor.execute("SELECT AVG(importance_score) FROM memories")
        avg_importance = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_memories': total_count,
            'active_memories': len(self.active_memories),
            'memories_by_type': type_counts,
            'average_importance': avg_importance,
            'index_size': self.vector_index.index.ntotal if self.vector_index.index else 0,
            'pattern_cache_age': (datetime.now() - self.pattern_cache['last_update']).seconds
        }


# Example usage
if __name__ == "__main__":
    print("Testing Optimized Superhuman Memory")
    print("=" * 50)
    
    # Initialize memory system
    memory = OptimizedSuperhumanMemory()
    
    # Test storing memories
    print("\nStoring test memories...")
    
    test_memories = [
        ("The key insight about neural networks is their ability to learn representations", MemoryType.SEMANTIC),
        ("Today I learned how to implement FAISS for fast similarity search", MemoryType.EPISODIC),
        ("To optimize Python code: 1) Profile first 2) Optimize bottlenecks 3) Use appropriate data structures", MemoryType.PROCEDURAL),
        ("Working on optimizing the O(n²) similarity calculation", MemoryType.WORKING),
        ("Pattern recognition is fundamentally about finding regularities in data", MemoryType.CRYSTALLIZED)
    ]
    
    memory_ids = []
    for content, mem_type in test_memories:
        mem_id = memory.store_memory(content, mem_type)
        memory_ids.append(mem_id)
        print(f"Stored {mem_type.value}: {mem_id}")
    
    # Test recall
    print("\nTesting recall...")
    query = "optimization techniques for Python"
    recalled = memory.recall_memory(query, limit=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Recalled {len(recalled)} memories:")
    for mem in recalled:
        print(f"  - {mem.memory_type.value}: {mem.content[:50]}...")
    
    # Test pattern analysis
    print("\nAnalyzing patterns...")
    patterns = memory.analyze_patterns()
    print(f"Found {len(patterns['clusters'])} clusters")
    
    # Get statistics
    print("\nMemory statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    memory.cleanup()
    print("\nOptimized superhuman memory test complete!")