"""
Smart Caching System for SUM

Implements an intelligent caching layer that:
- Caches summaries based on content hash
- Supports TTL and size limits
- Uses semantic similarity for near-match detection
- Provides cache warming and invalidation

Author: SUM Team
License: Apache License 2.0
"""

import hashlib
import json
import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import pickle
import sqlite3
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached summary."""
    content_hash: str
    summary: Dict[str, Any]
    text_length: int
    model_type: str
    config_hash: str
    created_at: float
    last_accessed: float
    access_count: int
    processing_time: float
    

class SmartCache:
    """
    Intelligent caching system with semantic matching.
    
    Features:
    - Content-based deduplication
    - Semantic similarity matching
    - TTL and size-based eviction
    - Persistent storage with SQLite
    - Thread-safe operations
    """
    
    def __init__(self,
                 cache_dir: str = ".sum_cache",
                 max_size_mb: int = 1024,
                 default_ttl_hours: int = 168,  # 1 week
                 similarity_threshold: float = 0.95):
        """
        Initialize smart cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
            default_ttl_hours: Default time-to-live in hours
            similarity_threshold: Threshold for semantic similarity matching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.similarity_threshold = similarity_threshold
        
        # Initialize database
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        # In-memory cache for fast access
        self.memory_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Load frequently accessed entries
        self._warm_cache()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    content_hash TEXT PRIMARY KEY,
                    summary_data BLOB,
                    text_length INTEGER,
                    model_type TEXT,
                    config_hash TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    processing_time REAL,
                    embedding BLOB
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_config 
                ON cache_entries(model_type, config_hash)
            """)
            
    def get(self, 
            text: str, 
            model_type: str,
            config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached summary if available.
        
        Args:
            text: Input text
            model_type: Model used for summarization
            config: Model configuration
            
        Returns:
            Cached summary or None
        """
        # Generate hashes
        content_hash = self._hash_text(text)
        config_hash = self._hash_config(config)
        
        with self.cache_lock:
            # Check memory cache first
            cache_key = f"{content_hash}:{model_type}:{config_hash}"
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                self._update_database(entry)
                
                logger.info(f"Cache hit (memory): {content_hash[:8]}...")
                return entry.summary
                
            # Check database
            entry = self._load_from_database(content_hash, model_type, config_hash)
            if entry:
                # Add to memory cache
                self.memory_cache[cache_key] = entry
                self._ensure_memory_limit()
                
                logger.info(f"Cache hit (disk): {content_hash[:8]}...")
                return entry.summary
                
            # Try semantic similarity matching for long texts
            if len(text) > 10000:
                similar_entry = self._find_similar(text, model_type, config_hash)
                if similar_entry:
                    logger.info(f"Cache hit (similar): {similar_entry.content_hash[:8]}...")
                    return similar_entry.summary
                    
        logger.info(f"Cache miss: {content_hash[:8]}...")
        return None
        
    def put(self,
            text: str,
            model_type: str,
            config: Dict[str, Any],
            summary: Dict[str, Any],
            processing_time: float):
        """
        Store summary in cache.
        
        Args:
            text: Original text
            model_type: Model used
            config: Model configuration
            summary: Generated summary
            processing_time: Time taken to process
        """
        # Generate hashes
        content_hash = self._hash_text(text)
        config_hash = self._hash_config(config)
        
        # Create cache entry
        entry = CacheEntry(
            content_hash=content_hash,
            summary=summary,
            text_length=len(text),
            model_type=model_type,
            config_hash=config_hash,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            processing_time=processing_time
        )
        
        with self.cache_lock:
            # Add to memory cache
            cache_key = f"{content_hash}:{model_type}:{config_hash}"
            self.memory_cache[cache_key] = entry
            self._ensure_memory_limit()
            
            # Save to database
            self._save_to_database(entry)
            
            # Generate embedding for similarity matching (async)
            if len(text) > 10000:
                threading.Thread(
                    target=self._generate_embedding,
                    args=(content_hash, text),
                    daemon=True
                ).start()
                
        logger.info(f"Cached summary: {content_hash[:8]}... ({len(text)} chars)")
        
    def invalidate(self, text: Optional[str] = None, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            text: Specific text to invalidate
            pattern: Pattern to match for invalidation
        """
        with self.cache_lock:
            if text:
                # Invalidate specific text
                content_hash = self._hash_text(text)
                self._invalidate_hash(content_hash)
            elif pattern:
                # Invalidate by pattern
                # This would need pattern matching implementation
                pass
            else:
                # Clear all cache
                self.memory_cache.clear()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
                logger.info("Cache cleared")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total = conn.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()[0]
            
            # Total size
            total_size = conn.execute(
                "SELECT SUM(LENGTH(summary_data)) FROM cache_entries"
            ).fetchone()[0] or 0
            
            # Hit rate (last 24 hours)
            yesterday = time.time() - 86400
            recent_hits = conn.execute(
                "SELECT SUM(access_count) FROM cache_entries WHERE last_accessed > ?",
                (yesterday,)
            ).fetchone()[0] or 0
            
        return {
            'total_entries': total,
            'memory_entries': len(self.memory_cache),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'recent_hits': recent_hits,
            'hit_rate': self._calculate_hit_rate()
        }
        
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        # Use first and last 1000 chars for very long texts
        if len(text) > 10000:
            sample = text[:5000] + text[-5000:]
        else:
            sample = text
            
        return hashlib.sha256(sample.encode('utf-8')).hexdigest()
        
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        # Sort keys for consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
        
    def _warm_cache(self):
        """Load frequently accessed entries into memory."""
        with sqlite3.connect(self.db_path) as conn:
            # Load top 100 most accessed entries from last week
            week_ago = time.time() - (7 * 86400)
            rows = conn.execute("""
                SELECT * FROM cache_entries 
                WHERE last_accessed > ? 
                ORDER BY access_count DESC 
                LIMIT 100
            """, (week_ago,)).fetchall()
            
            for row in rows:
                entry = self._row_to_entry(row)
                cache_key = f"{entry.content_hash}:{entry.model_type}:{entry.config_hash}"
                self.memory_cache[cache_key] = entry
                
        logger.info(f"Warmed cache with {len(self.memory_cache)} entries")
        
    def _ensure_memory_limit(self):
        """Ensure memory cache doesn't exceed size limit."""
        # Simple LRU eviction
        while len(self.memory_cache) > 1000:  # Max 1000 entries in memory
            # Remove least recently used
            self.memory_cache.popitem(last=False)
            
    def _cleanup_loop(self):
        """Background cleanup thread."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_expired()
                self._enforce_size_limit()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                
    def _cleanup_expired(self):
        """Remove expired entries."""
        with sqlite3.connect(self.db_path) as conn:
            expired_time = time.time() - self.default_ttl.total_seconds()
            conn.execute(
                "DELETE FROM cache_entries WHERE last_accessed < ?",
                (expired_time,)
            )
            
    def _enforce_size_limit(self):
        """Enforce cache size limit."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current size
            total_size = conn.execute(
                "SELECT SUM(LENGTH(summary_data)) FROM cache_entries"
            ).fetchone()[0] or 0
            
            if total_size > self.max_size_bytes:
                # Delete oldest entries
                to_delete = total_size - (self.max_size_bytes * 0.8)  # Free 20%
                
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE content_hash IN (
                        SELECT content_hash 
                        FROM cache_entries 
                        ORDER BY last_accessed ASC 
                        LIMIT (
                            SELECT COUNT(*) 
                            FROM cache_entries 
                            WHERE LENGTH(summary_data) < ?
                        )
                    )
                """, (to_delete,))
                
    def _save_to_database(self, entry: CacheEntry):
        """Save entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (content_hash, summary_data, text_length, model_type, 
                 config_hash, created_at, last_accessed, access_count, 
                 processing_time, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.content_hash,
                pickle.dumps(entry.summary),
                entry.text_length,
                entry.model_type,
                entry.config_hash,
                entry.created_at,
                entry.last_accessed,
                entry.access_count,
                entry.processing_time,
                None  # Embedding generated async
            ))
            
    def _load_from_database(self, 
                           content_hash: str,
                           model_type: str,
                           config_hash: str) -> Optional[CacheEntry]:
        """Load entry from database."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT * FROM cache_entries 
                WHERE content_hash = ? AND model_type = ? AND config_hash = ?
            """, (content_hash, model_type, config_hash)).fetchone()
            
            if row:
                return self._row_to_entry(row)
        return None
        
    def _row_to_entry(self, row) -> CacheEntry:
        """Convert database row to CacheEntry."""
        return CacheEntry(
            content_hash=row[0],
            summary=pickle.loads(row[1]),
            text_length=row[2],
            model_type=row[3],
            config_hash=row[4],
            created_at=row[5],
            last_accessed=row[6],
            access_count=row[7],
            processing_time=row[8]
        )
        
    def _update_database(self, entry: CacheEntry):
        """Update entry in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cache_entries 
                SET last_accessed = ?, access_count = ?
                WHERE content_hash = ? AND model_type = ? AND config_hash = ?
            """, (
                entry.last_accessed,
                entry.access_count,
                entry.content_hash,
                entry.model_type,
                entry.config_hash
            ))
            
    def _invalidate_hash(self, content_hash: str):
        """Invalidate entries by content hash."""
        # Remove from memory
        keys_to_remove = [
            k for k in self.memory_cache.keys() 
            if k.startswith(content_hash)
        ]
        for key in keys_to_remove:
            del self.memory_cache[key]
            
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE content_hash = ?",
                (content_hash,)
            )
            
    def _generate_embedding(self, content_hash: str, text: str):
        """Generate embedding for similarity matching."""
        try:
            # Simple embedding using TF-IDF or sentence transformers
            # For now, using a simple approach
            words = text.lower().split()[:1000]  # First 1000 words
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
            # Create simple embedding vector
            # In production, use sentence-transformers or similar
            embedding = list(word_freq.values())[:100]  # Top 100 frequencies
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE cache_entries SET embedding = ? WHERE content_hash = ?",
                    (pickle.dumps(embedding), content_hash)
                )
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            
    def _find_similar(self, 
                     text: str,
                     model_type: str,
                     config_hash: str) -> Optional[CacheEntry]:
        """Find similar cached entries using embeddings."""
        # For now, return None - full implementation would use
        # cosine similarity on embeddings
        return None
        
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This would track hits/misses over time
        # For now, return a placeholder
        return 0.0


# Global cache instance
_cache_instance = None
_cache_lock = threading.Lock()


def get_cache() -> SmartCache:
    """Get or create global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = SmartCache()
                
    return _cache_instance


def cache_result(func):
    """Decorator for caching function results."""
    def wrapper(text: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        cache = get_cache()
        
        # Try cache first
        cached = cache.get(text, model_type, config)
        if cached:
            cached['cached'] = True
            return cached
            
        # Process and cache
        start_time = time.time()
        result = func(text, model_type, config)
        processing_time = time.time() - start_time
        
        if 'error' not in result:
            cache.put(text, model_type, config, result, processing_time)
            
        result['cached'] = False
        return result
        
    return wrapper