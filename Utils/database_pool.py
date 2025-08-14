"""
Database connection pooling for improved performance and reliability
"""
import sqlite3
import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
import threading
import queue
import time

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """
    Connection pool for SQLite databases
    
    Note: SQLite has limitations with concurrent writes, but pooling helps with read performance
    """
    
    def __init__(self, 
                 database_path: str,
                 pool_size: int = 5,
                 max_overflow: int = 10,
                 timeout: float = 30.0):
        self.database_path = database_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        
        # Connection pool
        self._pool = queue.Queue(maxsize=pool_size)
        self._overflow = 0
        self._lock = threading.Lock()
        
        # Initialize pool with connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            
        # Connection statistics
        self.stats = {
            'connections_created': pool_size,
            'connections_active': 0,
            'connections_idle': pool_size,
            'wait_time_total': 0,
            'queries_executed': 0
        }
        
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimizations"""
        conn = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            isolation_level=None,  # Autocommit mode
            check_same_thread=False
        )
        
        # Enable optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        
        # Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
        
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        start_time = time.time()
        
        try:
            # Try to get from pool
            try:
                conn = self._pool.get(timeout=self.timeout)
                wait_time = time.time() - start_time
                self.stats['wait_time_total'] += wait_time
                
            except queue.Empty:
                # Pool exhausted, try overflow
                with self._lock:
                    if self._overflow < self.max_overflow:
                        self._overflow += 1
                        conn = self._create_connection()
                        self.stats['connections_created'] += 1
                    else:
                        raise Exception("Connection pool exhausted")
                        
            # Update stats
            with self._lock:
                self.stats['connections_active'] += 1
                self.stats['connections_idle'] = self._pool.qsize()
                
            yield conn
            
        finally:
            if conn:
                # Return connection to pool
                try:
                    # Check if connection is still valid
                    conn.execute("SELECT 1")
                    self._pool.put(conn)
                except:
                    # Connection is broken, create new one
                    try:
                        conn.close()
                    except:
                        pass
                    
                    if self._pool.qsize() < self.pool_size:
                        new_conn = self._create_connection()
                        self._pool.put(new_conn)
                        
                # Update stats
                with self._lock:
                    self.stats['connections_active'] -= 1
                    self.stats['connections_idle'] = self._pool.qsize()
                    
    def execute(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """Execute a query using a pooled connection"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            self.stats['queries_executed'] += 1
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return []
                
    def executemany(self, query: str, params_list: List[tuple]) -> None:
        """Execute many queries efficiently"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            self.stats['queries_executed'] += len(params_list)
            
    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass
                
    def get_stats(self) -> dict:
        """Get pool statistics"""
        return {
            **self.stats,
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'overflow_active': self._overflow
        }

class AsyncPostgreSQLPool:
    """
    Asynchronous connection pool for PostgreSQL using asyncpg
    
    This is for future use when migrating from SQLite to PostgreSQL
    """
    
    def __init__(self,
                 host: str,
                 port: int,
                 database: str,
                 user: str,
                 password: str,
                 min_size: int = 10,
                 max_size: int = 20,
                 max_queries: int = 50000,
                 max_inactive_connection_lifetime: float = 300.0):
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        
        self._pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize the connection pool"""
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            max_queries=self.max_queries,
            max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            command_timeout=60
        )
        
        # Create indexes for better performance
        async with self._pool.acquire() as conn:
            await self._create_indexes(conn)
            
    async def _create_indexes(self, conn):
        """Create database indexes for optimal performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_vector ON memories USING ivfflat (embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_graph(entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_relation ON knowledge_graph(relationship_type)"
        ]
        
        for index in indexes:
            try:
                await conn.execute(index)
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
                
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self._pool:
            raise Exception("Connection pool not initialized")
            
        async with self._pool.acquire() as conn:
            yield conn
            
    async def execute(self, query: str, *args) -> str:
        """Execute a query that doesn't return rows"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
            
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute a query and fetch all results"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
            
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and fetch a single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
            
    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
            
    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            
    async def get_stats(self) -> dict:
        """Get pool statistics"""
        if not self._pool:
            return {}
            
        return {
            'size': self._pool.get_size(),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'free_connections': self._pool.get_idle_size(),
            'used_connections': self._pool.get_size() - self._pool.get_idle_size()
        }

class DatabaseManager:
    """
    Manages database connections and provides a unified interface
    """
    
    def __init__(self, db_type: str = 'sqlite', **kwargs):
        self.db_type = db_type
        
        if db_type == 'sqlite':
            self.pool = SQLiteConnectionPool(**kwargs)
        elif db_type == 'postgresql':
            self.pool = AsyncPostgreSQLPool(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
    async def initialize(self):
        """Initialize the database connection pool"""
        if self.db_type == 'postgresql':
            await self.pool.initialize()
            
    def get_connection(self):
        """Get a database connection (sync)"""
        if self.db_type == 'sqlite':
            return self.pool.get_connection()
        else:
            raise Exception("Use acquire() for async databases")
            
    async def acquire(self):
        """Acquire a database connection (async)"""
        if self.db_type == 'postgresql':
            return self.pool.acquire()
        else:
            raise Exception("Use get_connection() for sync databases")
            
    def execute(self, query: str, params=None):
        """Execute a query (sync)"""
        if self.db_type == 'sqlite':
            return self.pool.execute(query, params)
        else:
            raise Exception("Use async execute methods for async databases")
            
    async def async_execute(self, query: str, *args):
        """Execute a query (async)"""
        if self.db_type == 'postgresql':
            return await self.pool.execute(query, *args)
        else:
            raise Exception("Use sync execute methods for sync databases")
            
    def close(self):
        """Close all database connections"""
        if self.db_type == 'sqlite':
            self.pool.close_all()
            
    async def async_close(self):
        """Close all database connections (async)"""
        if self.db_type == 'postgresql':
            await self.pool.close()
            
    def get_stats(self) -> dict:
        """Get database pool statistics"""
        return self.pool.get_stats()