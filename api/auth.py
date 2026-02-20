"""
API Authentication System for SUM

Provides secure API key authentication with rate limiting,
usage tracking, and admin capabilities.

Author: SUM Team
License: Apache License 2.0
"""

import os
import secrets
import hashlib
import time
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import sqlite3
import threading

from flask import request, jsonify, current_app
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """Represents an API key."""
    key_id: str
    key_hash: str
    name: str
    created_at: float
    last_used: float
    is_active: bool
    rate_limit: int  # requests per minute
    daily_limit: int  # requests per day
    total_requests: int
    permissions: List[str]
    metadata: Dict[str, Any]


class APIAuthManager:
    """
    Manages API authentication and authorization.
    
    Features:
    - Secure API key generation and validation
    - Rate limiting per key
    - Usage tracking and analytics
    - Permission-based access control
    - Key rotation and expiration
    """
    
    def __init__(self, 
                 db_path: str = "api_keys.db",
                 default_rate_limit: int = 60,
                 default_daily_limit: int = 10000):
        """
        Initialize API authentication manager.
        
        Args:
            db_path: Path to SQLite database
            default_rate_limit: Default requests per minute
            default_daily_limit: Default requests per day
        """
        self.db_path = db_path
        self.default_rate_limit = default_rate_limit
        self.default_daily_limit = default_daily_limit
        
        # Thread-safe locks
        self.db_lock = threading.Lock()
        self.rate_limit_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Create default admin key if none exists
        self._ensure_admin_key()
        
    def _init_database(self):
        """Initialize the API keys database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used REAL,
                    is_active INTEGER DEFAULT 1,
                    rate_limit INTEGER,
                    daily_limit INTEGER,
                    total_requests INTEGER DEFAULT 0,
                    permissions TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time REAL,
                    request_size INTEGER,
                    response_size INTEGER,
                    FOREIGN KEY (key_id) REFERENCES api_keys(key_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp 
                ON api_usage(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_key_id 
                ON api_usage(key_id, timestamp)
            """)
            
    def generate_api_key(self, 
                        name: str,
                        permissions: List[str] = None,
                        rate_limit: Optional[int] = None,
                        daily_limit: Optional[int] = None,
                        metadata: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Generate a new API key.
        
        Args:
            name: Friendly name for the key
            permissions: List of allowed permissions
            rate_limit: Custom rate limit (requests/minute)
            daily_limit: Custom daily limit
            metadata: Additional metadata
            
        Returns:
            Tuple of (key_id, api_key)
        """
        # Generate secure random key
        key_id = secrets.token_urlsafe(16)
        secret = secrets.token_urlsafe(32)
        # New format: sum_{key_id}.{secret} for efficient lookup with salted hash
        api_key = f"sum_{key_id}.{secret}"
        key_hash = self._hash_key(api_key)
        
        # Default permissions
        if permissions is None:
            permissions = ['read', 'summarize']
            
        # Create key record
        key_record = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            last_used=0,
            is_active=True,
            rate_limit=rate_limit or self.default_rate_limit,
            daily_limit=daily_limit or self.default_daily_limit,
            total_requests=0,
            permissions=permissions,
            metadata=metadata or {}
        )
        
        # Save to database
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_keys 
                    (key_id, key_hash, name, created_at, last_used, 
                     is_active, rate_limit, daily_limit, total_requests, 
                     permissions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key_record.key_id,
                    key_record.key_hash,
                    key_record.name,
                    key_record.created_at,
                    key_record.last_used,
                    key_record.is_active,
                    key_record.rate_limit,
                    key_record.daily_limit,
                    key_record.total_requests,
                    json.dumps(key_record.permissions),
                    json.dumps(key_record.metadata)
                ))
                
        logger.info(f"Generated API key '{name}' with ID {key_id}")
        return key_id, api_key
        
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return key info.
        Supports both new salted keys and legacy unsalted keys.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        if not api_key or not api_key.startswith('sum_'):
            return None
            
        with sqlite3.connect(self.db_path) as conn:
            # Try new format first: sum_{key_id}.{secret}
            if '.' in api_key:
                parts = api_key.split('.')
                if len(parts) == 2:
                    prefix_with_id = parts[0]
                    if prefix_with_id.startswith('sum_'):
                        key_id = prefix_with_id[4:]

                        row = conn.execute("""
                            SELECT * FROM api_keys
                            WHERE key_id = ? AND is_active = 1
                        """, (key_id,)).fetchone()

                        if row:
                            key_info = self._row_to_api_key(row)
                            # Verify with secure salted hash if present
                            if key_info.key_hash.startswith(('pbkdf2:', 'scrypt:', 'argon2:')):
                                if check_password_hash(key_info.key_hash, api_key):
                                    return key_info

            # Fallback for legacy unsalted SHA256 keys or if dot format failed
            legacy_hash = hashlib.sha256(api_key.encode()).hexdigest()
            row = conn.execute("""
                SELECT * FROM api_keys 
                WHERE key_hash = ? AND is_active = 1
            """, (legacy_hash,)).fetchone()
            
            if row:
                return self._row_to_api_key(row)
                
        return None
        
    def check_rate_limit(self, key_id: str, rate_limit: int) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            key_id: API key ID
            rate_limit: Requests per minute limit
            
        Returns:
            True if within limit, False otherwise
        """
        now = time.time()
        minute_ago = now - 60
        
        with self.cache_lock:
            # Clean old entries
            if key_id in self.rate_limit_cache:
                self.rate_limit_cache[key_id] = [
                    ts for ts in self.rate_limit_cache[key_id] 
                    if ts > minute_ago
                ]
            else:
                self.rate_limit_cache[key_id] = []
                
            # Check limit
            if len(self.rate_limit_cache[key_id]) >= rate_limit:
                return False
                
            # Add current request
            self.rate_limit_cache[key_id].append(now)
            return True
            
    def check_daily_limit(self, key_id: str, daily_limit: int) -> bool:
        """Check if request is within daily limit."""
        day_ago = time.time() - 86400
        
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM api_usage 
                WHERE key_id = ? AND timestamp > ?
            """, (key_id, day_ago)).fetchone()[0]
            
        return count < daily_limit
        
    def log_usage(self, 
                  key_id: str,
                  endpoint: str,
                  method: str,
                  status_code: int,
                  response_time: float,
                  request_size: int = 0,
                  response_size: int = 0):
        """Log API usage for analytics."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_usage 
                    (key_id, timestamp, endpoint, method, status_code, 
                     response_time, request_size, response_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key_id, time.time(), endpoint, method, 
                    status_code, response_time, request_size, response_size
                ))
                
                # Update last used and total requests
                conn.execute("""
                    UPDATE api_keys 
                    SET last_used = ?, total_requests = total_requests + 1
                    WHERE key_id = ?
                """, (time.time(), key_id))
                
    def get_usage_stats(self, key_id: str, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        since = time.time() - (days * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            # Total requests
            total = conn.execute("""
                SELECT COUNT(*) FROM api_usage 
                WHERE key_id = ? AND timestamp > ?
            """, (key_id, since)).fetchone()[0]
            
            # Requests by endpoint
            endpoints = conn.execute("""
                SELECT endpoint, COUNT(*) as count 
                FROM api_usage 
                WHERE key_id = ? AND timestamp > ?
                GROUP BY endpoint
                ORDER BY count DESC
            """, (key_id, since)).fetchall()
            
            # Average response time
            avg_time = conn.execute("""
                SELECT AVG(response_time) 
                FROM api_usage 
                WHERE key_id = ? AND timestamp > ?
            """, (key_id, since)).fetchone()[0] or 0
            
            # Error rate
            errors = conn.execute("""
                SELECT COUNT(*) FROM api_usage 
                WHERE key_id = ? AND timestamp > ? AND status_code >= 400
            """, (key_id, since)).fetchone()[0]
            
        return {
            'total_requests': total,
            'endpoints': dict(endpoints),
            'avg_response_time': round(avg_time, 3),
            'error_rate': round(errors / max(1, total) * 100, 2),
            'period_days': days
        }
        
    def revoke_key(self, key_id: str):
        """Revoke an API key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE api_keys 
                SET is_active = 0 
                WHERE key_id = ?
            """, (key_id,))
            
        logger.info(f"Revoked API key {key_id}")
        
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual keys)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT key_id, name, created_at, last_used, 
                       is_active, rate_limit, daily_limit, 
                       total_requests, permissions
                FROM api_keys
                ORDER BY created_at DESC
            """).fetchall()
            
        keys = []
        for row in rows:
            keys.append({
                'key_id': row[0],
                'name': row[1],
                'created_at': datetime.fromtimestamp(row[2]).isoformat(),
                'last_used': datetime.fromtimestamp(row[3]).isoformat() if row[3] else None,
                'is_active': bool(row[4]),
                'rate_limit': row[5],
                'daily_limit': row[6],
                'total_requests': row[7],
                'permissions': json.loads(row[8])
            })
            
        return keys
        
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage using secure salted hashing."""
        return generate_password_hash(api_key)
        
    def _row_to_api_key(self, row) -> APIKey:
        """Convert database row to APIKey object."""
        return APIKey(
            key_id=row[0],
            key_hash=row[1],
            name=row[2],
            created_at=row[3],
            last_used=row[4],
            is_active=bool(row[5]),
            rate_limit=row[6],
            daily_limit=row[7],
            total_requests=row[8],
            permissions=json.loads(row[9]),
            metadata=json.loads(row[10]) if row[10] else {}
        )
        
    def _ensure_admin_key(self):
        """Ensure at least one admin key exists."""
        with sqlite3.connect(self.db_path) as conn:
            admin_exists = conn.execute("""
                SELECT COUNT(*) FROM api_keys 
                WHERE permissions LIKE '%admin%' AND is_active = 1
            """).fetchone()[0]
            
        if not admin_exists:
            # Create default admin key
            key_id, api_key = self.generate_api_key(
                name="Default Admin Key",
                permissions=['admin', 'read', 'write', 'summarize'],
                rate_limit=1000,
                daily_limit=100000,
                metadata={'auto_generated': True}
            )
            
            # Save key to file for initial setup
            key_file = Path("admin_api_key.txt")
            key_file.write_text(f"Admin API Key: {api_key}\nKey ID: {key_id}\n")
            key_file.chmod(0o600)  # Restrict permissions
            
            logger.warning(f"Generated admin API key. Check {key_file} for details.")


# Global auth manager instance
_auth_manager = None


def get_auth_manager() -> APIAuthManager:
    """Get or create global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = APIAuthManager()
    return _auth_manager


def require_api_key(permissions: List[str] = None):
    """
    Decorator to require API key authentication.
    
    Args:
        permissions: Required permissions for the endpoint
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API key from header or query parameter
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
                
            # Validate key
            auth_manager = get_auth_manager()
            key_info = auth_manager.validate_api_key(api_key)
            
            if not key_info:
                return jsonify({'error': 'Invalid API key'}), 401
                
            # Check permissions
            if permissions:
                missing = set(permissions) - set(key_info.permissions)
                if missing:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required': list(missing)
                    }), 403
                    
            # Check rate limits
            if not auth_manager.check_rate_limit(key_info.key_id, key_info.rate_limit):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'limit': f'{key_info.rate_limit} requests per minute'
                }), 429
                
            if not auth_manager.check_daily_limit(key_info.key_id, key_info.daily_limit):
                return jsonify({
                    'error': 'Daily limit exceeded',
                    'limit': f'{key_info.daily_limit} requests per day'
                }), 429
                
            # Add key info to request context
            request.api_key_info = key_info
            
            # Track timing
            start_time = time.time()
            
            # Call the actual function
            response = func(*args, **kwargs)
            
            # Log usage
            response_time = time.time() - start_time
            status_code = response[1] if isinstance(response, tuple) else 200
            
            auth_manager.log_usage(
                key_id=key_info.key_id,
                endpoint=request.endpoint,
                method=request.method,
                status_code=status_code,
                response_time=response_time,
                request_size=request.content_length or 0
            )
            
            return response
            
        return wrapper
    return decorator


def optional_api_key():
    """Decorator to optionally use API key for better limits."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for API key
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            if api_key:
                auth_manager = get_auth_manager()
                key_info = auth_manager.validate_api_key(api_key)
                if key_info:
                    request.api_key_info = key_info
                    # Apply key-specific rate limits
                    if not auth_manager.check_rate_limit(key_info.key_id, key_info.rate_limit):
                        return jsonify({'error': 'Rate limit exceeded'}), 429
            else:
                # Apply default public rate limits (more restrictive)
                request.api_key_info = None
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator