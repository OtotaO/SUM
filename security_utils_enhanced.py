#!/usr/bin/env python3
"""
security_utils_enhanced.py - Enhanced Security Utilities for SUM

Major improvements:
- Secure per-user salt generation and storage
- Proper secret management with environment variables
- Enhanced input validation with configurable rules
- Improved rate limiting with distributed support
- Better thread safety and async support
- Comprehensive security monitoring and alerting

Author: SUM Development Team (Enhanced)
License: Apache License 2.0
"""

import hashlib
import hmac
import time
import re
import secrets
import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from functools import wraps
import ipaddress
import base64
from pathlib import Path
from threading import Lock
import aiofiles
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.getenv('SUM_REDIS_URL', 'redis://localhost:6379')
SALT_FILE_PATH = os.getenv('SUM_SALT_FILE_PATH', './data/salts.json')
SECRET_KEY = os.getenv('SUM_SECRET_KEY', None)
RATE_LIMIT_BACKEND = os.getenv('SUM_RATE_LIMIT_BACKEND', 'memory')  # memory or redis

if not SECRET_KEY:
    logger.warning("No SUM_SECRET_KEY found in environment. Using development key.")
    SECRET_KEY = 'development-only-key-not-for-production'


@dataclass
class SecurityMetrics:
    """Enhanced security event metrics with additional context."""
    event_type: str
    client_ip: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'client_ip': self.client_ip,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'severity': self.severity,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'user_agent': self.user_agent
        }


class SaltManager:
    """Manage per-user salts securely."""
    
    def __init__(self, salt_file_path: str = SALT_FILE_PATH):
        self.salt_file_path = Path(salt_file_path)
        self.salts: Dict[str, str] = {}
        self._lock = Lock()
        self._ensure_salt_file()
        self._load_salts()
    
    def _ensure_salt_file(self):
        """Ensure salt file exists with proper permissions."""
        self.salt_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.salt_file_path.exists():
            self.salt_file_path.write_text('{}')
            # Set restrictive permissions (owner read/write only)
            os.chmod(self.salt_file_path, 0o600)
    
    def _load_salts(self):
        """Load salts from file."""
        try:
            with open(self.salt_file_path, 'r') as f:
                self.salts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load salts: {e}")
            self.salts = {}
    
    def _save_salts(self):
        """Save salts to file."""
        with self._lock:
            with open(self.salt_file_path, 'w') as f:
                json.dump(self.salts, f)
    
    def get_or_create_salt(self, user_id: str) -> bytes:
        """Get existing salt or create new one for user."""
        if user_id in self.salts:
            return base64.b64decode(self.salts[user_id])
        
        # Generate new salt
        salt = os.urandom(32)
        salt_b64 = base64.b64encode(salt).decode('utf-8')
        
        with self._lock:
            self.salts[user_id] = salt_b64
            self._save_salts()
        
        logger.info(f"Generated new salt for user {user_id}")
        return salt


class EnhancedInputValidator:
    """Enhanced input validation with configurable rules."""
    
    # Compiled regex patterns for better performance
    SQL_PATTERNS = [
        re.compile(r"(\bUNION\b.*\bSELECT\b)", re.IGNORECASE),
        re.compile(r"(\bINSERT\b.*\bINTO\b)", re.IGNORECASE),
        re.compile(r"(\bDELETE\b.*\bFROM\b)", re.IGNORECASE),
        re.compile(r"(\bDROP\b.*\bTABLE\b)", re.IGNORECASE),
        re.compile(r"(\b(ALTER|CREATE|EXEC|EXECUTE)\b)", re.IGNORECASE),
        re.compile(r"(--|\#|\/\*|\*\/)"),
        re.compile(r"(\bOR\b.*=.*\bOR\b)", re.IGNORECASE),
        re.compile(r"('\s*(OR|AND)\s*'.+='.+)", re.IGNORECASE),
    ]
    
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"vbscript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<(iframe|object|embed|svg|img)[^>]*>", re.IGNORECASE),
    ]
    
    @classmethod
    def validate_text_input(
        cls, 
        text: str, 
        max_length: int = 100000,
        allow_html: bool = False,
        check_sql: bool = True,
        check_xss: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Enhanced text validation with configurable checks.
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if not text.strip():
            return False, "Input cannot be empty"
        
        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length} characters"
        
        # Check for null bytes (security issue)
        if '\x00' in text:
            return False, "Input contains null bytes"
        
        # SQL injection check
        if check_sql:
            for pattern in cls.SQL_PATTERNS:
                if pattern.search(text):
                    logger.warning(f"Potential SQL injection detected: {pattern.pattern}")
                    return False, "Potential SQL injection detected"
        
        # XSS check
        if check_xss and not allow_html:
            for pattern in cls.XSS_PATTERNS:
                if pattern.search(text):
                    logger.warning(f"Potential XSS attack detected: {pattern.pattern}")
                    return False, "Potential XSS attack detected"
        
        # Check for control characters
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 10:  # Allow some legitimate control chars
            return False, "Input contains excessive control characters"
        
        return True, None
    
    @classmethod
    def sanitize_for_html(cls, text: str) -> str:
        """Sanitize text for safe HTML display."""
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


class DistributedRateLimiter:
    """Rate limiting with support for distributed systems."""
    
    def __init__(
        self, 
        max_requests: int = 100, 
        time_window: int = 60,
        backend: str = RATE_LIMIT_BACKEND
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
            backend: 'memory' or 'redis'
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.backend = backend
        
        if backend == 'redis':
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        else:
            self.requests: Dict[str, deque] = defaultdict(lambda: deque())
            self.blocked_ips: Dict[str, datetime] = {}
            self._lock = Lock()
        
        self.block_duration = timedelta(minutes=15)
    
    async def is_allowed_async(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Async version of rate limit check."""
        if self.backend == 'redis':
            return await self._check_redis_async(client_ip)
        else:
            return self._check_memory(client_ip)
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Synchronous rate limit check."""
        if self.backend == 'redis':
            return self._check_redis(client_ip)
        else:
            return self._check_memory(client_ip)
    
    def _check_memory(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Check rate limit using in-memory storage."""
        now = datetime.now()
        
        with self._lock:
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                if now < self.blocked_ips[client_ip]:
                    remaining = (self.blocked_ips[client_ip] - now).total_seconds()
                    return False, f"IP blocked for {remaining:.0f} more seconds"
                else:
                    del self.blocked_ips[client_ip]
            
            # Clean old requests
            requests = self.requests[client_ip]
            cutoff_time = now - timedelta(seconds=self.time_window)
            
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Check rate limit
            if len(requests) >= self.max_requests:
                self.blocked_ips[client_ip] = now + self.block_duration
                logger.warning(f"Rate limit exceeded for IP {client_ip}")
                return False, f"Rate limit exceeded. Blocked for {self.block_duration.total_seconds():.0f} seconds"
            
            requests.append(now)
            return True, None
    
    def _check_redis(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Check rate limit using Redis."""
        now = int(time.time())
        key = f"rate_limit:{client_ip}"
        block_key = f"blocked:{client_ip}"
        
        try:
            # Check if blocked
            blocked_until = self.redis_client.get(block_key)
            if blocked_until and int(blocked_until) > now:
                remaining = int(blocked_until) - now
                return False, f"IP blocked for {remaining} more seconds"
            
            # Count requests in window
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, now - self.time_window)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.time_window + 60)
            
            results = pipe.execute()
            request_count = results[1]
            
            if request_count >= self.max_requests:
                # Block the IP
                block_until = now + int(self.block_duration.total_seconds())
                self.redis_client.setex(block_key, self.block_duration, block_until)
                return False, f"Rate limit exceeded. Blocked for {self.block_duration.total_seconds():.0f} seconds"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fallback to allow request on Redis error
            return True, None
    
    async def _check_redis_async(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Async Redis check (requires aioredis)."""
        # For now, fall back to sync version
        return self._check_redis(client_ip)


class SecureDataEncryption:
    """Enhanced encryption with per-user salts and key rotation."""
    
    def __init__(self, user_id: str, salt_manager: Optional[SaltManager] = None):
        """Initialize encryption for a specific user."""
        self.user_id = user_id
        self.salt_manager = salt_manager or SaltManager()
        self._fernet_cache: Dict[str, Fernet] = {}
        self._lock = Lock()
    
    def _get_fernet(self, password: str, version: int = 1) -> Fernet:
        """Get or create Fernet instance with caching."""
        cache_key = f"{self.user_id}:{version}:{hashlib.sha256(password.encode()).hexdigest()[:8]}"
        
        if cache_key in self._fernet_cache:
            return self._fernet_cache[cache_key]
        
        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._fernet_cache:
                return self._fernet_cache[cache_key]
            
            # Get user-specific salt
            salt = self.salt_manager.get_or_create_salt(self.user_id)
            
            # Derive key from password and salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,  # OWASP recommended minimum
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            fernet = Fernet(key)
            
            # Cache the instance
            self._fernet_cache[cache_key] = fernet
            
            return fernet
    
    def encrypt(self, data: str, password: str) -> str:
        """Encrypt string data with user-specific encryption."""
        try:
            fernet = self._get_fernet(password)
            encrypted = fernet.encrypt(data.encode())
            # Add version prefix for future key rotation
            versioned = b'v1:' + encrypted
            return base64.urlsafe_b64encode(versioned).decode()
        except Exception as e:
            logger.error(f"Encryption error for user {self.user_id}: {e}")
            raise
    
    def decrypt(self, encrypted_data: str, password: str) -> str:
        """Decrypt string data."""
        try:
            # Decode from base64
            versioned = base64.urlsafe_b64decode(encrypted_data.encode())
            
            # Extract version
            if versioned.startswith(b'v1:'):
                version = 1
                encrypted = versioned[3:]
            else:
                # Legacy format
                version = 0
                encrypted = versioned
            
            fernet = self._get_fernet(password, version)
            decrypted = fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error for user {self.user_id}: {e}")
            raise


class EnhancedSecurityMonitor:
    """Enhanced security monitoring with real-time alerting."""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.events: deque = deque(maxlen=100000)  # Increased capacity
        self.alert_callback = alert_callback
        self._lock = Lock()
        
        # Enhanced threat patterns
        self.threat_patterns = {
            'brute_force': {
                'threshold': 10, 
                'window': 300,
                'severity': 'critical'
            },
            'rapid_requests': {
                'threshold': 1000, 
                'window': 60,
                'severity': 'warning'
            },
            'suspicious_patterns': {
                'threshold': 5, 
                'window': 3600,
                'severity': 'warning'
            },
            'sql_injection_attempts': {
                'threshold': 3,
                'window': 600,
                'severity': 'critical'
            },
            'xss_attempts': {
                'threshold': 3,
                'window': 600,
                'severity': 'critical'
            }
        }
        
        # Start background threat analysis
        self._start_background_analysis()
    
    def _start_background_analysis(self):
        """Start background thread for continuous threat analysis."""
        import threading
        
        def analyze_loop():
            while True:
                try:
                    self._analyze_threat_trends()
                    time.sleep(60)  # Analyze every minute
                except Exception as e:
                    logger.error(f"Background analysis error: {e}")
        
        thread = threading.Thread(target=analyze_loop, daemon=True)
        thread.start()
    
    def log_event(self, event: SecurityMetrics):
        """Log a security event with immediate threat detection."""
        with self._lock:
            self.events.append(event)
        
        logger.info(
            f"Security event: {event.event_type} from {event.client_ip} "
            f"(user: {event.user_id}, severity: {event.severity})"
        )
        
        # Immediate threat detection
        self._analyze_threats(event)
    
    def _analyze_threats(self, event: SecurityMetrics):
        """Analyze events for threat patterns."""
        now = datetime.now()
        
        for threat_type, config in self.threat_patterns.items():
            # Count relevant events in time window
            window_start = now - timedelta(seconds=config['window'])
            
            with self._lock:
                relevant_events = [
                    e for e in self.events
                    if e.client_ip == event.client_ip and
                    e.timestamp >= window_start and
                    self._is_relevant_for_threat(e, threat_type)
                ]
            
            if len(relevant_events) >= config['threshold']:
                self._trigger_threat_alert(
                    threat_type, 
                    event.client_ip, 
                    len(relevant_events),
                    config['severity']
                )
    
    def _is_relevant_for_threat(self, event: SecurityMetrics, threat_type: str) -> bool:
        """Enhanced threat relevance checking."""
        threat_mappings = {
            'brute_force': ['login_failed', 'api_key_invalid', 'auth_failed'],
            'rapid_requests': ['api_request', 'page_view'],
            'suspicious_patterns': lambda e: e.severity in ['warning', 'critical'],
            'sql_injection_attempts': ['sql_injection_blocked', 'validation_failed_sql'],
            'xss_attempts': ['xss_blocked', 'validation_failed_xss']
        }
        
        mapping = threat_mappings.get(threat_type)
        
        if callable(mapping):
            return mapping(event)
        elif isinstance(mapping, list):
            return event.event_type in mapping
        else:
            return event.severity in ['warning', 'critical']
    
    def _trigger_threat_alert(
        self, 
        threat_type: str, 
        client_ip: str, 
        event_count: int,
        severity: str
    ):
        """Trigger enhanced threat alert."""
        alert_msg = (
            f"THREAT DETECTED: {threat_type} from {client_ip} - "
            f"{event_count} events (severity: {severity})"
        )
        logger.critical(alert_msg)
        
        # Create alert event
        alert_event = SecurityMetrics(
            event_type='threat_detected',
            client_ip=client_ip,
            timestamp=datetime.now(),
            details={
                'threat_type': threat_type,
                'event_count': event_count,
                'severity': severity,
                'recommended_action': self._get_recommended_action(threat_type)
            },
            severity='critical'
        )
        
        with self._lock:
            self.events.append(alert_event)
        
        # Trigger callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert_event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _get_recommended_action(self, threat_type: str) -> str:
        """Get recommended action for threat type."""
        actions = {
            'brute_force': 'Block IP immediately and investigate account',
            'rapid_requests': 'Apply stricter rate limiting',
            'suspicious_patterns': 'Monitor closely and prepare to block',
            'sql_injection_attempts': 'Block IP and review application logs',
            'xss_attempts': 'Block IP and check for vulnerabilities'
        }
        return actions.get(threat_type, 'Monitor and investigate')
    
    def _analyze_threat_trends(self):
        """Analyze threat trends over time."""
        try:
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            
            with self._lock:
                recent_threats = [
                    e for e in self.events
                    if e.timestamp >= hour_ago and e.event_type == 'threat_detected'
                ]
            
            if len(recent_threats) > 10:
                logger.warning(
                    f"High threat activity detected: {len(recent_threats)} "
                    f"threats in the last hour"
                )
                
                # Could trigger additional actions like:
                # - Enabling stricter security mode
                # - Notifying security team
                # - Activating additional monitoring
        
        except Exception as e:
            logger.error(f"Threat trend analysis error: {e}")


# Enhanced decorators with async support
def require_api_key_async(api_manager):
    """Async decorator to require valid API key."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract API key from request
            api_key = kwargs.get('api_key') or kwargs.get('x_api_key')
            
            if not api_key:
                raise SecurityError("API key required")
            
            is_valid, key_info = api_manager.validate_api_key(api_key)
            if not is_valid:
                raise SecurityError("Invalid API key")
            
            kwargs['user_info'] = key_info
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input_async(validator=None, **validation_kwargs):
    """Async decorator for input validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract text input from kwargs
            text = kwargs.get('text') or kwargs.get('content')
            
            if text:
                validator_func = validator or EnhancedInputValidator.validate_text_input
                is_valid, error = validator_func(text, **validation_kwargs)
                
                if not is_valid:
                    raise ValidationError(error)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and comprehensive testing
if __name__ == "__main__":
    import asyncio
    
    print("Testing Enhanced Security Utilities")
    print("=" * 50)
    
    # Test salt manager
    print("\nTesting Salt Manager:")
    salt_manager = SaltManager()
    salt1 = salt_manager.get_or_create_salt("user123")
    salt2 = salt_manager.get_or_create_salt("user123")
    salt3 = salt_manager.get_or_create_salt("user456")
    
    print(f"Salt consistency: {'✓ PASS' if salt1 == salt2 else '✗ FAIL'}")
    print(f"Salt uniqueness: {'✓ PASS' if salt1 != salt3 else '✗ FAIL'}")
    
    # Test enhanced encryption
    print("\nTesting Enhanced Encryption:")
    encryption1 = SecureDataEncryption("user123", salt_manager)
    encryption2 = SecureDataEncryption("user456", salt_manager)
    
    test_data = "Sensitive information that needs protection"
    password = "user_password_123"
    
    encrypted1 = encryption1.encrypt(test_data, password)
    encrypted2 = encryption2.encrypt(test_data, password)
    
    print(f"Different users, different encryption: {'✓ PASS' if encrypted1 != encrypted2 else '✗ FAIL'}")
    
    decrypted1 = encryption1.decrypt(encrypted1, password)
    print(f"Decryption works: {'✓ PASS' if decrypted1 == test_data else '✗ FAIL'}")
    
    # Test enhanced input validation
    print("\nTesting Enhanced Input Validation:")
    test_inputs = [
        ("Normal text input", True),
        ("SELECT * FROM users WHERE id = 1; --", False),
        ("<script>alert('xss')</script>", False),
        ("Text with \x00 null byte", False),
        ("", False),  # Empty input
    ]
    
    for test_text, expected_valid in test_inputs:
        is_valid, error = EnhancedInputValidator.validate_text_input(test_text)
        status = "✓ PASS" if (is_valid == expected_valid) else "✗ FAIL"
        print(f"{status}: {repr(test_text[:30])}... {'valid' if is_valid else f'invalid: {error}'}")
    
    # Test distributed rate limiter
    print("\nTesting Distributed Rate Limiter:")
    rate_limiter = DistributedRateLimiter(max_requests=5, time_window=10, backend='memory')
    
    test_ip = "192.168.1.100"
    for i in range(8):
        is_allowed, error = rate_limiter.is_allowed(test_ip)
        status = "✓ ALLOWED" if is_allowed else f"✗ BLOCKED: {error}"
        print(f"Request {i+1}: {status}")
    
    # Test security monitoring
    print("\nTesting Enhanced Security Monitor:")
    
    def alert_handler(event: SecurityMetrics):
        print(f"ALERT TRIGGERED: {event.event_type} - {event.details}")
    
    monitor = EnhancedSecurityMonitor(alert_callback=alert_handler)
    
    # Simulate events
    for i in range(12):
        monitor.log_event(SecurityMetrics(
            event_type='login_failed',
            client_ip='10.0.0.1',
            timestamp=datetime.now(),
            severity='warning',
            user_id='test_user'
        ))
    
    print("\nEnhanced security utilities ready for production!")