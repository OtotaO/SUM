#!/usr/bin/env python3
"""
security_utils.py - Security Utilities for SUM

Provides security features including input validation, rate limiting,
and secure data handling for production deployments.

Author: SUM Development Team
License: Apache License 2.0
"""

import hashlib
import hmac
import time
import re
import secrets
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from functools import wraps
import ipaddress
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class SecurityMetrics:
    """Security event metrics."""
    event_type: str
    client_ip: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'client_ip': self.client_ip,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'severity': self.severity
        }


class InputValidator:
    """Validate and sanitize user inputs."""
    
    # Common injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\b(ALTER|CREATE|EXEC|EXECUTE)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\'\s*(OR|AND)\s*\'.+=\'.+)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    @classmethod
    def validate_text_input(cls, text: str, max_length: int = 100000) -> tuple[bool, Optional[str]]:
        """
        Validate text input for security issues.
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length} characters"
        
        # Check for SQL injection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Potential SQL injection detected"
        
        # Check for XSS
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Potential XSS attack detected"
        
        # Check for excessive special characters (potential obfuscation)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False, "Input contains excessive special characters"
        
        return True, None
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = re.sub(r'\.\.', '', sanitized)  # Remove ..
        sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')
        
        return sanitized
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) and len(email) <= 320
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL format and safety."""
        # Basic URL pattern
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        
        if not re.match(pattern, url):
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'192\.168\.',
            r'10\.',
            r'file://',
            r'ftp://',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.warning(f"Suspicious URL pattern detected: {pattern}")
                return False
        
        return True


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        self.block_duration = timedelta(minutes=15)
    
    def is_allowed(self, client_ip: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed for client IP.
        
        Returns:
            (is_allowed, error_message)
        """
        now = datetime.now()
        
        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            if now < self.blocked_ips[client_ip]:
                remaining = (self.blocked_ips[client_ip] - now).total_seconds()
                return False, f"IP blocked for {remaining:.0f} more seconds"
            else:
                # Unblock IP
                del self.blocked_ips[client_ip]
        
        # Clean old requests
        requests = self.requests[client_ip]
        cutoff_time = now - timedelta(seconds=self.time_window)
        
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= self.max_requests:
            # Block IP
            self.blocked_ips[client_ip] = now + self.block_duration
            logger.warning(f"Rate limit exceeded for IP {client_ip}, blocking for {self.block_duration}")
            
            return False, f"Rate limit exceeded. Blocked for {self.block_duration.total_seconds():.0f} seconds"
        
        # Add current request
        requests.append(now)
        return True, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = datetime.now()
        active_ips = len([ip for ip, timestamps in self.requests.items() if timestamps])
        blocked_ips = len([ip for ip, block_time in self.blocked_ips.items() if now < block_time])
        
        return {
            'active_ips': active_ips,
            'blocked_ips': blocked_ips,
            'total_tracked_ips': len(self.requests),
            'max_requests_per_window': self.max_requests,
            'time_window_seconds': self.time_window
        }


class APIKeyManager:
    """Manage API keys for authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, List[datetime]] = defaultdict(list)
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'permissions': permissions or ['read', 'write'],
            'active': True
        }
        return api_key
    
    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate API key and return user info."""
        if api_key not in self.api_keys:
            return False, None
        
        key_info = self.api_keys[api_key]
        if not key_info['active']:
            return False, None
        
        # Track usage
        self.key_usage[api_key].append(datetime.now())
        
        # Clean old usage records (keep last 1000)
        self.key_usage[api_key] = self.key_usage[api_key][-1000:]
        
        return True, key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            return True
        return False


class DataEncryption:
    """Encrypt sensitive data at rest."""
    
    def __init__(self, password: str):
        """Initialize with password."""
        self.fernet = self._create_fernet(password)
    
    def _create_fernet(self, password: str) -> Fernet:
        """Create Fernet instance from password."""
        password_bytes = password.encode()
        salt = b'salt_1234567890123456'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()


class SecurityMonitor:
    """Monitor security events and detect threats."""
    
    def __init__(self):
        self.events: deque = deque(maxlen=10000)
        self.threat_patterns = {
            'brute_force': {'threshold': 10, 'window': 300},  # 10 failed attempts in 5 minutes
            'rapid_requests': {'threshold': 1000, 'window': 60},  # 1000 requests in 1 minute
            'suspicious_patterns': {'threshold': 5, 'window': 3600}  # 5 suspicious patterns in 1 hour
        }
    
    def log_event(self, event: SecurityMetrics):
        """Log a security event."""
        self.events.append(event)
        logger.info(f"Security event: {event.event_type} from {event.client_ip}")
        
        # Check for threat patterns
        self._analyze_threats(event)
    
    def _analyze_threats(self, event: SecurityMetrics):
        """Analyze events for threat patterns."""
        now = datetime.now()
        
        for threat_type, config in self.threat_patterns.items():
            # Count relevant events in time window
            window_start = now - timedelta(seconds=config['window'])
            relevant_events = [
                e for e in self.events
                if e.client_ip == event.client_ip and
                e.timestamp >= window_start and
                self._is_relevant_for_threat(e, threat_type)
            ]
            
            if len(relevant_events) >= config['threshold']:
                self._trigger_threat_alert(threat_type, event.client_ip, len(relevant_events))
    
    def _is_relevant_for_threat(self, event: SecurityMetrics, threat_type: str) -> bool:
        """Check if event is relevant for threat type."""
        if threat_type == 'brute_force':
            return event.event_type in ['login_failed', 'api_key_invalid']
        elif threat_type == 'rapid_requests':
            return event.event_type == 'api_request'
        elif threat_type == 'suspicious_patterns':
            return event.severity in ['warning', 'critical']
        return False
    
    def _trigger_threat_alert(self, threat_type: str, client_ip: str, event_count: int):
        """Trigger threat alert."""
        logger.critical(f"THREAT DETECTED: {threat_type} from {client_ip} - {event_count} events")
        
        # In production, this could:
        # - Send alert to security team
        # - Automatically block IP
        # - Trigger incident response
        
        alert_event = SecurityMetrics(
            event_type='threat_detected',
            client_ip=client_ip,
            timestamp=datetime.now(),
            details={
                'threat_type': threat_type,
                'event_count': event_count
            },
            severity='critical'
        )
        self.events.append(alert_event)
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]
        
        # Group by severity
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        by_ip = defaultdict(int)
        
        for event in recent_events:
            by_severity[event.severity] += 1
            by_type[event.event_type] += 1
            by_ip[event.client_ip] += 1
        
        return {
            'total_events': len(recent_events),
            'by_severity': dict(by_severity),
            'by_type': dict(by_type),
            'top_ips': sorted(by_ip.items(), key=lambda x: x[1], reverse=True)[:10],
            'time_range_hours': hours
        }


# Decorator for security validation
def require_api_key(api_manager: APIKeyManager):
    """Decorator to require valid API key."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from request (implementation depends on framework)
            api_key = kwargs.get('api_key') or kwargs.get('x_api_key')
            
            if not api_key:
                raise SecurityError("API key required")
            
            is_valid, key_info = api_manager.validate_api_key(api_key)
            if not is_valid:
                raise SecurityError("Invalid API key")
            
            # Add user info to kwargs
            kwargs['user_info'] = key_info
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(limiter: RateLimiter):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract client IP (implementation depends on framework)
            client_ip = kwargs.get('client_ip', '127.0.0.1')
            
            is_allowed, error_msg = limiter.is_allowed(client_ip)
            if not is_allowed:
                raise RateLimitError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Custom exceptions
class SecurityError(Exception):
    """Base security exception."""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded."""
    pass


class ValidationError(SecurityError):
    """Input validation failed."""
    pass


# Example usage and testing
if __name__ == "__main__":
    print("Testing Security Utilities")
    print("=" * 40)
    
    # Test input validation
    validator = InputValidator()
    
    test_inputs = [
        "Normal text input",
        "SELECT * FROM users WHERE id = 1; --",
        "<script>alert('xss')</script>",
        "Very long text " * 1000,
        "Excessive!!@#$%^&*(){}[]|\\:;\"'<>,.?/~`"
    ]
    
    print("\\nInput Validation Tests:")
    for i, test_input in enumerate(test_inputs, 1):
        is_valid, error = validator.validate_text_input(test_input)
        status = "✓ PASS" if is_valid else f"✗ FAIL: {error}"
        print(f"{i}. {status}")
    
    # Test rate limiting
    rate_limiter = RateLimiter(max_requests=5, time_window=10)
    
    print("\\nRate Limiting Tests:")
    test_ip = "192.168.1.100"
    
    for i in range(8):
        is_allowed, error = rate_limiter.is_allowed(test_ip)
        status = "✓ ALLOWED" if is_allowed else f"✗ BLOCKED: {error}"
        print(f"Request {i+1}: {status}")
    
    # Test API key management
    api_manager = APIKeyManager()
    
    print("\\nAPI Key Management:")
    api_key = api_manager.generate_api_key("user123", ["read", "write"])
    print(f"Generated API key: {api_key[:20]}...")
    
    is_valid, info = api_manager.validate_api_key(api_key)
    print(f"Validation: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    api_manager.revoke_api_key(api_key)
    is_valid, info = api_manager.validate_api_key(api_key)
    print(f"After revocation: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    # Test encryption
    encryption = DataEncryption("my_secret_password")
    
    print("\\nData Encryption:")
    original = "Sensitive data that needs protection"
    encrypted = encryption.encrypt(original)
    decrypted = encryption.decrypt(encrypted)
    
    print(f"Original: {original}")
    print(f"Encrypted: {encrypted[:30]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {'✓ YES' if original == decrypted else '✗ NO'}")
    
    print("\\nSecurity utilities ready!")