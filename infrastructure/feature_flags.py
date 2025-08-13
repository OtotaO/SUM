"""
Feature Flags System for SUM Platform
======================================

Enterprise-grade feature flag management for:
- Gradual feature rollouts
- A/B testing
- User segmentation
- Emergency kill switches
- Configuration management
- Real-time updates without deployment

Based on LaunchDarkly and Optimizely patterns.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import json
import logging
import random
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"          # Simple on/off
    PERCENTAGE = "percentage"    # Percentage-based rollout
    VARIANT = "variant"          # Multi-variant (A/B/C testing)
    CONFIG = "config"            # Configuration value
    KILL_SWITCH = "kill_switch"  # Emergency disable


class TargetingRule:
    """
    Targeting rule for feature flag evaluation.
    
    Supports various operators for flexible targeting.
    """
    
    OPERATORS = {
        'eq': lambda a, b: a == b,
        'neq': lambda a, b: a != b,
        'gt': lambda a, b: a > b,
        'gte': lambda a, b: a >= b,
        'lt': lambda a, b: a < b,
        'lte': lambda a, b: a <= b,
        'in': lambda a, b: a in b,
        'not_in': lambda a, b: a not in b,
        'contains': lambda a, b: b in str(a),
        'regex': lambda a, b: bool(re.match(b, str(a))),
        'semver_eq': lambda a, b: a == b,  # Simplified, use proper semver library
        'semver_gt': lambda a, b: a > b,   # Simplified
        'semver_lt': lambda a, b: a < b,   # Simplified
    }
    
    def __init__(self, attribute: str, operator: str, value: Any):
        """
        Initialize targeting rule.
        
        Args:
            attribute: User attribute to check
            operator: Comparison operator
            value: Value to compare against
        """
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")
        
        self.attribute = attribute
        self.operator = operator
        self.value = value
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate rule against user context.
        
        Args:
            context: User context dictionary
        
        Returns:
            True if rule matches
        """
        if self.attribute not in context:
            return False
        
        user_value = context[self.attribute]
        operator_func = self.OPERATORS[self.operator]
        
        try:
            return operator_func(user_value, self.value)
        except Exception as e:
            logger.warning(f"Rule evaluation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'attribute': self.attribute,
            'operator': self.operator,
            'value': self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetingRule':
        """Create from dictionary"""
        return cls(
            attribute=data['attribute'],
            operator=data['operator'],
            value=data['value']
        )


@dataclass
class FeatureFlag:
    """
    Feature flag definition with targeting and variants.
    """
    
    key: str                                    # Unique identifier
    name: str                                    # Human-readable name
    description: str = ""                        # Description
    flag_type: FlagType = FlagType.BOOLEAN      # Flag type
    enabled: bool = False                        # Global enable/disable
    
    # Variants for A/B testing
    variants: List[Dict[str, Any]] = field(default_factory=list)
    default_variant: str = "control"
    
    # Targeting rules
    rules: List[TargetingRule] = field(default_factory=list)
    rule_operator: str = "all"  # "all" or "any"
    
    # Percentage rollout
    rollout_percentage: float = 0.0
    
    # User/group overrides
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    group_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    config_value: Any = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Other flag keys
    
    # Metrics
    evaluation_count: int = 0
    last_evaluated: Optional[datetime] = None
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluate flag for given user context.
        
        Args:
            context: User context with attributes
        
        Returns:
            Flag value based on evaluation
        """
        self.evaluation_count += 1
        self.last_evaluated = datetime.now()
        
        # Check if globally disabled
        if not self.enabled:
            if self.flag_type == FlagType.BOOLEAN:
                return False
            elif self.flag_type == FlagType.VARIANT:
                return self.default_variant
            elif self.flag_type == FlagType.CONFIG:
                return self.config_value
            else:
                return False
        
        # Check user overrides
        user_id = context.get('user_id')
        if user_id and user_id in self.user_overrides:
            return self.user_overrides[user_id]
        
        # Check group overrides
        groups = context.get('groups', [])
        for group in groups:
            if group in self.group_overrides:
                return self.group_overrides[group]
        
        # Evaluate targeting rules
        if self.rules:
            rules_pass = self._evaluate_rules(context)
            if not rules_pass:
                if self.flag_type == FlagType.BOOLEAN:
                    return False
                elif self.flag_type == FlagType.VARIANT:
                    return self.default_variant
                else:
                    return self.config_value
        
        # Handle based on flag type
        if self.flag_type == FlagType.BOOLEAN:
            return self._evaluate_boolean(context)
        elif self.flag_type == FlagType.PERCENTAGE:
            return self._evaluate_percentage(context)
        elif self.flag_type == FlagType.VARIANT:
            return self._evaluate_variant(context)
        elif self.flag_type == FlagType.CONFIG:
            return self.config_value
        elif self.flag_type == FlagType.KILL_SWITCH:
            return not self.enabled  # Inverted for kill switch
        else:
            return False
    
    def _evaluate_rules(self, context: Dict[str, Any]) -> bool:
        """Evaluate targeting rules"""
        if not self.rules:
            return True
        
        if self.rule_operator == "all":
            return all(rule.evaluate(context) for rule in self.rules)
        elif self.rule_operator == "any":
            return any(rule.evaluate(context) for rule in self.rules)
        else:
            return True
    
    def _evaluate_boolean(self, context: Dict[str, Any]) -> bool:
        """Evaluate boolean flag"""
        if self.rollout_percentage > 0:
            return self._evaluate_percentage(context)
        return True
    
    def _evaluate_percentage(self, context: Dict[str, Any]) -> bool:
        """Evaluate percentage-based rollout"""
        # Use consistent hashing for stable assignment
        user_id = context.get('user_id', 'anonymous')
        hash_input = f"{self.key}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) + 1
        
        return bucket <= self.rollout_percentage
    
    def _evaluate_variant(self, context: Dict[str, Any]) -> str:
        """Evaluate variant assignment"""
        if not self.variants:
            return self.default_variant
        
        # Use consistent hashing for stable assignment
        user_id = context.get('user_id', 'anonymous')
        hash_input = f"{self.key}:variant:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Calculate variant based on weights
        total_weight = sum(v.get('weight', 1) for v in self.variants)
        bucket = (hash_value % total_weight) + 1
        
        cumulative_weight = 0
        for variant in self.variants:
            cumulative_weight += variant.get('weight', 1)
            if bucket <= cumulative_weight:
                return variant['key']
        
        return self.default_variant
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'name': self.name,
            'description': self.description,
            'flag_type': self.flag_type.value,
            'enabled': self.enabled,
            'variants': self.variants,
            'default_variant': self.default_variant,
            'rules': [r.to_dict() for r in self.rules],
            'rule_operator': self.rule_operator,
            'rollout_percentage': self.rollout_percentage,
            'user_overrides': self.user_overrides,
            'group_overrides': self.group_overrides,
            'config_value': self.config_value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': list(self.tags),
            'depends_on': self.depends_on,
            'evaluation_count': self.evaluation_count,
            'last_evaluated': self.last_evaluated.isoformat() if self.last_evaluated else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureFlag':
        """Create from dictionary"""
        flag = cls(
            key=data['key'],
            name=data['name'],
            description=data.get('description', ''),
            flag_type=FlagType(data['flag_type']),
            enabled=data.get('enabled', False),
            variants=data.get('variants', []),
            default_variant=data.get('default_variant', 'control'),
            rule_operator=data.get('rule_operator', 'all'),
            rollout_percentage=data.get('rollout_percentage', 0.0),
            user_overrides=data.get('user_overrides', {}),
            group_overrides=data.get('group_overrides', {}),
            config_value=data.get('config_value'),
            tags=set(data.get('tags', [])),
            depends_on=data.get('depends_on', []),
            evaluation_count=data.get('evaluation_count', 0)
        )
        
        # Parse rules
        flag.rules = [
            TargetingRule.from_dict(r) 
            for r in data.get('rules', [])
        ]
        
        # Parse dates
        if 'created_at' in data:
            flag.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            flag.updated_at = datetime.fromisoformat(data['updated_at'])
        if data.get('last_evaluated'):
            flag.last_evaluated = datetime.fromisoformat(data['last_evaluated'])
        
        return flag


class FeatureFlagManager:
    """
    Central manager for feature flags with persistence and caching.
    
    Features:
    - Database persistence
    - In-memory caching
    - Real-time updates
    - Metrics collection
    - Dependency resolution
    """
    
    def __init__(self, 
                 db_path: str = "feature_flags.db",
                 cache_ttl: float = 60.0,
                 enable_metrics: bool = True):
        """
        Initialize feature flag manager.
        
        Args:
            db_path: Path to SQLite database
            cache_ttl: Cache time-to-live in seconds
            enable_metrics: Enable metrics collection
        """
        self.db_path = db_path
        self.cache_ttl = cache_ttl
        self.enable_metrics = enable_metrics
        
        # In-memory cache
        self.flags: Dict[str, FeatureFlag] = {}
        self.cache_timestamp = 0
        self.lock = threading.Lock()
        
        # Metrics
        self.metrics = defaultdict(lambda: {
            'evaluations': 0,
            'true_evaluations': 0,
            'false_evaluations': 0,
            'variants': defaultdict(int)
        })
        
        # Initialize database
        self._init_database()
        
        # Load flags from database
        self.reload_flags()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_flags (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flag_metrics (
                flag_key TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evaluations INTEGER DEFAULT 0,
                true_count INTEGER DEFAULT 0,
                false_count INTEGER DEFAULT 0,
                variants TEXT,
                PRIMARY KEY (flag_key, timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_flag(self, flag: FeatureFlag) -> bool:
        """
        Create a new feature flag.
        
        Args:
            flag: Feature flag to create
        
        Returns:
            True if created successfully
        """
        with self.lock:
            if flag.key in self.flags:
                logger.warning(f"Flag {flag.key} already exists")
                return False
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "INSERT INTO feature_flags (key, data) VALUES (?, ?)",
                    (flag.key, json.dumps(flag.to_dict()))
                )
                conn.commit()
                
                # Update cache
                self.flags[flag.key] = flag
                
                logger.info(f"Created feature flag: {flag.key}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to create flag {flag.key}: {e}")
                conn.rollback()
                return False
            
            finally:
                conn.close()
    
    def update_flag(self, flag: FeatureFlag) -> bool:
        """
        Update an existing feature flag.
        
        Args:
            flag: Updated feature flag
        
        Returns:
            True if updated successfully
        """
        with self.lock:
            if flag.key not in self.flags:
                logger.warning(f"Flag {flag.key} does not exist")
                return False
            
            flag.updated_at = datetime.now()
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "UPDATE feature_flags SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?",
                    (json.dumps(flag.to_dict()), flag.key)
                )
                conn.commit()
                
                # Update cache
                self.flags[flag.key] = flag
                
                logger.info(f"Updated feature flag: {flag.key}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to update flag {flag.key}: {e}")
                conn.rollback()
                return False
            
            finally:
                conn.close()
    
    def delete_flag(self, key: str) -> bool:
        """
        Delete a feature flag.
        
        Args:
            key: Flag key to delete
        
        Returns:
            True if deleted successfully
        """
        with self.lock:
            if key not in self.flags:
                return False
            
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("DELETE FROM feature_flags WHERE key = ?", (key,))
                conn.commit()
                
                # Remove from cache
                del self.flags[key]
                
                logger.info(f"Deleted feature flag: {key}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to delete flag {key}: {e}")
                conn.rollback()
                return False
            
            finally:
                conn.close()
    
    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag by key.
        
        Args:
            key: Flag key
        
        Returns:
            Feature flag or None if not found
        """
        # Check cache validity
        if time.time() - self.cache_timestamp > self.cache_ttl:
            self.reload_flags()
        
        return self.flags.get(key)
    
    def evaluate(self, 
                key: str,
                context: Dict[str, Any],
                default: Any = False) -> Any:
        """
        Evaluate a feature flag.
        
        Args:
            key: Flag key
            context: User context
            default: Default value if flag not found
        
        Returns:
            Flag value based on evaluation
        """
        flag = self.get_flag(key)
        
        if not flag:
            logger.debug(f"Flag {key} not found, returning default: {default}")
            return default
        
        # Check dependencies
        if flag.depends_on:
            for dep_key in flag.depends_on:
                if not self.evaluate(dep_key, context, False):
                    logger.debug(f"Dependency {dep_key} not met for flag {key}")
                    return default
        
        # Evaluate flag
        value = flag.evaluate(context)
        
        # Record metrics
        if self.enable_metrics:
            self._record_metric(key, value)
        
        return value
    
    def _record_metric(self, key: str, value: Any):
        """Record evaluation metric"""
        metric = self.metrics[key]
        metric['evaluations'] += 1
        
        if isinstance(value, bool):
            if value:
                metric['true_evaluations'] += 1
            else:
                metric['false_evaluations'] += 1
        elif isinstance(value, str):
            metric['variants'][value] += 1
    
    def reload_flags(self):
        """Reload flags from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT key, data FROM feature_flags")
            rows = cursor.fetchall()
            
            self.flags.clear()
            
            for key, data_str in rows:
                try:
                    data = json.loads(data_str)
                    flag = FeatureFlag.from_dict(data)
                    self.flags[key] = flag
                except Exception as e:
                    logger.error(f"Failed to load flag {key}: {e}")
            
            conn.close()
            self.cache_timestamp = time.time()
            
            logger.info(f"Loaded {len(self.flags)} feature flags")
    
    def list_flags(self, tags: Optional[Set[str]] = None) -> List[FeatureFlag]:
        """
        List all feature flags.
        
        Args:
            tags: Filter by tags
        
        Returns:
            List of feature flags
        """
        flags = list(self.flags.values())
        
        if tags:
            flags = [f for f in flags if tags.intersection(f.tags)]
        
        return flags
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature flag metrics"""
        return {
            'total_flags': len(self.flags),
            'enabled_flags': sum(1 for f in self.flags.values() if f.enabled),
            'flag_metrics': dict(self.metrics),
            'cache_age': time.time() - self.cache_timestamp
        }
    
    def save_metrics(self):
        """Save metrics to database"""
        if not self.enable_metrics:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, metric in self.metrics.items():
            cursor.execute(
                """INSERT INTO flag_metrics 
                   (flag_key, evaluations, true_count, false_count, variants)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    key,
                    metric['evaluations'],
                    metric['true_evaluations'],
                    metric['false_evaluations'],
                    json.dumps(dict(metric['variants']))
                )
            )
        
        conn.commit()
        conn.close()
        
        # Clear metrics after saving
        self.metrics.clear()


# Global feature flag manager instance
feature_flags = FeatureFlagManager()


# Convenience decorators
def feature_flag(key: str, default: Any = False):
    """
    Decorator to conditionally execute functions based on feature flags.
    
    Args:
        key: Feature flag key
        default: Default value if flag not found
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context from kwargs or use default
            context = kwargs.get('ff_context', {'user_id': 'anonymous'})
            
            if feature_flags.evaluate(key, context, default):
                return func(*args, **kwargs)
            else:
                logger.debug(f"Feature {key} disabled, skipping {func.__name__}")
                return None
        
        return wrapper
    
    return decorator


def variant(key: str, default: str = "control"):
    """
    Decorator for A/B testing with variants.
    
    Args:
        key: Feature flag key
        default: Default variant
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context from kwargs
            context = kwargs.get('ff_context', {'user_id': 'anonymous'})
            
            variant_key = feature_flags.evaluate(key, context, default)
            
            # Add variant to kwargs
            kwargs['variant'] = variant_key
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Helper function for easy flag evaluation
def is_enabled(key: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a feature flag is enabled.
    
    Args:
        key: Feature flag key
        context: User context
    
    Returns:
        True if enabled
    """
    context = context or {'user_id': 'anonymous'}
    return feature_flags.evaluate(key, context, False)


# Example usage
if __name__ == "__main__":
    # Create example flags
    
    # Simple boolean flag
    simple_flag = FeatureFlag(
        key="new_ui",
        name="New UI Design",
        description="Enable new UI design",
        flag_type=FlagType.BOOLEAN,
        enabled=True,
        rollout_percentage=50  # 50% rollout
    )
    feature_flags.create_flag(simple_flag)
    
    # A/B test flag
    ab_test_flag = FeatureFlag(
        key="algorithm_test",
        name="Algorithm A/B Test",
        description="Test different summarization algorithms",
        flag_type=FlagType.VARIANT,
        enabled=True,
        variants=[
            {'key': 'control', 'weight': 50},
            {'key': 'variant_a', 'weight': 25},
            {'key': 'variant_b', 'weight': 25}
        ]
    )
    feature_flags.create_flag(ab_test_flag)
    
    # Targeted flag with rules
    targeted_flag = FeatureFlag(
        key="premium_features",
        name="Premium Features",
        description="Enable premium features for specific users",
        flag_type=FlagType.BOOLEAN,
        enabled=True,
        rules=[
            TargetingRule("subscription", "eq", "premium"),
            TargetingRule("country", "in", ["US", "UK", "CA"])
        ],
        rule_operator="all"
    )
    feature_flags.create_flag(targeted_flag)
    
    # Test evaluation
    contexts = [
        {'user_id': 'user1', 'subscription': 'free', 'country': 'US'},
        {'user_id': 'user2', 'subscription': 'premium', 'country': 'US'},
        {'user_id': 'user3', 'subscription': 'premium', 'country': 'FR'}
    ]
    
    for context in contexts:
        print(f"\nUser {context['user_id']}:")
        print(f"  New UI: {is_enabled('new_ui', context)}")
        print(f"  Algorithm: {feature_flags.evaluate('algorithm_test', context)}")
        print(f"  Premium: {is_enabled('premium_features', context)}")
    
    # Show metrics
    print("\nMetrics:", feature_flags.get_metrics())
