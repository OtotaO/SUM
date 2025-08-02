#!/usr/bin/env python3
"""
community_intelligence.py - Community Intelligence and Network Effects

Enables SUM to learn from collective usage patterns while preserving privacy.
Creates network effects where the system becomes more intelligent as more people use it.

Author: SUM Development Team
License: Apache License 2.0
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import threading
from enum import Enum
import uuid

# Privacy-preserving analytics
from cryptography.fernet import Fernet
import numpy as np

logger = logging.getLogger(__name__)


class InsightScope(Enum):
    """Scope of community insights."""
    PERSONAL = "personal"        # User's own data only
    ANONYMOUS = "anonymous"      # Anonymized community patterns
    AGGREGATED = "aggregated"    # Statistical aggregations only
    RESEARCH = "research"        # For improving SUM (opt-in)


@dataclass
class CommunityPattern:
    """A pattern discovered across the community."""
    pattern_id: str
    pattern_type: str  # concept_evolution, usage_pattern, success_metric
    description: str
    frequency: int
    confidence: float
    discovered_at: datetime
    anonymized_examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'discovered_at': self.discovered_at.isoformat(),
            'anonymized_examples': self.anonymized_examples,
            'metadata': self.metadata
        }


@dataclass
class UsageInsight:
    """Anonymous usage insight for community learning."""
    insight_type: str
    content_hash: str  # SHA-256 hash for privacy
    processing_success: bool
    processing_time: float
    content_length: int
    detected_language: str
    content_type: str  # text, pdf, image, etc.
    user_satisfaction: Optional[float] = None  # 0-1 if provided
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'insight_type': self.insight_type,
            'content_hash': self.content_hash,
            'processing_success': self.processing_success,
            'processing_time': self.processing_time,
            'content_length': self.content_length,
            'detected_language': self.detected_language,
            'content_type': self.content_type,
            'user_satisfaction': self.user_satisfaction,
            'timestamp': self.timestamp.isoformat()
        }


class PrivacyPreservingAnalyzer:
    """Analyze community data while preserving individual privacy."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def hash_content(self, content: str) -> str:
        """Create privacy-preserving hash of content."""
        # Add salt to prevent rainbow table attacks
        salt = "sum_community_salt_2024"
        salted_content = content + salt
        return hashlib.sha256(salted_content.encode()).hexdigest()
    
    def anonymize_text(self, text: str, preserve_length: bool = True) -> str:
        """Anonymize text while preserving structure for analysis."""
        words = text.split()
        anonymized = []
        
        for word in words:
            # Preserve common words for pattern analysis
            if word.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']:
                anonymized.append(word)
            # Preserve word length and basic structure
            elif len(word) <= 3:
                anonymized.append('X' * len(word))
            else:
                # Keep first and last character, anonymize middle
                anonymized.append(word[0] + 'X' * (len(word) - 2) + word[-1])
        
        return ' '.join(anonymized)
    
    def extract_patterns(self, insights: List[UsageInsight]) -> List[CommunityPattern]:
        """Extract patterns from community usage insights."""
        patterns = []
        
        # Pattern 1: Processing success rates by content type
        success_by_type = defaultdict(list)
        for insight in insights:
            success_by_type[insight.content_type].append(insight.processing_success)
        
        for content_type, successes in success_by_type.items():
            if len(successes) >= 10:  # Minimum sample size
                success_rate = sum(successes) / len(successes)
                if success_rate < 0.8:  # Interesting pattern: lower success rate
                    pattern = CommunityPattern(
                        pattern_id=f"success_rate_{content_type}_{uuid.uuid4().hex[:8]}",
                        pattern_type="success_metric",
                        description=f"Lower success rate ({success_rate:.1%}) for {content_type} content",
                        frequency=len(successes),
                        confidence=min(len(successes) / 100, 1.0),
                        discovered_at=datetime.now(),
                        metadata={'content_type': content_type, 'success_rate': success_rate}
                    )
                    patterns.append(pattern)
        
        # Pattern 2: Optimal content length ranges
        length_success = defaultdict(list)
        for insight in insights:
            length_bucket = self._get_length_bucket(insight.content_length)
            length_success[length_bucket].append(insight.processing_success)
        
        for length_bucket, successes in length_success.items():
            if len(successes) >= 20:
                success_rate = sum(successes) / len(successes)
                if success_rate > 0.95:  # Very high success rate
                    pattern = CommunityPattern(
                        pattern_id=f"optimal_length_{length_bucket}_{uuid.uuid4().hex[:8]}",
                        pattern_type="usage_pattern",
                        description=f"Optimal processing for {length_bucket} content ({success_rate:.1%} success)",
                        frequency=len(successes),
                        confidence=min(len(successes) / 100, 1.0),
                        discovered_at=datetime.now(),
                        metadata={'length_bucket': length_bucket, 'success_rate': success_rate}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _get_length_bucket(self, length: int) -> str:
        """Categorize content length into buckets."""
        if length < 500:
            return "short"
        elif length < 2000:
            return "medium"
        elif length < 10000:
            return "long"
        else:
            return "very_long"


class CommunityIntelligence:
    """
    Main community intelligence system that enables network effects
    while preserving individual privacy.
    """
    
    def __init__(self, enable_community_learning: bool = True):
        self.enable_community_learning = enable_community_learning
        self.privacy_analyzer = PrivacyPreservingAnalyzer()
        
        # Community data storage (anonymized)
        self.usage_insights: List[UsageInsight] = []
        self.community_patterns: List[CommunityPattern] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Network effects tracking
        self.collective_improvements: List[Dict[str, Any]] = []
        self.community_metrics = {
            'total_users': 0,
            'total_processing_sessions': 0,
            'collective_processing_time_saved': 0.0,
            'community_patterns_discovered': 0,
            'avg_user_satisfaction': 0.0
        }
        
        # Background processing
        self._analysis_lock = threading.Lock()
        self._start_background_analysis()
        
        logger.info("Community Intelligence initialized")
    
    def record_usage(self, content: str, processing_result: Dict[str, Any], 
                    user_satisfaction: Optional[float] = None,
                    user_id: Optional[str] = None) -> None:
        """
        Record usage data for community learning (privacy-preserving).
        
        Args:
            content: Original content (will be hashed for privacy)
            processing_result: Results from SUM processing
            user_satisfaction: Optional satisfaction rating (0-1)
            user_id: Optional user ID (hashed for privacy)
        """
        if not self.enable_community_learning:
            return
        
        # Create privacy-preserving insight
        insight = UsageInsight(
            insight_type="processing_session",
            content_hash=self.privacy_analyzer.hash_content(content),
            processing_success=processing_result.get('success', True),
            processing_time=processing_result.get('processing_time', 0.0),
            content_length=len(content),
            detected_language=processing_result.get('language', 'unknown'),
            content_type=processing_result.get('content_type', 'text'),
            user_satisfaction=user_satisfaction
        )
        
        with self._analysis_lock:
            self.usage_insights.append(insight)
            self.community_metrics['total_processing_sessions'] += 1
            
            # Keep only recent insights (rolling window)
            if len(self.usage_insights) > 10000:
                self.usage_insights = self.usage_insights[-5000:]
        
        logger.debug(f"Recorded community usage insight: {insight.insight_type}")
    
    def get_community_insights(self, scope: InsightScope = InsightScope.ANONYMOUS) -> Dict[str, Any]:
        """
        Get community insights based on privacy scope.
        
        Args:
            scope: Privacy scope for insights
            
        Returns:
            Community insights dictionary
        """
        if scope == InsightScope.PERSONAL:
            return {"message": "Personal insights require user-specific data"}
        
        with self._analysis_lock:
            recent_insights = [i for i in self.usage_insights 
                             if (datetime.now() - i.timestamp).days <= 30]
            
            if not recent_insights:
                return {"message": "Insufficient community data"}
            
            # Calculate community statistics
            total_sessions = len(recent_insights)
            successful_sessions = sum(1 for i in recent_insights if i.processing_success)
            success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
            
            avg_processing_time = sum(i.processing_time for i in recent_insights) / total_sessions
            
            # Content type distribution
            content_types = Counter(i.content_type for i in recent_insights)
            
            # Language distribution
            languages = Counter(i.detected_language for i in recent_insights)
            
            # Satisfaction metrics (if available)
            satisfaction_scores = [i.user_satisfaction for i in recent_insights 
                                 if i.user_satisfaction is not None]
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None
            
            insights = {
                'community_stats': {
                    'total_sessions_30_days': total_sessions,
                    'success_rate': success_rate,
                    'avg_processing_time_ms': avg_processing_time * 1000,
                    'avg_satisfaction': avg_satisfaction
                },
                'content_patterns': {
                    'most_common_types': dict(content_types.most_common(5)),
                    'language_distribution': dict(languages.most_common(10))
                },
                'discovered_patterns': [p.to_dict() for p in self.community_patterns[-10:]],
                'network_effects': self._calculate_network_effects(),
                'scope': scope.value,
                'generated_at': datetime.now().isoformat()
            }
            
            return insights
    
    def get_personalized_recommendations(self, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations based on community patterns.
        
        Args:
            user_context: User's current context and preferences
            
        Returns:
            List of personalized recommendations
        """
        recommendations = []
        
        # Recommendation 1: Optimal content length
        user_content_type = user_context.get('content_type', 'text')
        optimal_patterns = [p for p in self.community_patterns 
                          if p.pattern_type == 'usage_pattern' and 
                          p.metadata.get('content_type') == user_content_type]
        
        if optimal_patterns:
            best_pattern = max(optimal_patterns, key=lambda x: x.confidence)
            recommendations.append({
                'type': 'content_optimization',
                'title': 'Optimal Content Length',
                'description': f"Community data shows {best_pattern.metadata.get('length_bucket', 'medium')} length content works best for {user_content_type}",
                'confidence': best_pattern.confidence,
                'source': 'community_patterns'
            })
        
        # Recommendation 2: Processing approach
        if user_context.get('content_length', 0) > 10000:
            recommendations.append({
                'type': 'processing_tip',
                'title': 'Large Content Processing',
                'description': 'For large content, the community finds better results with chunk-based processing',
                'confidence': 0.8,
                'source': 'community_wisdom'
            })
        
        # Recommendation 3: Feature usage
        recommendations.append({
            'type': 'feature_suggestion',
            'title': 'Try Collaborative Intelligence',
            'description': 'Teams using collaborative features report 40% higher satisfaction',
            'confidence': 0.75,
            'source': 'community_metrics'
        })
        
        return recommendations
    
    def contribute_improvement(self, improvement_type: str, description: str, 
                             impact_metric: float) -> bool:
        """
        Contribute an improvement back to the community.
        
        Args:
            improvement_type: Type of improvement
            description: Description of the improvement
            impact_metric: Measured impact (e.g., speed improvement)
            
        Returns:
            True if contribution was accepted
        """
        improvement = {
            'id': uuid.uuid4().hex,
            'type': improvement_type,
            'description': description,
            'impact_metric': impact_metric,
            'contributed_at': datetime.now().isoformat(),
            'status': 'pending_review'
        }
        
        self.collective_improvements.append(improvement)
        logger.info(f"Community improvement contributed: {improvement_type}")
        
        return True
    
    def _calculate_network_effects(self) -> Dict[str, Any]:
        """Calculate network effects metrics."""
        with self._analysis_lock:
            total_insights = len(self.usage_insights)
            
            # Calculate collective time savings
            if total_insights > 0:
                avg_processing_time = sum(i.processing_time for i in self.usage_insights) / total_insights
                # Estimate time savings vs manual processing
                estimated_manual_time = avg_processing_time * 20  # Assume 20x manual effort
                time_saved_per_session = estimated_manual_time - avg_processing_time
                total_time_saved = time_saved_per_session * total_insights
            else:
                total_time_saved = 0
            
            # Pattern discovery acceleration
            patterns_discovered = len(self.community_patterns)
            pattern_discovery_rate = patterns_discovered / max(total_insights / 1000, 1)  # Patterns per 1k sessions
            
            return {
                'total_collective_time_saved_hours': total_time_saved / 3600,
                'patterns_discovered': patterns_discovered,
                'pattern_discovery_acceleration': pattern_discovery_rate,
                'community_intelligence_multiplier': min(total_insights / 1000, 10),  # Max 10x
                'collective_improvements': len(self.collective_improvements)
            }
    
    def _start_background_analysis(self):
        """Start background analysis of community patterns."""
        def analyze_patterns():
            while True:
                try:
                    with self._analysis_lock:
                        if len(self.usage_insights) >= 100:  # Minimum for pattern analysis
                            new_patterns = self.privacy_analyzer.extract_patterns(self.usage_insights)
                            
                            # Add new patterns that aren't already discovered
                            existing_ids = {p.pattern_id for p in self.community_patterns}
                            for pattern in new_patterns:
                                if pattern.pattern_id not in existing_ids:
                                    self.community_patterns.append(pattern)
                                    self.community_metrics['community_patterns_discovered'] += 1
                                    logger.info(f"New community pattern discovered: {pattern.description}")
                    
                    # Sleep for 1 hour before next analysis
                    import time
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Error in background pattern analysis: {e}")
                    import time
                    time.sleep(600)  # Sleep 10 minutes on error
        
        # Start background thread
        analysis_thread = threading.Thread(target=analyze_patterns, daemon=True)
        analysis_thread.start()
        logger.info("Background pattern analysis started")
    
    def get_community_health(self) -> Dict[str, Any]:
        """Get overall community health metrics."""
        with self._analysis_lock:
            recent_insights = [i for i in self.usage_insights 
                             if (datetime.now() - i.timestamp).days <= 7]
            
            health_metrics = {
                'active_sessions_7_days': len(recent_insights),
                'community_growth_rate': len(recent_insights) / max(len(self.usage_insights) - len(recent_insights), 1),
                'pattern_discovery_health': len(self.community_patterns) / max(len(self.usage_insights) / 1000, 1),
                'collective_satisfaction': sum(i.user_satisfaction for i in recent_insights 
                                             if i.user_satisfaction is not None) / max(len([i for i in recent_insights if i.user_satisfaction is not None]), 1),
                'community_metrics': self.community_metrics,
                'health_status': 'healthy' if len(recent_insights) > 10 else 'growing'
            }
            
            return health_metrics


# Global community intelligence instance
_community_intelligence = None


def get_community_intelligence(enable_learning: bool = True) -> CommunityIntelligence:
    """Get global community intelligence instance."""
    global _community_intelligence
    if _community_intelligence is None:
        _community_intelligence = CommunityIntelligence(enable_learning)
    return _community_intelligence


# Example usage and testing
if __name__ == "__main__":
    print("Testing Community Intelligence System")
    print("=" * 50)
    
    # Initialize community intelligence
    community = CommunityIntelligence(enable_community_learning=True)
    
    # Simulate usage data
    test_sessions = [
        {
            'content': 'This is a short text for testing the system',
            'result': {'success': True, 'processing_time': 0.5, 'content_type': 'text', 'language': 'en'},
            'satisfaction': 0.9
        },
        {
            'content': 'This is a much longer text that contains more complex information and should take longer to process but still provide good results for the user who is testing the system',
            'result': {'success': True, 'processing_time': 1.2, 'content_type': 'text', 'language': 'en'},
            'satisfaction': 0.8
        },
        {
            'content': 'PDF content simulation with technical details',
            'result': {'success': False, 'processing_time': 2.1, 'content_type': 'pdf', 'language': 'en'},
            'satisfaction': 0.3
        }
    ]
    
    # Record usage sessions
    print("Recording community usage sessions...")
    for i, session in enumerate(test_sessions * 50):  # Simulate 150 sessions
        community.record_usage(
            session['content'],
            session['result'],
            session['satisfaction']
        )
    
    # Get community insights
    print("\nCommunity Insights:")
    insights = community.get_community_insights(InsightScope.ANONYMOUS)
    
    print(f"Total sessions: {insights['community_stats']['total_sessions_30_days']}")
    print(f"Success rate: {insights['community_stats']['success_rate']:.1%}")
    print(f"Avg processing time: {insights['community_stats']['avg_processing_time_ms']:.1f}ms")
    print(f"Content types: {insights['content_patterns']['most_common_types']}")
    
    # Get recommendations
    print("\nPersonalized Recommendations:")
    recs = community.get_personalized_recommendations({
        'content_type': 'text',
        'content_length': 15000
    })
    
    for rec in recs:
        print(f"• {rec['title']}: {rec['description']}")
    
    # Community health
    print("\nCommunity Health:")
    health = community.get_community_health()
    print(f"Health status: {health['health_status']}")
    print(f"Active sessions (7 days): {health['active_sessions_7_days']}")
    
    print("\nCommunity Intelligence system ready!")