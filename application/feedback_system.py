"""
feedback_system.py - User Feedback and Learning System

Simple, practical feedback system that:
- Collects user ratings on summaries
- Tracks which summaries are most useful
- Adjusts parameters based on feedback
- Keeps it simple and maintainable

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    id: str
    content_hash: str
    summary_type: str
    rating: int  # 1-5 stars
    helpful: Optional[bool] = None
    tags: Optional[List[str]] = None
    comment: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class FeedbackSystem:
    """
    Simple feedback collection and learning system.
    Focuses on practical improvements based on user preferences.
    """
    
    def __init__(self, storage_path: str = "./feedback"):
        """Initialize feedback system with persistent storage."""
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing feedback
        self.feedback_file = os.path.join(storage_path, "feedback.json")
        self.preferences_file = os.path.join(storage_path, "preferences.json")
        
        self.feedback_entries = self._load_feedback()
        self.user_preferences = self._load_preferences()
        
        # Simple statistics
        self.stats = {
            'total_feedback': len(self.feedback_entries),
            'average_rating': 0.0,
            'preferred_density': 'medium',
            'helpful_count': 0,
            'not_helpful_count': 0
        }
        self._update_stats()
    
    def _load_feedback(self) -> Dict[str, FeedbackEntry]:
        """Load feedback from disk."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    return {
                        fid: FeedbackEntry(**entry) 
                        for fid, entry in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")
        return {}
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load learned preferences."""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
        
        # Default preferences
        return {
            'preferred_density': 'medium',
            'min_summary_length': 50,
            'max_summary_length': 200,
            'prefer_bullet_points': False,
            'prefer_technical_terms': True,
            'density_scores': {
                'minimal': 3.0,
                'short': 3.5,
                'medium': 4.0,
                'detailed': 3.5
            }
        }
    
    def add_feedback(self,
                    content_hash: str,
                    summary_type: str,
                    rating: int,
                    helpful: Optional[bool] = None,
                    comment: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> str:
        """
        Add user feedback for a summary.
        
        Args:
            content_hash: Hash of the original content
            summary_type: Type of summary (minimal, short, etc.)
            rating: 1-5 star rating
            helpful: Whether the summary was helpful
            comment: Optional user comment
            tags: Optional tags about the feedback
            
        Returns:
            Feedback ID
        """
        # Create feedback entry
        feedback_id = f"{content_hash}_{summary_type}_{int(time.time())}"
        
        entry = FeedbackEntry(
            id=feedback_id,
            content_hash=content_hash,
            summary_type=summary_type,
            rating=max(1, min(5, rating)),  # Clamp to 1-5
            helpful=helpful,
            comment=comment,
            tags=tags or []
        )
        
        # Store feedback
        self.feedback_entries[feedback_id] = entry
        
        # Update preferences based on feedback
        self._update_preferences(entry)
        
        # Save to disk
        self._save_feedback()
        self._save_preferences()
        
        # Update stats
        self._update_stats()
        
        logger.info(f"Added feedback: {feedback_id} (rating: {rating})")
        return feedback_id
    
    def _update_preferences(self, entry: FeedbackEntry):
        """Update user preferences based on feedback."""
        # Update density scores
        if entry.summary_type in self.user_preferences['density_scores']:
            # Simple exponential moving average
            alpha = 0.1  # Learning rate
            old_score = self.user_preferences['density_scores'][entry.summary_type]
            new_score = (1 - alpha) * old_score + alpha * entry.rating
            self.user_preferences['density_scores'][entry.summary_type] = round(new_score, 2)
        
        # Update preferred density
        best_density = max(
            self.user_preferences['density_scores'].items(),
            key=lambda x: x[1]
        )
        self.user_preferences['preferred_density'] = best_density[0]
        
        # Learn from comments (simple keyword analysis)
        if entry.comment:
            comment_lower = entry.comment.lower()
            
            # Check for length preferences
            if 'too long' in comment_lower or 'shorter' in comment_lower:
                self.user_preferences['max_summary_length'] = max(
                    50,
                    int(self.user_preferences['max_summary_length'] * 0.9)
                )
            elif 'too short' in comment_lower or 'more detail' in comment_lower:
                self.user_preferences['max_summary_length'] = min(
                    500,
                    int(self.user_preferences['max_summary_length'] * 1.1)
                )
            
            # Check for format preferences
            if 'bullet' in comment_lower or 'list' in comment_lower:
                self.user_preferences['prefer_bullet_points'] = True
            
            # Check for technical preferences
            if 'technical' in comment_lower or 'jargon' in comment_lower:
                if 'less' in comment_lower or 'simpl' in comment_lower:
                    self.user_preferences['prefer_technical_terms'] = False
                else:
                    self.user_preferences['prefer_technical_terms'] = True
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get current learned preferences."""
        return self.user_preferences.copy()
    
    def get_recommended_settings(self, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recommended summarization settings based on feedback.
        
        Args:
            content_type: Optional content type for specific recommendations
            
        Returns:
            Recommended settings
        """
        recommendations = {
            'density': self.user_preferences['preferred_density'],
            'target_length': self.user_preferences['max_summary_length'],
            'format_as_bullets': self.user_preferences['prefer_bullet_points'],
            'include_technical': self.user_preferences['prefer_technical_terms'],
            'confidence': self._calculate_confidence()
        }
        
        # Adjust for content type if provided
        if content_type:
            if content_type == 'technical':
                recommendations['include_technical'] = True
                recommendations['density'] = 'detailed'
            elif content_type == 'news':
                recommendations['density'] = 'short'
                recommendations['format_as_bullets'] = True
            elif content_type == 'academic':
                recommendations['density'] = 'medium'
                recommendations['include_technical'] = True
        
        return recommendations
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return self.stats.copy()
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in recommendations based on feedback amount."""
        # Simple confidence based on number of feedback entries
        feedback_count = len(self.feedback_entries)
        
        if feedback_count < 10:
            return 0.3
        elif feedback_count < 50:
            return 0.6
        elif feedback_count < 100:
            return 0.8
        else:
            return 0.9
    
    def _update_stats(self):
        """Update statistics."""
        if self.feedback_entries:
            ratings = [entry.rating for entry in self.feedback_entries.values()]
            self.stats['average_rating'] = sum(ratings) / len(ratings)
            
            helpful_count = sum(
                1 for entry in self.feedback_entries.values() 
                if entry.helpful is True
            )
            not_helpful_count = sum(
                1 for entry in self.feedback_entries.values() 
                if entry.helpful is False
            )
            
            self.stats['helpful_count'] = helpful_count
            self.stats['not_helpful_count'] = not_helpful_count
            self.stats['total_feedback'] = len(self.feedback_entries)
            self.stats['preferred_density'] = self.user_preferences['preferred_density']
    
    def _save_feedback(self):
        """Save feedback to disk."""
        try:
            data = {
                fid: asdict(entry) 
                for fid, entry in self.feedback_entries.items()
            }
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def _save_preferences(self):
        """Save preferences to disk."""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def export_insights(self) -> Dict[str, Any]:
        """Export insights from feedback for analysis."""
        insights = {
            'stats': self.get_feedback_stats(),
            'preferences': self.get_preferences(),
            'top_rated_summaries': self._get_top_rated(),
            'common_issues': self._analyze_common_issues(),
            'recommendation_confidence': self._calculate_confidence()
        }
        return insights
    
    def _get_top_rated(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top-rated summary types."""
        # Group by summary type and calculate average
        type_ratings = defaultdict(list)
        
        for entry in self.feedback_entries.values():
            type_ratings[entry.summary_type].append(entry.rating)
        
        # Calculate averages
        type_averages = []
        for summary_type, ratings in type_ratings.items():
            avg_rating = sum(ratings) / len(ratings)
            type_averages.append({
                'type': summary_type,
                'average_rating': round(avg_rating, 2),
                'count': len(ratings)
            })
        
        # Sort by rating
        type_averages.sort(key=lambda x: x['average_rating'], reverse=True)
        
        return type_averages[:limit]
    
    def _analyze_common_issues(self) -> List[str]:
        """Analyze common issues from feedback comments."""
        issues = []
        
        # Count common keywords in comments
        keyword_counts = defaultdict(int)
        
        for entry in self.feedback_entries.values():
            if entry.comment and entry.rating <= 2:  # Low ratings
                comment_lower = entry.comment.lower()
                
                # Check for common issues
                if 'long' in comment_lower:
                    keyword_counts['too_long'] += 1
                if 'short' in comment_lower:
                    keyword_counts['too_short'] += 1
                if 'miss' in comment_lower or 'lost' in comment_lower:
                    keyword_counts['missing_info'] += 1
                if 'technical' in comment_lower or 'complex' in comment_lower:
                    keyword_counts['too_technical'] += 1
        
        # Convert to issues list
        for issue, count in keyword_counts.items():
            if count >= 3:  # Threshold for common issue
                issues.append(f"{issue.replace('_', ' ').title()} ({count} reports)")
        
        return issues


# Global instance
_feedback_system = None


def get_feedback_system() -> FeedbackSystem:
    """Get or create the global feedback system."""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = FeedbackSystem()
    return _feedback_system


# Integration function for easy use
def apply_feedback_preferences(summary_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply learned preferences to summarization parameters.
    
    Args:
        summary_params: Original summarization parameters
        
    Returns:
        Updated parameters with preferences applied
    """
    feedback = get_feedback_system()
    preferences = feedback.get_preferences()
    
    # Apply preferences
    if 'density' not in summary_params:
        summary_params['density'] = preferences['preferred_density']
    
    if 'max_length' not in summary_params:
        summary_params['max_length'] = preferences['max_summary_length']
    
    # Add format hints
    summary_params['format_hints'] = {
        'bullet_points': preferences['prefer_bullet_points'],
        'technical_terms': preferences['prefer_technical_terms']
    }
    
    return summary_params


if __name__ == "__main__":
    # Example usage
    feedback = get_feedback_system()
    
    # Add some feedback
    feedback_id = feedback.add_feedback(
        content_hash="abc123",
        summary_type="medium",
        rating=5,
        helpful=True,
        comment="Perfect length and detail"
    )
    
    print(f"Added feedback: {feedback_id}")
    
    # Get recommendations
    recommendations = feedback.get_recommended_settings()
    print(f"Recommendations: {recommendations}")
    
    # Get insights
    insights = feedback.export_insights()
    print(f"Insights: {json.dumps(insights, indent=2)}")