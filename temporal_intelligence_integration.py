#!/usr/bin/env python3
"""
temporal_intelligence_integration.py - Complete Temporal Intelligence Integration

This module provides the main integration layer for the revolutionary temporal 
intelligence system, making it easy to add time-aware understanding to any 
SUM application.

Features:
- Simple integration with existing SUM systems
- Automatic temporal pattern detection
- Beautiful temporal insights and visualizations
- Time-aware recommendations
- Persistent temporal state management

Usage:
    from temporal_intelligence_integration import TemporalSUM
    
    # Create enhanced SUM with temporal intelligence
    temporal_sum = TemporalSUM()
    
    # Capture thoughts normally - temporal intelligence works automatically
    temporal_sum.capture("Machine learning is fascinating...")
    
    # Get time-aware insights
    insights = temporal_sum.get_temporal_insights()
    
    # Create beautiful visualizations
    temporal_sum.create_visualizations()

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import core components
from knowledge_os import KnowledgeOperatingSystem, Thought
from temporal_intelligence_engine import TemporalIntelligenceEngine, TemporalSUMIntegration
from temporal_visualization_dashboard import TemporalVisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TemporalIntegration')


class TemporalSUM:
    """
    The complete temporal intelligence-enhanced SUM system.
    
    This class provides a simple, unified interface to the revolutionary temporal
    intelligence capabilities, making it easy to add time-aware understanding
    to any application.
    """
    
    def __init__(self, data_dir: str = "temporal_sum_data", enable_visualizations: bool = True):
        """
        Initialize the temporal intelligence-enhanced SUM system.
        
        Args:
            data_dir: Directory for storing temporal intelligence data
            enable_visualizations: Whether to enable visualization capabilities
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize core systems
        self.knowledge_os = KnowledgeOperatingSystem(str(self.data_dir / "knowledge"))
        self.temporal_integration = TemporalSUMIntegration(self.knowledge_os)
        
        # Initialize visualization engine if enabled
        self.viz_engine = None
        if enable_visualizations:
            try:
                self.viz_engine = TemporalVisualizationEngine(
                    self.temporal_integration.temporal_engine,
                    str(self.data_dir / "visualizations")
                )
            except Exception as e:
                logger.warning(f"Visualization engine initialization failed: {e}")
        
        logger.info(f"TemporalSUM initialized with data directory: {data_dir}")
    
    # Core capture and processing methods
    def capture(self, content: str, source: str = "direct") -> Optional[Thought]:
        """
        Capture a thought with automatic temporal intelligence processing.
        
        Args:
            content: The thought content to capture
            source: Source of the thought (direct, voice, import, etc.)
            
        Returns:
            The captured thought with temporal enrichment
        """
        return self.knowledge_os.capture_thought(content, source)
    
    def capture_multiple(self, thoughts: List[str], source: str = "direct") -> List[Thought]:
        """
        Capture multiple thoughts at once.
        
        Args:
            thoughts: List of thought contents
            source: Source of the thoughts
            
        Returns:
            List of captured thoughts
        """
        captured = []
        for thought_content in thoughts:
            thought = self.capture(thought_content, source)
            if thought:
                captured.append(thought)
        return captured
    
    def capture_with_timestamp(self, content: str, timestamp: datetime, source: str = "direct") -> Optional[Thought]:
        """
        Capture a thought with a specific timestamp for historical data import.
        
        Args:
            content: The thought content
            timestamp: When the thought occurred
            source: Source of the thought
            
        Returns:
            The captured thought
        """
        thought = self.capture(content, source)
        if thought:
            thought.timestamp = timestamp
            # Re-process with correct timestamp
            self.temporal_integration.temporal_engine.process_new_thought(thought)
        return thought
    
    # Insight and analysis methods
    def get_temporal_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive temporal intelligence insights.
        
        Returns:
            Dictionary containing all temporal insights
        """
        return self.temporal_integration.get_enhanced_insights()
    
    def get_concept_evolution(self, concept: str = None) -> Dict[str, Any]:
        """
        Get evolution data for a specific concept or all concepts.
        
        Args:
            concept: Specific concept to analyze (optional)
            
        Returns:
            Concept evolution data
        """
        evolutions = self.temporal_integration.temporal_engine.concept_evolutions
        
        if concept:
            concept_key = concept.lower().replace(' ', '-')
            if concept_key in evolutions:
                evolution = evolutions[concept_key]
                return {
                    'concept': concept,
                    'first_appearance': evolution.first_appearance.isoformat(),
                    'last_activity': evolution.last_activity.isoformat(),
                    'depth_progression': evolution.depth_progression,
                    'breakthrough_moments': [dt.isoformat() for dt in evolution.breakthrough_moments],
                    'projected_next_depth': evolution.projected_next_depth,
                    'recommendations': evolution.recommended_exploration_areas
                }
            else:
                return {'error': f'No evolution data found for concept: {concept}'}
        else:
            # Return summary of all concepts
            return {
                'total_concepts': len(evolutions),
                'concepts': list(evolutions.keys()),
                'most_evolved': sorted(
                    evolutions.items(),
                    key=lambda x: len(x[1].depth_progression),
                    reverse=True
                )[:10]
            }
    
    def get_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Get detected seasonal thinking patterns.
        
        Returns:
            Seasonal pattern data
        """
        patterns = self.temporal_integration.temporal_engine.seasonal_patterns
        
        if not patterns:
            return {'message': 'No seasonal patterns detected yet. Continue capturing thoughts over time.'}
        
        return {
            'total_patterns': len(patterns),
            'patterns': [
                {
                    'id': pattern.pattern_id,
                    'type': pattern.pattern_type,
                    'description': pattern.human_description,
                    'strength': pattern.pattern_strength,
                    'peak_periods': pattern.peak_periods,
                    'concepts': pattern.associated_concepts
                }
                for pattern in patterns.values()
            ]
        }
    
    def get_intellectual_momentum(self) -> Dict[str, Any]:
        """
        Get intellectual momentum analysis.
        
        Returns:
            Momentum tracking data
        """
        momentum_trackers = self.temporal_integration.temporal_engine.momentum_trackers
        
        if not momentum_trackers:
            return {'message': 'No momentum detected yet. Explore topics consistently to build momentum.'}
        
        active_momentum = [
            {
                'area': momentum.research_area.replace('-', ' ').title(),
                'velocity': momentum.velocity,
                'mass': momentum.mass,
                'critical_mass': momentum.current_critical_mass,
                'breakthrough_probability': momentum.current_critical_mass,
                'estimated_breakthrough': momentum.estimated_breakthrough_date.isoformat() if momentum.estimated_breakthrough_date else None,
                'flow_sessions': len(momentum.flow_sessions),
                'suggestions': momentum.momentum_optimization_suggestions
            }
            for momentum in momentum_trackers.values()
        ]
        
        # Sort by critical mass
        active_momentum.sort(key=lambda x: x['critical_mass'], reverse=True)
        
        return {
            'active_areas': len(active_momentum),
            'momentum_areas': active_momentum
        }
    
    def get_cognitive_rhythms(self) -> Dict[str, Any]:
        """
        Get cognitive rhythm analysis.
        
        Returns:
            Cognitive rhythm data
        """
        return self.temporal_integration.temporal_engine.rhythm_analyzer.get_optimal_thinking_times()
    
    def get_future_projections(self) -> Dict[str, Any]:
        """
        Get future interest projections.
        
        Returns:
            Future projection data
        """
        projections = self.temporal_integration.temporal_engine.future_projections
        
        if not projections:
            return {'message': 'Future projections will appear as thinking patterns develop over time.'}
        
        latest = max(projections, key=lambda p: p.generated_date)
        
        return {
            'projection_date': latest.generated_date.isoformat(),
            'horizon_days': latest.projection_horizon.days,
            'confidence': latest.prediction_confidence,
            'emerging_interests': [
                {'interest': interest.replace('-', ' ').title(), 'probability': prob}
                for interest, prob in latest.emerging_interests
            ],
            'declining_interests': [
                {'interest': interest.replace('-', ' ').title(), 'probability': prob}
                for interest, prob in latest.declining_interests
            ],
            'recommendations': latest.recommended_next_steps
        }
    
    def get_knowledge_aging_status(self) -> Dict[str, Any]:
        """
        Get knowledge aging and resurrection analysis.
        
        Returns:
            Knowledge aging status
        """
        aging_data = self.temporal_integration.temporal_engine.aging_knowledge
        
        if not aging_data:
            return {'message': 'Knowledge aging analysis will develop as you capture more thoughts over time.'}
        
        current_time = datetime.now()
        
        # Analyze aging status
        ready_for_review = []
        recently_resurrected = []
        
        for aging_id, aging in aging_data.items():
            if aging.next_optimal_review and aging.next_optimal_review <= current_time:
                ready_for_review.append({
                    'knowledge_id': aging_id,
                    'age_days': (current_time - aging.original_capture_date).days,
                    'current_relevance': aging.current_relevance_score,
                    'review_due': aging.next_optimal_review.isoformat()
                })
            
            if aging.resurrection_events and aging.resurrection_events[-1] > current_time - timedelta(days=7):
                recently_resurrected.append({
                    'knowledge_id': aging_id,
                    'resurrection_date': aging.resurrection_events[-1].isoformat(),
                    'context': aging.resurrection_contexts[-1] if aging.resurrection_contexts else '',
                    'triggers': aging.resurrection_triggers[-3:]
                })
        
        return {
            'total_knowledge_items': len(aging_data),
            'ready_for_review': len(ready_for_review),
            'recently_resurrected': len(recently_resurrected),
            'review_items': ready_for_review[:10],  # Top 10
            'recent_resurrections': recently_resurrected[:5]  # Recent 5
        }
    
    # Search and query methods
    def search_thoughts(self, query: str) -> List[Dict[str, Any]]:
        """
        Search thoughts with temporal context.
        
        Args:
            query: Search query
            
        Returns:
            List of matching thoughts with temporal information
        """
        thoughts = self.knowledge_os.search_thoughts(query)
        
        return [
            {
                'id': thought.id,
                'content': thought.content,
                'timestamp': thought.timestamp.isoformat(),
                'concepts': thought.concepts,
                'importance': thought.importance,
                'connections': len(thought.connections),
                'age_days': (datetime.now() - thought.timestamp).days
            }
            for thought in thoughts
        ]
    
    def get_thoughts_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Get all thoughts related to a specific concept.
        
        Args:
            concept: The concept to search for
            
        Returns:
            List of related thoughts
        """
        concept_key = concept.lower().replace(' ', '-')
        related_thoughts = []
        
        for thought in self.knowledge_os.active_thoughts.values():
            if concept_key in thought.concepts:
                related_thoughts.append({
                    'id': thought.id,
                    'content': thought.content,
                    'timestamp': thought.timestamp.isoformat(),
                    'importance': thought.importance,
                    'concepts': thought.concepts
                })
        
        # Sort by timestamp (newest first)
        related_thoughts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return related_thoughts
    
    def get_thoughts_by_timeframe(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get thoughts within a specific timeframe.
        
        Args:
            start_date: Start of timeframe
            end_date: End of timeframe
            
        Returns:
            List of thoughts in timeframe
        """
        timeframe_thoughts = []
        
        for thought in self.knowledge_os.active_thoughts.values():
            if start_date <= thought.timestamp <= end_date:
                timeframe_thoughts.append({
                    'id': thought.id,
                    'content': thought.content,
                    'timestamp': thought.timestamp.isoformat(),
                    'concepts': thought.concepts,
                    'importance': thought.importance
                })
        
        timeframe_thoughts.sort(key=lambda x: x['timestamp'])
        
        return timeframe_thoughts
    
    # Visualization methods
    def create_visualizations(self, output_dir: str = None) -> Optional[Dict[str, str]]:
        """
        Create all temporal intelligence visualizations.
        
        Args:
            output_dir: Custom output directory (optional)
            
        Returns:
            Dictionary of visualization file paths
        """
        if not self.viz_engine:
            logger.warning("Visualization engine not available")
            return None
        
        if output_dir:
            self.viz_engine.output_dir = Path(output_dir)
            self.viz_engine.output_dir.mkdir(exist_ok=True)
        
        try:
            dashboard_data = self.viz_engine.create_comprehensive_dashboard()
            logger.info(f"Visualizations created in: {self.viz_engine.output_dir}")
            return dashboard_data['individual_visualizations']
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def create_concept_evolution_chart(self, concept: str = None) -> Optional[str]:
        """
        Create concept evolution visualization.
        
        Args:
            concept: Specific concept to visualize (optional)
            
        Returns:
            Path to visualization file
        """
        if not self.viz_engine:
            return None
        
        try:
            return self.viz_engine.create_concept_evolution_timeline(concept)
        except Exception as e:
            logger.error(f"Concept evolution chart creation failed: {e}")
            return None
    
    def create_seasonal_patterns_chart(self) -> Optional[str]:
        """
        Create seasonal patterns visualization.
        
        Returns:
            Path to visualization file
        """
        if not self.viz_engine:
            return None
        
        try:
            return self.viz_engine.create_seasonal_pattern_heatmap()
        except Exception as e:
            logger.error(f"Seasonal patterns chart creation failed: {e}")
            return None
    
    def create_momentum_chart(self) -> Optional[str]:
        """
        Create intellectual momentum visualization.
        
        Returns:
            Path to visualization file
        """
        if not self.viz_engine:
            return None
        
        try:
            return self.viz_engine.create_momentum_trajectory_graph()
        except Exception as e:
            logger.error(f"Momentum chart creation failed: {e}")
            return None
    
    # Utility and management methods
    def save_state(self) -> bool:
        """
        Save the current temporal intelligence state.
        
        Returns:
            True if successful
        """
        try:
            self.temporal_integration.temporal_engine.save_temporal_state()
            logger.info("Temporal intelligence state saved")
            return True
        except Exception as e:
            logger.error(f"State save failed: {e}")
            return False
    
    def export_insights(self, output_file: str = None) -> str:
        """
        Export comprehensive insights to JSON file.
        
        Args:
            output_file: Custom output file path (optional)
            
        Returns:
            Path to exported file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"temporal_insights_{timestamp}.json"
        
        insights = self.get_temporal_insights()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(insights, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Insights exported to: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status information
        """
        # Count active components
        thoughts_count = len(self.knowledge_os.active_thoughts)
        concepts_count = len(self.temporal_integration.temporal_engine.concept_evolutions)
        patterns_count = len(self.temporal_integration.temporal_engine.seasonal_patterns)
        momentum_count = len(self.temporal_integration.temporal_engine.momentum_trackers)
        aging_count = len(self.temporal_integration.temporal_engine.aging_knowledge)
        projections_count = len(self.temporal_integration.temporal_engine.future_projections)
        
        return {
            'system_status': 'active',
            'data_directory': str(self.data_dir),
            'visualizations_enabled': self.viz_engine is not None,
            'components': {
                'thoughts_captured': thoughts_count,
                'concepts_tracked': concepts_count,
                'seasonal_patterns': patterns_count,
                'momentum_areas': momentum_count,
                'aging_knowledge_items': aging_count,
                'future_projections': projections_count
            },
            'temporal_intelligence_status': 'operational' if concepts_count > 0 else 'collecting_data',
            'recommendations': self._get_status_recommendations(thoughts_count, concepts_count)
        }
    
    def _get_status_recommendations(self, thoughts_count: int, concepts_count: int) -> List[str]:
        """Generate status-based recommendations."""
        recommendations = []
        
        if thoughts_count == 0:
            recommendations.append("Start capturing thoughts to begin temporal intelligence analysis")
        elif thoughts_count < 10:
            recommendations.append("Capture more thoughts to improve pattern detection accuracy")
        
        if concepts_count == 0:
            recommendations.append("Continue thinking to develop concept evolution tracking")
        elif concepts_count < 5:
            recommendations.append("Explore diverse topics to enhance temporal insights")
        
        if thoughts_count > 50:
            recommendations.append("Consider creating visualizations to see temporal patterns")
        
        return recommendations
    
    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic state saving."""
        self.save_state()


# Convenience functions for quick setup
def create_temporal_sum(data_dir: str = "temporal_sum_data", 
                       enable_visualizations: bool = True) -> TemporalSUM:
    """
    Create a new TemporalSUM instance with default settings.
    
    Args:
        data_dir: Directory for data storage
        enable_visualizations: Whether to enable visualizations
        
    Returns:
        Configured TemporalSUM instance
    """
    return TemporalSUM(data_dir, enable_visualizations)


def quick_demo() -> str:
    """
    Run a quick demonstration of temporal intelligence.
    
    Returns:
        Path to demo results
    """
    demo_dir = "temporal_sum_quick_demo"
    
    with create_temporal_sum(demo_dir) as temporal_sum:
        # Capture some example thoughts
        sample_thoughts = [
            "Artificial intelligence is revolutionizing how we work and think.",
            "Machine learning algorithms can find patterns humans might miss.",
            "Deep learning neural networks mimic how the brain processes information.",
            "The ethics of AI development are becoming increasingly important.",
            "Natural language processing enables computers to understand human speech.",
            "Computer vision allows machines to interpret visual information.",
            "Reinforcement learning teaches AI through trial and error.",
            "The future of AI includes both opportunities and challenges."
        ]
        
        print("🧠 Capturing sample thoughts...")
        for thought in sample_thoughts:
            temporal_sum.capture(thought)
        
        # Get insights
        print("🔮 Generating temporal insights...")
        insights = temporal_sum.get_temporal_insights()
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        viz_paths = temporal_sum.create_visualizations()
        
        # Export insights
        export_path = temporal_sum.export_insights()
        
        print(f"✨ Quick demo completed!")
        print(f"📊 Insights exported to: {export_path}")
        if viz_paths:
            print(f"🎨 Visualizations created in: {temporal_sum.data_dir / 'visualizations'}")
        
        return demo_dir


if __name__ == "__main__":
    # Run quick demo
    demo_dir = quick_demo()
    print(f"\n🎉 Demo completed! Check results in: {demo_dir}")