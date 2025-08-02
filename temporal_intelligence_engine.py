#!/usr/bin/env python3
"""
temporal_intelligence_engine.py - Temporal Intelligence System for SUM

This system transforms SUM from understanding WHAT you think to understanding
HOW your thinking evolves over time. It provides advanced insights into:

🧠 Concept Evolution - How your understanding deepens and changes
⏰ Seasonal Patterns - "You tend to think about X in December"  
🚀 Intellectual Momentum - Building toward breakthroughs
⏳ Knowledge Aging - Old insights resurface when relevant
🔮 Future Projection - Predictive interest modeling

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
import sqlite3
import hashlib
import pickle
import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from collections import defaultdict, Counter, deque
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Import existing SUM components
from knowledge_os import KnowledgeOperatingSystem, Thought, KnowledgeCluster
from predictive_intelligence import UserProfile, ResearchThread
from temporal_knowledge_analysis import TemporalAnalysis

# Configure beautiful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('TemporalIntelligence')


@dataclass
class ConceptEvolution:
    """Tracks how a concept evolves in the user's understanding over time."""
    concept: str
    first_appearance: datetime
    last_activity: datetime
    
    # Evolution metrics
    depth_progression: List[float] = field(default_factory=list)  # How deep understanding goes
    complexity_trajectory: List[float] = field(default_factory=list)  # Simple -> Complex
    confidence_evolution: List[float] = field(default_factory=list)  # Certainty over time
    context_expansion: List[Set[str]] = field(default_factory=list)  # Related concepts over time
    
    # Milestones
    breakthrough_moments: List[datetime] = field(default_factory=list)
    learning_plateaus: List[Tuple[datetime, datetime]] = field(default_factory=list)
    insight_density_peaks: List[datetime] = field(default_factory=list)
    
    # Predictive
    projected_next_depth: float = 0.0
    estimated_breakthrough_probability: float = 0.0
    recommended_exploration_areas: List[str] = field(default_factory=list)


@dataclass
class SeasonalPattern:
    """Captures cyclical patterns in thinking and interests."""
    pattern_id: str
    pattern_type: str  # "monthly", "weekly", "daily", "annual"
    
    # Pattern characteristics
    peak_periods: List[str] = field(default_factory=list)  # When this pattern peaks
    trough_periods: List[str] = field(default_factory=list)  # When it's minimal
    associated_concepts: List[str] = field(default_factory=list)
    trigger_contexts: List[str] = field(default_factory=list)  # What triggers this pattern
    
    # Strength and reliability
    pattern_strength: float = 0.0  # How strong the cyclical pattern is
    prediction_confidence: float = 0.0  # How reliably we can predict it
    historical_accuracy: float = 0.0  # How often predictions were right
    
    # Beautiful descriptions
    human_description: str = ""  # "You tend to think about philosophy in quiet December evenings"
    next_predicted_peak: Optional[datetime] = None


@dataclass
class IntellectualMomentum:
    """Tracks momentum building toward intellectual breakthroughs."""
    momentum_id: str
    research_area: str
    started: datetime
    
    # Momentum characteristics
    velocity: float = 0.0  # Rate of progress
    acceleration: float = 0.0  # Change in velocity
    mass: float = 0.0  # Depth of engagement
    direction_vector: List[float] = field(default_factory=list)  # Where it's heading
    
    # Critical mass detection
    current_critical_mass: float = 0.0
    breakthrough_threshold: float = 0.8
    estimated_breakthrough_date: Optional[datetime] = None
    
    # Flow state indicators
    flow_sessions: List[datetime] = field(default_factory=list)
    sustained_focus_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    insight_generation_rate: float = 0.0
    
    # Recommendations
    momentum_optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class KnowledgeAging:
    """Tracks how knowledge ages and when it becomes relevant again."""
    knowledge_id: str
    original_capture_date: datetime
    
    # Aging characteristics
    relevance_decay_rate: float = 0.0  # How quickly it becomes less relevant
    current_relevance_score: float = 1.0
    forgetting_curve_params: Tuple[float, float] = (1.0, 0.5)  # (initial_strength, decay_rate)
    
    # Resurrection patterns
    resurrection_events: List[datetime] = field(default_factory=list)
    resurrection_contexts: List[str] = field(default_factory=list)
    resurrection_triggers: List[str] = field(default_factory=list)
    
    # Spaced repetition optimization
    next_optimal_review: Optional[datetime] = None
    review_intervals: List[timedelta] = field(default_factory=list)
    successful_recalls: int = 0
    failed_recalls: int = 0


@dataclass
class FutureProjection:
    """Predicts future interests and learning paths."""
    projection_id: str
    generated_date: datetime
    projection_horizon: timedelta
    
    # Predicted interests
    emerging_interests: List[Tuple[str, float]] = field(default_factory=list)  # (interest, probability)
    declining_interests: List[Tuple[str, float]] = field(default_factory=list)
    stable_interests: List[str] = field(default_factory=list)
    
    # Learning path predictions
    recommended_next_steps: List[str] = field(default_factory=list)
    predicted_knowledge_gaps: List[str] = field(default_factory=list)
    optimal_learning_sequence: List[str] = field(default_factory=list)
    
    # Confidence and validation
    prediction_confidence: float = 0.0
    historical_accuracy: float = 0.0  # How accurate past predictions were
    uncertainty_factors: List[str] = field(default_factory=list)


class TemporalKnowledgeGraph:
    """A knowledge graph with time-weighted connections and temporal reasoning."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.temporal_weights = {}  # Edge weights that decay over time
        self.concept_timelines = defaultdict(list)  # concept -> [(timestamp, strength)]
        self.connection_history = defaultdict(list)  # (concept1, concept2) -> [connection_events]
        
    def add_temporal_node(self, concept: str, timestamp: datetime, strength: float = 1.0):
        """Add a concept node with temporal information."""
        self.graph.add_node(concept, 
                          first_seen=getattr(self.graph.nodes.get(concept, {}), 'first_seen', timestamp),
                          last_seen=timestamp,
                          total_strength=getattr(self.graph.nodes.get(concept, {}), 'total_strength', 0) + strength)
        
        self.concept_timelines[concept].append((timestamp, strength))
    
    def add_temporal_edge(self, source: str, target: str, timestamp: datetime, 
                         connection_type: str = "related", strength: float = 1.0):
        """Add a time-weighted connection between concepts."""
        edge_id = f"{source}_{target}_{timestamp.isoformat()}"
        
        self.graph.add_edge(source, target, 
                          key=edge_id,
                          timestamp=timestamp,
                          connection_type=connection_type,
                          strength=strength,
                          decay_rate=0.1)  # How quickly this connection fades
        
        self.connection_history[(source, target)].append({
            'timestamp': timestamp,
            'type': connection_type,
            'strength': strength
        })
    
    def get_temporal_centrality(self, timestamp: datetime, time_window: timedelta) -> Dict[str, float]:
        """Get concept centrality within a specific time window."""
        # Create subgraph for the time window
        start_time = timestamp - time_window
        end_time = timestamp + time_window
        
        relevant_edges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_time = data['timestamp']
            if start_time <= edge_time <= end_time:
                # Apply temporal decay
                time_diff = abs((timestamp - edge_time).total_seconds())
                decay_factor = math.exp(-time_diff / time_window.total_seconds())
                relevant_edges.append((u, v, data['strength'] * decay_factor))
        
        # Build temporary graph and calculate centrality
        temp_graph = nx.Graph()
        for u, v, weight in relevant_edges:
            if temp_graph.has_edge(u, v):
                temp_graph[u][v]['weight'] += weight
            else:
                temp_graph.add_edge(u, v, weight=weight)
        
        if temp_graph.number_of_nodes() > 0:
            return nx.eigenvector_centrality(temp_graph, weight='weight')
        else:
            return {}
    
    def predict_future_connections(self, concept: str, horizon: timedelta) -> List[Tuple[str, float]]:
        """Predict what concepts might connect to this one in the future."""
        # Analyze historical connection patterns
        concept_connections = []
        
        for neighbor in self.graph.neighbors(concept):
            connection_events = self.connection_history.get((concept, neighbor), [])
            if len(connection_events) >= 2:
                # Calculate connection velocity (how quickly connections strengthen)
                times = [event['timestamp'] for event in connection_events]
                strengths = [event['strength'] for event in connection_events]
                
                # Simple linear regression to predict future strength
                if len(times) > 1:
                    time_deltas = [(t - times[0]).total_seconds() for t in times]
                    if max(time_deltas) > 0:
                        slope = np.polyfit(time_deltas, strengths, 1)[0]
                        future_seconds = horizon.total_seconds()
                        predicted_strength = strengths[-1] + slope * future_seconds
                        
                        if predicted_strength > 0:
                            concept_connections.append((neighbor, predicted_strength))
        
        # Sort by predicted strength
        concept_connections.sort(key=lambda x: x[1], reverse=True)
        return concept_connections[:10]  # Top 10 predictions


class CognitiveRhythmAnalyzer:
    """Analyzes cognitive rhythms to optimize learning and insight timing."""
    
    def __init__(self):
        self.hourly_patterns = defaultdict(list)  # hour -> [performance_scores]
        self.daily_patterns = defaultdict(list)   # day_of_week -> [performance_scores]
        self.monthly_patterns = defaultdict(list) # month -> [performance_scores]
        self.flow_state_indicators = []
        
    def record_cognitive_session(self, timestamp: datetime, 
                               thoughts_captured: int,
                               insights_generated: int,
                               processing_quality: float,
                               focus_duration: float):
        """Record a cognitive session for rhythm analysis."""
        # Calculate performance score
        performance_score = (
            thoughts_captured * 0.3 +
            insights_generated * 0.4 +
            processing_quality * 0.2 +
            min(focus_duration / 60, 1.0) * 0.1  # Normalize to hour
        )
        
        # Record patterns
        self.hourly_patterns[timestamp.hour].append(performance_score)
        self.daily_patterns[timestamp.weekday()].append(performance_score)
        self.monthly_patterns[timestamp.month].append(performance_score)
        
        # Detect flow state (high performance + sustained focus)
        if performance_score > 0.8 and focus_duration > 25:  # 25+ minutes of high performance
            self.flow_state_indicators.append({
                'timestamp': timestamp,
                'performance': performance_score,
                'duration': focus_duration
            })
    
    def get_optimal_thinking_times(self) -> Dict[str, Any]:
        """Get the optimal times for different types of cognitive work."""
        # Calculate average performance by time period
        hourly_averages = {
            hour: np.mean(scores) for hour, scores in self.hourly_patterns.items()
            if len(scores) >= 3  # Need at least 3 data points
        }
        
        daily_averages = {
            day: np.mean(scores) for day, scores in self.daily_patterns.items()
            if len(scores) >= 2
        }
        
        monthly_averages = {
            month: np.mean(scores) for month, scores in self.monthly_patterns.items()
            if len(scores) >= 2
        }
        
        # Find peak times
        peak_hour = max(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else 10
        peak_day = max(daily_averages.items(), key=lambda x: x[1])[0] if daily_averages else 1  # Tuesday
        peak_month = max(monthly_averages.items(), key=lambda x: x[1])[0] if monthly_averages else 3  # March
        
        # Analyze flow state patterns
        flow_hours = [f['timestamp'].hour for f in self.flow_state_indicators]
        flow_hour_distribution = Counter(flow_hours)
        
        return {
            'peak_performance_hour': peak_hour,
            'peak_performance_day': peak_day,
            'peak_performance_month': peak_month,
            'flow_state_hours': dict(flow_hour_distribution.most_common(5)),
            'recommendations': self._generate_rhythm_recommendations(
                peak_hour, peak_day, peak_month, flow_hour_distribution
            ),
            'hourly_patterns': hourly_averages,
            'daily_patterns': daily_averages,
            'monthly_patterns': monthly_averages
        }
    
    def _generate_rhythm_recommendations(self, peak_hour: int, peak_day: int, 
                                       peak_month: int, flow_hours: Counter) -> List[str]:
        """Generate beautiful recommendations based on cognitive rhythms."""
        recommendations = []
        
        # Hour recommendations
        if 5 <= peak_hour <= 9:
            recommendations.append(f"Your mind shines brightest at {peak_hour}:00 AM. Reserve this sacred time for your most important thinking.")
        elif 10 <= peak_hour <= 14:
            recommendations.append(f"Your cognitive peak at {peak_hour}:00 is perfect for complex analysis and breakthrough insights.")
        elif 15 <= peak_hour <= 19:
            recommendations.append(f"Your afternoon peak at {peak_hour}:00 is ideal for connecting ideas and creative synthesis.")
        else:
            recommendations.append(f"Your late-hour peak at {peak_hour}:00 suggests deep contemplative work thrives in quiet solitude.")
        
        # Day recommendations
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        recommendations.append(f"Your cognitive rhythm peaks on {day_names[peak_day]}s - schedule your most challenging intellectual work then.")
        
        # Flow state recommendations
        if flow_hours:
            most_common_flow_hour = flow_hours.most_common(1)[0][0]
            recommendations.append(f"You enter flow most often around {most_common_flow_hour}:00. Protect this time for deep, uninterrupted work.")
        
        return recommendations


class TemporalIntelligenceEngine:
    """
    Advanced temporal intelligence system that understands how thinking evolves over time.
    
    This engine transforms SUM from understanding WHAT you think to understanding 
    HOW your thinking evolves, providing insights that only emerge from temporal analysis.
    """
    
    def __init__(self, knowledge_os: KnowledgeOperatingSystem, data_dir: str = "temporal_intelligence_data"):
        self.knowledge_os = knowledge_os
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core components
        self.temporal_graph = TemporalKnowledgeGraph()
        self.rhythm_analyzer = CognitiveRhythmAnalyzer()
        
        # Storage
        self.db_path = self.data_dir / "temporal_intelligence.db"
        self._init_database()
        
        # In-memory temporal state
        self.concept_evolutions = {}  # concept -> ConceptEvolution
        self.seasonal_patterns = {}   # pattern_id -> SeasonalPattern
        self.momentum_trackers = {}   # area -> IntellectualMomentum
        self.aging_knowledge = {}     # knowledge_id -> KnowledgeAging
        self.future_projections = [] # List[FutureProjection]
        
        # Background processing
        self.processing_thread = None
        self.is_processing = False
        
        # Load existing data
        self._load_temporal_data()
        
        # Start temporal processing
        self._start_temporal_processing()
        
        logger.info("Temporal Intelligence Engine initialized - Time-aware understanding activated")
    
    def _init_database(self):
        """Initialize the temporal intelligence database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS concept_evolutions (
                    concept TEXT PRIMARY KEY,
                    first_appearance TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    depth_progression TEXT DEFAULT '[]',
                    complexity_trajectory TEXT DEFAULT '[]',
                    confidence_evolution TEXT DEFAULT '[]',
                    breakthrough_moments TEXT DEFAULT '[]',
                    projected_next_depth REAL DEFAULT 0.0
                );
                
                CREATE TABLE IF NOT EXISTS seasonal_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    peak_periods TEXT DEFAULT '[]',
                    associated_concepts TEXT DEFAULT '[]',
                    pattern_strength REAL DEFAULT 0.0,
                    human_description TEXT DEFAULT '',
                    next_predicted_peak TEXT
                );
                
                CREATE TABLE IF NOT EXISTS intellectual_momentum (
                    momentum_id TEXT PRIMARY KEY,
                    research_area TEXT NOT NULL,
                    started TEXT NOT NULL,
                    velocity REAL DEFAULT 0.0,
                    acceleration REAL DEFAULT 0.0,
                    current_critical_mass REAL DEFAULT 0.0,
                    estimated_breakthrough_date TEXT
                );
                
                CREATE TABLE IF NOT EXISTS knowledge_aging (
                    knowledge_id TEXT PRIMARY KEY,
                    original_capture_date TEXT NOT NULL,
                    relevance_decay_rate REAL DEFAULT 0.0,
                    current_relevance_score REAL DEFAULT 1.0,
                    resurrection_events TEXT DEFAULT '[]',
                    next_optimal_review TEXT
                );
                
                CREATE TABLE IF NOT EXISTS future_projections (
                    projection_id TEXT PRIMARY KEY,
                    generated_date TEXT NOT NULL,
                    projection_horizon_days INTEGER NOT NULL,
                    emerging_interests TEXT DEFAULT '[]',
                    prediction_confidence REAL DEFAULT 0.0
                );
                
                CREATE TABLE IF NOT EXISTS cognitive_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    thoughts_captured INTEGER DEFAULT 0,
                    insights_generated INTEGER DEFAULT 0,
                    processing_quality REAL DEFAULT 0.0,
                    focus_duration REAL DEFAULT 0.0
                );
            """)
    
    def _load_temporal_data(self):
        """Load existing temporal intelligence data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load concept evolutions
                cursor = conn.execute("SELECT * FROM concept_evolutions")
                for row in cursor.fetchall():
                    data = dict(zip([col[0] for col in cursor.description], row))
                    
                    evolution = ConceptEvolution(
                        concept=data['concept'],
                        first_appearance=datetime.fromisoformat(data['first_appearance']),
                        last_activity=datetime.fromisoformat(data['last_activity']),
                        depth_progression=json.loads(data.get('depth_progression', '[]')),
                        complexity_trajectory=json.loads(data.get('complexity_trajectory', '[]')),
                        confidence_evolution=json.loads(data.get('confidence_evolution', '[]')),
                        breakthrough_moments=[datetime.fromisoformat(dt) for dt in json.loads(data.get('breakthrough_moments', '[]'))],
                        projected_next_depth=data.get('projected_next_depth', 0.0)
                    )
                    
                    self.concept_evolutions[data['concept']] = evolution
                
                # Load seasonal patterns
                cursor = conn.execute("SELECT * FROM seasonal_patterns")
                for row in cursor.fetchall():
                    data = dict(zip([col[0] for col in cursor.description], row))
                    
                    pattern = SeasonalPattern(
                        pattern_id=data['pattern_id'],
                        pattern_type=data['pattern_type'],
                        peak_periods=json.loads(data.get('peak_periods', '[]')),
                        associated_concepts=json.loads(data.get('associated_concepts', '[]')),
                        pattern_strength=data.get('pattern_strength', 0.0),
                        human_description=data.get('human_description', ''),
                        next_predicted_peak=datetime.fromisoformat(data['next_predicted_peak']) if data.get('next_predicted_peak') else None
                    )
                    
                    self.seasonal_patterns[data['pattern_id']] = pattern
                
                # Load cognitive sessions for rhythm analysis
                cursor = conn.execute("SELECT * FROM cognitive_sessions ORDER BY timestamp DESC LIMIT 1000")
                for row in cursor.fetchall():
                    data = dict(zip([col[0] for col in cursor.description], row))
                    
                    self.rhythm_analyzer.record_cognitive_session(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        thoughts_captured=data.get('thoughts_captured', 0),
                        insights_generated=data.get('insights_generated', 0),
                        processing_quality=data.get('processing_quality', 0.0),
                        focus_duration=data.get('focus_duration', 0.0)
                    )
            
            logger.info(f"Loaded temporal data: {len(self.concept_evolutions)} concept evolutions, "
                       f"{len(self.seasonal_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error loading temporal data: {e}")
    
    def _start_temporal_processing(self):
        """Start background temporal processing."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._temporal_processing_loop, daemon=True)
        self.processing_thread.start()
    
    def _temporal_processing_loop(self):
        """Continuous temporal analysis in the background."""
        while self.is_processing:
            try:
                # Update concept evolutions every minute
                self._update_concept_evolutions()
                
                # Analyze seasonal patterns every hour
                if datetime.now().minute == 0:
                    self._analyze_seasonal_patterns()
                
                # Update momentum tracking every 30 minutes
                if datetime.now().minute in [0, 30]:
                    self._update_intellectual_momentum()
                
                # Age knowledge and update relevance scores every 6 hours
                if datetime.now().hour % 6 == 0 and datetime.now().minute == 0:
                    self._update_knowledge_aging()
                
                # Generate future projections daily
                if datetime.now().hour == 3 and datetime.now().minute == 0:  # 3 AM daily
                    self._generate_future_projections()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Temporal processing error: {e}")
                time.sleep(60)
    
    def process_new_thought(self, thought: Thought):
        """Process a new thought through the temporal intelligence system."""
        timestamp = thought.timestamp
        
        # Update concept evolutions
        for concept in thought.concepts:
            self._update_concept_evolution(concept, thought, timestamp)
            
            # Add to temporal graph
            self.temporal_graph.add_temporal_node(concept, timestamp, thought.importance)
        
        # Create temporal connections between concepts
        for i, concept1 in enumerate(thought.concepts):
            for concept2 in thought.concepts[i+1:]:
                self.temporal_graph.add_temporal_edge(
                    concept1, concept2, timestamp, "co-occurrence", thought.importance
                )
        
        # Update aging for existing knowledge
        self._trigger_knowledge_resurrection_check(thought)
        
        # Record cognitive session data
        self._record_cognitive_session(thought)
    
    def _update_concept_evolution(self, concept: str, thought: Thought, timestamp: datetime):
        """Update the evolution tracking for a concept."""
        if concept not in self.concept_evolutions:
            self.concept_evolutions[concept] = ConceptEvolution(
                concept=concept,
                first_appearance=timestamp,
                last_activity=timestamp
            )
        
        evolution = self.concept_evolutions[concept]
        evolution.last_activity = timestamp
        
        # Calculate depth progression (based on thought complexity and importance)
        current_depth = len(thought.content.split()) / 100 * thought.importance
        evolution.depth_progression.append(current_depth)
        
        # Calculate complexity (based on concept co-occurrence)
        complexity = len(thought.concepts) * thought.importance
        evolution.complexity_trajectory.append(complexity)
        
        # Estimate confidence (based on processing quality and connections)
        confidence = thought.importance * (1 + len(thought.connections) * 0.1)
        evolution.confidence_evolution.append(min(confidence, 1.0))
        
        # Detect breakthrough moments (significant jumps in understanding)
        if len(evolution.depth_progression) > 1:
            depth_increase = current_depth - evolution.depth_progression[-2]
            if depth_increase > 0.3:  # Significant jump
                evolution.breakthrough_moments.append(timestamp)
        
        # Update context expansion
        current_context = set(thought.concepts)
        evolution.context_expansion.append(current_context)
        
        # Calculate projected next depth using simple linear regression
        if len(evolution.depth_progression) >= 3:
            x = np.arange(len(evolution.depth_progression))
            y = np.array(evolution.depth_progression)
            slope, intercept = np.polyfit(x, y, 1)
            evolution.projected_next_depth = slope * len(evolution.depth_progression) + intercept
        
        # Estimate breakthrough probability based on momentum
        if len(evolution.depth_progression) >= 5:
            recent_progress = np.mean(evolution.depth_progression[-3:]) - np.mean(evolution.depth_progression[-6:-3])
            evolution.estimated_breakthrough_probability = min(max(recent_progress * 2, 0), 1)
    
    def _update_concept_evolutions(self):
        """Update all concept evolutions."""
        # This runs continuously and updates temporal patterns
        current_time = datetime.now()
        
        for concept, evolution in self.concept_evolutions.items():
            # Calculate time since last activity
            time_since_activity = current_time - evolution.last_activity
            
            # Generate recommendations based on evolution patterns
            if len(evolution.depth_progression) >= 3:
                recent_trend = np.mean(evolution.depth_progression[-3:]) - np.mean(evolution.depth_progression[-6:-3])
                
                if recent_trend > 0.1:
                    evolution.recommended_exploration_areas = [
                        f"Deep dive into {concept.replace('-', ' ')} fundamentals",
                        f"Explore advanced {concept.replace('-', ' ')} applications",
                        f"Connect {concept.replace('-', ' ')} to related fields"
                    ]
                elif recent_trend < -0.1:
                    evolution.recommended_exploration_areas = [
                        f"Revisit {concept.replace('-', ' ')} basics",
                        f"Find new perspectives on {concept.replace('-', ' ')}",
                        f"Apply {concept.replace('-', ' ')} to practical problems"
                    ]
    
    def _analyze_seasonal_patterns(self):
        """Analyze and update seasonal thinking patterns."""
        current_time = datetime.now()
        
        # Analyze hourly patterns
        hourly_concepts = defaultdict(list)
        daily_concepts = defaultdict(list)
        monthly_concepts = defaultdict(list)
        
        # Gather data from recent thoughts
        recent_thoughts = self.knowledge_os.get_recent_thoughts(1000)  # Last 1000 thoughts
        
        for thought in recent_thoughts:
            hour = thought.timestamp.hour
            day = thought.timestamp.weekday()
            month = thought.timestamp.month
            
            for concept in thought.concepts:
                hourly_concepts[hour].append(concept)
                daily_concepts[day].append(concept)
                monthly_concepts[month].append(concept)
        
        # Detect hourly patterns
        for hour, concepts in hourly_concepts.items():
            if len(concepts) >= 10:  # Need sufficient data
                concept_freq = Counter(concepts)
                dominant_concepts = [concept for concept, count in concept_freq.most_common(3)]
                
                pattern_id = f"hourly_{hour}"
                if pattern_id not in self.seasonal_patterns:
                    self.seasonal_patterns[pattern_id] = SeasonalPattern(
                        pattern_id=pattern_id,
                        pattern_type="hourly"
                    )
                
                pattern = self.seasonal_patterns[pattern_id]
                pattern.peak_periods = [f"{hour}:00"]
                pattern.associated_concepts = dominant_concepts
                pattern.pattern_strength = min(len(concepts) / 50.0, 1.0)  # Normalize
                pattern.human_description = f"You tend to think about {', '.join(dominant_concepts[:2])} around {hour}:00"
        
        # Similar analysis for daily and monthly patterns
        self._analyze_daily_patterns(daily_concepts)
        self._analyze_monthly_patterns(monthly_concepts)
    
    def _analyze_daily_patterns(self, daily_concepts: dict):
        """Analyze daily thinking patterns."""
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day, concepts in daily_concepts.items():
            if len(concepts) >= 20:
                concept_freq = Counter(concepts)
                dominant_concepts = [concept for concept, count in concept_freq.most_common(3)]
                
                pattern_id = f"daily_{day}"
                if pattern_id not in self.seasonal_patterns:
                    self.seasonal_patterns[pattern_id] = SeasonalPattern(
                        pattern_id=pattern_id,
                        pattern_type="daily"
                    )
                
                pattern = self.seasonal_patterns[pattern_id]
                pattern.peak_periods = [day_names[day]]
                pattern.associated_concepts = dominant_concepts
                pattern.pattern_strength = min(len(concepts) / 100.0, 1.0)
                pattern.human_description = f"Your {day_names[day]} mind gravitates toward {', '.join(dominant_concepts[:2])}"
    
    def _analyze_monthly_patterns(self, monthly_concepts: dict):
        """Analyze monthly thinking patterns."""
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        for month, concepts in monthly_concepts.items():
            if len(concepts) >= 30:
                concept_freq = Counter(concepts)
                dominant_concepts = [concept for concept, count in concept_freq.most_common(3)]
                
                pattern_id = f"monthly_{month}"
                if pattern_id not in self.seasonal_patterns:
                    self.seasonal_patterns[pattern_id] = SeasonalPattern(
                        pattern_id=pattern_id,
                        pattern_type="monthly"
                    )
                
                pattern = self.seasonal_patterns[pattern_id]
                pattern.peak_periods = [month_names[month-1]]
                pattern.associated_concepts = dominant_concepts
                pattern.pattern_strength = min(len(concepts) / 200.0, 1.0)
                pattern.human_description = f"You tend to explore {', '.join(dominant_concepts[:2])} in {month_names[month-1]}"
                
                # Predict next peak (next year)
                next_year = datetime.now().year + 1
                pattern.next_predicted_peak = datetime(next_year, month, 15)  # Mid-month
    
    def _update_intellectual_momentum(self):
        """Update intellectual momentum tracking."""
        # Group recent thoughts by research area (dominant concept)
        recent_thoughts = self.knowledge_os.get_recent_thoughts(500)
        research_areas = defaultdict(list)
        
        for thought in recent_thoughts:
            if thought.concepts:
                primary_concept = thought.concepts[0]  # Use primary concept as research area
                research_areas[primary_concept].append(thought)
        
        # Update momentum for each research area
        for area, thoughts in research_areas.items():
            if len(thoughts) >= 3:  # Need minimum thoughts for momentum analysis
                momentum_id = f"momentum_{area}"
                
                if momentum_id not in self.momentum_trackers:
                    self.momentum_trackers[momentum_id] = IntellectualMomentum(
                        momentum_id=momentum_id,
                        research_area=area,
                        started=min(t.timestamp for t in thoughts)
                    )
                
                momentum = self.momentum_trackers[momentum_id]
                
                # Calculate velocity (thoughts per day)
                time_span = (max(t.timestamp for t in thoughts) - min(t.timestamp for t in thoughts)).days
                if time_span > 0:
                    momentum.velocity = len(thoughts) / time_span
                
                # Calculate mass (depth of engagement)
                total_importance = sum(t.importance for t in thoughts)
                momentum.mass = total_importance / len(thoughts)
                
                # Calculate critical mass
                momentum.current_critical_mass = min(momentum.velocity * momentum.mass / 10, 1.0)
                
                # Predict breakthrough if critical mass is high
                if momentum.current_critical_mass > momentum.breakthrough_threshold:
                    days_to_breakthrough = max(1, int(10 * (1 - momentum.current_critical_mass)))
                    momentum.estimated_breakthrough_date = datetime.now() + timedelta(days=days_to_breakthrough)
                
                # Detect flow sessions (sustained high-quality thinking)
                flow_sessions = []
                current_session_start = None
                
                for thought in sorted(thoughts, key=lambda t: t.timestamp):
                    if thought.importance > 0.7 and thought.word_count > 50:
                        if current_session_start is None:
                            current_session_start = thought.timestamp
                    else:
                        if current_session_start:
                            flow_sessions.append(current_session_start)
                            current_session_start = None
                
                momentum.flow_sessions = flow_sessions
                
                # Generate optimization suggestions
                momentum.momentum_optimization_suggestions = self._generate_momentum_suggestions(momentum, thoughts)
    
    def _generate_momentum_suggestions(self, momentum: IntellectualMomentum, thoughts: List[Thought]) -> List[str]:
        """Generate suggestions to optimize intellectual momentum."""
        suggestions = []
        area_name = momentum.research_area.replace('-', ' ').title()
        
        if momentum.velocity < 0.5:  # Low velocity
            suggestions.append(f"Your exploration of {area_name} could benefit from more frequent engagement. Try daily micro-sessions.")
        
        if momentum.mass < 0.5:  # Low depth
            suggestions.append(f"Consider diving deeper into {area_name}. Your current thoughts could be more substantial.")
        
        if momentum.current_critical_mass > 0.6:  # High momentum
            suggestions.append(f"You're building excellent momentum in {area_name}! A breakthrough might be near - maintain this pace.")
        
        if len(momentum.flow_sessions) > 0:
            suggestions.append(f"You've had {len(momentum.flow_sessions)} flow sessions with {area_name}. Schedule dedicated time to maximize these insights.")
        
        return suggestions
    
    def _update_knowledge_aging(self):
        """Update knowledge aging and relevance scores."""
        current_time = datetime.now()
        
        # Get all thoughts for aging analysis
        all_thoughts = list(self.knowledge_os.active_thoughts.values())
        
        for thought in all_thoughts:
            aging_id = f"aging_{thought.id}"
            
            if aging_id not in self.aging_knowledge:
                self.aging_knowledge[aging_id] = KnowledgeAging(
                    knowledge_id=aging_id,
                    original_capture_date=thought.timestamp
                )
            
            aging = self.aging_knowledge[aging_id]
            
            # Calculate age in days
            age_days = (current_time - aging.original_capture_date).days
            
            # Apply forgetting curve: R(t) = R0 * e^(-t/S)
            # Where R0 is initial strength, t is time, S is decay constant
            initial_strength, decay_rate = aging.forgetting_curve_params
            aging.current_relevance_score = initial_strength * math.exp(-age_days * decay_rate / 30)
            
            # Check for resurrection opportunities
            if aging.current_relevance_score < 0.3:  # Low relevance
                # Calculate optimal review time using spaced repetition
                if aging.successful_recalls > 0:
                    # Increase interval based on successful recalls
                    interval_days = min(2 ** aging.successful_recalls, 365)
                else:
                    interval_days = 1  # Start with daily review
                
                aging.next_optimal_review = current_time + timedelta(days=interval_days)
    
    def _trigger_knowledge_resurrection_check(self, new_thought: Thought):
        """Check if the new thought should resurrect old knowledge."""
        # Find aged knowledge that might be relevant to this new thought
        for aging_id, aging in self.aging_knowledge.items():
            if aging.current_relevance_score < 0.5:  # Aged knowledge
                # Extract original thought
                thought_id = aging_id.replace('aging_', '')
                if thought_id in self.knowledge_os.active_thoughts:
                    old_thought = self.knowledge_os.active_thoughts[thought_id]
                    
                    # Check for concept overlap
                    concept_overlap = set(new_thought.concepts).intersection(set(old_thought.concepts))
                    
                    if concept_overlap or self._semantic_similarity(new_thought.content, old_thought.content) > 0.6:
                        # Resurrect the knowledge
                        aging.resurrection_events.append(datetime.now())
                        aging.resurrection_contexts.append(new_thought.content[:100])
                        aging.resurrection_triggers.extend(list(concept_overlap))
                        aging.current_relevance_score = 0.8  # Boost relevance
                        
                        logger.info(f"Resurrected knowledge: {old_thought.content[:50]}... due to new thought about {concept_overlap}")
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _generate_future_projections(self):
        """Generate predictions about future interests and learning paths."""
        projection_time = datetime.now()
        
        # Analyze recent trends
        recent_thoughts = self.knowledge_os.get_recent_thoughts(200)
        if not recent_thoughts:
            return
        
        # Calculate concept momentum
        concept_frequencies = defaultdict(list)
        for thought in recent_thoughts:
            days_ago = (projection_time - thought.timestamp).days
            for concept in thought.concepts:
                concept_frequencies[concept].append(days_ago)
        
        # Predict emerging interests
        emerging_interests = []
        declining_interests = []
        
        for concept, day_list in concept_frequencies.items():
            if len(day_list) >= 3:
                # Calculate trend (negative slope means increasing recent activity)
                x = np.array(day_list)
                y = np.arange(len(day_list))
                
                if len(set(x)) > 1:  # Avoid division by zero
                    correlation = np.corrcoef(x, y)[0, 1]
                    
                    # Predict probability based on trend
                    if correlation < -0.3:  # Strong negative correlation = emerging
                        probability = min(abs(correlation), 1.0)
                        emerging_interests.append((concept, probability))
                    elif correlation > 0.3:  # Strong positive correlation = declining
                        probability = min(correlation, 1.0)
                        declining_interests.append((concept, probability))
        
        # Sort by probability
        emerging_interests.sort(key=lambda x: x[1], reverse=True)
        declining_interests.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = []
        if emerging_interests:
            top_emerging = emerging_interests[0][0].replace('-', ' ').title()
            recommendations.append(f"Deep dive into {top_emerging} - your interest is accelerating")
        
        # Create projection
        projection = FutureProjection(
            projection_id=f"projection_{int(projection_time.timestamp())}",
            generated_date=projection_time,
            projection_horizon=timedelta(days=30),
            emerging_interests=emerging_interests[:10],
            declining_interests=declining_interests[:10],
            recommended_next_steps=recommendations,
            prediction_confidence=0.7  # Base confidence
        )
        
        self.future_projections.append(projection)
        
        # Keep only recent projections
        cutoff_date = projection_time - timedelta(days=90)
        self.future_projections = [p for p in self.future_projections if p.generated_date > cutoff_date]
    
    def _record_cognitive_session(self, thought: Thought):
        """Record cognitive session data for rhythm analysis."""
        # Estimate session quality based on thought characteristics
        processing_quality = thought.importance
        focus_duration = min(thought.word_count / 10, 60)  # Estimate based on word count
        
        self.rhythm_analyzer.record_cognitive_session(
            timestamp=thought.timestamp,
            thoughts_captured=1,
            insights_generated=1 if thought.importance > 0.7 else 0,
            processing_quality=processing_quality,
            focus_duration=focus_duration
        )
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cognitive_sessions 
                    (session_id, timestamp, thoughts_captured, insights_generated, processing_quality, focus_duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    f"session_{thought.id}",
                    thought.timestamp.isoformat(),
                    1,
                    1 if thought.importance > 0.7 else 0,
                    processing_quality,
                    focus_duration
                ))
        except Exception as e:
            logger.error(f"Error saving cognitive session: {e}")
    
    def get_temporal_insights(self) -> Dict[str, Any]:
        """Get comprehensive temporal intelligence insights."""
        current_time = datetime.now()
        
        insights = {
            'timestamp': current_time.isoformat(),
            'concept_evolution_summary': self._get_concept_evolution_summary(),
            'seasonal_patterns': self._get_seasonal_patterns_summary(),
            'intellectual_momentum': self._get_momentum_summary(),
            'knowledge_aging': self._get_aging_summary(),
            'future_projections': self._get_projection_summary(),
            'cognitive_rhythms': self.rhythm_analyzer.get_optimal_thinking_times(),
            'temporal_connections': self._get_temporal_connections_summary(),
            'beautiful_narrative': self._generate_temporal_narrative()
        }
        
        return insights
    
    def _get_concept_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of concept evolution."""
        if not self.concept_evolutions:
            return {'message': 'Begin thinking to see how your understanding evolves over time'}
        
        # Find concepts with the most evolution
        evolved_concepts = []
        for concept, evolution in self.concept_evolutions.items():
            if len(evolution.depth_progression) > 1:
                depth_change = evolution.depth_progression[-1] - evolution.depth_progression[0]
                evolved_concepts.append((concept, depth_change, len(evolution.breakthrough_moments)))
        
        evolved_concepts.sort(key=lambda x: x[1], reverse=True)
        
        summary = {
            'total_concepts_tracked': len(self.concept_evolutions),
            'most_evolved_concepts': [
                {
                    'concept': concept.replace('-', ' ').title(),
                    'depth_increase': f"{depth_change:.2f}",
                    'breakthroughs': breakthroughs,
                    'status': 'Rapidly evolving' if depth_change > 0.5 else 'Steady growth' if depth_change > 0.1 else 'Stable'
                }
                for concept, depth_change, breakthroughs in evolved_concepts[:5]
            ],
            'breakthrough_concepts': [
                {
                    'concept': concept.replace('-', ' ').title(),
                    'breakthrough_count': len(evolution.breakthrough_moments),
                    'latest_breakthrough': evolution.breakthrough_moments[-1].strftime('%B %d') if evolution.breakthrough_moments else None
                }
                for concept, evolution in self.concept_evolutions.items()
                if evolution.breakthrough_moments
            ][:5]
        }
        
        return summary
    
    def _get_seasonal_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of seasonal thinking patterns."""
        if not self.seasonal_patterns:
            return {'message': 'More data needed to detect seasonal patterns in your thinking'}
        
        strongest_patterns = sorted(
            self.seasonal_patterns.values(),
            key=lambda p: p.pattern_strength,
            reverse=True
        )[:5]
        
        return {
            'patterns_detected': len(self.seasonal_patterns),
            'strongest_patterns': [
                {
                    'description': pattern.human_description,
                    'strength': f"{pattern.pattern_strength:.1%}",
                    'type': pattern.pattern_type,
                    'concepts': pattern.associated_concepts[:3]
                }
                for pattern in strongest_patterns
            ],
            'next_predicted_peaks': [
                {
                    'pattern': pattern.human_description,
                    'predicted_peak': pattern.next_predicted_peak.strftime('%B %d, %Y') if pattern.next_predicted_peak else 'Calculating...'
                }
                for pattern in strongest_patterns[:3]
                if pattern.next_predicted_peak
            ]
        }
    
    def _get_momentum_summary(self) -> Dict[str, Any]:
        """Get summary of intellectual momentum."""
        if not self.momentum_trackers:
            return {'message': 'Start exploring topics consistently to build intellectual momentum'}
        
        active_momentum = [(k, v) for k, v in self.momentum_trackers.items() 
                          if v.current_critical_mass > 0.1]
        active_momentum.sort(key=lambda x: x[1].current_critical_mass, reverse=True)
        
        return {
            'active_research_areas': len(active_momentum),
            'highest_momentum': [
                {
                    'area': momentum.research_area.replace('-', ' ').title(),
                    'critical_mass': f"{momentum.current_critical_mass:.1%}",
                    'velocity': f"{momentum.velocity:.1f} thoughts/day",
                    'breakthrough_probability': f"{momentum.current_critical_mass:.1%}",
                    'estimated_breakthrough': momentum.estimated_breakthrough_date.strftime('%B %d') if momentum.estimated_breakthrough_date else 'Calculating...',
                    'suggestions': momentum.momentum_optimization_suggestions[:2]
                }
                for _, momentum in active_momentum[:5]
            ],
            'flow_state_sessions': sum(len(m.flow_sessions) for _, m in active_momentum)
        }
    
    def _get_aging_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge aging."""
        if not self.aging_knowledge:
            return {'message': 'Knowledge aging analysis will develop as you capture more thoughts'}
        
        # Find knowledge ready for resurrection
        ready_for_review = []
        recently_resurrected = []
        
        current_time = datetime.now()
        
        for aging_id, aging in self.aging_knowledge.items():
            if aging.next_optimal_review and aging.next_optimal_review <= current_time:
                ready_for_review.append(aging)
            
            if aging.resurrection_events and aging.resurrection_events[-1] > current_time - timedelta(days=7):
                recently_resurrected.append(aging)
        
        return {
            'total_knowledge_tracked': len(self.aging_knowledge),
            'ready_for_review': len(ready_for_review),
            'recently_resurrected': len(recently_resurrected),
            'resurrection_opportunities': [
                {
                    'knowledge_id': aging.knowledge_id.replace('aging_', ''),
                    'age_days': (current_time - aging.original_capture_date).days,
                    'current_relevance': f"{aging.current_relevance_score:.1%}",
                    'next_review': aging.next_optimal_review.strftime('%B %d') if aging.next_optimal_review else 'Soon'
                }
                for aging in ready_for_review[:5]
            ],
            'recent_resurrections': [
                {
                    'context': aging.resurrection_contexts[-1][:100] + '...' if aging.resurrection_contexts else 'Unknown',
                    'triggers': aging.resurrection_triggers[-3:] if aging.resurrection_triggers else []
                }
                for aging in recently_resurrected[:3]
            ]
        }
    
    def _get_projection_summary(self) -> Dict[str, Any]:
        """Get summary of future projections."""
        if not self.future_projections:
            return {'message': 'Future projections will appear as your thinking patterns establish'}
        
        latest_projection = max(self.future_projections, key=lambda p: p.generated_date)
        
        return {
            'projection_date': latest_projection.generated_date.strftime('%B %d, %Y'),
            'confidence': f"{latest_projection.prediction_confidence:.1%}",
            'emerging_interests': [
                {
                    'interest': interest.replace('-', ' ').title(),
                    'probability': f"{prob:.1%}"
                }
                for interest, prob in latest_projection.emerging_interests[:5]
            ],
            'declining_interests': [
                {
                    'interest': interest.replace('-', ' ').title(),
                    'probability': f"{prob:.1%}"
                }
                for interest, prob in latest_projection.declining_interests[:3]
            ],
            'recommendations': latest_projection.recommended_next_steps[:3],
            'horizon': f"{latest_projection.projection_horizon.days} days"
        }
    
    def _get_temporal_connections_summary(self) -> Dict[str, Any]:
        """Get summary of temporal connections in the knowledge graph."""
        current_time = datetime.now()
        
        # Get temporal centrality for recent period
        centrality = self.temporal_graph.get_temporal_centrality(current_time, timedelta(days=30))
        
        if not centrality:
            return {'message': 'Temporal connections will emerge as concepts develop relationships'}
        
        # Sort by centrality
        top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_nodes': self.temporal_graph.graph.number_of_nodes(),
            'total_connections': self.temporal_graph.graph.number_of_edges(),
            'most_central_concepts': [
                {
                    'concept': concept.replace('-', ' ').title(),
                    'centrality_score': f"{score:.3f}",
                    'connections': len(list(self.temporal_graph.graph.neighbors(concept)))
                }
                for concept, score in top_concepts
            ],
            'predicted_future_connections': [
                {
                    'from_concept': concept.replace('-', ' ').title(),
                    'predicted_connections': [
                        {'concept': conn[0].replace('-', ' ').title(), 'probability': f"{conn[1]:.2f}"}
                        for conn in self.temporal_graph.predict_future_connections(concept, timedelta(days=30))[:3]
                    ]
                }
                for concept, _ in top_concepts[:2]
            ]
        }
    
    def _generate_temporal_narrative(self) -> str:
        """Generate a beautiful narrative about the user's temporal thinking journey."""
        if not self.concept_evolutions:
            return "Your temporal intelligence journey awaits. As you capture thoughts over time, I'll reveal the beautiful patterns of how your understanding evolves, when insights peak, and what breakthroughs await."
        
        # Gather narrative elements
        total_concepts = len(self.concept_evolutions)
        evolved_concepts = [c for c, e in self.concept_evolutions.items() if len(e.depth_progression) > 1]
        breakthrough_count = sum(len(e.breakthrough_moments) for e in self.concept_evolutions.values())
        
        oldest_concept = min(self.concept_evolutions.values(), key=lambda e: e.first_appearance)
        newest_concept = max(self.concept_evolutions.values(), key=lambda e: e.first_appearance)
        
        journey_days = (newest_concept.first_appearance - oldest_concept.first_appearance).days + 1
        
        # Find dominant seasonal pattern
        strongest_pattern = None
        if self.seasonal_patterns:
            strongest_pattern = max(self.seasonal_patterns.values(), key=lambda p: p.pattern_strength)
        
        # Generate narrative
        narrative = f"Over {journey_days} days, your thinking has explored {total_concepts} distinct concepts, "
        narrative += f"with {len(evolved_concepts)} showing measurable evolution in depth and understanding. "
        
        if breakthrough_count > 0:
            narrative += f"You've experienced {breakthrough_count} breakthrough moments where understanding suddenly deepened. "
        
        if strongest_pattern:
            narrative += f"Your mind shows beautiful patterns - {strongest_pattern.human_description.lower()}. "
        
        # Add momentum insight
        if self.momentum_trackers:
            high_momentum = [m for m in self.momentum_trackers.values() if m.current_critical_mass > 0.5]
            if high_momentum:
                area_name = high_momentum[0].research_area.replace('-', ' ')
                narrative += f"Right now, you're building powerful momentum in {area_name}, "
                narrative += f"suggesting a breakthrough may be approaching. "
        
        narrative += "This is the poetry of thought evolution - each idea a note in the symphony of your expanding understanding."
        
        return narrative
    
    def save_temporal_state(self):
        """Save current temporal intelligence state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save concept evolutions
                for concept, evolution in self.concept_evolutions.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO concept_evolutions
                        (concept, first_appearance, last_activity, depth_progression, 
                         complexity_trajectory, confidence_evolution, breakthrough_moments, projected_next_depth)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        concept,
                        evolution.first_appearance.isoformat(),
                        evolution.last_activity.isoformat(),
                        json.dumps(evolution.depth_progression),
                        json.dumps(evolution.complexity_trajectory),
                        json.dumps(evolution.confidence_evolution),
                        json.dumps([dt.isoformat() for dt in evolution.breakthrough_moments]),
                        evolution.projected_next_depth
                    ))
                
                # Save seasonal patterns
                for pattern_id, pattern in self.seasonal_patterns.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO seasonal_patterns
                        (pattern_id, pattern_type, peak_periods, associated_concepts, 
                         pattern_strength, human_description, next_predicted_peak)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_id,
                        pattern.pattern_type,
                        json.dumps(pattern.peak_periods),
                        json.dumps(pattern.associated_concepts),
                        pattern.pattern_strength,
                        pattern.human_description,
                        pattern.next_predicted_peak.isoformat() if pattern.next_predicted_peak else None
                    ))
                
                # Save intellectual momentum
                for momentum_id, momentum in self.momentum_trackers.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO intellectual_momentum
                        (momentum_id, research_area, started, velocity, acceleration,
                         current_critical_mass, estimated_breakthrough_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        momentum_id,
                        momentum.research_area,
                        momentum.started.isoformat(),
                        momentum.velocity,
                        momentum.acceleration,
                        momentum.current_critical_mass,
                        momentum.estimated_breakthrough_date.isoformat() if momentum.estimated_breakthrough_date else None
                    ))
                
                # Save knowledge aging
                for aging_id, aging in self.aging_knowledge.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO knowledge_aging
                        (knowledge_id, original_capture_date, relevance_decay_rate,
                         current_relevance_score, resurrection_events, next_optimal_review)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        aging_id,
                        aging.original_capture_date.isoformat(),
                        aging.relevance_decay_rate,
                        aging.current_relevance_score,
                        json.dumps([dt.isoformat() for dt in aging.resurrection_events]),
                        aging.next_optimal_review.isoformat() if aging.next_optimal_review else None
                    ))
                
                # Save future projections
                for projection in self.future_projections:
                    conn.execute("""
                        INSERT OR REPLACE INTO future_projections
                        (projection_id, generated_date, projection_horizon_days,
                         emerging_interests, prediction_confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        projection.projection_id,
                        projection.generated_date.isoformat(),
                        projection.projection_horizon.days,
                        json.dumps(projection.emerging_interests),
                        projection.prediction_confidence
                    ))
            
            logger.info("Temporal intelligence state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving temporal state: {e}")


# Integration with existing SUM systems
class TemporalSUMIntegration:
    """Integration layer between Temporal Intelligence and existing SUM systems."""
    
    def __init__(self, knowledge_os: KnowledgeOperatingSystem):
        self.knowledge_os = knowledge_os
        self.temporal_engine = TemporalIntelligenceEngine(knowledge_os)
        
        # Hook into knowledge OS thought processing
        self._integrate_with_knowledge_os()
    
    def _integrate_with_knowledge_os(self):
        """Integrate temporal processing with knowledge OS."""
        # Override the capture_thought method to include temporal processing
        original_capture = self.knowledge_os.capture_thought
        
        def enhanced_capture_thought(content: str, source: str = "direct") -> Thought:
            # Call original capture
            thought = original_capture(content, source)
            
            if thought:
                # Process through temporal intelligence
                self.temporal_engine.process_new_thought(thought)
            
            return thought
        
        # Replace the method
        self.knowledge_os.capture_thought = enhanced_capture_thought
    
    def get_enhanced_insights(self) -> Dict[str, Any]:
        """Get insights that combine Knowledge OS and Temporal Intelligence."""
        knowledge_insights = self.knowledge_os.get_system_insights()
        temporal_insights = self.temporal_engine.get_temporal_insights()
        
        # Combine insights
        enhanced_insights = {
            'timestamp': datetime.now().isoformat(),
            'knowledge_os_insights': knowledge_insights,
            'temporal_insights': temporal_insights,
            'combined_narrative': self._generate_combined_narrative(knowledge_insights, temporal_insights),
            'actionable_recommendations': self._generate_actionable_recommendations(knowledge_insights, temporal_insights)
        }
        
        return enhanced_insights
    
    def _generate_combined_narrative(self, knowledge_insights: Dict, temporal_insights: Dict) -> str:
        """Generate a narrative combining both knowledge and temporal insights."""
        base_narrative = knowledge_insights.get('beautiful_summary', '')
        temporal_narrative = temporal_insights.get('beautiful_narrative', '')
        
        if base_narrative and temporal_narrative:
            return f"{base_narrative}\n\n{temporal_narrative}"
        else:
            return base_narrative or temporal_narrative or "Your enhanced thinking journey is beginning..."
    
    def _generate_actionable_recommendations(self, knowledge_insights: Dict, temporal_insights: Dict) -> List[str]:
        """Generate actionable recommendations based on combined insights."""
        recommendations = []
        
        # From cognitive rhythms
        rhythm_data = temporal_insights.get('cognitive_rhythms', {})
        if 'recommendations' in rhythm_data:
            recommendations.extend(rhythm_data['recommendations'][:2])
        
        # From momentum tracking
        momentum_data = temporal_insights.get('intellectual_momentum', {})
        if 'highest_momentum' in momentum_data:
            for area in momentum_data['highest_momentum'][:2]:
                recommendations.extend(area.get('suggestions', [])[:1])
        
        # From future projections
        projection_data = temporal_insights.get('future_projections', {})
        if 'recommendations' in projection_data:
            recommendations.extend(projection_data['recommendations'][:2])
        
        return recommendations[:5]  # Limit to top 5 recommendations


if __name__ == "__main__":
    # Demonstration of the Temporal Intelligence System
    print("🧠⏰ Temporal Intelligence Engine - Revolutionary Time-Aware Understanding")
    print("=" * 80)
    
    # Initialize system
    knowledge_os = KnowledgeOperatingSystem()
    temporal_integration = TemporalSUMIntegration(knowledge_os)
    
    # Simulate thought capture over time to demonstrate temporal features
    sample_thoughts = [
        ("Machine learning is fascinating. I want to understand deep learning better.", "2024-01-15"),
        ("Reading about neural networks today. The math is complex but beautiful.", "2024-01-18"),
        ("Had a breakthrough understanding backpropagation! It's like learning in reverse.", "2024-01-22"),
        ("Thinking about applications of ML in creativity. Could AI help with art?", "2024-02-01"),
        ("Deep learning models seem to learn hierarchical features automatically.", "2024-02-15"),
        ("Philosophy question: What does it mean for a machine to 'understand'?", "2024-03-01"),
        ("Connecting ML to consciousness studies. Are we building minds?", "2024-03-10"),
        ("Meditation today made me think about attention mechanisms in transformers.", "2024-03-15"),
    ]
    
    print("Simulating thought capture over time...")
    for content, date_str in sample_thoughts:
        # Simulate thoughts captured at different times
        thought = knowledge_os.capture_thought(content)
        if thought:
            # Manually set timestamp for demonstration
            thought.timestamp = datetime.fromisoformat(f"{date_str} 14:30:00")
            temporal_integration.temporal_engine.process_new_thought(thought)
            print(f"  📝 {date_str}: {content[:50]}...")
    
    print("\n" + "="*80)
    print("🔮 TEMPORAL INTELLIGENCE INSIGHTS")
    print("="*80)
    
    # Get enhanced insights
    insights = temporal_integration.get_enhanced_insights()
    
    # Display temporal insights
    temporal_data = insights['temporal_insights']
    
    print(f"\n📈 CONCEPT EVOLUTION")
    print("-" * 40)
    concept_summary = temporal_data['concept_evolution_summary']
    if 'most_evolved_concepts' in concept_summary:
        for concept in concept_summary['most_evolved_concepts']:
            print(f"  • {concept['concept']}: {concept['status']} ({concept['depth_increase']} depth increase)")
    
    print(f"\n⏰ SEASONAL PATTERNS")
    print("-" * 40)
    pattern_summary = temporal_data['seasonal_patterns']
    if 'strongest_patterns' in pattern_summary:
        for pattern in pattern_summary['strongest_patterns']:
            print(f"  • {pattern['description']} (Strength: {pattern['strength']})")
    
    print(f"\n🚀 INTELLECTUAL MOMENTUM")
    print("-" * 40)
    momentum_summary = temporal_data['intellectual_momentum']
    if 'highest_momentum' in momentum_summary:
        for area in momentum_summary['highest_momentum']:
            print(f"  • {area['area']}: {area['critical_mass']} critical mass")
            for suggestion in area['suggestions']:
                print(f"    💡 {suggestion}")
    
    print(f"\n🔮 FUTURE PROJECTIONS")
    print("-" * 40)
    projection_summary = temporal_data['future_projections']
    if 'emerging_interests' in projection_summary:
        print("  Emerging interests:")
        for interest in projection_summary['emerging_interests']:
            print(f"    • {interest['interest']} ({interest['probability']} probability)")
    
    print(f"\n🧠 COGNITIVE RHYTHMS")
    print("-" * 40)
    rhythm_data = temporal_data['cognitive_rhythms']
    if 'recommendations' in rhythm_data:
        for rec in rhythm_data['recommendations']:
            print(f"  💡 {rec}")
    
    print(f"\n✨ TEMPORAL NARRATIVE")
    print("-" * 40)
    narrative = temporal_data.get('beautiful_narrative', '')
    if narrative:
        print(f"  {narrative}")
    
    print(f"\n🎯 ACTIONABLE RECOMMENDATIONS")
    print("-" * 40)
    recommendations = insights.get('actionable_recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
    print("✨ Temporal Intelligence Engine Ready - Your thinking now understands time!")
    print("="*80)