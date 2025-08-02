#!/usr/bin/env python3
"""
Collaborative Intelligence Engine for SUM
========================================

Real-time collaborative intelligence platform that transforms individual 
intelligence amplification into collective wisdom generation.

Features:
- Shared knowledge clusters with real-time collaboration
- Live co-thinking sessions with multi-user support
- Team insight generation and pattern recognition
- Knowledge inheritance and mentorship systems
- Privacy-first anonymous insight sharing

Author: SUM Development Team
License: Apache License 2.0
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
from enum import Enum
import hashlib

# Core SUM integration
from invisible_ai_engine import InvisibleAI
from predictive_intelligence import PredictiveIntelligenceEngine
from temporal_intelligence_engine import TemporalIntelligenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborationPermission(Enum):
    """Permission levels for collaborative spaces."""
    READ = "read"
    CONTRIBUTE = "contribute" 
    MODERATE = "moderate"
    ADMIN = "admin"


class SessionState(Enum):
    """States for collaborative sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    PRIVATE = "private"


@dataclass
class Participant:
    """Represents a participant in collaborative intelligence."""
    user_id: str
    name: str
    joined_at: datetime
    permission: CollaborationPermission
    last_active: datetime = field(default_factory=datetime.now)
    contributions: int = 0
    insights_generated: int = 0
    connections_discovered: int = 0
    thinking_style: str = "unknown"  # analytical, creative, systematic, exploratory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'joined_at': self.joined_at.isoformat(),
            'permission': self.permission.value,
            'last_active': self.last_active.isoformat(),
            'contributions': self.contributions,
            'insights_generated': self.insights_generated,
            'connections_discovered': self.connections_discovered,
            'thinking_style': self.thinking_style
        }


@dataclass
class CollaborativeContribution:
    """Represents a contribution to collaborative intelligence."""
    id: str
    participant_id: str
    content: str
    content_type: str  # text, image, audio, video, insight
    timestamp: datetime
    processed_result: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    connections: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    reactions: Dict[str, int] = field(default_factory=dict)  # emoji -> count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'participant_id': self.participant_id,
            'content': self.content,
            'content_type': self.content_type,
            'timestamp': self.timestamp.isoformat(),
            'processed_result': self.processed_result,
            'confidence_score': self.confidence_score,
            'connections': self.connections,
            'insights': self.insights,
            'reactions': self.reactions
        }


@dataclass
class SharedKnowledgeCluster:
    """Represents a shared knowledge cluster."""
    id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    participants: List[Participant] = field(default_factory=list)
    contributions: List[CollaborativeContribution] = field(default_factory=list)
    shared_insights: List[str] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    state: SessionState = SessionState.ACTIVE
    tags: List[str] = field(default_factory=list)
    privacy_level: str = "team"  # public, team, private
    
    def get_active_participants(self) -> List[Participant]:
        """Get participants active in the last 5 minutes."""
        cutoff = datetime.now() - timedelta(minutes=5)
        return [p for p in self.participants if p.last_active > cutoff]
    
    def get_recent_contributions(self, hours: int = 24) -> List[CollaborativeContribution]:
        """Get contributions from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [c for c in self.contributions if c.timestamp > cutoff]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'participants': [p.to_dict() for p in self.participants],
            'contributions': [c.to_dict() for c in self.contributions],
            'shared_insights': self.shared_insights,
            'knowledge_graph': self.knowledge_graph,
            'state': self.state.value,
            'tags': self.tags,
            'privacy_level': self.privacy_level
        }


class CollaborativeIntelligenceEngine:
    """
    Engine for real-time collaborative intelligence.
    
    Transforms individual intelligence amplification into collective wisdom 
    generation through real-time collaboration, shared knowledge spaces,
    and privacy-first insight sharing.
    """
    
    def __init__(self):
        """Initialize the collaborative intelligence engine."""
        self.invisible_ai = InvisibleAI()
        self.predictive_engine = PredictiveIntelligenceEngine()
        self.temporal_engine = TemporalIntelligenceEngine()
        
        # Collaborative state
        self.knowledge_clusters: Dict[str, SharedKnowledgeCluster] = {}
        self.active_sessions: Dict[str, Set[str]] = defaultdict(set)  # cluster_id -> participant_ids
        self.participant_connections: Dict[str, List[str]] = defaultdict(list)
        
        # Real-time event handling
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = deque(maxlen=1000)
        
        # Privacy and security
        self.anonymous_insights: List[Dict[str, Any]] = []
        self.insight_patterns: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.collaboration_metrics = {
            'total_clusters': 0,
            'active_sessions': 0,
            'insights_generated': 0,
            'connections_discovered': 0,
            'participants_served': 0
        }
        
        logger.info("Collaborative Intelligence Engine initialized")
    
    def create_knowledge_cluster(self, 
                                name: str, 
                                description: str, 
                                creator_id: str,
                                creator_name: str,
                                privacy_level: str = "team") -> SharedKnowledgeCluster:
        """
        Create a new shared knowledge cluster.
        
        Args:
            name: Name of the knowledge cluster
            description: Description of the cluster's purpose
            creator_id: ID of the creating user
            creator_name: Name of the creating user
            privacy_level: Privacy level (public, team, private)
            
        Returns:
            Created SharedKnowledgeCluster
        """
        cluster_id = f"cluster_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create admin participant
        admin = Participant(
            user_id=creator_id,
            name=creator_name,
            joined_at=datetime.now(),
            permission=CollaborationPermission.ADMIN
        )
        
        cluster = SharedKnowledgeCluster(
            id=cluster_id,
            name=name,
            description=description,
            created_by=creator_id,
            created_at=datetime.now(),
            participants=[admin],
            privacy_level=privacy_level
        )
        
        self.knowledge_clusters[cluster_id] = cluster
        self.collaboration_metrics['total_clusters'] += 1
        
        # Emit creation event
        self._emit_event('cluster_created', {
            'cluster_id': cluster_id,
            'cluster': cluster.to_dict(),
            'creator': admin.to_dict()
        })
        
        logger.info(f"Created knowledge cluster: {name} (ID: {cluster_id})")
        return cluster
    
    def join_knowledge_cluster(self, 
                              cluster_id: str, 
                              user_id: str, 
                              user_name: str,
                              permission: CollaborationPermission = CollaborationPermission.CONTRIBUTE) -> bool:
        """
        Join an existing knowledge cluster.
        
        Args:
            cluster_id: ID of the cluster to join
            user_id: ID of the joining user
            user_name: Name of the joining user
            permission: Permission level for the user
            
        Returns:
            True if successfully joined, False otherwise
        """
        if cluster_id not in self.knowledge_clusters:
            return False
        
        cluster = self.knowledge_clusters[cluster_id]
        
        # Check if user already in cluster
        existing_participant = next((p for p in cluster.participants if p.user_id == user_id), None)
        if existing_participant:
            existing_participant.last_active = datetime.now()
            return True
        
        # Add new participant
        participant = Participant(
            user_id=user_id,
            name=user_name,
            joined_at=datetime.now(),
            permission=permission
        )
        
        cluster.participants.append(participant)
        self.active_sessions[cluster_id].add(user_id)
        
        # Update metrics
        if user_id not in [p.user_id for c in self.knowledge_clusters.values() for p in c.participants]:
            self.collaboration_metrics['participants_served'] += 1
        
        # Emit join event
        self._emit_event('participant_joined', {
            'cluster_id': cluster_id,
            'participant': participant.to_dict(),
            'active_count': len(cluster.get_active_participants())
        })
        
        logger.info(f"User {user_name} joined cluster {cluster.name}")
        return True
    
    async def add_contribution(self, 
                              cluster_id: str, 
                              participant_id: str, 
                              content: str,
                              content_type: str = "text") -> Optional[CollaborativeContribution]:
        """
        Add a contribution to a collaborative knowledge cluster.
        
        Args:
            cluster_id: ID of the cluster
            participant_id: ID of the contributing participant
            content: Content of the contribution
            content_type: Type of content (text, image, audio, video)
            
        Returns:
            Created CollaborativeContribution or None if failed
        """
        if cluster_id not in self.knowledge_clusters:
            return None
        
        cluster = self.knowledge_clusters[cluster_id]
        participant = next((p for p in cluster.participants if p.user_id == participant_id), None)
        
        if not participant:
            return None
        
        # Create contribution
        contribution_id = f"contrib_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        contribution = CollaborativeContribution(
            id=contribution_id,
            participant_id=participant_id,
            content=content,
            content_type=content_type,
            timestamp=datetime.now()
        )
        
        # Process content using Invisible AI
        try:
            processing_result = self.invisible_ai.process_content(content)
            contribution.processed_result = processing_result
            contribution.confidence_score = processing_result.get('confidence', 0.0)
            
            # Extract insights and connections
            if 'insights' in processing_result:
                contribution.insights = [i.get('text', '') for i in processing_result['insights']]
            
            # Find connections to other contributions
            contribution.connections = await self._find_connections(cluster, contribution)
            
        except Exception as e:
            logger.error(f"Error processing contribution: {e}")
            contribution.confidence_score = 0.5  # Default for unprocessed content
        
        # Add to cluster
        cluster.contributions.append(contribution)
        participant.contributions += 1
        participant.last_active = datetime.now()
        
        # Generate team insights
        await self._generate_team_insights(cluster, contribution)
        
        # Update knowledge graph
        await self._update_collaborative_knowledge_graph(cluster, contribution)
        
        # Emit contribution event
        self._emit_event('contribution_added', {
            'cluster_id': cluster_id,
            'contribution': contribution.to_dict(),
            'participant': participant.to_dict(),
            'team_insights': cluster.shared_insights[-3:]  # Last 3 insights
        })
        
        logger.info(f"Added contribution to cluster {cluster.name} by {participant.name}")
        return contribution
    
    async def start_live_session(self, cluster_id: str, session_name: str) -> bool:
        """
        Start a live co-thinking session.
        
        Args:
            cluster_id: ID of the cluster for the session
            session_name: Name of the live session
            
        Returns:
            True if session started successfully
        """
        if cluster_id not in self.knowledge_clusters:
            return False
        
        cluster = self.knowledge_clusters[cluster_id]
        cluster.state = SessionState.ACTIVE
        
        # Initialize session metrics
        session_data = {
            'session_name': session_name,
            'started_at': datetime.now().isoformat(),
            'participants': len(cluster.get_active_participants()),
            'contributions_this_session': 0,
            'insights_this_session': 0
        }
        
        self.collaboration_metrics['active_sessions'] += 1
        
        # Emit session start event
        self._emit_event('live_session_started', {
            'cluster_id': cluster_id,
            'session_data': session_data,
            'cluster': cluster.to_dict()
        })
        
        logger.info(f"Started live session '{session_name}' for cluster {cluster.name}")
        return True
    
    async def _find_connections(self, 
                               cluster: SharedKnowledgeCluster, 
                               contribution: CollaborativeContribution) -> List[str]:
        """Find connections between this contribution and existing ones."""
        connections = []
        
        if not contribution.processed_result:
            return connections
        
        current_concepts = set()
        if 'concepts' in contribution.processed_result:
            current_concepts.update(contribution.processed_result['concepts'])
        
        # Find contributions with overlapping concepts
        for existing_contrib in cluster.contributions[-20:]:  # Last 20 contributions
            if existing_contrib.id == contribution.id:
                continue
                
            if existing_contrib.processed_result and 'concepts' in existing_contrib.processed_result:
                existing_concepts = set(existing_contrib.processed_result['concepts'])
                overlap = current_concepts.intersection(existing_concepts)
                
                if len(overlap) >= 2:  # At least 2 shared concepts
                    connections.append(existing_contrib.id)
        
        return connections[:5]  # Limit to 5 strongest connections
    
    async def _generate_team_insights(self, 
                                     cluster: SharedKnowledgeCluster, 
                                     new_contribution: CollaborativeContribution):
        """Generate insights from team collaboration."""
        try:
            # Analyze recent collaborative patterns
            recent_contributions = cluster.get_recent_contributions(hours=2)
            
            if len(recent_contributions) >= 3:  # Need multiple contributions
                # Combine recent content for analysis
                combined_content = "\n\n".join([
                    f"[{c.timestamp.strftime('%H:%M')}] {c.content}" 
                    for c in recent_contributions[-5:]
                ])
                
                # Use predictive intelligence to find team patterns
                team_analysis = await self._analyze_team_patterns(combined_content, cluster)
                
                if team_analysis and team_analysis.get('insights'):
                    for insight in team_analysis['insights']:
                        team_insight = f"Team insight: {insight['text']} (confidence: {insight.get('score', 0.0):.2f})"
                        cluster.shared_insights.append(team_insight)
                        
                        # Update participant who triggered insight
                        participant = next((p for p in cluster.participants 
                                          if p.user_id == new_contribution.participant_id), None)
                        if participant:
                            participant.insights_generated += 1
                
                # Limit insights to last 50
                cluster.shared_insights = cluster.shared_insights[-50:]
                
        except Exception as e:
            logger.error(f"Error generating team insights: {e}")
    
    async def _analyze_team_patterns(self, content: str, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Analyze patterns in team collaboration."""
        try:
            # Create team context
            team_context = {
                'participant_count': len(cluster.get_active_participants()),
                'thinking_styles': [p.thinking_style for p in cluster.participants],
                'recent_activity': len(cluster.get_recent_contributions(hours=1)),
                'cluster_focus': cluster.name
            }
            
            # Process with invisible AI for team dynamics
            result = self.invisible_ai.process_content(content, context=team_context)
            
            # Enhance with collaborative analysis
            if result and 'insights' in result:
                collaborative_insights = []
                for insight in result['insights']:
                    # Add team dimension to insights
                    enhanced_insight = {
                        'text': f"Team pattern: {insight.get('text', '')}",
                        'score': insight.get('score', 0.0),
                        'type': 'collaborative',
                        'participants_involved': len(cluster.get_active_participants())
                    }
                    collaborative_insights.append(enhanced_insight)
                
                result['insights'] = collaborative_insights
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing team patterns: {e}")
            return {}
    
    async def _update_collaborative_knowledge_graph(self, 
                                                   cluster: SharedKnowledgeCluster, 
                                                   contribution: CollaborativeContribution):
        """Update the collaborative knowledge graph."""
        try:
            if not contribution.processed_result:
                return
            
            # Initialize graph if needed
            if not cluster.knowledge_graph:
                cluster.knowledge_graph = {
                    'nodes': {},
                    'edges': [],
                    'clusters': [],
                    'participants': {}
                }
            
            # Add concepts as nodes
            if 'concepts' in contribution.processed_result:
                for concept in contribution.processed_result['concepts']:
                    if concept not in cluster.knowledge_graph['nodes']:
                        cluster.knowledge_graph['nodes'][concept] = {
                            'id': concept,
                            'count': 0,
                            'contributors': set(),
                            'first_mentioned': contribution.timestamp.isoformat()
                        }
                    
                    # Update node data
                    cluster.knowledge_graph['nodes'][concept]['count'] += 1
                    cluster.knowledge_graph['nodes'][concept]['contributors'].add(contribution.participant_id)
            
            # Add connections as edges
            for connection_id in contribution.connections:
                edge = {
                    'source': contribution.id,
                    'target': connection_id,
                    'weight': 1.0,
                    'created_at': contribution.timestamp.isoformat()
                }
                cluster.knowledge_graph['edges'].append(edge)
            
            # Convert sets to lists for serialization
            for node_data in cluster.knowledge_graph['nodes'].values():
                if 'contributors' in node_data and isinstance(node_data['contributors'], set):
                    node_data['contributors'] = list(node_data['contributors'])
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")
    
    def get_collaborative_insights(self, cluster_id: str) -> Dict[str, Any]:
        """Get collaborative insights for a cluster."""
        if cluster_id not in self.knowledge_clusters:
            return {}
        
        cluster = self.knowledge_clusters[cluster_id]
        active_participants = cluster.get_active_participants()
        recent_contributions = cluster.get_recent_contributions(hours=24)
        
        # Generate collaborative statistics
        insights = {
            'cluster_overview': {
                'name': cluster.name,
                'total_participants': len(cluster.participants),
                'active_participants': len(active_participants),
                'total_contributions': len(cluster.contributions),
                'recent_contributions': len(recent_contributions),
                'shared_insights': len(cluster.shared_insights)
            },
            'collaboration_patterns': self._analyze_collaboration_patterns(cluster),
            'knowledge_evolution': self._analyze_knowledge_evolution(cluster),
            'participant_dynamics': self._analyze_participant_dynamics(cluster),
            'breakthrough_indicators': self._detect_collaborative_breakthroughs(cluster)
        }
        
        return insights
    
    def _analyze_collaboration_patterns(self, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Analyze collaboration patterns in the cluster."""
        if not cluster.contributions:
            return {}
        
        # Time-based patterns
        contributions_by_hour = defaultdict(int)
        for contrib in cluster.contributions:
            hour = contrib.timestamp.hour
            contributions_by_hour[hour] += 1
        
        peak_hours = sorted(contributions_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Participant interaction patterns
        participant_interactions = defaultdict(int)
        for contrib in cluster.contributions:
            for connection_id in contrib.connections:
                # Find the participant who made the connected contribution
                connected_contrib = next((c for c in cluster.contributions if c.id == connection_id), None)
                if connected_contrib and connected_contrib.participant_id != contrib.participant_id:
                    pair = tuple(sorted([contrib.participant_id, connected_contrib.participant_id]))
                    participant_interactions[pair] += 1
        
        return {
            'peak_collaboration_hours': [hour for hour, count in peak_hours],
            'most_connected_pairs': dict(list(participant_interactions.items())[:5]),
            'average_connections_per_contribution': sum(len(c.connections) for c in cluster.contributions) / len(cluster.contributions),
            'collaboration_momentum': len(cluster.get_recent_contributions(hours=2))
        }
    
    def _analyze_knowledge_evolution(self, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Analyze how knowledge evolves in the cluster."""
        if not cluster.contributions:
            return {}
        
        # Track concept evolution over time
        concept_timeline = []
        concept_first_appearance = {}
        
        for contrib in sorted(cluster.contributions, key=lambda x: x.timestamp):
            if contrib.processed_result and 'concepts' in contrib.processed_result:
                for concept in contrib.processed_result['concepts']:
                    if concept not in concept_first_appearance:
                        concept_first_appearance[concept] = contrib.timestamp
                        concept_timeline.append({
                            'concept': concept,
                            'introduced_at': contrib.timestamp.isoformat(),
                            'introduced_by': contrib.participant_id
                        })
        
        # Identify concept clusters and evolution
        recent_concepts = []
        if cluster.contributions:
            for contrib in cluster.contributions[-10:]:  # Last 10 contributions
                if contrib.processed_result and 'concepts' in contrib.processed_result:
                    recent_concepts.extend(contrib.processed_result['concepts'])
        
        concept_frequency = defaultdict(int)
        for concept in recent_concepts:
            concept_frequency[concept] += 1
        
        emerging_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'concept_evolution_timeline': concept_timeline[-10:],  # Last 10 new concepts
            'emerging_concepts': emerging_concepts,
            'knowledge_depth_indicators': {
                'total_unique_concepts': len(concept_first_appearance),
                'concepts_introduced_today': len([c for c in concept_timeline 
                                                 if datetime.fromisoformat(c['introduced_at']).date() == datetime.now().date()])
            }
        }
    
    def _analyze_participant_dynamics(self, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Analyze participant dynamics and contributions."""
        if not cluster.participants:
            return {}
        
        # Participation analysis
        participant_stats = []
        for participant in cluster.participants:
            participant_contribs = [c for c in cluster.contributions if c.participant_id == participant.user_id]
            
            stats = {
                'participant_id': participant.user_id,
                'name': participant.name,
                'contributions': len(participant_contribs),
                'insights_generated': participant.insights_generated,
                'connections_discovered': participant.connections_discovered,
                'last_active': participant.last_active.isoformat(),
                'thinking_style': participant.thinking_style,
                'avg_confidence': sum(c.confidence_score for c in participant_contribs) / len(participant_contribs) if participant_contribs else 0.0
            }
            participant_stats.append(stats)
        
        # Sort by contribution activity
        participant_stats.sort(key=lambda x: x['contributions'], reverse=True)
        
        return {
            'most_active_contributors': participant_stats[:5],
            'thinking_style_distribution': {
                style: len([p for p in cluster.participants if p.thinking_style == style])
                for style in ['analytical', 'creative', 'systematic', 'exploratory', 'unknown']
            },
            'collaboration_health': self._calculate_collaboration_health(cluster)
        }
    
    def _calculate_collaboration_health(self, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Calculate health metrics for collaboration."""
        active_participants = cluster.get_active_participants()
        recent_contributions = cluster.get_recent_contributions(hours=6)
        
        # Distribution of contributions
        contribution_distribution = defaultdict(int)
        for contrib in recent_contributions:
            contribution_distribution[contrib.participant_id] += 1
        
        # Calculate balance (how evenly distributed contributions are)
        if contribution_distribution:
            values = list(contribution_distribution.values())
            balance_score = 1.0 - (max(values) - min(values)) / sum(values) if sum(values) > 0 else 0.0
        else:
            balance_score = 0.0
        
        return {
            'active_participants': len(active_participants),
            'recent_activity_level': len(recent_contributions),
            'contribution_balance': balance_score,
            'engagement_score': min(1.0, len(recent_contributions) / (len(active_participants) or 1) / 5),  # Normalize to 0-1
            'health_status': 'healthy' if len(active_participants) >= 2 and len(recent_contributions) >= 3 else 'needs_attention'
        }
    
    def _detect_collaborative_breakthroughs(self, cluster: SharedKnowledgeCluster) -> Dict[str, Any]:
        """Detect potential breakthrough moments in collaboration."""
        breakthrough_indicators = {
            'high_insight_density': False,
            'rapid_concept_introduction': False,
            'increased_connections': False,
            'synchronized_activity': False,
            'breakthrough_score': 0.0
        }
        
        recent_contributions = cluster.get_recent_contributions(hours=2)
        if len(recent_contributions) < 3:
            return breakthrough_indicators
        
        # High insight density
        recent_insights = [c for c in recent_contributions if c.insights]
        if len(recent_insights) / len(recent_contributions) > 0.6:
            breakthrough_indicators['high_insight_density'] = True
            breakthrough_indicators['breakthrough_score'] += 0.3
        
        # Rapid concept introduction
        unique_concepts = set()
        for contrib in recent_contributions:
            if contrib.processed_result and 'concepts' in contrib.processed_result:
                unique_concepts.update(contrib.processed_result['concepts'])
        
        if len(unique_concepts) > len(recent_contributions) * 2:  # More than 2 concepts per contribution
            breakthrough_indicators['rapid_concept_introduction'] = True
            breakthrough_indicators['breakthrough_score'] += 0.3
        
        # Increased connections
        avg_connections = sum(len(c.connections) for c in recent_contributions) / len(recent_contributions)
        if avg_connections > 2.0:
            breakthrough_indicators['increased_connections'] = True
            breakthrough_indicators['breakthrough_score'] += 0.2
        
        # Synchronized activity (multiple people contributing in short time)
        time_windows = defaultdict(set)
        for contrib in recent_contributions:
            window = contrib.timestamp.replace(minute=contrib.timestamp.minute // 10 * 10, second=0, microsecond=0)
            time_windows[window].add(contrib.participant_id)
        
        max_simultaneous = max(len(participants) for participants in time_windows.values()) if time_windows else 0
        if max_simultaneous >= 3:
            breakthrough_indicators['synchronized_activity'] = True
            breakthrough_indicators['breakthrough_score'] += 0.2
        
        return breakthrough_indicators
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        self.event_queue.append(event)
        
        # Call registered handlers
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for specific event types."""
        self.event_handlers[event_type].append(handler)
    
    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get overall collaboration metrics."""
        active_clusters = len([c for c in self.knowledge_clusters.values() 
                              if c.state == SessionState.ACTIVE])
        
        total_active_participants = sum(len(c.get_active_participants()) 
                                       for c in self.knowledge_clusters.values())
        
        total_recent_contributions = sum(len(c.get_recent_contributions(hours=24)) 
                                        for c in self.knowledge_clusters.values())
        
        self.collaboration_metrics.update({
            'active_clusters': active_clusters,
            'total_active_participants': total_active_participants,
            'recent_contributions_24h': total_recent_contributions,
            'avg_participants_per_cluster': total_active_participants / max(active_clusters, 1)
        })
        
        return self.collaboration_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'engine_status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_collaboration_metrics(),
            'active_clusters': [
                {
                    'id': cluster.id,
                    'name': cluster.name,
                    'participants': len(cluster.get_active_participants()),
                    'recent_activity': len(cluster.get_recent_contributions(hours=1))
                }
                for cluster in self.knowledge_clusters.values()
                if cluster.state == SessionState.ACTIVE
            ],
            'event_queue_size': len(self.event_queue),
            'registered_handlers': {event_type: len(handlers) 
                                   for event_type, handlers in self.event_handlers.items()}
        }


# Example usage and testing
if __name__ == "__main__":
    async def demo_collaborative_intelligence():
        """Demonstrate collaborative intelligence capabilities."""
        print("🤝 SUM Collaborative Intelligence Engine Demo")
        print("=" * 60)
        
        # Initialize engine
        collab_engine = CollaborativeIntelligenceEngine()
        
        # Create a knowledge cluster
        cluster = collab_engine.create_knowledge_cluster(
            name="AI Research Collaboration",
            description="Collaborative space for AI research and insights",
            creator_id="user_1",
            creator_name="Dr. Alice Smith",
            privacy_level="team"
        )
        
        print(f"✅ Created cluster: {cluster.name}")
        
        # Add participants
        collab_engine.join_knowledge_cluster(cluster.id, "user_2", "Bob Johnson")
        collab_engine.join_knowledge_cluster(cluster.id, "user_3", "Carol Wilson")
        
        print(f"✅ Added participants. Total: {len(cluster.participants)}")
        
        # Start live session
        await collab_engine.start_live_session(cluster.id, "Monday Research Session")
        print("✅ Started live co-thinking session")
        
        # Add collaborative contributions
        contributions = [
            ("user_1", "Machine learning models are showing breakthrough performance in natural language understanding, particularly with transformer architectures."),
            ("user_2", "The attention mechanism in transformers allows for better context understanding, which could revolutionize how we approach semantic analysis."),
            ("user_3", "I've been thinking about the implications for knowledge representation. These advances could enable more sophisticated knowledge graphs."),
            ("user_1", "Exactly! The combination of transformers and knowledge graphs could create powerful reasoning systems."),
            ("user_2", "We should explore how temporal dynamics affect these models. Time-aware AI could be the next frontier.")
        ]
        
        print("\n📝 Adding collaborative contributions...")
        for user_id, content in contributions:
            contrib = await collab_engine.add_contribution(cluster.id, user_id, content)
            if contrib:
                print(f"   ✅ {user_id}: Added contribution (confidence: {contrib.confidence_score:.2f})")
        
        # Get collaborative insights
        print("\n🧠 Generating collaborative insights...")
        insights = collab_engine.get_collaborative_insights(cluster.id)
        
        print(f"\n📊 Collaboration Overview:")
        overview = insights['cluster_overview']
        print(f"   • Total contributions: {overview['total_contributions']}")
        print(f"   • Active participants: {overview['active_participants']}")
        print(f"   • Shared insights: {overview['shared_insights']}")
        
        if 'collaboration_patterns' in insights:
            patterns = insights['collaboration_patterns']
            print(f"\n🔗 Collaboration Patterns:")
            print(f"   • Average connections per contribution: {patterns.get('average_connections_per_contribution', 0):.1f}")
            print(f"   • Current momentum: {patterns.get('collaboration_momentum', 0)} recent contributions")
        
        if 'breakthrough_indicators' in insights:
            breakthrough = insights['breakthrough_indicators']
            print(f"\n🚀 Breakthrough Indicators:")
            print(f"   • Breakthrough score: {breakthrough['breakthrough_score']:.2f}")
            print(f"   • High insight density: {breakthrough['high_insight_density']}")
            print(f"   • Rapid concept introduction: {breakthrough['rapid_concept_introduction']}")
        
        # Show system status
        print(f"\n🌟 System Status:")
        status = collab_engine.get_system_status()
        print(f"   • Engine status: {status['engine_status']}")
        print(f"   • Active clusters: {status['metrics']['active_clusters']}")
        print(f"   • Total active participants: {status['metrics']['total_active_participants']}")
        
        print(f"\n🎉 Collaborative Intelligence Demo Complete!")
        print("The future of collective wisdom generation is here! 🧠✨")
    
    # Run the demo
    asyncio.run(demo_collaborative_intelligence())