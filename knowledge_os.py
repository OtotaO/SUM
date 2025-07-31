#!/usr/bin/env python3
"""
knowledge_os.py - SUM Knowledge Operating System

The foundation of a cognitive amplification platform that makes thinking effortless
and insights profound. This system handles the complete information lifecycle:
capture, process, compress, and amplify.

Design Philosophy:
- Effortless capture: As natural as breathing
- Invisible intelligence: Magic happens behind the scenes
- Profound insights: Surface connections humans miss
- Intuitive prose: Every interaction feels conversational
- Joy of use: Both human and machine find beauty in the process

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from collections import defaultdict, Counter
import re

# Core SUM components
from SUM import HierarchicalDensificationEngine
from ollama_manager import OllamaManager, ProcessingRequest

# Configure beautiful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('KnowledgeOS')


@dataclass
class Thought:
    """A single captured thought - the atomic unit of knowledge."""
    id: str
    content: str
    timestamp: datetime
    source: str = "direct"  # direct, voice, import, email
    raw_content: str = ""   # Original unprocessed content
    
    # Automatic enrichment
    concepts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)  # IDs of connected thoughts
    importance: float = 0.0  # 0-1 score
    
    # Processing metadata
    processed: bool = False
    processing_time: float = 0.0
    word_count: int = 0
    
    def __post_init__(self):
        """Enrich thought with basic metadata."""
        if not self.raw_content:
            self.raw_content = self.content
        self.word_count = len(self.content.split())
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a beautiful, human-readable ID."""
        # Use timestamp + content hash for uniqueness with beauty
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:6]
        return f"thought_{timestamp_str}_{content_hash}"
    
    def to_prose(self) -> str:
        """Convert thought to beautiful prose for display."""
        time_str = self.timestamp.strftime("%B %d at %I:%M %p")
        
        prose = f"On {time_str}, you captured:\n\n"
        prose += f'"{self.content}"\n\n'
        
        if self.concepts:
            prose += f"Key concepts: {', '.join(self.concepts)}\n"
        
        if self.connections:
            prose += f"Connected to {len(self.connections)} other thoughts\n"
        
        return prose


@dataclass
class KnowledgeCluster:
    """A collection of related thoughts that have reached critical mass."""
    id: str
    name: str
    thoughts: List[str]  # Thought IDs
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Densification metadata
    original_word_count: int = 0
    compressed_word_count: int = 0
    compression_ratio: float = 0.0
    density_score: float = 0.0  # How much insight per word


@dataclass
class CaptureSession:
    """A user's thinking session - tracks the flow of thoughts."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    thoughts_captured: int = 0
    insights_generated: int = 0
    connections_discovered: int = 0
    session_summary: str = ""


class IntuitiveCaptureEngine:
    """
    The heart of effortless thought capture - makes writing feel like thinking.
    
    This engine handles the delicate art of capturing human thoughts without
    interrupting the flow of thinking. Every interaction is designed to feel
    natural and encourage deeper reflection.
    """
    
    def __init__(self):
        self.active_session = None
        self.capture_suggestions = [
            "What's on your mind?",
            "Share a thought...",
            "What are you discovering?",
            "Capture this moment...",
            "What insight just emerged?",
            "Tell me what you're thinking...",
            "What patterns do you notice?",
            "What question is forming?",
            "Describe what you're seeing...",
            "What connection just sparked?"
        ]
        
    def get_capture_prompt(self, context: Dict[str, Any] = None) -> str:
        """Generate an intuitive, contextual prompt for thought capture."""
        if not context:
            return self._random_gentle_prompt()
        
        # Context-aware prompts
        recent_thoughts = context.get('recent_thoughts', 0)
        time_of_day = datetime.now().hour
        last_topic = context.get('last_topic', '')
        
        if recent_thoughts == 0:
            if 5 <= time_of_day < 12:
                return "Good morning. What's emerging in your thinking today?"
            elif 12 <= time_of_day < 17:
                return "What insights are you gathering this afternoon?"
            elif 17 <= time_of_day < 22:
                return "What reflections from today are worth capturing?"
            else:
                return "What thoughts are keeping you up tonight?"
        
        elif recent_thoughts < 3:
            return "What else is unfolding in your mind?"
        
        elif last_topic:
            return f"Still thinking about {last_topic}? Or has your mind wandered somewhere new?"
        
        else:
            return "Your thoughts are flowing. What's next?"
    
    def _random_gentle_prompt(self) -> str:
        """Return a gentle, encouraging prompt."""
        import random
        return random.choice(self.capture_suggestions)
    
    def start_session(self) -> CaptureSession:
        """Begin a new thinking session."""
        session = CaptureSession(
            session_id=f"session_{int(time.time())}",
            start_time=datetime.now()
        )
        self.active_session = session
        logger.info(f"Started new capture session: {session.session_id}")
        return session
    
    def capture_thought(self, content: str, source: str = "direct") -> Thought:
        """Capture a thought with elegant processing."""
        if not content.strip():
            return None
        
        # Create the thought
        thought = Thought(
            content=content.strip(),
            timestamp=datetime.now(),
            source=source,
            id=""  # Will be auto-generated
        )
        
        # Update active session
        if self.active_session:
            self.active_session.thoughts_captured += 1
        
        logger.info(f"Captured thought: {thought.id}")
        return thought
    
    def end_session(self) -> Optional[CaptureSession]:
        """End the current thinking session with a beautiful summary."""
        if not self.active_session:
            return None
        
        self.active_session.end_time = datetime.now()
        session_duration = self.active_session.end_time - self.active_session.start_time
        
        # Generate session summary
        if self.active_session.thoughts_captured > 0:
            duration_str = f"{session_duration.total_seconds() / 60:.0f} minutes"
            self.active_session.session_summary = (
                f"In {duration_str}, you captured {self.active_session.thoughts_captured} thoughts, "
                f"discovered {self.active_session.connections_discovered} connections, "
                f"and generated {self.active_session.insights_generated} insights."
            )
        
        completed_session = self.active_session
        self.active_session = None
        
        logger.info(f"Session completed: {completed_session.session_summary}")
        return completed_session


class BackgroundIntelligenceEngine:
    """
    The invisible mind that processes thoughts automatically.
    
    This engine works silently in the background, finding patterns,
    making connections, and preparing insights without interrupting
    the user's flow of thought.
    """
    
    def __init__(self, hierarchical_engine: HierarchicalDensificationEngine):
        self.hierarchical_engine = hierarchical_engine
        self.processing_queue = []
        self.processing_thread = None
        self.is_processing = False
        
        # Intelligence patterns
        self.concept_memory = defaultdict(list)  # concept -> [thought_ids]
        self.connection_patterns = defaultdict(int)  # (concept1, concept2) -> strength
        self.temporal_patterns = defaultdict(list)  # hour -> [concepts]
        
        # Start background processing
        self._start_background_processing()
    
    def _start_background_processing(self):
        """Start the background intelligence processing thread."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_continuously, daemon=True)
        self.processing_thread.start()
        logger.info("Background intelligence engine started")
    
    def _process_continuously(self):
        """Continuously process thoughts in the background."""
        while self.is_processing:
            if self.processing_queue:
                thought = self.processing_queue.pop(0)
                try:
                    self._process_thought_deeply(thought)
                except Exception as e:
                    logger.error(f"Background processing error: {e}")
            
            time.sleep(1)  # Gentle processing cycle
    
    def enqueue_thought(self, thought: Thought):
        """Add a thought to the processing queue."""
        self.processing_queue.append(thought)
    
    def _process_thought_deeply(self, thought: Thought):
        """Perform deep processing on a single thought."""
        start_time = time.time()
        
        try:
            # Extract concepts using hierarchical engine
            result = self.hierarchical_engine.process_text(thought.content, {
                'max_concepts': 5,
                'max_insights': 3
            })
            
            # Enrich the thought
            if 'hierarchical_summary' in result:
                concepts = result['hierarchical_summary'].get('level_1_concepts', [])
                thought.concepts = [c.lower().replace(' ', '-') for c in concepts if c]
            
            # Generate intelligent tags
            thought.tags = self._generate_intelligent_tags(thought)
            
            # Calculate importance
            thought.importance = self._calculate_importance(thought, result)
            
            # Update memory patterns
            self._update_memory_patterns(thought)
            
            # Find connections
            thought.connections = self._find_connections(thought)
            
            thought.processed = True
            thought.processing_time = time.time() - start_time
            
            logger.debug(f"Processed {thought.id}: {len(thought.concepts)} concepts, "
                        f"{len(thought.connections)} connections")
            
        except Exception as e:
            logger.error(f"Error processing thought {thought.id}: {e}")
    
    def _generate_intelligent_tags(self, thought: Thought) -> List[str]:
        """Generate intelligent, meaningful tags for a thought."""
        tags = []
        
        # Time-based tags
        hour = thought.timestamp.hour
        if 5 <= hour < 12:
            tags.append('morning-thoughts')
        elif 12 <= hour < 17:
            tags.append('afternoon-insights')
        elif 17 <= hour < 22:
            tags.append('evening-reflections')
        else:
            tags.append('late-night-ideas')
        
        # Content-based tags
        content_lower = thought.content.lower()
        
        # Question detection
        if '?' in thought.content:
            tags.append('question')
        
        # Insight detection
        insight_words = ['realize', 'understand', 'notice', 'discover', 'insight', 'aha']
        if any(word in content_lower for word in insight_words):
            tags.append('insight')
        
        # Emotional content
        emotional_words = ['feel', 'emotion', 'excited', 'frustrated', 'happy', 'worried']
        if any(word in content_lower for word in emotional_words):
            tags.append('emotional')
        
        # Action items
        action_words = ['should', 'need to', 'must', 'will', 'plan to', 'going to']
        if any(phrase in content_lower for phrase in action_words):
            tags.append('action-item')
        
        return tags
    
    def _calculate_importance(self, thought: Thought, processing_result: Dict) -> float:
        """Calculate the importance score of a thought."""
        importance = 0.0
        
        # Base importance from content length and structure
        word_count = thought.word_count
        if word_count > 50:
            importance += 0.3
        elif word_count > 20:
            importance += 0.2
        else:
            importance += 0.1
        
        # Concept richness
        concept_count = len(thought.concepts)
        importance += min(concept_count * 0.1, 0.3)
        
        # Insights from hierarchical processing
        insights = processing_result.get('key_insights', [])
        if insights:
            avg_insight_score = sum(i.get('score', 0) for i in insights) / len(insights)
            importance += avg_insight_score * 0.4
        
        # Question bonus (questions are often important)
        if 'question' in thought.tags:
            importance += 0.2
        
        # Insight bonus
        if 'insight' in thought.tags:
            importance += 0.3
        
        return min(importance, 1.0)
    
    def _update_memory_patterns(self, thought: Thought):
        """Update the system's memory patterns with this thought."""
        # Update concept memory
        for concept in thought.concepts:
            self.concept_memory[concept].append(thought.id)
        
        # Update temporal patterns
        hour = thought.timestamp.hour
        self.temporal_patterns[hour].extend(thought.concepts)
        
        # Update connection patterns
        for i, concept1 in enumerate(thought.concepts):
            for j, concept2 in enumerate(thought.concepts[i+1:], i+1):
                pattern_key = tuple(sorted([concept1, concept2]))
                self.connection_patterns[pattern_key] += 1
    
    def _find_connections(self, thought: Thought) -> List[str]:
        """Find connections to other thoughts based on concepts and patterns."""
        connections = []
        
        # Find thoughts with overlapping concepts
        for concept in thought.concepts:
            if concept in self.concept_memory:
                related_thoughts = self.concept_memory[concept]
                # Add most recent related thoughts (up to 3)
                connections.extend(related_thoughts[-3:])
        
        # Remove duplicates and self-references
        connections = list(set(connections))
        if thought.id in connections:
            connections.remove(thought.id)
        
        return connections[:5]  # Limit to 5 strongest connections
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get a summary of the background intelligence state."""
        return {
            'concepts_tracked': len(self.concept_memory),
            'connection_patterns': len(self.connection_patterns),
            'processing_queue_size': len(self.processing_queue),
            'most_common_concepts': dict(Counter({
                concept: len(thoughts) 
                for concept, thoughts in self.concept_memory.items()
            }).most_common(10))
        }


class ThresholdDensificationEngine:
    """
    Intelligently compresses knowledge when it reaches cognitive limits.
    
    This engine monitors the user's knowledge accumulation and automatically
    triggers densification when information reaches a threshold that would
    overwhelm human cognitive capacity.
    """
    
    def __init__(self, hierarchical_engine: HierarchicalDensificationEngine):
        self.hierarchical_engine = hierarchical_engine
        
        # Configurable thresholds
        self.word_count_threshold = 2000  # Words per cluster
        self.thought_count_threshold = 15  # Thoughts per cluster
        self.time_threshold_days = 7      # Days before suggesting densification
        
        # Densification history
        self.densification_history = []
    
    def should_densify(self, thoughts: List[Thought], concept: str = None) -> Dict[str, Any]:
        """Determine if a collection of thoughts should be densified."""
        if not thoughts:
            return {'should_densify': False, 'reason': 'No thoughts to analyze'}
        
        total_words = sum(t.word_count for t in thoughts)
        thought_count = len(thoughts)
        oldest_thought = min(thoughts, key=lambda t: t.timestamp)
        age_days = (datetime.now() - oldest_thought.timestamp).days
        
        # Multiple threshold checks
        triggers = []
        
        if total_words > self.word_count_threshold:
            triggers.append(f"Word count ({total_words}) exceeds threshold ({self.word_count_threshold})")
        
        if thought_count > self.thought_count_threshold:
            triggers.append(f"Thought count ({thought_count}) exceeds threshold ({self.thought_count_threshold})")
        
        if age_days > self.time_threshold_days:
            triggers.append(f"Oldest thought is {age_days} days old (threshold: {self.time_threshold_days})")
        
        # Calculate density score
        unique_concepts = set()
        for thought in thoughts:
            unique_concepts.update(thought.concepts)
        
        concept_density = len(unique_concepts) / max(total_words / 100, 1)  # Concepts per 100 words
        
        should_densify = len(triggers) >= 1 and concept_density > 0.5
        
        return {
            'should_densify': should_densify,
            'triggers': triggers,
            'metrics': {
                'total_words': total_words,
                'thought_count': thought_count,
                'age_days': age_days,
                'unique_concepts': len(unique_concepts),
                'concept_density': concept_density
            },
            'suggestion': self._generate_densification_suggestion(triggers, concept or 'thoughts')
        }
    
    def _generate_densification_suggestion(self, triggers: List[str], concept: str) -> str:
        """Generate a beautiful, human suggestion for densification."""
        if not triggers:
            return ""
        
        suggestion = f"Your thoughts about {concept} are growing rich and complex. "
        
        if len(triggers) == 1:
            suggestion += "Consider distilling them into key insights to maintain clarity."
        else:
            suggestion += "It might be time to weave them together into a cohesive understanding."
        
        suggestion += f" This will preserve all the wisdom while making it easier to build upon."
        
        return suggestion
    
    def densify_thoughts(self, thoughts: List[Thought], cluster_name: str) -> KnowledgeCluster:
        """Perform intelligent densification on a collection of thoughts."""
        start_time = time.time()
        
        # Combine all thought content
        combined_content = "\n\n".join([
            f"Thought from {t.timestamp.strftime('%B %d')}: {t.content}"
            for t in thoughts
        ])
        
        # Process with hierarchical engine
        result = self.hierarchical_engine.process_text(combined_content, {
            'max_concepts': 10,
            'max_summary_tokens': 300,
            'max_insights': 8
        })
        
        # Extract summary and insights
        summary = result.get('hierarchical_summary', {}).get('level_2_core', '')
        insights = result.get('key_insights', [])
        key_insights = [insight['text'] for insight in insights if insight.get('score', 0) > 0.6]
        
        # Calculate compression metrics
        original_words = sum(t.word_count for t in thoughts)
        compressed_words = len(summary.split())
        compression_ratio = 1 - (compressed_words / max(original_words, 1))
        
        # Create knowledge cluster
        cluster = KnowledgeCluster(
            id=f"cluster_{int(time.time())}",
            name=cluster_name,
            thoughts=[t.id for t in thoughts],
            summary=summary,
            key_insights=key_insights,
            original_word_count=original_words,
            compressed_word_count=compressed_words,
            compression_ratio=compression_ratio,
            density_score=len(key_insights) / max(compressed_words / 100, 1)
        )
        
        # Record densification
        self.densification_history.append({
            'cluster_id': cluster.id,
            'timestamp': datetime.now(),
            'thoughts_processed': len(thoughts),
            'compression_ratio': compression_ratio,
            'processing_time': time.time() - start_time
        })
        
        logger.info(f"Densified {len(thoughts)} thoughts into cluster '{cluster_name}' "
                   f"with {compression_ratio:.1%} compression")
        
        return cluster


class KnowledgeOperatingSystem:
    """
    The main Knowledge OS that orchestrates all cognitive amplification.
    
    This is where the magic happens - thoughts flow in effortlessly,
    intelligence processes them invisibly, and profound insights emerge naturally.
    """
    
    def __init__(self, data_dir: str = "knowledge_os_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize core engines
        self.hierarchical_engine = HierarchicalDensificationEngine()
        self.capture_engine = IntuitiveCaptureEngine()
        self.intelligence_engine = BackgroundIntelligenceEngine(self.hierarchical_engine)
        self.densification_engine = ThresholdDensificationEngine(self.hierarchical_engine)
        
        # Storage
        self.db_path = self.data_dir / "knowledge.db"
        self._init_database()
        
        # In-memory stores for active session
        self.active_thoughts = {}  # id -> Thought
        self.knowledge_clusters = {}  # id -> KnowledgeCluster
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"Knowledge OS initialized with {len(self.active_thoughts)} thoughts")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS thoughts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT DEFAULT 'direct',
                    concepts TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    connections TEXT DEFAULT '[]',
                    importance REAL DEFAULT 0.0,
                    processed BOOLEAN DEFAULT FALSE,
                    raw_content TEXT DEFAULT ''
                );
                
                CREATE TABLE IF NOT EXISTS knowledge_clusters (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    thoughts TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    key_insights TEXT DEFAULT '[]',
                    created TEXT NOT NULL,
                    original_word_count INTEGER DEFAULT 0,
                    compressed_word_count INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 0.0
                );
                
                CREATE TABLE IF NOT EXISTS capture_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    thoughts_captured INTEGER DEFAULT 0,
                    insights_generated INTEGER DEFAULT 0,
                    connections_discovered INTEGER DEFAULT 0,
                    session_summary TEXT DEFAULT ''
                );
            """)
    
    def _load_existing_data(self):
        """Load existing thoughts and clusters from database."""
        with sqlite3.connect(self.db_path) as conn:
            # Load thoughts
            cursor = conn.execute("SELECT * FROM thoughts ORDER BY timestamp DESC LIMIT 1000")
            for row in cursor.fetchall():
                thought_data = dict(zip([col[0] for col in cursor.description], row))
                
                # Parse JSON fields
                thought_data['concepts'] = json.loads(thought_data.get('concepts', '[]'))
                thought_data['tags'] = json.loads(thought_data.get('tags', '[]'))
                thought_data['connections'] = json.loads(thought_data.get('connections', '[]'))
                thought_data['timestamp'] = datetime.fromisoformat(thought_data['timestamp'])
                
                thought = Thought(**thought_data)
                self.active_thoughts[thought.id] = thought
            
            # Load clusters
            cursor = conn.execute("SELECT * FROM knowledge_clusters ORDER BY created DESC")
            for row in cursor.fetchall():
                cluster_data = dict(zip([col[0] for col in cursor.description], row))
                
                # Parse JSON fields
                cluster_data['thoughts'] = json.loads(cluster_data.get('thoughts', '[]'))
                cluster_data['key_insights'] = json.loads(cluster_data.get('key_insights', '[]'))
                cluster_data['created'] = datetime.fromisoformat(cluster_data['created'])
                cluster_data['last_updated'] = cluster_data['created']  # Set default
                
                cluster = KnowledgeCluster(**cluster_data)
                self.knowledge_clusters[cluster.id] = cluster
    
    def capture_thought(self, content: str, source: str = "direct") -> Thought:
        """Capture a thought with full system integration."""
        thought = self.capture_engine.capture_thought(content, source)
        if not thought:
            return None
        
        # Store in active memory
        self.active_thoughts[thought.id] = thought
        
        # Queue for background processing
        self.intelligence_engine.enqueue_thought(thought)
        
        # Persist to database
        self._save_thought(thought)
        
        return thought
    
    def _save_thought(self, thought: Thought):
        """Save thought to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO thoughts 
                (id, content, timestamp, source, concepts, tags, connections, 
                 importance, processed, raw_content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thought.id,
                thought.content,
                thought.timestamp.isoformat(),
                thought.source,
                json.dumps(thought.concepts),
                json.dumps(thought.tags),
                json.dumps(thought.connections),
                thought.importance,
                thought.processed,
                thought.raw_content
            ))
    
    def get_capture_prompt(self) -> str:
        """Get an intuitive prompt for thought capture."""
        context = {
            'recent_thoughts': len([t for t in self.active_thoughts.values() 
                                  if (datetime.now() - t.timestamp).seconds < 3600]),
            'last_topic': self._get_recent_topic()
        }
        return self.capture_engine.get_capture_prompt(context)
    
    def _get_recent_topic(self) -> str:
        """Get the most recent topic/concept being explored."""
        recent_thoughts = sorted(
            self.active_thoughts.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )[:5]
        
        all_concepts = []
        for thought in recent_thoughts:
            all_concepts.extend(thought.concepts)
        
        if all_concepts:
            concept_counts = Counter(all_concepts)
            return concept_counts.most_common(1)[0][0].replace('-', ' ')
        
        return ""
    
    def check_densification_opportunities(self) -> List[Dict[str, Any]]:
        """Check for thoughts that should be densified."""
        opportunities = []
        
        # Group thoughts by dominant concept
        concept_groups = defaultdict(list)
        for thought in self.active_thoughts.values():
            if thought.concepts:
                # Use the first (most important) concept as the grouping key
                primary_concept = thought.concepts[0]
                concept_groups[primary_concept].append(thought)
        
        # Check each group for densification opportunities
        for concept, thoughts in concept_groups.items():
            if len(thoughts) >= 3:  # Minimum thoughts for meaningful densification
                analysis = self.densification_engine.should_densify(thoughts, concept)
                if analysis['should_densify']:
                    opportunities.append({
                        'concept': concept,
                        'thoughts': thoughts,
                        'analysis': analysis
                    })
        
        return opportunities
    
    def densify_concept(self, concept: str) -> Optional[KnowledgeCluster]:
        """Densify all thoughts related to a specific concept."""
        # Find all thoughts with this concept
        related_thoughts = [
            thought for thought in self.active_thoughts.values()
            if concept in thought.concepts
        ]
        
        if len(related_thoughts) < 2:
            return None
        
        # Create beautiful cluster name
        cluster_name = concept.replace('-', ' ').title()
        
        # Perform densification
        cluster = self.densification_engine.densify_thoughts(related_thoughts, cluster_name)
        
        # Store cluster
        self.knowledge_clusters[cluster.id] = cluster
        self._save_cluster(cluster)
        
        return cluster
    
    def _save_cluster(self, cluster: KnowledgeCluster):
        """Save knowledge cluster to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_clusters
                (id, name, thoughts, summary, key_insights, created,
                 original_word_count, compressed_word_count, compression_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.id,
                cluster.name,
                json.dumps(cluster.thoughts),
                cluster.summary,
                json.dumps(cluster.key_insights),
                cluster.created.isoformat(),
                cluster.original_word_count,
                cluster.compressed_word_count,
                cluster.compression_ratio
            ))
    
    def get_recent_thoughts(self, limit: int = 10) -> List[Thought]:
        """Get the most recent thoughts."""
        return sorted(
            self.active_thoughts.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )[:limit]
    
    def search_thoughts(self, query: str) -> List[Thought]:
        """Search thoughts by content or concepts."""
        query_lower = query.lower()
        matches = []
        
        for thought in self.active_thoughts.values():
            # Search in content
            if query_lower in thought.content.lower():
                matches.append(thought)
            # Search in concepts
            elif any(query_lower in concept.lower() for concept in thought.concepts):
                matches.append(thought)
        
        # Sort by relevance (exact matches first, then by importance)
        matches.sort(key=lambda t: (
            -int(query_lower in t.content.lower()),
            -t.importance,
            -t.timestamp.timestamp()
        ))
        
        return matches
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get profound insights about the user's thinking patterns."""
        thoughts = list(self.active_thoughts.values())
        
        if not thoughts:
            return {'message': 'Begin capturing thoughts to discover insights about your thinking patterns.'}
        
        # Temporal patterns
        hour_distribution = defaultdict(int)
        concept_evolution = defaultdict(list)
        
        for thought in thoughts:
            hour_distribution[thought.timestamp.hour] += 1
            
            for concept in thought.concepts:
                concept_evolution[concept].append(thought.timestamp)
        
        # Find peak thinking hours
        peak_hour = max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else 12
        
        # Find evolving concepts
        evolving_concepts = []
        for concept, timestamps in concept_evolution.items():
            if len(timestamps) >= 3:
                span_days = (max(timestamps) - min(timestamps)).days
                if span_days > 0:
                    evolving_concepts.append((concept, len(timestamps), span_days))
        
        evolving_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Generate beautiful insights
        insights = {
            'thinking_patterns': {
                'peak_hour': peak_hour,
                'peak_description': self._describe_peak_hour(peak_hour),
                'total_thoughts': len(thoughts),
                'processed_thoughts': len([t for t in thoughts if t.processed]),
                'average_importance': sum(t.importance for t in thoughts) / len(thoughts)
            },
            'concept_evolution': [
                {
                    'concept': concept.replace('-', ' ').title(),
                    'frequency': frequency,
                    'span_days': span_days,
                    'description': f"You've been exploring {concept.replace('-', ' ')} for {span_days} days with {frequency} thoughts"
                }
                for concept, frequency, span_days in evolving_concepts[:5]
            ],
            'intelligence_summary': self.intelligence_engine.get_intelligence_summary(),
            'beautiful_summary': self._generate_beautiful_summary(thoughts)
        }
        
        return insights
    
    def _describe_peak_hour(self, hour: int) -> str:
        """Generate a beautiful description of peak thinking time."""
        if 5 <= hour < 9:
            return f"Your mind is most active in the early morning around {hour}:00 AM - a time of fresh perspective and clear thinking."
        elif 9 <= hour < 12:
            return f"Your peak thinking happens mid-morning around {hour}:00 AM - when focus and creativity intersect beautifully."
        elif 12 <= hour < 17:
            return f"Your thoughts flow strongest in the afternoon around {hour}:00 PM - building on the day's momentum."
        elif 17 <= hour < 22:
            return f"Your most profound thinking emerges in the evening around {hour}:00 PM - a time of reflection and synthesis."
        else:
            return f"Your mind comes alive in the quiet hours around {hour}:00 - when the world sleeps, your thoughts awaken."
    
    def _generate_beautiful_summary(self, thoughts: List[Thought]) -> str:
        """Generate a beautiful, poetic summary of the user's thinking journey."""
        if not thoughts:
            return "Your thinking journey awaits..."
        
        total_words = sum(t.word_count for t in thoughts)
        unique_concepts = set()
        for thought in thoughts:
            unique_concepts.update(thought.concepts)
        
        oldest = min(thoughts, key=lambda t: t.timestamp)
        newest = max(thoughts, key=lambda t: t.timestamp)
        
        days_span = (newest.timestamp - oldest.timestamp).days + 1
        
        return (f"Over {days_span} days, you've woven {len(thoughts)} thoughts into "
                f"{total_words:,} words, exploring {len(unique_concepts)} unique concepts. "
                f"Your mind has been a garden of ideas, each thought a seed that connects "
                f"to create something greater than the sum of its parts.")


# Beautiful CLI interface for testing
def create_beautiful_cli():
    """Create a beautiful command-line interface for the Knowledge OS."""
    print("\n" + "="*60)
    print("🧠 Welcome to the Knowledge Operating System")
    print("   Where thoughts become wisdom")
    print("="*60)
    
    # Initialize the system
    knowledge_os = KnowledgeOperatingSystem()
    
    print(f"\n{knowledge_os.get_capture_prompt()}")
    
    while True:
        try:
            # Get user input
            user_input = input("\n💭 ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n✨ Thank you for thinking with us. Your insights are preserved.")
                break
            
            elif user_input.lower() in ['insights', 'summary']:
                insights = knowledge_os.get_system_insights()
                print(f"\n📊 {insights['beautiful_summary']}")
                continue
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                results = knowledge_os.search_thoughts(query)
                print(f"\n🔍 Found {len(results)} thoughts about '{query}':")
                for i, thought in enumerate(results[:3], 1):
                    print(f"  {i}. {thought.content[:100]}...")
                continue
            
            elif user_input.lower() in ['densify', 'compress']:
                opportunities = knowledge_os.check_densification_opportunities()
                if opportunities:
                    print(f"\n🎯 Found {len(opportunities)} densification opportunities:")
                    for opp in opportunities[:3]:
                        print(f"  • {opp['concept'].replace('-', ' ').title()}: {len(opp['thoughts'])} thoughts")
                        print(f"    {opp['analysis']['suggestion']}")
                else:
                    print("\n✨ Your thoughts are already well-organized. Keep capturing!")
                continue
            
            # Capture the thought
            thought = knowledge_os.capture_thought(user_input)
            
            if thought:
                print(f"✨ Captured: {thought.id}")
                
                # Show a new prompt
                print(f"\n{knowledge_os.get_capture_prompt()}")
            
        except KeyboardInterrupt:
            print("\n\n✨ Thank you for thinking with us. Your insights are preserved.")
            break
        except Exception as e:
            print(f"\n❌ Something went wrong: {e}")
            continue


if __name__ == "__main__":
    create_beautiful_cli()