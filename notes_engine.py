#!/usr/bin/env python3
"""
Notes Engine - Intelligent Note-Taking with Auto-Tagging and Crystallization
===========================================================================

A deceptively simple note-taking system that leverages SUM's full AI capabilities
for automatic tagging, periodic distillation, and knowledge crystallization.

Because sometimes the most powerful features are the ones mentioned casually.

Author: SUM Revolutionary Team
License: Apache License 2.0
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import threading
import schedule
from enum import Enum

# Core SUM integration
from invisible_ai_engine import InvisibleAI
from predictive_intelligence import PredictiveIntelligenceEngine
from temporal_intelligence_engine import TemporalIntelligenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationPolicy(Enum):
    """Distillation policies for different note types."""
    NEVER = "never"                    # Never auto-distill (e.g., diary)
    SIZE_BASED = "size_based"          # When N related notes exist
    TIME_BASED = "time_based"          # Every X hours/days
    SMART = "smart"                    # AI decides based on content patterns
    MANUAL_ONLY = "manual_only"        # Only when user requests


class CrystallizationPolicy(Enum):
    """Crystallization policies for different note types."""
    NEVER = "never"                    # Never crystallize
    PATTERN_BASED = "pattern_based"    # When patterns emerge
    THRESHOLD_BASED = "threshold_based" # When frequency hits threshold
    INSIGHT_DRIVEN = "insight_driven"  # When breakthrough insights detected
    MANUAL_ONLY = "manual_only"        # Only when user requests


@dataclass
class NotePolicy:
    """Processing policy for different note types."""
    tag_pattern: str  # Tag pattern to match (e.g., "diary", "ideas", "meeting-*")
    distillation: DistillationPolicy
    crystallization: CrystallizationPolicy
    distill_threshold: int = 3           # For size-based distillation
    distill_frequency: str = "1hour"     # For time-based distillation
    crystallize_threshold: int = 5       # For threshold-based crystallization
    allow_insights: bool = True          # Allow analytical insights
    privacy_level: str = "personal"      # personal, shared, public


@dataclass
class Note:
    """A simple note with AI-powered enhancements."""
    id: str
    content: str
    created_at: datetime
    title: str = ""
    tags: List[str] = field(default_factory=list)
    auto_tags: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    importance: float = 0.0
    connections: List[str] = field(default_factory=list)
    distilled: bool = False
    crystallized: bool = False
    policy_tag: str = "general"         # Which policy applies to this note
    distillation_blocked: bool = False   # User explicitly blocked distillation
    crystallization_blocked: bool = False # User explicitly blocked crystallization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'title': self.title,
            'tags': self.tags,
            'auto_tags': self.auto_tags,
            'concepts': self.concepts,
            'importance': self.importance,
            'connections': self.connections,
            'distilled': self.distilled,
            'crystallized': self.crystallized,
            'policy_tag': self.policy_tag,
            'distillation_blocked': self.distillation_blocked,
            'crystallization_blocked': self.crystallization_blocked
        }


@dataclass
class CrystallizedKnowledge:
    """Crystallized wisdom from multiple notes."""
    id: str
    title: str
    essence: str
    source_notes: List[str]
    created_at: datetime
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'essence': self.essence,
            'source_notes': self.source_notes,
            'created_at': self.created_at.isoformat(),
            'key_insights': self.key_insights,
            'confidence': self.confidence
        }


class NotesEngine:
    """
    Deceptively simple notes that leverage SUM's full AI capabilities.
    
    Features:
    - Automatic tagging and concept extraction
    - Periodic distillation of related notes
    - Knowledge crystallization when patterns emerge
    - Seamless integration with all SUM intelligence
    """
    
    def __init__(self):
        """Initialize the notes engine."""
        self.invisible_ai = InvisibleAI()
        self.predictive_engine = PredictiveIntelligenceEngine()
        self.temporal_engine = TemporalIntelligenceEngine()
        
        # Storage
        self.notes: Dict[str, Note] = {}
        self.crystallized: Dict[str, CrystallizedKnowledge] = {}
        
        # Auto-processing
        self.processing_queue = []
        self.is_running = False
        
        # Note policies for different types
        self.note_policies: Dict[str, NotePolicy] = self._initialize_default_policies()
        
        # Schedule periodic operations with intelligent timing
        self._setup_intelligent_scheduling()
        
        self._start_background_processing()
        logger.info("Notes engine initialized - ready for effortless capture")
    
    def _initialize_default_policies(self) -> Dict[str, NotePolicy]:
        """Initialize default processing policies for different note types."""
        return {
            "ideas": NotePolicy(
                tag_pattern="ideas",
                distillation=DistillationPolicy.TIME_BASED,
                crystallization=CrystallizationPolicy.PATTERN_BASED,
                distill_frequency="2hours",
                distill_threshold=3,
                crystallize_threshold=5
            ),
            "diary": NotePolicy(
                tag_pattern="diary",
                distillation=DistillationPolicy.MANUAL_ONLY,
                crystallization=CrystallizationPolicy.NEVER,
                allow_insights=True,  # Still allow analytical insights
                privacy_level="personal"
            ),
            "meeting": NotePolicy(
                tag_pattern="meeting",
                distillation=DistillationPolicy.SIZE_BASED,
                crystallization=CrystallizationPolicy.THRESHOLD_BASED,
                distill_threshold=5,
                distill_frequency="1week",
                crystallize_threshold=10
            ),
            "research": NotePolicy(
                tag_pattern="research",
                distillation=DistillationPolicy.SMART,
                crystallization=CrystallizationPolicy.INSIGHT_DRIVEN,
                distill_threshold=4,
                crystallize_threshold=7
            ),
            "todo": NotePolicy(
                tag_pattern="todo",
                distillation=DistillationPolicy.SIZE_BASED,
                crystallization=CrystallizationPolicy.NEVER,
                distill_threshold=10
            ),
            "general": NotePolicy(
                tag_pattern="*",
                distillation=DistillationPolicy.TIME_BASED,
                crystallization=CrystallizationPolicy.PATTERN_BASED,
                distill_frequency="6hours",
                crystallize_threshold=5
            )
        }
    
    def _setup_intelligent_scheduling(self):
        """Setup intelligent scheduling based on policies."""
        # Schedule different distillation frequencies
        schedule.every(1).hours.do(lambda: self._distill_by_policy("1hour"))
        schedule.every(2).hours.do(lambda: self._distill_by_policy("2hours"))
        schedule.every(6).hours.do(lambda: self._distill_by_policy("6hours"))
        schedule.every().day.do(lambda: self._distill_by_policy("1day"))
        schedule.every().week.do(lambda: self._distill_by_policy("1week"))
        
        # Crystallization checks
        schedule.every(3).hours.do(self._crystallize_by_policies)
        
        # Smart distillation check (AI decides)
        schedule.every(30).minutes.do(self._smart_distillation_check)
    
    def add_note(self, content: str, title: str = "", tags: List[str] = None, policy_tag: str = None) -> Note:
        """
        Add a note - simple interface, powerful processing.
        
        Args:
            content: The note content
            title: Optional title (auto-generated if empty)
            tags: Optional manual tags
            policy_tag: Which policy to apply (auto-detected from tags if None)
            
        Returns:
            Created Note with AI enhancements
        """
        note_id = f"note_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        note = Note(
            id=note_id,
            content=content.strip(),
            created_at=datetime.now(),
            title=title or self._generate_title(content),
            tags=tags or [],
            policy_tag=policy_tag or self._detect_policy_tag(tags or [], content)
        )
        
        # Queue for AI processing
        self.processing_queue.append(note)
        self.notes[note_id] = note
        
        logger.info(f"Added note: {note.title[:50]}... [Policy: {note.policy_tag}]")
        return note
    
    def _detect_policy_tag(self, tags: List[str], content: str) -> str:
        """Auto-detect which policy applies to this note."""
        # Check explicit tags first
        for tag in tags:
            for policy_name, policy in self.note_policies.items():
                if policy.tag_pattern == tag or (policy.tag_pattern == "*" and policy_name == "general"):
                    return policy_name
        
        # Smart detection from content
        content_lower = content.lower()
        
        # Diary detection
        diary_indicators = ['today i', 'feeling', 'mood', 'personal', 'dear diary', 'this morning']
        if any(indicator in content_lower for indicator in diary_indicators):
            return "diary"
        
        # Meeting detection
        meeting_indicators = ['meeting', 'agenda', 'action items', 'discussed', 'attendees']
        if any(indicator in content_lower for indicator in meeting_indicators):
            return "meeting"
        
        # Research detection
        research_indicators = ['hypothesis', 'experiment', 'findings', 'study', 'analysis', 'research']
        if any(indicator in content_lower for indicator in research_indicators):
            return "research"
        
        # Ideas detection
        ideas_indicators = ['idea:', 'what if', 'brainstorm', 'concept', 'innovation', 'creative']
        if any(indicator in content_lower for indicator in ideas_indicators):
            return "ideas"
        
        # Todo detection
        todo_indicators = ['todo', 'task', 'need to', 'must do', 'action:', '[ ]', 'deadline']
        if any(indicator in content_lower for indicator in todo_indicators):
            return "todo"
        
        return "general"
    
    def _generate_title(self, content: str) -> str:
        """Generate a concise title from content."""
        try:
            result = self.invisible_ai.process_content(content[:200])  # First 200 chars
            if result and 'summary' in result:
                # Extract first sentence or key phrase
                title = result['summary'].split('.')[0].strip()
                return title[:60] if len(title) > 60 else title
        except:
            pass
        
        # Fallback: first few words
        words = content.split()[:8]
        return ' '.join(words) + ('...' if len(words) == 8 else '')
    
    def _start_background_processing(self):
        """Start background AI processing thread."""
        def process_continuously():
            self.is_running = True
            while self.is_running:
                # Process queued notes
                if self.processing_queue:
                    note = self.processing_queue.pop(0)
                    self._enhance_note_with_ai(note)
                
                # Run scheduled tasks
                schedule.run_pending()
                
                time.sleep(10)  # Check every 10 seconds
        
        thread = threading.Thread(target=process_continuously, daemon=True)
        thread.start()
    
    def _enhance_note_with_ai(self, note: Note):
        """Enhance note with AI-powered insights."""
        try:
            # Process with invisible AI
            result = self.invisible_ai.process_content(note.content)
            
            if result:
                # Extract concepts
                if 'concepts' in result:
                    note.concepts = result['concepts'][:10]  # Top 10 concepts
                
                # Generate auto-tags
                if 'keywords' in result:
                    auto_tags = [kw.lower().replace(' ', '-') for kw in result['keywords'][:5]]
                    note.auto_tags = auto_tags
                
                # Calculate importance
                note.importance = self._calculate_importance(note, result)
                
                # Find connections to other notes
                note.connections = self._find_note_connections(note)
                
                logger.debug(f"Enhanced note {note.id} with {len(note.concepts)} concepts")
                
        except Exception as e:
            logger.error(f"Error enhancing note {note.id}: {e}")
    
    def _calculate_importance(self, note: Note, ai_result: Dict[str, Any]) -> float:
        """Calculate note importance based on AI analysis."""
        importance = 0.0
        
        # Base importance from content length
        word_count = len(note.content.split())
        if word_count > 100:
            importance += 0.3
        elif word_count > 50:
            importance += 0.2
        else:
            importance += 0.1
        
        # Concept richness
        concept_count = len(note.concepts)
        importance += min(concept_count * 0.05, 0.2)
        
        # AI confidence
        confidence = ai_result.get('confidence', 0.5)
        importance += confidence * 0.3
        
        # Question or action items bonus
        if '?' in note.content:
            importance += 0.1
        if any(word in note.content.lower() for word in ['todo', 'action', 'must', 'should', 'need to']):
            importance += 0.2
        
        return min(importance, 1.0)
    
    def _find_note_connections(self, note: Note) -> List[str]:
        """Find connections to other notes based on concepts."""
        connections = []
        note_concepts = set(note.concepts)
        
        for other_note in self.notes.values():
            if other_note.id == note.id:
                continue
            
            other_concepts = set(other_note.concepts)
            overlap = note_concepts.intersection(other_concepts)
            
            if len(overlap) >= 2:  # At least 2 shared concepts
                connections.append(other_note.id)
        
        return connections[:5]  # Top 5 connections
    
    def _distill_by_policy(self, frequency: str):
        """Distill notes based on specific frequency policy."""
        logger.info(f"Running {frequency} distillation...")
        
        # Find policies that match this frequency
        applicable_policies = []
        for policy_name, policy in self.note_policies.items():
            if (policy.distillation == DistillationPolicy.TIME_BASED and 
                policy.distill_frequency == frequency):
                applicable_policies.append(policy_name)
        
        if not applicable_policies:
            return
        
        # Distill notes for each applicable policy
        for policy_name in applicable_policies:
            self._distill_notes_by_policy_type(policy_name)
    
    def _smart_distillation_check(self):
        """AI-driven smart distillation check."""
        logger.debug("Running smart distillation check...")
        
        for policy_name, policy in self.note_policies.items():
            if policy.distillation == DistillationPolicy.SMART:
                self._smart_distill_policy_type(policy_name)
    
    def _smart_distill_policy_type(self, policy_name: str):
        """Smart distillation for a specific policy type."""
        policy = self.note_policies[policy_name]
        policy_notes = [n for n in self.notes.values() 
                       if n.policy_tag == policy_name and not n.distilled and not n.distillation_blocked]
        
        if len(policy_notes) < 3:
            return
        
        # Use AI to detect if distillation would be valuable
        recent_notes = sorted(policy_notes, key=lambda x: x.created_at, reverse=True)[:10]
        combined_content = "\n\n".join([f"[{n.created_at.strftime('%m/%d %H:%M')}] {n.content[:200]}" for n in recent_notes])
        
        try:
            # Ask AI if these notes would benefit from distillation
            analysis_prompt = f"Analyze these {policy_name} notes. Would distilling them create valuable insights? Consider: concept clustering, pattern emergence, redundancy reduction."
            result = self.invisible_ai.process_content(f"{analysis_prompt}\n\n{combined_content}")
            
            if result and result.get('confidence', 0) > 0.7:
                # Check for distillation indicators in the AI response
                response_text = str(result.get('summary', '')).lower()
                distill_indicators = ['pattern', 'cluster', 'redundant', 'synthesize', 'consolidate', 'connect']
                
                if any(indicator in response_text for indicator in distill_indicators):
                    logger.info(f"Smart distillation triggered for {policy_name} notes")
                    self._distill_notes_by_policy_type(policy_name)
        
        except Exception as e:
            logger.error(f"Error in smart distillation check: {e}")
    
    def _distill_notes_by_policy_type(self, policy_name: str):
        """Distill notes for a specific policy type."""
        policy = self.note_policies[policy_name]
        
        # Get notes that match this policy and aren't already distilled
        policy_notes = [n for n in self.notes.values() 
                       if n.policy_tag == policy_name and not n.distilled and not n.distillation_blocked]
        
        if len(policy_notes) < policy.distill_threshold:
            return
        
        # Group by concept similarity for distillation
        concept_groups = defaultdict(list)
        
        for note in policy_notes:
            if not note.concepts:
                continue
            
            # Group by dominant concept
            primary_concept = note.concepts[0] if note.concepts else 'misc'
            concept_groups[primary_concept].append(note)
        
        # Distill groups that meet threshold
        for concept, notes in concept_groups.items():
            if len(notes) >= policy.distill_threshold:
                self._distill_note_group(concept, notes, policy_name)
    
    def _distill_related_notes(self):
        """Legacy method - now redirects to policy-based distillation."""
        logger.info("Running legacy distillation (redirecting to policy-based)...")
        self._distill_notes_by_policy_type("general")
        
        # Group notes by concept similarity
        concept_groups = defaultdict(list)
        
        for note in self.notes.values():
            if note.distilled or not note.concepts:
                continue
            
            # Group by dominant concept
            primary_concept = note.concepts[0] if note.concepts else 'misc'
            concept_groups[primary_concept].append(note)
        
        # Distill groups with multiple notes
        for concept, notes in concept_groups.items():
            if len(notes) >= 3:  # Need at least 3 notes to distill
                self._distill_note_group(concept, notes)
    
    def _distill_note_group(self, concept: str, notes: List[Note], policy_name: str = "general"):
        """Distill a group of related notes with policy awareness."""
        try:
            # Combine note content
            combined_content = "\n\n".join([
                f"Note from {note.created_at.strftime('%Y-%m-%d')}: {note.content}"
                for note in notes
            ])
            
            # Process with hierarchical engine
            result = self.invisible_ai.process_content(combined_content)
            
            if result and 'summary' in result:
                # Create distilled insight
                distilled_title = f"Distilled Insights: {concept.title()}"
                distilled_content = f"Key insights from {len(notes)} notes about {concept}:\n\n{result['summary']}"
                
                # Add as new note with appropriate policy
                distilled_note = self.add_note(
                    content=distilled_content,
                    title=distilled_title,
                    tags=['distilled', concept.lower(), policy_name],
                    policy_tag=policy_name
                )
                
                # Mark source notes as distilled
                for note in notes:
                    note.distilled = True
                
                logger.info(f"Distilled {len(notes)} notes about '{concept}' into: {distilled_title}")
                
        except Exception as e:
            logger.error(f"Error distilling notes for concept '{concept}': {e}")
    
    def _crystallize_by_policies(self):
        """Crystallize knowledge based on policies."""
        logger.info("Running policy-based crystallization...")
        
        for policy_name, policy in self.note_policies.items():
            if policy.crystallization != CrystallizationPolicy.NEVER:
                self._crystallize_policy_type(policy_name)
    
    def _crystallize_policy_type(self, policy_name: str):
        """Crystallize knowledge for a specific policy type."""
        policy = self.note_policies[policy_name]
        policy_notes = [n for n in self.notes.values() 
                       if n.policy_tag == policy_name and not n.crystallization_blocked]
        
        if len(policy_notes) < policy.crystallize_threshold:
            return
        
        # Different crystallization strategies based on policy
        if policy.crystallization == CrystallizationPolicy.PATTERN_BASED:
            self._crystallize_patterns(policy_notes, policy_name)
        elif policy.crystallization == CrystallizationPolicy.THRESHOLD_BASED:
            self._crystallize_by_threshold(policy_notes, policy_name, policy.crystallize_threshold)
        elif policy.crystallization == CrystallizationPolicy.INSIGHT_DRIVEN:
            self._crystallize_insights(policy_notes, policy_name)
    
    def _crystallize_patterns(self, notes: List[Note], policy_name: str):
        """Crystallize based on recurring patterns."""
        # Find patterns in note concepts and tags
        all_concepts = []
        all_tags = []
        
        for note in notes:
            all_concepts.extend(note.concepts)
            all_tags.extend(note.tags + note.auto_tags)
        
        # Find frequently recurring patterns
        concept_counts = Counter(all_concepts)
        tag_counts = Counter(all_tags)
        
        policy = self.note_policies[policy_name]
        
        # Crystallize high-frequency patterns
        for pattern, count in concept_counts.most_common(5):
            if count >= policy.crystallize_threshold and not self._already_crystallized(pattern, policy_name):
                self._create_crystallized_knowledge(pattern, 'concept', policy_name)
        
        for pattern, count in tag_counts.most_common(3):
            if count >= policy.crystallize_threshold + 2 and not self._already_crystallized(pattern, policy_name):
                self._create_crystallized_knowledge(pattern, 'tag', policy_name)
    
    def _crystallize_by_threshold(self, notes: List[Note], policy_name: str, threshold: int):
        """Simple threshold-based crystallization."""
        self._crystallize_patterns(notes, policy_name)  # Use pattern-based for now
    
    def _crystallize_insights(self, notes: List[Note], policy_name: str):
        """Crystallize based on breakthrough insights."""
        # Look for notes with high importance or many connections
        breakthrough_notes = [n for n in notes if n.importance > 0.7 or len(n.connections) > 3]
        
        if len(breakthrough_notes) >= 3:
            # Find common concepts in breakthrough notes
            breakthrough_concepts = []
            for note in breakthrough_notes:
                breakthrough_concepts.extend(note.concepts)
            
            concept_counts = Counter(breakthrough_concepts)
            
            # Crystallize breakthrough patterns
            for pattern, count in concept_counts.most_common(3):
                if count >= 2 and not self._already_crystallized(pattern, policy_name):
                    self._create_crystallized_knowledge(pattern, 'breakthrough', policy_name)
    
    def _crystallize_knowledge(self):
        """Legacy method - now redirects to policy-based crystallization."""
        logger.info("Running legacy crystallization (redirecting to policy-based)...")
        self._crystallize_by_policies()
        
        # Find patterns in note concepts and tags
        all_concepts = []
        all_tags = []
        
        for note in self.notes.values():
            all_concepts.extend(note.concepts)
            all_tags.extend(note.tags + note.auto_tags)
        
        # Find frequently recurring patterns
        concept_counts = Counter(all_concepts)
        tag_counts = Counter(all_tags)
        
        # Crystallize high-frequency patterns
        for pattern, count in concept_counts.most_common(5):
            if count >= 5 and not self._already_crystallized(pattern):
                self._create_crystallized_knowledge(pattern, 'concept')
        
        for pattern, count in tag_counts.most_common(3):
            if count >= 7 and not self._already_crystallized(pattern):
                self._create_crystallized_knowledge(pattern, 'tag')
    
    def _already_crystallized(self, pattern: str, policy_name: str = None) -> bool:
        """Check if pattern is already crystallized for this policy type."""
        for crystal in self.crystallized.values():
            if pattern.lower() in crystal.title.lower():
                # If policy-specific check, ensure it's the same policy
                if policy_name and hasattr(crystal, 'policy_tag'):
                    return crystal.policy_tag == policy_name
                return True
        return False
    
    def _create_crystallized_knowledge(self, pattern: str, pattern_type: str, policy_name: str = "general"):
        """Create crystallized knowledge from a pattern with policy awareness."""
        try:
            # Find all notes related to this pattern
            related_notes = []
            
            for note in self.notes.values():
                # Filter by policy type
                if note.policy_tag != policy_name:
                    continue
                    
                if pattern_type == 'concept' and pattern in note.concepts:
                    related_notes.append(note)
                elif pattern_type == 'tag' and pattern in (note.tags + note.auto_tags):
                    related_notes.append(note)
                elif pattern_type == 'breakthrough' and (note.importance > 0.7 or len(note.connections) > 3):
                    if pattern in note.concepts:
                        related_notes.append(note)
            
            if len(related_notes) < 3:
                return
            
            # Extract wisdom from related notes
            combined_wisdom = "\n\n".join([note.content for note in related_notes])
            
            # Process to extract essence
            result = self.invisible_ai.process_content(
                f"Extract the core wisdom and key insights from these notes about '{pattern}':\n\n{combined_wisdom}"
            )
            
            if result and 'summary' in result:
                crystal_id = f"crystal_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                crystal = CrystallizedKnowledge(
                    id=crystal_id,
                    title=f"Crystallized Wisdom: {pattern.title()} [{policy_name.title()}]",
                    essence=result['summary'],
                    source_notes=[note.id for note in related_notes],
                    created_at=datetime.now(),
                    key_insights=result.get('insights', [])[:3],
                    confidence=result.get('confidence', 0.8)
                )
                
                # Add policy tag to crystal for tracking
                crystal.policy_tag = policy_name
                
                self.crystallized[crystal_id] = crystal
                
                # Mark source notes as crystallized
                for note in related_notes:
                    note.crystallized = True
                
                logger.info(f"Crystallized knowledge: {crystal.title} from {len(related_notes)} {policy_name} notes")
                
        except Exception as e:
            logger.error(f"Error crystallizing pattern '{pattern}' for policy '{policy_name}': {e}")
    
    def search_notes(self, query: str) -> List[Note]:
        """Search notes by content, tags, or concepts."""
        query_lower = query.lower()
        matches = []
        
        for note in self.notes.values():
            # Search in content
            if query_lower in note.content.lower():
                matches.append(note)
            # Search in tags
            elif any(query_lower in tag.lower() for tag in note.tags + note.auto_tags):
                matches.append(note)
            # Search in concepts
            elif any(query_lower in concept.lower() for concept in note.concepts):
                matches.append(note)
        
        # Sort by relevance (importance and recency)
        matches.sort(key=lambda n: (n.importance, n.created_at.timestamp()), reverse=True)
        return matches
    
    def get_recent_notes(self, limit: int = 20) -> List[Note]:
        """Get recent notes."""
        sorted_notes = sorted(self.notes.values(), key=lambda n: n.created_at, reverse=True)
        return sorted_notes[:limit]
    
    def get_crystallized_knowledge(self) -> List[CrystallizedKnowledge]:
        """Get all crystallized knowledge."""
        return sorted(self.crystallized.values(), key=lambda c: c.created_at, reverse=True)
    
    def set_note_policy(self, note_id: str, policy_tag: str) -> bool:
        """Change the policy for a specific note."""
        if note_id not in self.notes or policy_tag not in self.note_policies:
            return False
        
        self.notes[note_id].policy_tag = policy_tag
        logger.info(f"Changed note {note_id} policy to {policy_tag}")
        return True
    
    def block_distillation(self, note_id: str) -> bool:
        """Block a note from being auto-distilled."""
        if note_id not in self.notes:
            return False
        
        self.notes[note_id].distillation_blocked = True
        logger.info(f"Blocked distillation for note {note_id}")
        return True
    
    def block_crystallization(self, note_id: str) -> bool:
        """Block a note from being crystallized."""
        if note_id not in self.notes:
            return False
        
        self.notes[note_id].crystallization_blocked = True
        logger.info(f"Blocked crystallization for note {note_id}")
        return True
    
    def manual_distill(self, policy_tag: str = None, concept: str = None) -> List[str]:
        """Manually trigger distillation for a policy or concept."""
        distilled_ids = []
        
        if policy_tag and policy_tag in self.note_policies:
            logger.info(f"Manual distillation triggered for {policy_tag} notes")
            old_count = len([n for n in self.notes.values() if n.distilled])
            self._distill_notes_by_policy_type(policy_tag)
            new_count = len([n for n in self.notes.values() if n.distilled])
            distilled_ids.extend([f"{new_count - old_count} notes distilled for {policy_tag}"])
        
        if concept:
            # Find notes with this concept
            concept_notes = [n for n in self.notes.values() 
                           if concept.lower() in [c.lower() for c in n.concepts] and not n.distilled]
            if len(concept_notes) >= 2:
                self._distill_note_group(concept, concept_notes)
                distilled_ids.append(f"Distilled {len(concept_notes)} notes about '{concept}'")
        
        return distilled_ids
    
    def manual_crystallize(self, policy_tag: str = None, pattern: str = None) -> List[str]:
        """Manually trigger crystallization for a policy or pattern."""
        crystallized_ids = []
        
        if policy_tag and policy_tag in self.note_policies:
            logger.info(f"Manual crystallization triggered for {policy_tag} notes")
            old_count = len(self.crystallized)
            self._crystallize_policy_type(policy_tag)
            new_count = len(self.crystallized)
            if new_count > old_count:
                crystallized_ids.append(f"{new_count - old_count} knowledge crystals created for {policy_tag}")
        
        if pattern:
            # Force crystallization of this pattern
            old_count = len(self.crystallized)
            self._create_crystallized_knowledge(pattern, 'manual', policy_tag or 'general')
            new_count = len(self.crystallized)
            if new_count > old_count:
                crystallized_ids.append(f"Crystallized knowledge about '{pattern}'")
        
        return crystallized_ids
    
    def get_analytical_insights(self, policy_tag: str = None, days: int = 30) -> Dict[str, Any]:
        """Get analytical insights without distillation (perfect for diary notes)."""
        # Filter notes by policy and timeframe
        cutoff = datetime.now() - timedelta(days=days)
        target_notes = []
        
        for note in self.notes.values():
            if note.created_at < cutoff:
                continue
            if policy_tag and note.policy_tag != policy_tag:
                continue
            target_notes.append(note)
        
        if not target_notes:
            return {'insights': [], 'patterns': [], 'trends': []}
        
        try:
            # Analyze without distilling
            combined_content = "\n\n".join([
                f"[{note.created_at.strftime('%m/%d')}] {note.content[:300]}"
                for note in sorted(target_notes, key=lambda x: x.created_at)
            ])
            
            analysis_prompt = f"""Analyze these {policy_tag or 'personal'} notes from the last {days} days. 
            Provide insights about patterns, trends, and themes WITHOUT summarizing or condensing the content.
            Focus on: emotional patterns, recurring themes, personal growth, behavioral insights."""
            
            result = self.invisible_ai.process_content(f"{analysis_prompt}\n\n{combined_content}")
            
            insights = {
                'total_notes_analyzed': len(target_notes),
                'date_range': f"{cutoff.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
                'policy_type': policy_tag or 'mixed',
                'insights': [],
                'patterns': [],
                'trends': [],
                'key_concepts': [],
                'emotional_indicators': []
            }
            
            if result:
                if 'insights' in result:
                    insights['insights'] = [i.get('text', str(i)) for i in result['insights'][:5]]
                if 'summary' in result:
                    # Parse AI response for different types of insights
                    summary = result['summary']
                    if 'pattern' in summary.lower():
                        insights['patterns'].append(summary)
                    if 'trend' in summary.lower():
                        insights['trends'].append(summary)
            
            # Add concept frequency analysis
            all_concepts = []
            for note in target_notes:
                all_concepts.extend(note.concepts)
            
            concept_counts = Counter(all_concepts)
            insights['key_concepts'] = [{'concept': c, 'frequency': f} 
                                      for c, f in concept_counts.most_common(10)]
            
            # Emotional analysis for diary-like content
            if policy_tag == 'diary':
                emotional_words = ['happy', 'sad', 'excited', 'worried', 'grateful', 'frustrated', 'peaceful', 'anxious']
                for word in emotional_words:
                    count = sum(1 for note in target_notes if word in note.content.lower())
                    if count > 0:
                        insights['emotional_indicators'].append({'emotion': word, 'frequency': count})
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating analytical insights: {e}")
            return {'error': str(e), 'insights': [], 'patterns': [], 'trends': []}
    
    def create_custom_policy(self, name: str, policy: NotePolicy) -> bool:
        """Create a custom note policy."""
        if name in self.note_policies:
            logger.warning(f"Policy {name} already exists, updating...")
        
        self.note_policies[name] = policy
        logger.info(f"Created/updated custom policy: {name}")
        return True
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about notes by policy."""
        stats = {}
        
        for policy_name in self.note_policies.keys():
            policy_notes = [n for n in self.notes.values() if n.policy_tag == policy_name]
            distilled_count = len([n for n in policy_notes if n.distilled])
            crystallized_count = len([n for n in policy_notes if n.crystallized])
            blocked_distill = len([n for n in policy_notes if n.distillation_blocked])
            blocked_crystal = len([n for n in policy_notes if n.crystallization_blocked])
            
            stats[policy_name] = {
                'total_notes': len(policy_notes),
                'distilled_notes': distilled_count,
                'crystallized_notes': crystallized_count,
                'blocked_distillation': blocked_distill,
                'blocked_crystallization': blocked_crystal,
                'recent_notes': len([n for n in policy_notes 
                                   if (datetime.now() - n.created_at).days <= 7])
            }
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notes statistics."""
        total_notes = len(self.notes)
        distilled_notes = len([n for n in self.notes.values() if n.distilled])
        crystallized_notes = len([n for n in self.notes.values() if n.crystallized])
        
        # Tag analysis
        all_tags = []
        for note in self.notes.values():
            all_tags.extend(note.tags + note.auto_tags)
        
        tag_counts = Counter(all_tags)
        
        return {
            'total_notes': total_notes,
            'distilled_notes': distilled_notes,
            'crystallized_notes': crystallized_notes,
            'crystallized_knowledge': len(self.crystallized),
            'average_importance': sum(n.importance for n in self.notes.values()) / max(total_notes, 1),
            'top_tags': dict(tag_counts.most_common(10)),
            'notes_this_week': len([n for n in self.notes.values() 
                                  if (datetime.now() - n.created_at).days <= 7])
        }


# Simple API for easy integration
class SimpleNotes:
    """Dead simple notes interface."""
    
    def __init__(self):
        self.engine = NotesEngine()
    
    def note(self, content: str, title: str = "", policy: str = None) -> str:
        """Add a note. Returns note ID."""
        note = self.engine.add_note(content, title, policy_tag=policy)
        return note.id
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search notes. Returns list of note dictionaries."""
        notes = self.engine.search_notes(query)
        return [note.to_dict() for note in notes]
    
    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notes."""
        notes = self.engine.get_recent_notes(limit)
        return [note.to_dict() for note in notes]
    
    def wisdom(self, policy: str = None) -> List[Dict[str, Any]]:
        """Get crystallized wisdom, optionally filtered by policy."""
        crystals = self.engine.get_crystallized_knowledge()
        if policy:
            crystals = [c for c in crystals if hasattr(c, 'policy_tag') and c.policy_tag == policy]
        return [crystal.to_dict() for crystal in crystals]
    
    def insights(self, policy: str = None, days: int = 30) -> Dict[str, Any]:
        """Get analytical insights without distillation."""
        return self.engine.get_analytical_insights(policy, days)
    
    def distill(self, policy: str = None, concept: str = None) -> List[str]:
        """Manually trigger distillation."""
        return self.engine.manual_distill(policy, concept)
    
    def crystallize(self, policy: str = None, pattern: str = None) -> List[str]:
        """Manually trigger crystallization."""
        return self.engine.manual_crystallize(policy, pattern)
    
    def block_auto_processing(self, note_id: str, distill: bool = True, crystallize: bool = True) -> bool:
        """Block automatic processing for a note."""
        success = True
        if distill:
            success &= self.engine.block_distillation(note_id)
        if crystallize:
            success &= self.engine.block_crystallization(note_id)
        return success
    
    def policy_stats(self) -> Dict[str, Any]:
        """Get statistics by policy type."""
        return self.engine.get_policy_stats()


# Example usage
if __name__ == "__main__":
    # Create notes engine
    notes = SimpleNotes()
    
    print("📝 SUM Notes Engine - Deceptively Simple, Surprisingly Intelligent")
    print("=" * 70)
    
    # Add some example notes
    example_notes = [
        "Machine learning models are becoming more efficient, but we need to focus on interpretability for real-world applications.",
        "The meeting today revealed that our current approach to user onboarding has a 40% drop-off rate. Need to redesign the flow.",
        "Interesting pattern: customers who use feature X are 3x more likely to upgrade. Should we promote this feature more?",
        "Research shows that collaborative AI performs better than individual AI on complex reasoning tasks. Team intelligence matters.",
        "TODO: Follow up on the budget discussion from yesterday's meeting. Finance wants projections by Friday."
    ]
    
    print("Adding example notes...")
    for i, content in enumerate(example_notes, 1):
        note_id = notes.note(content)
        print(f"✅ Added note {i}: {note_id}")
    
    # Show recent notes
    print(f"\n📋 Recent Notes:")
    recent = notes.recent(3)
    for note in recent:
        print(f"• {note['title']}")
        print(f"  Tags: {', '.join(note['auto_tags'][:3])}")
        print(f"  Importance: {note['importance']:.2f}")
        print()
    
    # Search example
    print("🔍 Searching for 'machine learning':")
    results = notes.search("machine learning")
    for result in results:
        print(f"• Found: {result['title']}")
    
    print(f"\n🎉 Notes engine ready!")
    print("Try: notes.note('Your thought here')")
    print("Or:  notes.search('keyword')")
    print("✨  Auto-tagging, distillation, and crystallization happen automatically!")