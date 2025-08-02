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

# Core SUM integration
from invisible_ai_engine import InvisibleAI
from predictive_intelligence import PredictiveIntelligenceEngine
from temporal_intelligence_engine import TemporalIntelligenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            'crystallized': self.crystallized
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
        
        # Schedule periodic operations
        schedule.every(1).hours.do(self._distill_related_notes)
        schedule.every(6).hours.do(self._crystallize_knowledge)
        
        self._start_background_processing()
        logger.info("Notes engine initialized - ready for effortless capture")
    
    def add_note(self, content: str, title: str = "", tags: List[str] = None) -> Note:
        """
        Add a note - simple interface, powerful processing.
        
        Args:
            content: The note content
            title: Optional title (auto-generated if empty)
            tags: Optional manual tags
            
        Returns:
            Created Note with AI enhancements
        """
        note_id = f"note_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        note = Note(
            id=note_id,
            content=content.strip(),
            created_at=datetime.now(),
            title=title or self._generate_title(content),
            tags=tags or []
        )
        
        # Queue for AI processing
        self.processing_queue.append(note)
        self.notes[note_id] = note
        
        logger.info(f"Added note: {note.title[:50]}...")
        return note
    
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
    
    def _distill_related_notes(self):
        """Periodically distill related notes into consolidated insights."""
        logger.info("Running periodic note distillation...")
        
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
    
    def _distill_note_group(self, concept: str, notes: List[Note]):
        """Distill a group of related notes."""
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
                
                # Add as new note
                distilled_note = self.add_note(
                    content=distilled_content,
                    title=distilled_title,
                    tags=['distilled', concept.lower()]
                )
                
                # Mark source notes as distilled
                for note in notes:
                    note.distilled = True
                
                logger.info(f"Distilled {len(notes)} notes about '{concept}' into: {distilled_title}")
                
        except Exception as e:
            logger.error(f"Error distilling notes for concept '{concept}': {e}")
    
    def _crystallize_knowledge(self):
        """Crystallize recurring patterns into permanent knowledge."""
        logger.info("Running knowledge crystallization...")
        
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
    
    def _already_crystallized(self, pattern: str) -> bool:
        """Check if pattern is already crystallized."""
        return any(pattern.lower() in crystal.title.lower() 
                  for crystal in self.crystallized.values())
    
    def _create_crystallized_knowledge(self, pattern: str, pattern_type: str):
        """Create crystallized knowledge from a pattern."""
        try:
            # Find all notes related to this pattern
            related_notes = []
            
            for note in self.notes.values():
                if pattern_type == 'concept' and pattern in note.concepts:
                    related_notes.append(note)
                elif pattern_type == 'tag' and pattern in (note.tags + note.auto_tags):
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
                    title=f"Crystallized Wisdom: {pattern.title()}",
                    essence=result['summary'],
                    source_notes=[note.id for note in related_notes],
                    created_at=datetime.now(),
                    key_insights=result.get('insights', [])[:3],
                    confidence=result.get('confidence', 0.8)
                )
                
                self.crystallized[crystal_id] = crystal
                
                # Mark source notes as crystallized
                for note in related_notes:
                    note.crystallized = True
                
                logger.info(f"Crystallized knowledge: {crystal.title}")
                
        except Exception as e:
            logger.error(f"Error crystallizing pattern '{pattern}': {e}")
    
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
    
    def note(self, content: str, title: str = "") -> str:
        """Add a note. Returns note ID."""
        note = self.engine.add_note(content, title)
        return note.id
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search notes. Returns list of note dictionaries."""
        notes = self.engine.search_notes(query)
        return [note.to_dict() for note in notes]
    
    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notes."""
        notes = self.engine.get_recent_notes(limit)
        return [note.to_dict() for note in notes]
    
    def wisdom(self) -> List[Dict[str, Any]]:
        """Get crystallized wisdom."""
        crystals = self.engine.get_crystallized_knowledge()
        return [crystal.to_dict() for crystal in crystals]


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