#!/usr/bin/env python3
"""
test_notes_engine.py - Comprehensive Unit Tests for Notes Engine

Tests all aspects of the intelligent notes system including policies,
distillation, crystallization, and analytical insights.

Author: SUM Development Team
License: Apache License 2.0
"""

import unittest
import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import Mock, patch, MagicMock

from notes_engine import (
    Note, CrystallizedKnowledge, NotesEngine, SimpleNotes,
    DistillationPolicy, CrystallizationPolicy, NotePolicy
)


class TestNoteDataStructures(unittest.TestCase):
    """Test Note and CrystallizedKnowledge data structures."""
    
    def test_note_creation(self):
        """Test basic note creation."""
        note = Note(
            id="test_123",
            content="Test content",
            created_at=datetime.now(),
            title="Test Note",
            tags=["test"],
            policy_tag="general"
        )
        
        self.assertEqual(note.id, "test_123")
        self.assertEqual(note.content, "Test content")
        self.assertEqual(note.title, "Test Note")
        self.assertEqual(note.tags, ["test"])
        self.assertEqual(note.policy_tag, "general")
        self.assertFalse(note.distilled)
        self.assertFalse(note.crystallized)
    
    def test_note_to_dict(self):
        """Test note serialization."""
        note = Note(
            id="test_123",
            content="Test content",
            created_at=datetime.now(),
            title="Test Note"
        )
        
        note_dict = note.to_dict()
        self.assertIn('id', note_dict)
        self.assertIn('content', note_dict)
        self.assertIn('created_at', note_dict)
        self.assertIn('policy_tag', note_dict)
    
    def test_crystallized_knowledge_creation(self):
        """Test crystallized knowledge creation."""
        crystal = CrystallizedKnowledge(
            id="crystal_123",
            title="Test Crystal",
            essence="Core wisdom",
            source_notes=["note_1", "note_2"],
            created_at=datetime.now(),
            confidence=0.9
        )
        
        self.assertEqual(crystal.id, "crystal_123")
        self.assertEqual(crystal.essence, "Core wisdom")
        self.assertEqual(len(crystal.source_notes), 2)
        self.assertEqual(crystal.confidence, 0.9)


class TestNotesEngine(unittest.TestCase):
    """Test the NotesEngine core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the AI engines to avoid dependencies
        with patch('notes_engine.InvisibleAI'), \
             patch('notes_engine.PredictiveIntelligenceEngine'), \
             patch('notes_engine.TemporalIntelligenceEngine'):
            self.engine = NotesEngine()
        
        # Mock the AI processing
        self.engine.invisible_ai.process_content = Mock(return_value={
            'summary': 'Test summary',
            'concepts': ['concept1', 'concept2'],
            'keywords': ['keyword1', 'keyword2'],
            'confidence': 0.8,
            'insights': [{'text': 'Test insight'}]
        })
    
    def test_add_note_basic(self):
        """Test adding a basic note."""
        note = self.engine.add_note("Test content", "Test Title")
        
        self.assertIsNotNone(note)
        self.assertEqual(note.content, "Test content")
        self.assertEqual(note.title, "Test Title")
        self.assertIn(note.id, self.engine.notes)
    
    def test_add_note_with_policy(self):
        """Test adding notes with different policies."""
        diary_note = self.engine.add_note(
            "Today I felt happy", 
            policy_tag="diary"
        )
        self.assertEqual(diary_note.policy_tag, "diary")
        
        ideas_note = self.engine.add_note(
            "What if we could fly?",
            policy_tag="ideas"
        )
        self.assertEqual(ideas_note.policy_tag, "ideas")
    
    def test_policy_detection(self):
        """Test automatic policy detection from content."""
        # Diary detection
        diary_content = "Today I woke up feeling grateful for my family"
        policy = self.engine._detect_policy_tag([], diary_content)
        self.assertEqual(policy, "diary")
        
        # Meeting detection
        meeting_content = "Meeting agenda: discuss Q1 targets with team"
        policy = self.engine._detect_policy_tag([], meeting_content)
        self.assertEqual(policy, "meeting")
        
        # Research detection
        research_content = "Our hypothesis is that neural networks can learn"
        policy = self.engine._detect_policy_tag([], research_content)
        self.assertEqual(policy, "research")
        
        # Ideas detection
        ideas_content = "What if we created a new type of interface?"
        policy = self.engine._detect_policy_tag([], ideas_content)
        self.assertEqual(policy, "ideas")
        
        # Todo detection
        todo_content = "TODO: finish the report by Friday"
        policy = self.engine._detect_policy_tag([], todo_content)
        self.assertEqual(policy, "todo")
    
    def test_search_notes(self):
        """Test searching notes."""
        # Add some notes
        note1 = self.engine.add_note("Machine learning is fascinating")
        note2 = self.engine.add_note("Deep learning with neural networks")
        note3 = self.engine.add_note("Today I learned about transformers")
        
        # Search by content
        results = self.engine.search_notes("learning")
        self.assertEqual(len(results), 2)
        
        # Search by concepts (mocked)
        note1.concepts = ["machine-learning", "AI"]
        results = self.engine.search_notes("AI")
        self.assertEqual(len(results), 1)
    
    def test_block_distillation(self):
        """Test blocking distillation for specific notes."""
        note = self.engine.add_note("Private thoughts")
        
        # Block distillation
        success = self.engine.block_distillation(note.id)
        self.assertTrue(success)
        self.assertTrue(self.engine.notes[note.id].distillation_blocked)
    
    def test_block_crystallization(self):
        """Test blocking crystallization for specific notes."""
        note = self.engine.add_note("Temporary note")
        
        # Block crystallization
        success = self.engine.block_crystallization(note.id)
        self.assertTrue(success)
        self.assertTrue(self.engine.notes[note.id].crystallization_blocked)
    
    def test_manual_distill(self):
        """Test manual distillation trigger."""
        # Add notes for a specific policy
        for i in range(5):
            self.engine.add_note(f"Research note {i}", policy_tag="research")
        
        # Manual distill
        results = self.engine.manual_distill(policy_tag="research")
        self.assertIsInstance(results, list)
    
    def test_get_analytical_insights(self):
        """Test getting insights without distillation."""
        # Add diary notes
        for i in range(3):
            note = self.engine.add_note(
                f"Today I feel excited about the progress",
                policy_tag="diary"
            )
            note.created_at = datetime.now() - timedelta(days=i)
        
        # Get insights
        insights = self.engine.get_analytical_insights(
            policy_tag="diary",
            days=7
        )
        
        self.assertIn('total_notes_analyzed', insights)
        self.assertIn('policy_type', insights)
        self.assertEqual(insights['policy_type'], 'diary')
        self.assertIn('emotional_indicators', insights)
    
    def test_create_custom_policy(self):
        """Test creating custom policies."""
        custom_policy = NotePolicy(
            tag_pattern="journal",
            distillation=DistillationPolicy.NEVER,
            crystallization=CrystallizationPolicy.MANUAL_ONLY,
            privacy_level="private"
        )
        
        success = self.engine.create_custom_policy("journal", custom_policy)
        self.assertTrue(success)
        self.assertIn("journal", self.engine.note_policies)
    
    def test_get_policy_stats(self):
        """Test getting statistics by policy."""
        # Add notes with different policies
        self.engine.add_note("Diary entry", policy_tag="diary")
        self.engine.add_note("Meeting notes", policy_tag="meeting")
        self.engine.add_note("Research paper", policy_tag="research")
        
        stats = self.engine.get_policy_stats()
        
        self.assertIn('diary', stats)
        self.assertIn('meeting', stats)
        self.assertIn('research', stats)
        self.assertEqual(stats['diary']['total_notes'], 1)


class TestSimpleNotesAPI(unittest.TestCase):
    """Test the SimpleNotes API wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        with patch('notes_engine.InvisibleAI'), \
             patch('notes_engine.PredictiveIntelligenceEngine'), \
             patch('notes_engine.TemporalIntelligenceEngine'):
            self.notes = SimpleNotes()
            
        # Mock AI processing
        self.notes.engine.invisible_ai.process_content = Mock(return_value={
            'summary': 'Test summary',
            'concepts': ['test'],
            'keywords': ['test'],
            'confidence': 0.8
        })
    
    def test_simple_note_api(self):
        """Test simple note addition."""
        note_id = self.notes.note("Test content")
        self.assertIsInstance(note_id, str)
        self.assertTrue(note_id.startswith("note_"))
    
    def test_note_with_policy(self):
        """Test note with policy."""
        note_id = self.notes.note("Dear diary...", policy="diary")
        note = self.notes.engine.notes[note_id]
        self.assertEqual(note.policy_tag, "diary")
    
    def test_search_api(self):
        """Test search functionality."""
        # Add notes
        self.notes.note("Python programming")
        self.notes.note("Java programming")
        
        results = self.notes.search("programming")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
    
    def test_recent_notes(self):
        """Test getting recent notes."""
        # Add some notes
        for i in range(5):
            self.notes.note(f"Note {i}")
        
        recent = self.notes.recent(3)
        self.assertEqual(len(recent), 3)
    
    def test_insights_api(self):
        """Test getting insights."""
        # Add diary notes
        self.notes.note("Feeling happy today", policy="diary")
        
        insights = self.notes.insights(policy="diary", days=7)
        self.assertIn('total_notes_analyzed', insights)
    
    def test_block_auto_processing(self):
        """Test blocking auto processing."""
        note_id = self.notes.note("Private note")
        
        success = self.notes.block_auto_processing(note_id)
        self.assertTrue(success)
        
        note = self.notes.engine.notes[note_id]
        self.assertTrue(note.distillation_blocked)
        self.assertTrue(note.crystallization_blocked)


class TestDistillationPolicies(unittest.TestCase):
    """Test different distillation policies."""
    
    def setUp(self):
        """Set up test environment."""
        with patch('notes_engine.InvisibleAI'), \
             patch('notes_engine.PredictiveIntelligenceEngine'), \
             patch('notes_engine.TemporalIntelligenceEngine'):
            self.engine = NotesEngine()
    
    def test_size_based_distillation(self):
        """Test size-based distillation policy."""
        # Add meeting notes (size-based policy, threshold=5)
        for i in range(6):
            self.engine.add_note(
                f"Meeting note {i} about project planning",
                policy_tag="meeting"
            )
        
        # Should trigger distillation when threshold reached
        self.engine._distill_notes_by_policy_type("meeting")
        
        # Check if distillation occurred
        distilled_notes = [n for n in self.engine.notes.values() 
                          if n.distilled]
        self.assertGreater(len(distilled_notes), 0)
    
    def test_manual_only_distillation(self):
        """Test manual-only distillation policy."""
        # Add diary notes (manual-only policy)
        for i in range(10):
            self.engine.add_note(
                f"Diary entry {i}",
                policy_tag="diary"
            )
        
        # Automatic distillation should not occur
        self.engine._distill_notes_by_policy_type("diary")
        
        # No notes should be distilled
        distilled_notes = [n for n in self.engine.notes.values() 
                          if n.distilled and n.policy_tag == "diary"]
        self.assertEqual(len(distilled_notes), 0)


class TestCrystallizationPolicies(unittest.TestCase):
    """Test different crystallization policies."""
    
    def setUp(self):
        """Set up test environment."""
        with patch('notes_engine.InvisibleAI'), \
             patch('notes_engine.PredictiveIntelligenceEngine'), \
             patch('notes_engine.TemporalIntelligenceEngine'):
            self.engine = NotesEngine()
            
        # Mock AI for crystallization
        self.engine.invisible_ai.process_content = Mock(return_value={
            'summary': 'Crystallized wisdom',
            'insights': ['Key insight 1', 'Key insight 2'],
            'confidence': 0.9
        })
    
    def test_pattern_based_crystallization(self):
        """Test pattern-based crystallization."""
        # Add ideas notes with recurring concepts
        for i in range(6):
            note = self.engine.add_note(
                f"Idea {i} about innovation",
                policy_tag="ideas"
            )
            note.concepts = ["innovation", "creativity"]
        
        # Run crystallization
        self.engine._crystallize_policy_type("ideas")
        
        # Check if crystallization occurred
        self.assertGreater(len(self.engine.crystallized), 0)
    
    def test_never_crystallize_policy(self):
        """Test never crystallize policy."""
        # Add diary notes (never crystallize)
        for i in range(10):
            note = self.engine.add_note(
                f"Diary {i}",
                policy_tag="diary"
            )
            note.concepts = ["personal", "reflection"]
        
        # Run crystallization
        self.engine._crystallize_policy_type("diary")
        
        # No crystallization should occur
        diary_crystals = [c for c in self.engine.crystallized.values()
                         if hasattr(c, 'policy_tag') and c.policy_tag == "diary"]
        self.assertEqual(len(diary_crystals), 0)


# Run tests
if __name__ == "__main__":
    unittest.main()