"""
Comprehensive Test Suite for SUM Platform
==========================================

A production-grade test suite following best practices:
- Unit tests for each component
- Integration tests for workflows
- Performance benchmarks
- Edge case handling
- Mock external dependencies

Author: ototao
License: Apache License 2.0
"""

import unittest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import SumEngine
from sum_engines import AdvancedSummarizationEngine
from temporal_intelligence_engine import TemporalIntelligenceEngine
from predictive_intelligence import PredictiveIntelligence
from superhuman_memory import SuperhumanMemory, MemoryType
from knowledge_os import KnowledgeOperatingSystem, Thought


class TestSumEngine(unittest.TestCase):
    """Test suite for core SUM engine with Carmack optimizations"""
    
    def setUp(self):
        """Initialize test fixtures"""
        self.engine = SumEngine()
        self.sample_text = """
        Artificial intelligence is transforming how we process information.
        Machine learning algorithms can now understand context and meaning.
        This represents a fundamental shift in computing paradigms.
        The implications for knowledge work are profound and far-reaching.
        """
    
    def test_fast_summarization(self):
        """Test fast algorithm performance and accuracy"""
        start_time = time.time()
        result = self.engine.summarize(
            self.sample_text,
            max_length=50,
            algorithm='fast'
        )
        elapsed = time.time() - start_time
        
        # Performance assertion: should be under 100ms
        self.assertLess(elapsed, 0.1, "Fast algorithm should complete in <100ms")
        
        # Quality assertions
        self.assertIn('summary', result)
        self.assertIn('keywords', result)
        self.assertLess(len(result['summary'].split()), 60)
        self.assertGreater(len(result['keywords']), 0)
    
    def test_quality_summarization(self):
        """Test quality algorithm for semantic understanding"""
        result = self.engine.summarize(
            self.sample_text,
            max_length=50,
            algorithm='quality'
        )
        
        # Should identify key concepts
        self.assertIn('summary', result)
        self.assertIn('concepts', result)
        self.assertTrue(any('artificial intelligence' in c.lower() 
                          for c in result.get('concepts', [])))
    
    def test_hierarchical_large_text(self):
        """Test hierarchical processing for large documents"""
        # Generate large text
        large_text = self.sample_text * 100  # ~400 sentences
        
        result = self.engine.summarize(
            large_text,
            max_length=200,
            algorithm='hierarchical'
        )
        
        # Should handle large text efficiently
        self.assertIn('summary', result)
        self.assertIn('hierarchy_levels', result.get('metadata', {}))
        self.assertLess(len(result['summary'].split()), 250)
    
    def test_auto_algorithm_selection(self):
        """Test intelligent algorithm auto-selection"""
        # Small text should use fast
        small_result = self.engine.summarize("Short text.", algorithm='auto')
        self.assertEqual(small_result.get('metadata', {}).get('algorithm_used'), 'fast')
        
        # Large text should use hierarchical
        large_text = self.sample_text * 200
        large_result = self.engine.summarize(large_text, algorithm='auto')
        self.assertEqual(large_result.get('metadata', {}).get('algorithm_used'), 'hierarchical')
    
    def test_caching_performance(self):
        """Test LRU caching improves performance"""
        # First call - cache miss
        start1 = time.time()
        result1 = self.engine.summarize(self.sample_text)
        time1 = time.time() - start1
        
        # Second call - cache hit
        start2 = time.time()
        result2 = self.engine.summarize(self.sample_text)
        time2 = time.time() - start2
        
        # Cache hit should be much faster
        self.assertLess(time2, time1 * 0.1, "Cached call should be 10x faster")
        self.assertEqual(result1, result2, "Cached result should be identical")
    
    def test_error_handling(self):
        """Test robust error handling and fallbacks"""
        # Empty text
        result = self.engine.summarize("")
        self.assertIn('error', result)
        
        # None input
        result = self.engine.summarize(None)
        self.assertIn('error', result)
        
        # Extremely long text (should not crash)
        huge_text = "word " * 1000000
        result = self.engine.summarize(huge_text, max_length=100)
        self.assertIn('summary', result)
    
    def test_thread_safety(self):
        """Test concurrent access thread safety"""
        results = []
        errors = []
        
        def concurrent_summarize(text, index):
            try:
                result = self.engine.summarize(text + str(index))
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Launch concurrent threads
        import threading
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=concurrent_summarize,
                args=(self.sample_text, i)
            )
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed without errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 10, "All threads should complete")


class TestTemporalIntelligence(unittest.TestCase):
    """Test suite for Temporal Intelligence Engine"""
    
    def setUp(self):
        """Initialize temporal engine with test data"""
        self.engine = TemporalIntelligenceEngine()
        self.test_memories = [
            ("AI research breakthrough", datetime.now() - timedelta(days=30)),
            ("Machine learning applications", datetime.now() - timedelta(days=20)),
            ("Neural network optimization", datetime.now() - timedelta(days=10)),
            ("AGI discussions", datetime.now() - timedelta(days=5)),
            ("Quantum computing intersection", datetime.now())
        ]
    
    def test_temporal_pattern_detection(self):
        """Test detection of temporal patterns in knowledge"""
        # Add memories with temporal context
        for content, timestamp in self.test_memories:
            self.engine.add_memory(content, timestamp=timestamp)
        
        # Detect acceleration pattern
        patterns = self.engine.detect_temporal_patterns()
        
        self.assertIn('acceleration', patterns)
        self.assertIn('burst_topics', patterns)
        self.assertTrue(any('AI' in topic for topic in patterns.get('burst_topics', [])))
    
    def test_breakthrough_moment_detection(self):
        """Test identification of breakthrough moments"""
        # Simulate breakthrough moment
        self.engine.add_memory("Standard research", datetime.now() - timedelta(days=10))
        self.engine.add_memory("REVOLUTIONARY DISCOVERY!", datetime.now() - timedelta(days=5))
        self.engine.add_memory("Follow-up insights", datetime.now())
        
        breakthroughs = self.engine.detect_breakthroughs()
        
        self.assertGreater(len(breakthroughs), 0)
        self.assertIn('REVOLUTIONARY', str(breakthroughs[0]))
    
    def test_concept_evolution_tracking(self):
        """Test how concepts evolve over time"""
        # Add evolving concept
        concept_evolution = [
            ("Basic AI understanding", 30),
            ("AI can do simple tasks", 20),
            ("AI surpasses human performance", 10),
            ("AI consciousness questions", 5),
            ("AGI implications", 1)
        ]
        
        for content, days_ago in concept_evolution:
            self.engine.add_memory(
                content,
                timestamp=datetime.now() - timedelta(days=days_ago)
            )
        
        evolution = self.engine.track_concept_evolution("AI")
        
        self.assertIn('stages', evolution)
        self.assertGreater(len(evolution['stages']), 3)
        self.assertIn('complexity_increase', evolution)
    
    def test_relevance_surfacing(self):
        """Test surfacing old insights when relevant"""
        # Add old insight
        old_insight = "Distributed systems are key to scalability"
        self.engine.add_memory(
            old_insight,
            timestamp=datetime.now() - timedelta(days=365)
        )
        
        # Add recent context that makes old insight relevant
        self.engine.add_memory(
            "Need to scale our system to millions of users",
            timestamp=datetime.now()
        )
        
        relevant = self.engine.surface_relevant_insights("scaling")
        
        self.assertTrue(any(old_insight in str(m) for m in relevant))


class TestSuperhumanMemory(unittest.TestCase):
    """Test suite for Superhuman Memory System"""
    
    def setUp(self):
        """Initialize memory system with test configuration"""
        self.memory = SuperhumanMemory(test_mode=True)
    
    def test_perfect_recall(self):
        """Test perfect recall without information loss"""
        # Store complex memory
        complex_data = {
            'concept': 'Quantum Computing',
            'details': ['superposition', 'entanglement', 'qubits'],
            'relations': {'classical': 0.3, 'ai': 0.7},
            'metadata': {'importance': 0.9, 'timestamp': time.time()}
        }
        
        memory_id = self.memory.store(
            complex_data,
            memory_type=MemoryType.SEMANTIC
        )
        
        # Recall should be perfect
        recalled = self.memory.recall(memory_id)
        
        self.assertEqual(recalled, complex_data)
        self.assertIsInstance(recalled, dict)
        self.assertEqual(recalled['details'], complex_data['details'])
    
    def test_pattern_recognition(self):
        """Test advanced pattern recognition across memories"""
        # Store pattern sequence
        patterns = [
            "User logs in at 9am",
            "User checks emails first",
            "User opens project dashboard",
            "User starts coding session"
        ]
        
        for i, pattern in enumerate(patterns):
            self.memory.store(pattern, memory_type=MemoryType.EPISODIC)
            time.sleep(0.01)  # Ensure temporal ordering
        
        # Detect sequential pattern
        detected = self.memory.detect_patterns(pattern_type='sequential')
        
        self.assertIn('login_workflow', detected)
        self.assertEqual(len(detected['login_workflow']), 4)
    
    def test_predictive_activation(self):
        """Test predictive memory activation"""
        # Store correlated memories
        self.memory.store("Python programming", memory_type=MemoryType.SEMANTIC)
        self.memory.store("Django framework", memory_type=MemoryType.SEMANTIC)
        self.memory.store("REST API design", memory_type=MemoryType.SEMANTIC)
        
        # Query should predict related memories
        predictions = self.memory.predict_relevant("web development")
        
        self.assertTrue(any("Django" in str(p) for p in predictions))
        self.assertTrue(any("API" in str(p) for p in predictions))
    
    def test_memory_compression(self):
        """Test intelligent memory compression without loss"""
        # Store redundant memories
        for i in range(100):
            self.memory.store(
                f"Similar concept variation {i % 10}",
                memory_type=MemoryType.SEMANTIC
            )
        
        # Compress memories
        original_count = self.memory.count()
        self.memory.compress(threshold=0.8)
        compressed_count = self.memory.count()
        
        # Should reduce redundancy
        self.assertLess(compressed_count, original_count)
        
        # But preserve unique information
        unique_concepts = self.memory.get_unique_concepts()
        self.assertGreaterEqual(len(unique_concepts), 10)


class TestKnowledgeOS(unittest.TestCase):
    """Test suite for Knowledge Operating System"""
    
    def setUp(self):
        """Initialize Knowledge OS with test configuration"""
        self.kos = KnowledgeOperatingSystem(test_mode=True)
    
    def test_effortless_capture(self):
        """Test zero-friction thought capture"""
        thought = "Need to refactor the authentication system"
        
        captured = self.kos.capture(thought)
        
        self.assertIsNotNone(captured.id)
        self.assertEqual(captured.content, thought)
        self.assertIsNotNone(captured.metadata)
        self.assertIn('tags', captured.metadata)
    
    def test_background_intelligence(self):
        """Test automatic background processing"""
        # Capture related thoughts
        thoughts = [
            "Machine learning model accuracy is dropping",
            "Need more training data for the model",
            "Consider data augmentation techniques"
        ]
        
        for thought in thoughts:
            self.kos.capture(thought)
        
        # Let background processing run
        time.sleep(0.1)
        
        # Check for automatic connections
        insights = self.kos.get_insights()
        
        self.assertIn('connections', insights)
        self.assertIn('concepts', insights)
        self.assertTrue(any('model' in c.lower() for c in insights['concepts']))
    
    def test_threshold_densification(self):
        """Test automatic knowledge densification"""
        # Add many related thoughts to trigger densification
        base_concept = "distributed systems"
        for i in range(20):
            self.kos.capture(f"{base_concept} aspect {i}: details...")
        
        # Check for densification opportunity
        densification = self.kos.check_densification_needed()
        
        self.assertTrue(densification['needed'])
        self.assertIn(base_concept, densification['concepts'])
        
        # Perform densification
        result = self.kos.densify(base_concept)
        
        self.assertIn('compressed', result)
        self.assertIn('insights', result)
        self.assertLess(result['compression_ratio'], 0.5)
    
    @patch('knowledge_os.BackgroundIntelligenceEngine.process')
    def test_async_processing(self, mock_process):
        """Test asynchronous background processing"""
        mock_process.return_value = {'concepts': ['test']}
        
        # Capture thought
        thought = self.kos.capture("Async processing test")
        
        # Process should be called asynchronously
        time.sleep(0.1)
        mock_process.assert_called_once()


class TestPredictiveIntelligence(unittest.TestCase):
    """Test suite for Predictive Intelligence System"""
    
    def setUp(self):
        """Initialize predictive system"""
        self.predictor = PredictiveIntelligence()
        
        # Add historical context
        self.context_history = [
            ("Researching machine learning", {"topic": "ML", "depth": "beginner"}),
            ("Reading about neural networks", {"topic": "ML", "depth": "intermediate"}),
            ("Implementing backpropagation", {"topic": "ML", "depth": "advanced"})
        ]
    
    def test_need_prediction(self):
        """Test prediction of user's information needs"""
        # Build context
        for action, metadata in self.context_history:
            self.predictor.add_context(action, metadata)
        
        # Predict next needs
        predictions = self.predictor.predict_needs()
        
        self.assertGreater(len(predictions), 0)
        self.assertTrue(any('optimization' in p.lower() for p in predictions))
        self.assertTrue(any('architecture' in p.lower() for p in predictions))
    
    def test_suggestion_relevance(self):
        """Test relevance of suggestions"""
        # Add specific context
        self.predictor.add_context(
            "Working on REST API",
            {"project": "backend", "language": "python"}
        )
        
        suggestions = self.predictor.suggest_resources()
        
        self.assertTrue(any('django' in s.lower() or 'flask' in s.lower() 
                          for s in suggestions))
    
    def test_pattern_learning(self):
        """Test learning from user patterns"""
        # Simulate usage pattern
        morning_pattern = [
            ("Check emails", {"time": "09:00"}),
            ("Review tasks", {"time": "09:30"}),
            ("Start coding", {"time": "10:00"})
        ]
        
        for _ in range(5):  # Repeat pattern
            for action, metadata in morning_pattern:
                self.predictor.add_context(action, metadata)
        
        # Should recognize pattern
        pattern_prediction = self.predictor.predict_at_time("09:15")
        
        self.assertIn("tasks", pattern_prediction.lower())


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        from web.app_factory import create_app
        cls.app = create_app(testing=True)
        cls.client = cls.app.test_client()
    
    def test_summarize_endpoint(self):
        """Test /api/summarize endpoint"""
        response = self.client.post(
            '/api/summarize',
            json={'text': 'Test text for summarization', 'max_length': 50},
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('summary', data)
    
    def test_knowledge_capture_endpoint(self):
        """Test /api/knowledge/capture endpoint"""
        response = self.client.post(
            '/api/knowledge/capture',
            json={'thought': 'Test thought for capture'},
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('thought_id', data)
    
    def test_temporal_analysis_endpoint(self):
        """Test /api/temporal/analyze endpoint"""
        response = self.client.post(
            '/api/temporal/analyze',
            json={'timeframe': '30d'},
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('patterns', data)
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        # Make many requests quickly
        responses = []
        for _ in range(35):  # Exceeds typical rate limit
            response = self.client.post(
                '/api/summarize',
                json={'text': 'Test'},
                content_type='application/json'
            )
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        self.assertIn(429, responses, "Rate limiting should trigger")
    
    def test_batch_processing(self):
        """Test batch processing endpoint"""
        batch_request = {
            'documents': [
                {'id': '1', 'text': 'First document'},
                {'id': '2', 'text': 'Second document'},
                {'id': '3', 'text': 'Third document'}
            ]
        }
        
        response = self.client.post(
            '/api/summarize/batch',
            json=batch_request,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data['results']), 3)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def setUp(self):
        """Initialize components for benchmarking"""
        self.engine = SumEngine()
        self.memory = SuperhumanMemory(test_mode=True)
        self.kos = KnowledgeOperatingSystem(test_mode=True)
    
    def test_summarization_speed_benchmark(self):
        """Benchmark summarization speed across text sizes"""
        benchmarks = []
        
        text_sizes = [100, 500, 1000, 5000, 10000]  # words
        
        for size in text_sizes:
            text = " ".join(["word"] * size)
            
            start = time.time()
            self.engine.summarize(text, algorithm='fast')
            elapsed = time.time() - start
            
            benchmarks.append({
                'size': size,
                'time': elapsed,
                'words_per_second': size / elapsed
            })
        
        # Verify performance scales well
        for benchmark in benchmarks:
            # Should process at least 10,000 words per second
            self.assertGreater(
                benchmark['words_per_second'], 
                10000,
                f"Performance degraded at {benchmark['size']} words"
            )
    
    def test_memory_recall_speed(self):
        """Benchmark memory recall speed"""
        # Store many memories
        memory_ids = []
        for i in range(1000):
            mem_id = self.memory.store(
                f"Memory {i}",
                memory_type=MemoryType.SEMANTIC
            )
            memory_ids.append(mem_id)
        
        # Benchmark recall
        start = time.time()
        for mem_id in memory_ids[:100]:  # Recall 100 memories
            self.memory.recall(mem_id)
        elapsed = time.time() - start
        
        avg_recall_time = elapsed / 100
        
        # Should recall in under 1ms per memory
        self.assertLess(avg_recall_time, 0.001)
    
    def test_pattern_detection_performance(self):
        """Benchmark pattern detection performance"""
        # Add many patterns
        for i in range(500):
            self.memory.store(
                f"Pattern element {i % 50}",
                memory_type=MemoryType.EPISODIC
            )
        
        # Benchmark pattern detection
        start = time.time()
        patterns = self.memory.detect_patterns()
        elapsed = time.time() - start
        
        # Should detect patterns in under 1 second
        self.assertLess(elapsed, 1.0)
        self.assertGreater(len(patterns), 0)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience"""
    
    def test_corrupted_input_recovery(self):
        """Test recovery from corrupted input"""
        engine = SumEngine()
        
        # Various corrupted inputs
        corrupted_inputs = [
            None,
            "",
            "   ",
            "\x00\x01\x02",  # Binary data
            "🚀" * 10000,  # Excessive emojis
            {"not": "a string"},  # Wrong type
        ]
        
        for inp in corrupted_inputs:
            result = engine.summarize(inp)
            # Should not crash, should return error
            self.assertIsNotNone(result)
            if 'error' not in result:
                self.assertIn('summary', result)
    
    @patch('openai.ChatCompletion.create')
    def test_external_service_failure(self, mock_openai):
        """Test handling of external service failures"""
        mock_openai.side_effect = Exception("Service unavailable")
        
        engine = SumEngine()
        result = engine.summarize("Test text", use_ai=True)
        
        # Should fallback gracefully
        self.assertIn('summary', result)
        self.assertNotIn('error', result)
    
    def test_memory_overflow_protection(self):
        """Test protection against memory overflow"""
        memory = SuperhumanMemory(test_mode=True, max_memories=100)
        
        # Try to store excessive memories
        for i in range(200):
            memory.store(f"Memory {i}", memory_type=MemoryType.SEMANTIC)
        
        # Should not exceed limit
        self.assertLessEqual(memory.count(), 100)
        
        # Should preserve most important memories
        important = memory.get_important_memories()
        self.assertGreater(len(important), 0)


# Test runner
if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSumEngine,
        TestTemporalIntelligence,
        TestSuperhumanMemory,
        TestKnowledgeOS,
        TestPredictiveIntelligence,
        TestAPIIntegration,
        TestPerformanceBenchmarks,
        TestErrorRecovery
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
