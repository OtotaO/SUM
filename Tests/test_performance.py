#!/usr/bin/env python3
"""
test_performance.py - Performance benchmarks for SUM

Tests various components under load to ensure performance standards.
"""

import pytest
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import SUM components
from SUM import SimpleSUM, HierarchicalDensificationEngine
from superhuman_memory import SuperhumanMemorySystem, MemoryType
from community_intelligence import CommunityIntelligence
from notes_engine import SimpleNotes


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial intelligence has made remarkable progress in recent years, with breakthroughs 
    in natural language processing, computer vision, and machine learning. Large language 
    models like GPT-4 and Claude have demonstrated impressive capabilities in understanding 
    and generating human-like text. These advances are transforming industries from healthcare 
    to finance, enabling new applications and improving existing processes. However, challenges 
    remain in areas such as AI safety, interpretability, and ensuring equitable access to 
    these powerful technologies. As we continue to develop more sophisticated AI systems, 
    it's crucial to consider their societal impact and work towards responsible deployment.
    """ * 10  # Make it longer for performance testing


@pytest.fixture
def long_text():
    """Very long text for stress testing."""
    base_text = """
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being explicitly 
    programmed. Machine learning focuses on the development of computer programs that can 
    access data and use it to learn for themselves. The process of learning begins with 
    observations or data, such as examples, direct experience, or instruction, in order 
    to look for patterns in data and make better decisions in the future based on the 
    examples that we provide.
    """
    return base_text * 100  # Very long text


class TestBasicPerformance:
    """Test basic performance of core components."""
    
    def test_simple_sum_performance(self, benchmark, sample_text):
        """Benchmark SimpleSUM processing speed."""
        sum_engine = SimpleSUM()
        
        def process_text():
            return sum_engine.process_text(sample_text)
        
        result = benchmark(process_text)
        
        # Assertions
        assert result is not None
        assert len(result) > 0
        
        # Performance assertions
        assert benchmark.stats.stats.mean < 2.0  # Should complete in under 2 seconds
    
    def test_hierarchical_engine_performance(self, benchmark, sample_text):
        """Benchmark HierarchicalDensificationEngine processing speed."""
        engine = HierarchicalDensificationEngine()
        
        def process_text():
            return engine.process_text(sample_text)
        
        result = benchmark(process_text)
        
        # Assertions
        assert result is not None
        assert 'summary' in result
        
        # Performance assertions
        assert benchmark.stats.stats.mean < 3.0  # Should complete in under 3 seconds
    
    def test_memory_storage_performance(self, benchmark):
        """Benchmark superhuman memory storage performance."""
        memory_system = SuperhumanMemorySystem(":memory:")  # In-memory DB for testing
        
        def store_memory():
            return memory_system.store_memory(
                "This is a test memory for performance benchmarking.",
                MemoryType.SEMANTIC
            )
        
        result = benchmark(store_memory)
        
        # Assertions
        assert result is not None
        assert result.memory_id is not None
        
        # Performance assertions
        assert benchmark.stats.stats.mean < 0.1  # Should complete in under 100ms
    
    def test_memory_recall_performance(self, benchmark):
        """Benchmark memory recall performance."""
        memory_system = SuperhumanMemorySystem(":memory:")
        
        # Pre-populate with memories
        for i in range(100):
            memory_system.store_memory(
                f"Test memory {i} about artificial intelligence and machine learning.",
                MemoryType.SEMANTIC
            )
        
        def recall_memory():
            return memory_system.recall_memory("artificial intelligence", limit=10)
        
        results = benchmark(recall_memory)
        
        # Assertions
        assert len(results) > 0
        
        # Performance assertions
        assert benchmark.stats.stats.mean < 0.5  # Should complete in under 500ms


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    def test_concurrent_processing(self, sample_text):
        """Test concurrent text processing."""
        engine = HierarchicalDensificationEngine()
        num_threads = 10
        results = []
        
        def process_text():
            result = engine.process_text(sample_text)
            results.append(result)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_text) for _ in range(num_threads)]
            for future in futures:
                future.result()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assertions
        assert len(results) == num_threads
        assert all(result is not None for result in results)
        assert total_time < 10.0  # Should complete all in under 10 seconds
    
    def test_concurrent_memory_operations(self):
        """Test concurrent memory operations."""
        memory_system = SuperhumanMemorySystem(":memory:")
        num_operations = 50
        results = []
        
        def memory_operation(i):
            # Store a memory
            memory = memory_system.store_memory(
                f"Concurrent memory {i} for testing performance.",
                MemoryType.SEMANTIC
            )
            
            # Recall memories
            recalled = memory_system.recall_memory(f"memory {i}", limit=5)
            results.append((memory, recalled))
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(memory_operation, i) for i in range(num_operations)]
            for future in futures:
                future.result()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assertions
        assert len(results) == num_operations
        assert total_time < 15.0  # Should complete all operations in under 15 seconds
    
    def test_notes_system_concurrency(self):
        """Test notes system under concurrent load."""
        notes = SimpleNotes()
        num_notes = 20
        results = []
        
        def add_note(i):
            note_id = notes.note(f"Concurrent note {i} about performance testing.")
            search_results = notes.search(f"note {i}")
            results.append((note_id, search_results))
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_note, i) for i in range(num_notes)]
            for future in futures:
                future.result()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assertions
        assert len(results) == num_notes
        assert total_time < 10.0  # Should complete all in under 10 seconds


class TestScalabilityPerformance:
    """Test scalability with increasing load."""
    
    def test_text_length_scaling(self, long_text):
        """Test how performance scales with text length."""
        engine = HierarchicalDensificationEngine()
        
        text_lengths = [1000, 5000, 10000, 20000, 50000]
        processing_times = []
        
        for length in text_lengths:
            test_text = long_text[:length]
            
            start_time = time.time()
            result = engine.process_text(test_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Ensure result is valid
            assert result is not None
            assert 'summary' in result
        
        # Check that performance doesn't degrade exponentially
        # Processing time should be roughly linear with text length
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i-1]
            length_ratio = text_lengths[i] / text_lengths[i-1]
            
            # Time increase should not be more than 3x the length increase
            assert time_ratio <= length_ratio * 3
    
    def test_memory_scaling(self):
        """Test memory system performance with increasing number of memories."""
        memory_system = SuperhumanMemorySystem(":memory:")
        
        memory_counts = [10, 50, 100, 500, 1000]
        recall_times = []
        
        for count in memory_counts:
            # Add memories up to the target count
            current_count = len(memory_system.active_memories)
            for i in range(current_count, count):
                memory_system.store_memory(
                    f"Scalability test memory {i} about various topics including AI, ML, science, and technology.",
                    MemoryType.SEMANTIC
                )
            
            # Test recall performance
            start_time = time.time()
            results = memory_system.recall_memory("AI technology", limit=10)
            end_time = time.time()
            
            recall_time = end_time - start_time
            recall_times.append(recall_time)
            
            # Ensure results are valid
            assert len(results) > 0
        
        # Check that recall time doesn't increase dramatically
        # Should remain relatively stable or increase logarithmically
        for i in range(1, len(recall_times)):
            time_ratio = recall_times[i] / recall_times[0]  # Compare to baseline
            count_ratio = memory_counts[i] / memory_counts[0]
            
            # Time increase should be much less than linear with memory count
            assert time_ratio <= np.log(count_ratio) + 1


class TestMemoryUsagePerformance:
    """Test memory usage and resource efficiency."""
    
    def test_memory_efficiency(self, long_text):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = HierarchicalDensificationEngine()
        
        # Process multiple large texts
        for i in range(5):
            result = engine.process_text(long_text)
            assert result is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
    
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple memory systems
        for i in range(10):
            memory_system = SuperhumanMemorySystem(":memory:")
            
            # Add many memories
            for j in range(100):
                memory_system.store_memory(
                    f"Cleanup test memory {i}-{j}",
                    MemoryType.EPISODIC
                )
            
            # Delete the system
            del memory_system
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not have increased significantly after cleanup
        assert memory_increase < 100


class TestAsyncPerformance:
    """Test asynchronous operation performance."""
    
    @pytest.mark.asyncio
    async def test_async_processing(self, sample_text):
        """Test asynchronous processing performance."""
        engine = HierarchicalDensificationEngine()
        
        async def async_process(text):
            # Simulate async processing
            await asyncio.sleep(0.01)  # Small delay to simulate async I/O
            return engine.process_text(text)
        
        start_time = time.time()
        
        # Process multiple texts concurrently
        tasks = [async_process(sample_text) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assertions
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Should complete faster than sequential processing
        assert total_time < 5.0  # Should complete in under 5 seconds


class TestRealWorldScenarios:
    """Test performance in realistic usage scenarios."""
    
    def test_research_workflow(self, benchmark):
        """Benchmark a typical research workflow."""
        def research_workflow():
            # Initialize systems
            sum_engine = SimpleSUM()
            memory_system = SuperhumanMemorySystem(":memory:")
            notes = SimpleNotes()
            
            # Simulate processing research papers
            papers = [
                "Machine learning research paper about neural networks and deep learning architectures.",
                "Artificial intelligence paper discussing natural language processing and transformers.",
                "Computer science research on distributed systems and cloud computing frameworks."
            ]
            
            results = []
            for paper in papers:
                # Process with SUM
                summary = sum_engine.process_text(paper)
                results.append(summary)
                
                # Store in memory
                memory_system.store_memory(paper, MemoryType.SEMANTIC)
                
                # Add notes
                notes.note(f"Research insight: {summary[:100]}...")
            
            # Search memories
            ai_memories = memory_system.recall_memory("artificial intelligence", limit=5)
            
            # Search notes
            research_notes = notes.search("research")
            
            return {
                'summaries': results,
                'memories': ai_memories,
                'notes': research_notes
            }
        
        result = benchmark(research_workflow)
        
        # Assertions
        assert len(result['summaries']) == 3
        assert len(result['memories']) > 0
        assert len(result['notes']) > 0
        
        # Performance assertion
        assert benchmark.stats.stats.mean < 10.0  # Should complete in under 10 seconds
    
    def test_collaborative_session(self, benchmark):
        """Benchmark a collaborative intelligence session."""
        def collaborative_session():
            community = CommunityIntelligence()
            
            # Simulate multiple users contributing
            sessions = []
            for i in range(5):
                content = f"Collaborative insight {i} about AI development and future trends."
                result = {
                    'success': True,
                    'processing_time': 0.1 + (i * 0.02),
                    'content_type': 'text',
                    'language': 'en'
                }
                
                community.record_usage(content, result, user_satisfaction=0.8 + (i * 0.02))
                sessions.append((content, result))
            
            # Get community insights
            insights = community.get_community_insights()
            
            # Get recommendations
            recommendations = community.get_personalized_recommendations({
                'content_type': 'text',
                'content_length': 1000
            })
            
            return {
                'sessions': sessions,
                'insights': insights,
                'recommendations': recommendations
            }
        
        result = benchmark(collaborative_session)
        
        # Assertions
        assert len(result['sessions']) == 5
        assert 'community_stats' in result['insights']
        assert len(result['recommendations']) > 0
        
        # Performance assertion
        assert benchmark.stats.stats.mean < 2.0  # Should complete in under 2 seconds


# Performance targets for continuous monitoring
PERFORMANCE_TARGETS = {
    'simple_sum_max_time': 2.0,  # seconds
    'hierarchical_engine_max_time': 3.0,  # seconds
    'memory_storage_max_time': 0.1,  # seconds
    'memory_recall_max_time': 0.5,  # seconds
    'concurrent_processing_max_time': 10.0,  # seconds
    'memory_usage_increase_max': 500,  # MB
    'research_workflow_max_time': 10.0,  # seconds
}


def test_performance_targets():
    """Ensure all performance targets are documented and reasonable."""
    assert all(isinstance(target, (int, float)) for target in PERFORMANCE_TARGETS.values())
    assert all(target > 0 for target in PERFORMANCE_TARGETS.values())


if __name__ == "__main__":
    # Run basic performance tests
    print("Running SUM Performance Tests")
    print("=" * 40)
    
    # Run pytest with benchmark
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", 
        "--benchmark-only", "--benchmark-columns=min,max,mean,stddev"
    ])
    
    if result.returncode == 0:
        print("\n✅ All performance tests passed!")
    else:
        print("\n❌ Some performance tests failed!")
        exit(1)