"""
Comprehensive tests for robustness improvements
"""
import os
import sys
import pytest
import tempfile
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.file_validator import FileValidator
from Utils.streaming_file_processor import StreamingFileProcessor
from Utils.request_queue import RequestQueue, QueuedRequest
from Utils.error_recovery import ErrorRecoveryManager, with_error_recovery, CircuitBreaker
from Utils.database_pool import SQLiteConnectionPool, DatabaseManager


class TestFileValidator:
    """Test file validation functionality"""
    
    def setup_method(self):
        self.validator = FileValidator()
    
    def test_valid_text_file(self):
        """Test validation of valid text file"""
        # Create mock file
        from werkzeug.datastructures import FileStorage
        import io
        
        content = b"This is a test text file content"
        file = FileStorage(
            stream=io.BytesIO(content),
            filename="test.txt",
            content_type="text/plain"
        )
        
        is_valid, error, metadata = self.validator.validate_file(file)
        
        assert is_valid is True
        assert error is None
        assert metadata['extension'] == 'txt'
        assert metadata['size'] == len(content)
        assert metadata['mime_type'] == 'text/plain'
    
    def test_file_size_limit(self):
        """Test file size validation"""
        from werkzeug.datastructures import FileStorage
        import io
        
        # Create large file (11MB for text, limit is 10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        file = FileStorage(
            stream=io.BytesIO(large_content),
            filename="large.txt",
            content_type="text/plain"
        )
        
        is_valid, error, metadata = self.validator.validate_file(file)
        
        assert is_valid is False
        assert "too large" in error
        assert metadata['size'] == len(large_content)
    
    def test_invalid_file_type(self):
        """Test rejection of invalid file types"""
        from werkzeug.datastructures import FileStorage
        import io
        
        # Create executable file
        content = b"\x7fELF"  # ELF header
        file = FileStorage(
            stream=io.BytesIO(content),
            filename="malicious.exe",
            content_type="application/x-executable"
        )
        
        is_valid, error, metadata = self.validator.validate_file(file)
        
        assert is_valid is False
        assert metadata['extension'] == 'exe'
    
    def test_file_content_mismatch(self):
        """Test detection of file content mismatch"""
        from werkzeug.datastructures import FileStorage
        import io
        
        # PDF content with .txt extension
        content = b"%PDF-1.4"
        file = FileStorage(
            stream=io.BytesIO(content),
            filename="fake.txt",
            content_type="text/plain"
        )
        
        is_valid, error, metadata = self.validator.validate_file(file)
        
        assert is_valid is False
        assert "content doesn't match extension" in error


class TestStreamingFileProcessor:
    """Test streaming file processing"""
    
    def setup_method(self):
        self.processor = StreamingFileProcessor()
    
    def test_text_streaming(self):
        """Test streaming text file processing"""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(1000):
                f.write(f"Line {i}: This is a test line with some content\n")
            temp_path = f.name
        
        try:
            chunks_processed = 0
            
            def process_chunk(text, index):
                nonlocal chunks_processed
                chunks_processed += 1
                return len(text)
            
            with self.processor.process_file_stream(temp_path) as file_obj:
                results = list(self.processor.process_text_chunks(
                    file_obj, 
                    process_chunk
                ))
            
            assert chunks_processed > 0
            assert sum(results) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_csv_streaming(self):
        """Test streaming CSV processing"""
        import csv
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value'])
            for i in range(100):
                writer.writerow([i, f'Item {i}', i * 10])
            temp_path = f.name
        
        try:
            rows_processed = 0
            
            def process_row(row, index):
                nonlocal rows_processed
                rows_processed += 1
                return row
            
            with self.processor.process_file_stream(temp_path, 'rb') as file_obj:
                results = list(self.processor.process_csv_streaming(
                    file_obj,
                    process_row
                ))
            
            assert rows_processed == 100  # Excluding header
            assert len(results) == 100
            
        finally:
            os.unlink(temp_path)
    
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"x" * 1024 * 1024)  # 1MB file
            temp_path = f.name
        
        try:
            # Text processing estimate
            text_estimate = self.processor.estimate_memory_usage(temp_path, 'text')
            assert text_estimate > 1024 * 1024  # Should be more than file size
            
            # JSON processing estimate
            json_estimate = self.processor.estimate_memory_usage(temp_path, 'json')
            assert json_estimate > text_estimate  # JSON needs more memory
            
        finally:
            os.unlink(temp_path)


class TestRequestQueue:
    """Test request queue system"""
    
    @pytest.mark.asyncio
    async def test_basic_queue_operation(self):
        """Test basic queue enqueue and processing"""
        queue = RequestQueue(max_concurrent_requests=2)
        
        # Register test handler
        async def test_handler(payload):
            await asyncio.sleep(0.1)
            return f"Processed: {payload['data']}"
        
        queue.register_handler('test', test_handler)
        
        # Start queue
        await queue.start()
        
        try:
            # Enqueue requests
            job_id1 = await queue.enqueue('test', {'data': 'request1'})
            job_id2 = await queue.enqueue('test', {'data': 'request2'})
            
            assert job_id1 is not None
            assert job_id2 is not None
            
            # Wait for completion
            result1 = await queue.get_result(job_id1, timeout=5)
            result2 = await queue.get_result(job_id2, timeout=5)
            
            assert result1 == "Processed: request1"
            assert result2 == "Processed: request2"
            
            # Check metrics
            metrics = queue.get_metrics()
            assert metrics['completed_requests'] >= 2
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_priority_queue(self):
        """Test priority handling in queue"""
        queue = RequestQueue(max_concurrent_requests=1)
        
        processed_order = []
        
        async def slow_handler(payload):
            await asyncio.sleep(0.2)
            processed_order.append(payload['priority'])
            return "done"
        
        queue.register_handler('slow', slow_handler)
        await queue.start()
        
        try:
            # Enqueue with different priorities (lower number = higher priority)
            await queue.enqueue('slow', {'priority': 3}, priority=3)
            await queue.enqueue('slow', {'priority': 1}, priority=1)
            await queue.enqueue('slow', {'priority': 2}, priority=2)
            
            # Wait for all to complete
            await asyncio.sleep(1)
            
            # Check processing order - should be 3, 1, 2 (first one starts immediately)
            assert processed_order[0] == 3  # First request
            assert processed_order[1] == 1  # Highest priority
            assert processed_order[2] == 2  # Lower priority
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        """Test queue behavior when full"""
        queue = RequestQueue(max_concurrent_requests=1, max_queue_size=2)
        
        async def slow_handler(payload):
            await asyncio.sleep(1)
            return "done"
        
        queue.register_handler('slow', slow_handler)
        await queue.start()
        
        try:
            # Fill queue
            await queue.enqueue('slow', {'id': 1})
            await queue.enqueue('slow', {'id': 2})
            
            # Try to exceed queue size
            with pytest.raises(ValueError, match="queue is full"):
                await queue.enqueue('slow', {'id': 3})
                
        finally:
            await queue.stop()


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    def test_error_tracking(self):
        """Test error tracking functionality"""
        manager = ErrorRecoveryManager()
        
        # Track different errors
        try:
            raise ValueError("Test error 1")
        except Exception as e:
            context1 = manager.track_error(e, {'user_id': 'user1'})
        
        try:
            raise ConnectionError("Test error 2")
        except Exception as e:
            context2 = manager.track_error(e, {'user_id': 'user2'})
        
        # Check error history
        assert len(manager.error_history) == 2
        assert manager.error_counts['ValueError'] == 1
        assert manager.error_counts['ConnectionError'] == 1
        
        # Check error stats
        stats = manager.get_error_stats(60)
        assert stats['total_errors'] == 2
        assert 'ValueError' in stats['error_types']
    
    @pytest.mark.asyncio
    async def test_retry_decorator(self):
        """Test automatic retry with decorator"""
        call_count = 0
        
        @with_error_recovery(retry_count=3, recoverable_errors=[ValueError])
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        call_count = 0
        
        @breaker
        def flaky_service():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Service down")
        
        # First 3 calls should fail normally
        for i in range(3):
            with pytest.raises(ConnectionError):
                flaky_service()
        
        # Circuit should now be open
        with pytest.raises(Exception, match="Circuit breaker .* is OPEN"):
            flaky_service()
        
        assert call_count == 3  # No more calls after circuit opens


class TestDatabasePool:
    """Test database connection pooling"""
    
    def test_sqlite_pool(self):
        """Test SQLite connection pool"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            pool = SQLiteConnectionPool(db_path, pool_size=3)
            
            # Create table
            pool.execute("""
                CREATE TABLE IF NOT EXISTS test (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Test concurrent operations
            results = []
            
            def insert_data(i):
                pool.execute(
                    "INSERT INTO test (data) VALUES (?)",
                    (f"test_{i}",)
                )
                results.append(i)
            
            # Run multiple operations
            import threading
            threads = []
            for i in range(10):
                t = threading.Thread(target=insert_data, args=(i,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # Verify all inserts
            rows = pool.execute("SELECT COUNT(*) as count FROM test")
            assert rows[0]['count'] == 10
            
            # Check pool stats
            stats = pool.get_stats()
            assert stats['connections_created'] >= 3
            assert stats['queries_executed'] >= 11  # 1 create + 10 inserts + 1 select
            
        finally:
            pool.close_all()
            os.unlink(db_path)
    
    def test_connection_recovery(self):
        """Test connection recovery after failure"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            pool = SQLiteConnectionPool(db_path, pool_size=2)
            
            # Normal operation
            pool.execute("CREATE TABLE test (id INTEGER)")
            pool.execute("INSERT INTO test VALUES (1)")
            
            # Simulate connection failure by corrupting a connection
            with pool.get_connection() as conn:
                conn.close()  # Force close
            
            # Pool should recover and create new connection
            result = pool.execute("SELECT COUNT(*) as count FROM test")
            assert result[0]['count'] == 1
            
        finally:
            pool.close_all()
            os.unlink(db_path)


# Integration test
class TestRobustnessIntegration:
    """Test all robustness features working together"""
    
    @pytest.mark.asyncio
    async def test_file_processing_pipeline(self):
        """Test complete file processing pipeline with robustness"""
        # Setup components
        validator = FileValidator()
        processor = StreamingFileProcessor()
        queue = RequestQueue(max_concurrent_requests=2)
        error_manager = ErrorRecoveryManager()
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,text\n")
            for i in range(100):
                f.write(f"{i},This is test text for row {i}\n")
            temp_path = f.name
        
        try:
            # File processing handler
            async def process_file_handler(payload):
                file_path = payload['file_path']
                
                # Process with streaming
                rows = []
                with processor.process_file_stream(file_path, 'rb') as file_obj:
                    for row in processor.process_csv_streaming(
                        file_obj,
                        lambda r, i: r
                    ):
                        rows.append(row)
                
                return {
                    'rows_processed': len(rows),
                    'sample': rows[:5]
                }
            
            queue.register_handler('file_process', process_file_handler)
            await queue.start()
            
            # Enqueue file processing job
            job_id = await queue.enqueue('file_process', {
                'file_path': temp_path
            })
            
            # Get result
            result = await queue.get_result(job_id, timeout=10)
            
            assert result['rows_processed'] == 100
            assert len(result['sample']) == 5
            
        finally:
            await queue.stop()
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])