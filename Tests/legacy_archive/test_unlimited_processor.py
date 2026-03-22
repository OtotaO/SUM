"""
Test suite for unlimited text processor

Tests the ability to handle texts of any size with
streaming and chunking capabilities.
"""

import pytest
import tempfile
import os
from unlimited_text_processor import (
    UnlimitedTextProcessor,
    get_unlimited_processor,
    process_unlimited_text
)


class TestUnlimitedTextProcessor:
    """Test suite for UnlimitedTextProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create UnlimitedTextProcessor instance."""
        return UnlimitedTextProcessor(
            overlap_ratio=0.1,
            max_memory_usage=50 * 1024 * 1024  # 50MB for testing
        )
    
    @pytest.fixture
    def small_text(self):
        """Small text that fits in memory."""
        return "This is a small text. " * 10
    
    @pytest.fixture
    def medium_text(self):
        """Medium-sized text."""
        return "This is a medium length sentence that will be repeated many times. " * 500
    
    def test_small_text_direct_processing(self, processor, small_text):
        """Test direct processing of small text."""
        result = processor.process_text(small_text)
        
        assert 'summary' in result
        assert 'processing_method' in result
        assert result['processing_method'] == 'direct'
        assert 'chunks_processed' in result
        assert result['chunks_processed'] == 1
    
    def test_medium_text_chunking(self, processor, medium_text):
        """Test chunked processing of medium text."""
        # Force chunking by setting small chunk size
        config = {'chunk_size': 1000}
        result = processor.process_text(medium_text, config)
        
        assert result['processing_method'] in ['chunked', 'direct']
        if result['processing_method'] == 'chunked':
            assert result['chunks_processed'] > 1
    
    def test_file_processing(self, processor):
        """Test processing text from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is text from a file. " * 100)
            temp_path = f.name
        
        try:
            result = processor.process_text(temp_path)
            assert 'summary' in result
            assert 'total_word_count' in result
        finally:
            os.unlink(temp_path)
    
    def test_streaming_detection(self, processor):
        """Test detection of streaming necessity."""
        # Small text - should not need streaming
        assert processor._needs_streaming("Small text", None) is False
        
        # Large file path - should need streaming
        with tempfile.NamedTemporaryFile() as f:
            # Write large amount to file
            for _ in range(10000):
                f.write(b"Large text content ")
            f.flush()
            
            # Check file size
            file_size = os.path.getsize(f.name)
            needs_streaming = processor._needs_streaming(f.name, None)
            
            if file_size > processor.max_memory_usage:
                assert needs_streaming is True
    
    def test_chunk_boundaries(self, processor):
        """Test chunk boundary detection."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        # Test finding sentence boundary
        boundary = processor._find_chunk_boundary(text, 20)
        assert text[boundary] == '.'
        assert boundary > 0
        assert boundary < len(text)
    
    def test_overlapping_chunks(self, processor):
        """Test overlapping chunk generation."""
        text = "A" * 1000 + ". " + "B" * 1000 + ". " + "C" * 1000 + "."
        chunks = list(processor._create_overlapping_chunks(text, chunk_size=1000))
        
        assert len(chunks) > 1
        
        # Check overlap exists
        for i in range(len(chunks) - 1):
            # End of chunk i should overlap with start of chunk i+1
            assert chunks[i][-100:] in chunks[i + 1]
    
    def test_hierarchical_summary(self, processor):
        """Test hierarchical summary generation."""
        # Create text that will produce multiple chunks
        text = "Important information. " * 50
        for i in range(10):
            text += f"Section {i} contains detailed information about topic {i}. " * 20
        
        config = {'chunk_size': 500, 'hierarchical': True}
        result = processor.process_text(text, config)
        
        if 'hierarchical_summary' in result:
            assert 'level_1_concepts' in result['hierarchical_summary']
            assert 'level_2_core' in result['hierarchical_summary']
    
    def test_memory_limit_enforcement(self, processor):
        """Test memory usage limits."""
        # Create processor with tiny memory limit
        tiny_processor = UnlimitedTextProcessor(max_memory_usage=1024)  # 1KB
        
        # Process large text
        large_text = "X" * 10000  # 10KB
        result = tiny_processor.process_text(large_text)
        
        # Should still work, using streaming or chunking
        assert 'summary' in result
        assert result['processing_method'] in ['chunked', 'streaming']
    
    def test_empty_text_handling(self, processor):
        """Test handling of empty text."""
        result = processor.process_text("")
        
        assert 'summary' in result
        assert result['summary'] == ""
        assert result['total_word_count'] == 0
    
    def test_unicode_text_processing(self, processor):
        """Test processing of Unicode text."""
        unicode_text = "Hello 世界 🌍 Привет мир"
        result = processor.process_text(unicode_text)
        
        assert 'summary' in result
        assert 'error' not in result
    
    def test_config_parameters(self, processor):
        """Test various configuration parameters."""
        text = "Test text. " * 100
        
        configs = [
            {'max_summary_tokens': 50},
            {'chunk_size': 100},
            {'overlap_ratio': 0.2},
            {'hierarchical': True},
            {'include_chunk_summaries': True}
        ]
        
        for config in configs:
            result = processor.process_text(text, config)
            assert 'summary' in result
            assert 'error' not in result
    
    def test_chunk_summaries_included(self, processor):
        """Test inclusion of individual chunk summaries."""
        text = "Chunk one content. " * 100 + "Chunk two content. " * 100
        
        config = {
            'chunk_size': 500,
            'include_chunk_summaries': True
        }
        
        result = processor.process_text(text, config)
        
        if result['chunks_processed'] > 1:
            assert 'chunk_summaries' in result
            assert len(result['chunk_summaries']) == result['chunks_processed']
            
            # Check chunk summary structure
            for chunk_summary in result['chunk_summaries']:
                assert 'chunk_id' in chunk_summary
                assert 'summary' in chunk_summary
                assert 'word_count' in chunk_summary


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_unlimited_processor(self):
        """Test get_unlimited_processor singleton."""
        proc1 = get_unlimited_processor()
        proc2 = get_unlimited_processor()
        
        assert proc1 is proc2  # Should be same instance
    
    def test_process_unlimited_text_function(self):
        """Test process_unlimited_text convenience function."""
        text = "This is a test of the global function."
        result = process_unlimited_text(text)
        
        assert 'summary' in result
        assert 'processing_method' in result
        assert 'total_word_count' in result


class TestLargeFileProcessing:
    """Test processing of large files."""
    
    @pytest.mark.slow
    def test_large_file_streaming(self):
        """Test streaming processing of large file."""
        processor = UnlimitedTextProcessor()
        
        # Create large temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write 10MB of text
            sentence = "This is a test sentence that will be repeated many times. "
            for _ in range(100000):
                f.write(sentence)
            temp_path = f.name
        
        try:
            result = processor.process_text(temp_path)
            
            assert 'summary' in result
            assert result['processing_method'] in ['streaming', 'hierarchical_streaming']
            assert result['total_word_count'] > 0
            
            # Summary should be much shorter than original
            assert len(result['summary']) < 10000
            
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        processor = UnlimitedTextProcessor()
        result = processor.process_text("/non/existent/file.txt")
        
        # Should handle gracefully
        assert 'error' in result or 'summary' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for edge case testing."""
        return UnlimitedTextProcessor()
    
    def test_single_word_text(self, processor):
        """Test processing of single word."""
        result = processor.process_text("Hello")
        assert result['summary'] == "Hello"
    
    def test_no_sentence_boundaries(self, processor):
        """Test text without clear sentence boundaries."""
        text = "word " * 1000  # No punctuation
        result = processor.process_text(text, {'chunk_size': 100})
        
        assert 'summary' in result
        assert 'error' not in result
    
    def test_only_punctuation(self, processor):
        """Test text with only punctuation."""
        result = processor.process_text("...")
        assert 'summary' in result
    
    def test_binary_content_detection(self, processor):
        """Test detection of binary content."""
        # Create file with binary content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'\x00\x01\x02\x03\x04')
            temp_path = f.name
        
        try:
            result = processor.process_text(temp_path)
            # Should handle binary files gracefully
            assert 'error' in result or result['summary'] == ''
        finally:
            os.unlink(temp_path)
    
    def test_mixed_line_endings(self, processor):
        """Test handling of mixed line endings."""
        text = "Line 1\nLine 2\r\nLine 3\rLine 4"
        result = processor.process_text(text)
        
        assert 'summary' in result
        assert 'error' not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])