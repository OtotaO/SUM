import pytest
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_processor import MultiModalProcessor

class TestMultiModal:
    def test_text_processing(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Hello World", encoding='utf-8')
        processor = MultiModalProcessor()
        result = processor.process_file(str(p))
        assert result['content'] == "Hello World"
        assert result['type'] == "text"
        
    def test_image_processing_fallback(self, tmp_path):
        # Even without OCR, it should handle the file existence check and return error or result
        p = tmp_path / "test.jpg"
        p.write_bytes(b"fake image data")
        processor = MultiModalProcessor()
        # Mock OCR disabled if not present
        if not processor.ocr_enabled:
            result = processor.process_file(str(p))
            assert 'error' in result
            assert result['type'] == 'image'
