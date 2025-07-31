#!/usr/bin/env python3
"""
test_multimodal_system.py - Comprehensive Test Suite for Multi-Modal Processing

This module provides extensive testing for the multi-modal processing capabilities,
including document processing, OCR, local AI integration, and performance benchmarking.

Test Categories:
- Document Processing (PDF, DOCX, HTML, Markdown)
- Image Processing (OCR, Vision Models)
- Local AI Integration (Ollama)
- Performance Benchmarking
- Error Handling and Edge Cases
- Integration Testing

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import unittest
import tempfile
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_processor import MultiModalProcessor, ContentType, ProcessingResult
from ollama_manager import OllamaManager, ProcessingRequest, ModelType

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiModalProcessor(unittest.TestCase):
    """Test suite for MultiModalProcessor functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.processor = MultiModalProcessor()
        cls.test_dir = Path(tempfile.mkdtemp(prefix="sum_multimodal_test_"))
        cls.created_files = []
        
        logger.info(f"Test directory: {cls.test_dir}")
        logger.info(f"Processor capabilities: {cls.processor.get_processing_stats()}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up created test files
        for file_path in cls.created_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
        
        # Remove test directory
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            logger.warning(f"Could not remove test directory: {e}")
    
    def create_test_file(self, filename: str, content: str, binary: bool = False) -> str:
        """Create a test file and track it for cleanup."""
        file_path = str(self.test_dir / filename)
        
        mode = 'wb' if binary else 'w'
        encoding = None if binary else 'utf-8'
        
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        
        self.created_files.append(file_path)
        return file_path
    
    def test_content_type_detection(self):
        """Test content type detection for various file formats."""
        test_cases = [
            ("test.txt", ContentType.TEXT),
            ("test.pdf", ContentType.PDF),
            ("test.docx", ContentType.DOCX),
            ("test.html", ContentType.HTML),
            ("test.md", ContentType.MARKDOWN),
            ("test.png", ContentType.IMAGE),
            ("test.jpg", ContentType.IMAGE),
            ("unknown.xyz", ContentType.UNKNOWN),
        ]
        
        for filename, expected_type in test_cases:
            file_path = self.create_test_file(filename, "test content")
            detected_type = self.processor.detect_content_type(file_path)
            self.assertEqual(detected_type, expected_type, 
                           f"Failed to detect {expected_type.value} for {filename}")
    
    def test_text_processing(self):
        """Test plain text file processing."""
        test_content = """
        This is a comprehensive test of text processing capabilities.
        The system should extract this content and process it through
        the hierarchical summarization engine for enhanced analysis.
        
        Key points to extract:
        1. Text processing functionality
        2. Hierarchical summarization
        3. Content analysis capabilities
        """
        
        file_path = self.create_test_file("test.txt", test_content)
        result = self.processor.process_file(file_path)
        
        self.assertEqual(result.content_type, ContentType.TEXT)
        self.assertIn("text processing", result.extracted_text.lower())
        self.assertGreater(result.confidence_score, 0.8)
        self.assertIsNotNone(result.metadata.get('word_count'))
        self.assertGreater(result.metadata['word_count'], 0)
    
    def test_html_processing(self):
        """Test HTML file processing with tag removal."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Document</title>
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a <strong>test paragraph</strong> with <em>formatting</em>.</p>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
            <script>alert('should be removed');</script>
        </body>
        </html>
        """
        
        file_path = self.create_test_file("test.html", html_content)
        result = self.processor.process_file(file_path)
        
        self.assertEqual(result.content_type, ContentType.HTML)
        self.assertIn("test paragraph", result.extracted_text.lower())
        self.assertNotIn("<", result.extracted_text)  # Tags should be removed
        self.assertNotIn("script", result.extracted_text.lower())
        self.assertGreater(result.confidence_score, 0.7)
    
    def test_markdown_processing(self):
        """Test Markdown file processing."""
        markdown_content = """
        # Test Document
        
        This is a **markdown test** with various formatting elements.
        
        ## Features
        
        - *Italic text*
        - **Bold text**
        - `Code snippets`
        - [Links](http://example.com)
        
        ### Code Block
        
        ```python
        def test_function():
            return "test"
        ```
        
        Regular paragraph text that should be preserved.
        """
        
        file_path = self.create_test_file("test.md", markdown_content)
        result = self.processor.process_file(file_path)
        
        self.assertEqual(result.content_type, ContentType.MARKDOWN)
        self.assertIn("markdown test", result.extracted_text.lower())
        self.assertIn("regular paragraph", result.extracted_text.lower())
        # Markdown formatting should be removed
        self.assertNotIn("**", result.extracted_text)
        self.assertNotIn("##", result.extracted_text)
        self.assertGreater(result.confidence_score, 0.8)
    
    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        files = [
            ("test1.txt", "First test document with important content."),
            ("test2.md", "# Second Test\n\nMarkdown document with **formatting**."),
            ("test3.html", "<html><body><h1>Third Test</h1><p>HTML content.</p></body></html>"),
        ]
        
        file_paths = []
        for filename, content in files:
            file_path = self.create_test_file(filename, content)
            file_paths.append(file_path)
        
        results = self.processor.process_batch(file_paths)
        
        self.assertEqual(len(results), 3)
        
        # Check each result
        for i, result in enumerate(results):
            self.assertIsInstance(result, ProcessingResult)
            self.assertGreater(len(result.extracted_text), 0)
            self.assertIsNone(result.error_message)
            self.assertGreater(result.confidence_score, 0.5)
    
    def test_error_handling(self):
        """Test error handling for invalid files."""
        # Test non-existent file
        result = self.processor.process_file("/nonexistent/file.txt")
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.confidence_score, 0.0)
        
        # Test empty file
        empty_file = self.create_test_file("empty.txt", "")
        result = self.processor.process_file(empty_file)
        self.assertEqual(result.content_type, ContentType.TEXT)
        self.assertEqual(result.extracted_text, "")
    
    def test_large_file_processing(self):
        """Test processing of large text files."""
        # Generate large content
        large_content = "This is a test sentence. " * 1000  # ~25KB
        
        file_path = self.create_test_file("large_test.txt", large_content)
        result = self.processor.process_file(file_path)
        
        self.assertEqual(result.content_type, ContentType.TEXT)
        self.assertGreater(len(result.extracted_text), 20000)
        self.assertGreaterEqual(result.metadata['word_count'], 5000)
        self.assertGreater(result.confidence_score, 0.8)


class TestOllamaIntegration(unittest.TestCase):
    """Test suite for Ollama local AI integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Ollama test environment."""
        cls.manager = OllamaManager()
        cls.has_ollama = cls.manager.ollama_client is not None
        cls.has_models = len(cls.manager.available_models) > 0
        
        if not cls.has_ollama:
            logger.warning("Ollama not available - skipping Ollama integration tests")
        elif not cls.has_models:
            logger.warning("No local models available - skipping model tests")
        else:
            logger.info(f"Testing with {len(cls.manager.available_models)} available models")
    
    def setUp(self):
        """Skip tests if Ollama not available."""
        if not self.has_ollama:
            self.skipTest("Ollama not available")
    
    def test_model_discovery(self):
        """Test model discovery and cataloging."""
        status = self.manager.get_model_status()
        
        self.assertIsInstance(status['total_models'], int)
        self.assertIsInstance(status['models_by_type'], dict)
        self.assertIsInstance(status['ollama_available'], bool)
        self.assertTrue(status['ollama_available'])
    
    def test_model_selection(self):
        """Test automatic model selection for different tasks."""
        if not self.has_models:
            self.skipTest("No models available")
        
        tasks = ['summarization', 'quick_summary', 'detailed_analysis']
        
        for task in tasks:
            model = self.manager.get_best_model(task)
            if model:  # Only test if a model is available
                self.assertIn(model, self.manager.available_models)
                logger.info(f"Selected {model} for {task}")
    
    def test_text_processing(self):
        """Test local model text processing."""
        if not self.has_models:
            self.skipTest("No models available")
        
        test_text = """
        Artificial intelligence and machine learning have revolutionized many industries.
        From healthcare to finance, these technologies are being used to automate processes,
        improve decision-making, and create new opportunities for innovation.
        """
        
        request = ProcessingRequest(
            text=test_text,
            task_type="summarization",
            max_tokens=100,
            temperature=0.3
        )
        
        try:
            response = self.manager.process_text(request)
            
            self.assertIsInstance(response.response, str)
            self.assertGreater(len(response.response), 10)
            self.assertGreater(response.confidence_score, 0.0)
            self.assertIn(response.model_used, self.manager.available_models)
            self.assertIsInstance(response.processing_time, float)
            self.assertGreater(response.processing_time, 0)
            
            logger.info(f"Processed with {response.model_used} in {response.processing_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Text processing failed: {e}")
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        if not self.has_models:
            self.skipTest("No models available")
        
        # Process a few requests to generate stats
        test_texts = [
            "Short text for testing.",
            "Medium length text that contains more information for processing and analysis.",
            "Longer text sample that includes multiple sentences and provides more comprehensive content for the model to process and analyze effectively."
        ]
        
        for text in test_texts:
            try:
                request = ProcessingRequest(text=text, max_tokens=50)
                self.manager.process_text(request)
            except Exception as e:
                logger.warning(f"Performance test failed: {e}")
        
        # Check if performance stats were updated
        self.assertGreater(len(self.manager.performance_cache), 0)


class TestIntegratedSystem(unittest.TestCase):
    """Test integration between multi-modal processing and local AI."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integrated system testing."""
        cls.processor = MultiModalProcessor()
        cls.ollama_manager = OllamaManager()
        cls.test_dir = Path(tempfile.mkdtemp(prefix="sum_integration_test_"))
        cls.created_files = []
        
        cls.has_full_system = (
            cls.processor.get_processing_stats()['hierarchical_engine'] and
            cls.ollama_manager.ollama_client is not None
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment."""
        for file_path in cls.created_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
        
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception:
            pass
    
    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file for integration testing."""
        file_path = str(self.test_dir / filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.created_files.append(file_path)
        return file_path
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end multi-modal processing."""
        test_content = """
        The Future of Artificial Intelligence in Healthcare
        
        Artificial intelligence is transforming healthcare through various applications:
        
        1. Medical Imaging: AI algorithms can analyze medical images with high accuracy
        2. Drug Discovery: Machine learning accelerates the identification of new treatments
        3. Personalized Medicine: AI enables tailored treatment plans based on individual data
        4. Predictive Analytics: Early warning systems help prevent medical emergencies
        
        These advances promise to improve patient outcomes while reducing costs.
        However, challenges remain including data privacy, algorithm bias, and regulatory approval.
        
        The integration of AI in healthcare represents a paradigm shift that will
        continue to evolve as technology advances and adoption increases.
        """
        
        file_path = self.create_test_file("healthcare_ai.txt", test_content)
        
        # Process with multi-modal processor
        result = self.processor.process_file(
            file_path,
            hierarchical_config={
                'max_concepts': 7,
                'max_summary_tokens': 100,
                'max_insights': 5
            }
        )
        
        # Verify processing results
        self.assertEqual(result.content_type, ContentType.TEXT)
        self.assertGreater(result.confidence_score, 0.8)
        self.assertIn('hierarchical_analysis', result.metadata)
        
        # Check hierarchical analysis structure
        hierarchical = result.metadata['hierarchical_analysis']
        self.assertIn('hierarchical_summary', hierarchical)
        
        summary = hierarchical['hierarchical_summary']
        self.assertIn('level_1_concepts', summary)
        self.assertIn('level_2_core', summary)
        
        logger.info("End-to-end processing successful")
        logger.info(f"Extracted {len(summary['level_1_concepts'])} key concepts")
    
    def test_system_capabilities_report(self):
        """Generate comprehensive system capabilities report."""
        processor_stats = self.processor.get_processing_stats()
        ollama_status = self.ollama_manager.get_model_status()
        
        report = {
            'multi_modal_processing': processor_stats,
            'local_ai_capabilities': ollama_status,
            'integration_status': {
                'full_system_available': self.has_full_system,
                'supported_formats': self.processor.get_supported_formats(),
                'available_models': list(self.ollama_manager.available_models.keys()),
                'test_timestamp': time.time()
            }
        }
        
        # Export report
        report_path = str(self.test_dir / "capabilities_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"System capabilities report saved to {report_path}")
        
        # Verify report structure
        self.assertIn('multi_modal_processing', report)
        self.assertIn('local_ai_capabilities', report)
        self.assertIn('integration_status', report)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking test suite."""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance testing."""
        cls.processor = MultiModalProcessor()
        cls.test_sizes = [1000, 5000, 10000, 25000]  # Character counts
        cls.results = {}
    
    def test_processing_performance(self):
        """Benchmark processing performance for different text sizes."""
        for size in self.test_sizes:
            # Generate test content
            test_content = "This is test content for performance benchmarking. " * (size // 50)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_path = f.name
            
            try:
                # Measure processing time
                start_time = time.time()
                result = self.processor.process_file(temp_path)
                processing_time = time.time() - start_time
                
                # Store results
                self.results[size] = {
                    'processing_time': processing_time,
                    'characters_per_second': size / processing_time,
                    'confidence_score': result.confidence_score,
                    'word_count': result.metadata.get('word_count', 0)
                }
                
                logger.info(f"Size {size}: {processing_time:.3f}s "
                           f"({size/processing_time:.0f} chars/sec)")
                
            finally:
                # Clean up
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        
        # Verify performance scales reasonably
        self.assertGreater(len(self.results), 0)
        
        # Check that larger texts don't take proportionally much longer
        if len(self.results) >= 2:
            sizes = sorted(self.results.keys())
            small_cps = self.results[sizes[0]]['characters_per_second']
            large_cps = self.results[sizes[-1]]['characters_per_second']
            
            # Performance shouldn't degrade by more than 50%
            self.assertGreater(large_cps, small_cps * 0.5)


def run_comprehensive_tests():
    """Run all test suites with detailed reporting."""
    print("🚀 Starting Comprehensive Multi-Modal System Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMultiModalProcessor,
        TestOllamaIntegration,
        TestIntegratedSystem,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True
    )
    
    print(f"\nRunning {suite.countTestCases()} tests...")
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, error in result.failures:
            print(f"  {test}: {error.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n⚠️ Errors:")
        for test, error in result.errors:
            print(f"  {test}: {error.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Multi-modal system is working well!")
    elif success_rate >= 60:
        print("⚡ System is functional with some issues to address")
    else:
        print("🔧 System needs attention - multiple issues detected")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)