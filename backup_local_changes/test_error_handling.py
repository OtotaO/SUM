"""
test_error_handling.py - Unit tests for error handling utilities

This module provides comprehensive unit tests for the error handling utilities,
ensuring proper functionality for exception handling, validation, and error responses.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys
import logging
from unittest.mock import patch, MagicMock, call

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.error_handling import (
    SUMError,
    ConfigurationError,
    DataError,
    ProcessingError,
    ValidationError,
    APIError,
    handle_exceptions,
    validate_input,
    format_exception,
    safe_execute,
    create_error_response,
    register_error_handlers
)

class TestErrorHandling(unittest.TestCase):
    """Tests for the error handling utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        logging.basicConfig(level=logging.CRITICAL)  # Suppress log output during tests
        self.logger = logging.getLogger('test_logger')
    
    def test_sum_error_base_class(self):
        """Test the base SUMError class."""
        # Test with minimal arguments
        error = SUMError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertEqual(error.message, "Test error message")
        self.assertIsNone(error.code)
        self.assertEqual(error.details, {})
        
        # Test with all arguments
        error = SUMError("Test error message", code="TEST_ERROR", details={"key": "value"})
        self.assertEqual(str(error), "[TEST_ERROR] Test error message")
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.code, "TEST_ERROR")
        self.assertEqual(error.details, {"key": "value"})
        
        # Test to_dict method
        error_dict = error.to_dict()
        self.assertTrue(error_dict["error"])
        self.assertEqual(error_dict["message"], "Test error message")
        self.assertEqual(error_dict["code"], "TEST_ERROR")
        self.assertEqual(error_dict["details"], {"key": "value"})
        self.assertIn("timestamp", error_dict)
    
    def test_specific_error_classes(self):
        """Test specific error subclasses."""
        # Test ConfigurationError
        config_error = ConfigurationError("Config error", details={"param": "value"})
        self.assertEqual(config_error.code, "CONFIG_ERROR")
        self.assertEqual(config_error.message, "Config error")
        self.assertEqual(config_error.details, {"param": "value"})
        
        # Test DataError
        data_error = DataError("Data error")
        self.assertEqual(data_error.code, "DATA_ERROR")
        
        # Test ProcessingError
        processing_error = ProcessingError("Processing error")
        self.assertEqual(processing_error.code, "PROCESSING_ERROR")
        
        # Test ValidationError
        validation_error = ValidationError("Validation error")
        self.assertEqual(validation_error.code, "VALIDATION_ERROR")
        
        # Test APIError
        api_error = APIError("API error", status_code=404)
        self.assertEqual(api_error.code, "API_ERROR")
        self.assertEqual(api_error.status_code, 404)
        self.assertEqual(api_error.details["status_code"], 404)
    
    def test_handle_exceptions_decorator(self):
        """Test the handle_exceptions decorator."""
        # Test function that succeeds
        @handle_exceptions(logger_instance=self.logger)
        def successful_function():
            return "success"
        
        result = successful_function()
        self.assertEqual(result, "success")
        
        # Test function that raises an exception
        @handle_exceptions(logger_instance=self.logger, reraise=False, default_message="Failed")
        def failing_function():
            raise ValueError("Test error")
        
        with patch.object(self.logger, 'error') as mock_error:
            result = failing_function()
            mock_error.assert_called_once()
            self.assertIn("ValueError: Test error", mock_error.call_args[0][0])
            self.assertEqual(result, {'error': True, 'message': 'Failed'})
        
        # Test function that raises an exception with reraising
        @handle_exceptions(logger_instance=self.logger, reraise=True)
        def reraising_function():
            raise ValueError("Test error")
        
        with patch.object(self.logger, 'error') as mock_error:
            with self.assertRaises(ValueError):
                reraising_function()
            mock_error.assert_called_once()
        
        # Test with exception mapping
        @handle_exceptions(
            logger_instance=self.logger,
            exception_map={ValueError: ValidationError}
        )
        def mapping_function():
            raise ValueError("Mapped error")
        
        with patch.object(self.logger, 'error') as mock_error:
            with self.assertRaises(ValidationError) as context:
                mapping_function()
            self.assertEqual(str(context.exception), "Mapped error")
            mock_error.assert_called_once()
    
    def test_validate_input_decorator(self):
        """Test the validate_input decorator."""
        # Define validators
        validators = {
            'text': lambda x: isinstance(x, str) and len(x) > 0,
            'count': lambda x: isinstance(x, int) and x > 0
        }
        
        error_messages = {
            'text': "Text must be a non-empty string",
            'count': "Count must be a positive integer"
        }
        
        # Test function with valid inputs
        @validate_input(validators=validators, error_messages=error_messages)
        def valid_function(text, count):
            return f"{text} repeated {count} times"
        
        result = valid_function("test", 3)
        self.assertEqual(result, "test repeated 3 times")
        
        # Test with invalid text
        with self.assertRaises(ValidationError) as context:
            valid_function("", 3)
        self.assertEqual(str(context.exception), "Text must be a non-empty string")
        self.assertEqual(context.exception.details["parameter"], "text")
        
        # Test with invalid count
        with self.assertRaises(ValidationError) as context:
            valid_function("test", 0)
        self.assertEqual(str(context.exception), "Count must be a positive integer")
        self.assertEqual(context.exception.details["parameter"], "count")
        
        # Test with default error message
        @validate_input(validators={'param': lambda x: x > 10})
        def default_message_function(param):
            return param
        
        with self.assertRaises(ValidationError) as context:
            default_message_function(5)
        self.assertIn("Invalid value for parameter 'param'", str(context.exception))
    
    def test_format_exception(self):
        """Test the format_exception function."""
        # Test with SUMError
        error = SUMError("Test error", code="TEST_CODE", details={"key": "value"})
        formatted = format_exception(error)
        self.assertTrue(formatted["error"])
        self.assertEqual(formatted["message"], "Test error")
        self.assertEqual(formatted["code"], "TEST_CODE")
        self.assertEqual(formatted["details"], {"key": "value"})
        
        # Test with standard exception
        std_error = ValueError("Standard error")
        formatted = format_exception(std_error)
        self.assertTrue(formatted["error"])
        self.assertEqual(formatted["message"], "Standard error")
        self.assertEqual(formatted["type"], "ValueError")
    
    def test_safe_execute(self):
        """Test the safe_execute function."""
        # Test with successful function
        def success_func(a, b):
            return a + b
        
        result = safe_execute(success_func, 1, 2)
        self.assertEqual(result, 3)
        
        # Test with failing function
        def failing_func():
            raise ValueError("Test error")
        
        with patch.object(logging.getLogger('Utils.error_handling'), 'error') as mock_error:
            result = safe_execute(failing_func, default_return="default")
            self.assertEqual(result, "default")
            mock_error.assert_called_once()
            self.assertIn("Error executing failing_func", mock_error.call_args[0][0])
        
        # Test with failing function and no logging
        with patch.object(logging.getLogger('Utils.error_handling'), 'error') as mock_error:
            result = safe_execute(failing_func, default_return="default", log_error=False)
            self.assertEqual(result, "default")
            mock_error.assert_not_called()
    
    def test_create_error_response(self):
        """Test the create_error_response function."""
        # Test with minimal arguments
        response = create_error_response("Error message")
        self.assertTrue(response["error"])
        self.assertEqual(response["message"], "Error message")
        self.assertEqual(response["status_code"], 500)
        self.assertIn("timestamp", response)
        
        # Test with all arguments
        response = create_error_response(
            "Error message",
            code="ERROR_CODE",
            status_code=404,
            details={"key": "value"}
        )
        self.assertTrue(response["error"])
        self.assertEqual(response["message"], "Error message")
        self.assertEqual(response["code"], "ERROR_CODE")
        self.assertEqual(response["status_code"], 404)
        self.assertEqual(response["details"], {"key": "value"})
    
    def test_register_error_handlers(self):
        """Test the register_error_handlers function."""
        # Create a mock Flask app
        mock_app = MagicMock()
        
        # Call register_error_handlers
        register_error_handlers(mock_app)
        
        # Check that error handlers were registered
        self.assertEqual(mock_app.errorhandler.call_count, 4)
        mock_app.errorhandler.assert_has_calls([
            call(SUMError),
            call(400),
            call(404),
            call(500)
        ], any_order=True)
        
        # Test with None app
        register_error_handlers(None)  # Should not raise an exception


if __name__ == '__main__':
    unittest.main()
