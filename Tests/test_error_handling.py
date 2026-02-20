"""test_error_handling.py - Unit tests for the error handling utilities

This module provides comprehensive unit tests for the error handling utilities,
ensuring proper functionality for exception handling, validation, and error responses.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.error_handling import (
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
    create_error_response
)

class TestErrorHandling(unittest.TestCase):
    """Tests for the error handling utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
    
    def test_sum_error(self):
        """Test the base SUMError class."""
        # Create a basic error
        error = SUMError("Test error message")
        
        # Check error properties
        self.assertEqual(error.message, "Test error message")
        self.assertIsNone(error.code)
        self.assertEqual(error.details, {})
        
        # Check string representation
        self.assertEqual(str(error), "Test error message")
        
        # Create an error with code and details
        error = SUMError(
            "Test error with code",
            code="TEST_ERROR",
            details={"key": "value"}
        )
        
        # Check error properties
        self.assertEqual(error.message, "Test error with code")
        self.assertEqual(error.code, "TEST_ERROR")
        self.assertEqual(error.details, {"key": "value"})
        
        # Check string representation with code
        self.assertEqual(str(error), "[TEST_ERROR] Test error with code")
        
        # Check dictionary representation
        error_dict = error.to_dict()
        self.assertTrue(error_dict["error"])
        self.assertEqual(error_dict["message"], "Test error with code")
        self.assertEqual(error_dict["code"], "TEST_ERROR")
        self.assertEqual(error_dict["details"], {"key": "value"})
        self.assertIn("timestamp", error_dict)
    
    def test_specific_errors(self):
        """Test specific error subclasses."""
        # Test ConfigurationError
        config_error = ConfigurationError("Configuration error")
        self.assertEqual(config_error.code, "CONFIG_ERROR")
        
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
        # Define a test function that raises an exception
        @handle_exceptions(logger_instance=self.logger, reraise=False)
        def test_function():
            raise ValueError("Test exception")
        
        # Call the function and check the result
        result = test_function()
        self.assertEqual(result, None)
        
        # Define a test function that returns a dictionary
        @handle_exceptions(logger_instance=self.logger, reraise=False)
        def test_dict_function() -> Dict[str, Any]:
            raise ValueError("Test exception")
        
        # Call the function and check the result
        result = test_dict_function()
        self.assertTrue(result["error"])
        self.assertEqual(result["message"], "An unexpected error occurred")
        
        # Define a test function with exception mapping
        @handle_exceptions(
            logger_instance=self.logger,
            reraise=True,
            exception_map={ValueError: ConfigurationError}
        )
        def test_mapping_function():
            raise ValueError("Mapped exception")
        
        # Call the function and check that the exception is mapped
        with self.assertRaises(ConfigurationError):
            test_mapping_function()
    
    def test_validate_input_decorator(self):
        """Test the validate_input decorator."""
        # Define validators
        validators = {
            "name": lambda x: isinstance(x, str) and len(x) > 0,
            "age": lambda x: isinstance(x, int) and x >= 0
        }
        
        # Define error messages
        error_messages = {
            "name": "Name must be a non-empty string",
            "age": "Age must be a non-negative integer"
        }
        
        # Define a test function with validation
        @validate_input(validators, error_messages)
        def test_function(name, age):
            return {"name": name, "age": age}
        
        # Call the function with valid inputs
        result = test_function("John", 30)
        self.assertEqual(result, {"name": "John", "age": 30})
        
        # Call the function with invalid name
        with self.assertRaises(ValidationError) as context:
            test_function("", 30)
        self.assertEqual(str(context.exception), "[VALIDATION_ERROR] Name must be a non-empty string")
        
        # Call the function with invalid age
        with self.assertRaises(ValidationError) as context:
            test_function("John", -1)
        self.assertEqual(str(context.exception), "[VALIDATION_ERROR] Age must be a non-negative integer")
    
    def test_format_exception(self):
        """Test the format_exception function."""
        # Format a standard exception
        std_exception = ValueError("Standard exception")
        result = format_exception(std_exception)
        
        self.assertTrue(result["error"])
        self.assertEqual(result["message"], "Standard exception")
        self.assertEqual(result["type"], "ValueError")
        self.assertIn("timestamp", result)
        
        # Format a SUM-specific exception
        sum_exception = SUMError(
            "SUM exception",
            code="TEST_ERROR",
            details={"key": "value"}
        )
        result = format_exception(sum_exception)
        
        self.assertTrue(result["error"])
        self.assertEqual(result["message"], "SUM exception")
        self.assertEqual(result["code"], "TEST_ERROR")
        self.assertEqual(result["details"], {"key": "value"})
        self.assertIn("timestamp", result)
    
    def test_safe_execute(self):
        """Test the safe_execute function."""
        # Define test functions
        def successful_function(x, y):
            return x + y
        
        def failing_function(x, y):
            raise ValueError(f"Cannot add {x} and {y}")
        
        # Test successful execution
        result = safe_execute(successful_function, 2, 3)
        self.assertEqual(result, 5)
        
        # Test failed execution with default return
        result = safe_execute(failing_function, 2, "3", default_return=0)
        self.assertEqual(result, 0)
        
        # Test failed execution with no default return
        result = safe_execute(failing_function, 2, "3")
        self.assertIsNone(result)
        
        # Test with keyword arguments
        result = safe_execute(successful_function, x=2, y=3)
        self.assertEqual(result, 5)
    
    def test_create_error_response(self):
        """Test the create_error_response function."""
        # Create a basic error response
        response = create_error_response("Error message")
        
        self.assertTrue(response["error"])
        self.assertEqual(response["message"], "Error message")
        self.assertEqual(response["status_code"], 500)
        self.assertIn("timestamp", response)
        
        # Create an error response with code and status code
        response = create_error_response(
            "Error with code",
            code="TEST_ERROR",
            status_code=400
        )
        
        self.assertTrue(response["error"])
        self.assertEqual(response["message"], "Error with code")
        self.assertEqual(response["code"], "TEST_ERROR")
        self.assertEqual(response["status_code"], 400)
        
        # Create an error response with details
        response = create_error_response(
            "Error with details",
            details={"key": "value"}
        )
        
        self.assertTrue(response["error"])
        self.assertEqual(response["message"], "Error with details")
        self.assertEqual(response["details"], {"key": "value"})


if __name__ == '__main__':
    unittest.main()