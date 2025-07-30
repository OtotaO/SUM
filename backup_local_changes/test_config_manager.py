"""
test_config_manager.py - Unit tests for the ConfigManager

This module provides comprehensive unit tests for the ConfigManager class,
ensuring proper functionality for configuration loading, validation, and access.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    """Tests for the ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.base_config = {
            'app_name': 'SUM',
            'version': '1.0.0',
            'debug': False
        }
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a sample JSON config file
        self.sample_config = {
            'num_topics': 10,
            'max_summary_length': 300,
            'algorithms': ['lda', 'nmf', 'lsa'],
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'sum_db'
            }
        }
        
        self.config_path = self.temp_path / 'test_config.json'
        with open(self.config_path, 'w') as f:
            json.dump(self.sample_config, f)
            
        # Set up test environment variables
        os.environ['SUM_TEST_STRING'] = 'test_value'
        os.environ['SUM_TEST_INT'] = '42'
        os.environ['SUM_TEST_FLOAT'] = '3.14'
        os.environ['SUM_TEST_BOOL'] = 'true'
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        self.temp_dir.cleanup()
        
        # Clean up environment variables
        for var in ['SUM_TEST_STRING', 'SUM_TEST_INT', 'SUM_TEST_FLOAT', 'SUM_TEST_BOOL']:
            if var in os.environ:
                del os.environ[var]
    
    def test_initialization(self):
        """Test initialization with base configuration."""
        config = ConfigManager(self.base_config)
        
        # Check that base config was loaded
        self.assertEqual(config.get('app_name'), 'SUM')
        self.assertEqual(config.get('version'), '1.0.0')
        self.assertEqual(config.get('debug'), False)
        
        # Check initialization without base config
        empty_config = ConfigManager()
        self.assertEqual(empty_config.as_dict(), {})
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = ConfigManager()
        config.load_from_env()
        
        # Check that environment variables were loaded with correct types
        self.assertEqual(config.get('test_string'), 'test_value')
        self.assertEqual(config.get('test_int'), 42)
        self.assertEqual(config.get('test_float'), 3.14)
        self.assertEqual(config.get('test_bool'), True)
        
        # Test with custom prefix
        custom_prefix_config = ConfigManager()
        os.environ['CUSTOM_PREFIX_VALUE'] = 'custom_value'
        custom_prefix_config.load_from_env(prefix='CUSTOM_')
        self.assertEqual(custom_prefix_config.get('prefix_value'), 'custom_value')
        del os.environ['CUSTOM_PREFIX_VALUE']
        
        # Test with uppercase keys
        uppercase_config = ConfigManager()
        uppercase_config.load_from_env(lowercase=False)
        self.assertEqual(uppercase_config.get('TEST_STRING'), 'test_value')
    
    def test_load_from_json(self):
        """Test loading configuration from a JSON file."""
        config = ConfigManager()
        config.load_from_json(self.config_path)
        
        # Check that JSON config was loaded
        self.assertEqual(config.get('num_topics'), 10)
        self.assertEqual(config.get('max_summary_length'), 300)
        self.assertEqual(config.get('algorithms'), ['lda', 'nmf', 'lsa'])
        self.assertEqual(config.get('database')['host'], 'localhost')
        
        # Test with non-existent file
        config = ConfigManager()
        config.load_from_json(self.temp_path / 'non_existent.json')
        self.assertEqual(config.as_dict(), {})
    
    def test_load_from_dict(self):
        """Test loading configuration from a dictionary."""
        config = ConfigManager()
        config.load_from_dict({'key1': 'value1', 'key2': 'value2'})
        
        self.assertEqual(config.get('key1'), 'value1')
        self.assertEqual(config.get('key2'), 'value2')
    
    def test_get_set_methods(self):
        """Test get and set methods."""
        config = ConfigManager(self.base_config)
        
        # Test get with default
        self.assertEqual(config.get('non_existent', 'default'), 'default')
        
        # Test set
        config.set('new_key', 'new_value')
        self.assertEqual(config.get('new_key'), 'new_value')
        
        # Test overwrite
        config.set('app_name', 'NewName')
        self.assertEqual(config.get('app_name'), 'NewName')
    
    def test_dictionary_access(self):
        """Test dictionary-style access."""
        config = ConfigManager(self.base_config)
        
        # Test __getitem__
        self.assertEqual(config['app_name'], 'SUM')
        
        # Test __setitem__
        config['new_key'] = 'new_value'
        self.assertEqual(config['new_key'], 'new_value')
        
        # Test __contains__
        self.assertTrue('app_name' in config)
        self.assertFalse('non_existent' in config)
        
        # Test KeyError
        with self.assertRaises(KeyError):
            value = config['non_existent']
    
    def test_as_dict(self):
        """Test as_dict method."""
        config = ConfigManager(self.base_config)
        config_dict = config.as_dict()
        
        self.assertEqual(config_dict, self.base_config)
        
        # Ensure it's a copy, not a reference
        config_dict['modified'] = True
        self.assertNotIn('modified', config.as_dict())
    
    def test_validate(self):
        """Test configuration validation."""
        config = ConfigManager({
            'app_name': 'SUM',
            'version': '1.0.0',
            'debug': False,
            'port': 8080,
            'log_level': 'INFO',
            'max_connections': 100
        })
        
        # Valid schema
        schema = {
            'app_name': {'required': True, 'type': str},
            'version': {'required': True, 'type': str},
            'debug': {'type': bool},
            'port': {'type': int, 'min': 1024, 'max': 65535},
            'log_level': {'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
            'max_connections': {'type': int, 'min': 1}
        }
        
        errors = config.validate(schema)
        self.assertEqual(errors, [])
        
        # Test with invalid configuration
        invalid_config = ConfigManager({
            'app_name': 123,  # Wrong type
            'port': 80,  # Below min
            'log_level': 'TRACE',  # Not in allowed values
            'max_connections': 0  # Below min
        })
        
        errors = invalid_config.validate(schema)
        self.assertEqual(len(errors), 5)  # Missing version + wrong type for app_name + port below min + invalid log_level + max_connections below min
        
        # Test missing required field
        missing_required = ConfigManager({
            'debug': True
        })
        
        errors = missing_required.validate(schema)
        self.assertEqual(len(errors), 2)  # Missing app_name and version
    
    def test_save_to_json(self):
        """Test saving configuration to a JSON file."""
        config = ConfigManager(self.base_config)
        
        # Add some sensitive keys
        config.set('api_key', 'secret_key')
        config.set('password', 'secret_password')
        
        # Save to file
        save_path = self.temp_path / 'saved_config.json'
        result = config.save_to_json(save_path)
        self.assertTrue(result)
        self.assertTrue(save_path.exists())
        
        # Load the saved file and check contents
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['app_name'], 'SUM')
        self.assertEqual(saved_data['version'], '1.0.0')
        self.assertEqual(saved_data['debug'], False)
        
        # Check that sensitive keys were excluded
        self.assertNotIn('api_key', saved_data)
        self.assertNotIn('password', saved_data)
        
        # Test with include_sensitive=True
        result = config.save_to_json(save_path, include_sensitive=True)
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertIn('api_key', saved_data)
        self.assertIn('password', saved_data)
        
        # Test creating directories
        nested_path = self.temp_path / 'nested' / 'dir' / 'config.json'
        result = config.save_to_json(nested_path)
        self.assertTrue(result)
        self.assertTrue(nested_path.exists())
    
    def test_config_sources(self):
        """Test tracking of configuration sources."""
        config = ConfigManager(self.base_config)
        
        # Initially should have no sources
        self.assertEqual(config.config_sources, [])
        
        # Load from environment
        config.load_from_env()
        self.assertIn('environment', config.config_sources)
        
        # Load from JSON
        config.load_from_json(self.config_path)
        self.assertIn(str(self.config_path), config.config_sources)
        
        # Load from dict
        config.load_from_dict({'key': 'value'})
        self.assertIn('dictionary', config.config_sources)
    
    def test_method_chaining(self):
        """Test method chaining functionality."""
        config = ConfigManager()
        
        # Chain multiple methods
        result = config.load_from_env().load_from_json(self.config_path).load_from_dict({'key': 'value'})
        
        # Check that chaining returns self
        self.assertIs(result, config)
        
        # Check that all sources were loaded
        self.assertEqual(config.get('test_string'), 'test_value')  # From env
        self.assertEqual(config.get('num_topics'), 10)  # From JSON
        self.assertEqual(config.get('key'), 'value')  # From dict
    
    def test_string_representation(self):
        """Test string representation of ConfigManager."""
        config = ConfigManager(self.base_config)
        config.load_from_env()
        
        # Check that __str__ returns a string with sources and keys
        str_repr = str(config)
        self.assertIn('ConfigManager', str_repr)
        self.assertIn('sources=', str_repr)
        self.assertIn('keys=', str_repr)


if __name__ == '__main__':
    unittest.main()
