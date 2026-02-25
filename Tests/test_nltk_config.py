import os
import pytest
from unittest.mock import patch, MagicMock
import nltk
from config import Config
from summarization_engine import BasicSummarizationEngine

class TestNLTKConfig:
    def test_nltk_path_configuration(self):
        """Test that NLTK path can be configured via environment variable."""
        test_path = os.path.abspath("./test_nltk_data_custom_123")

        # Mock Config.NLTK_DATA_DIR
        with patch.object(Config, 'NLTK_DATA_DIR', test_path):
            with patch('nltk.download'):  # Avoid download
                # Ensure path is not in nltk.data.path initially
                if test_path in nltk.data.path:
                    nltk.data.path.remove(test_path)

                engine = BasicSummarizationEngine()

                assert test_path in nltk.data.path

                # Cleanup
                if test_path in nltk.data.path:
                    nltk.data.path.remove(test_path)
                if os.path.exists(test_path):
                    try:
                        os.rmdir(test_path)
                    except:
                        pass

    def test_nltk_resources_download(self):
        """Test that resources are downloaded to the configured path."""
        test_path = os.path.abspath("./test_nltk_data_dl_123")

        with patch.object(Config, 'NLTK_DATA_DIR', test_path):
            with patch('nltk.download') as mock_download:
                engine = BasicSummarizationEngine()

                # Check that download was called with the correct download_dir
                calls = mock_download.call_args_list
                assert len(calls) > 0
                for call in calls:
                    assert call.kwargs.get('download_dir') == test_path

            # Cleanup
            if test_path in nltk.data.path:
                nltk.data.path.remove(test_path)
            if os.path.exists(test_path):
                try:
                    os.rmdir(test_path)
                except:
                    pass
