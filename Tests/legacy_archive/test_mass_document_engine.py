import sys
from unittest.mock import MagicMock

# Mock dependencies of mass_document_engine to avoid ModuleNotFoundError
sys.modules['utils.universal_file_processor'] = MagicMock()
sys.modules['summarization_engine'] = MagicMock()
sys.modules['numpy'] = MagicMock()

import pytest
import hashlib
from pathlib import Path
from unittest.mock import patch

# Now import the class under test
from mass_document_engine import MassDocumentEngine

class TestMassDocumentEngine:
    @pytest.fixture
    def engine(self):
        return MassDocumentEngine()

    def test_generate_doc_id_consistency(self, engine):
        """Test that the same file attributes yield the same ID."""
        file_path = "test_file.txt"

        # Mock stat results
        mock_stat = MagicMock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = 1600000000.0

        with patch.object(Path, "stat", return_value=mock_stat):
            id1 = engine._generate_doc_id(file_path)
            id2 = engine._generate_doc_id(file_path)

            assert id1 == id2
            assert len(id1) == 16

            # Manually calculate expected hash
            expected_id_string = f"test_file.txt_1024_1600000000.0"
            expected_hash = hashlib.md5(expected_id_string.encode()).hexdigest()[:16]
            assert id1 == expected_hash

    def test_generate_doc_id_differentiation(self, engine):
        """Test that different file attributes yield different IDs."""
        file_path = "test_file.txt"

        # Base stat
        stat_base = MagicMock()
        stat_base.st_size = 1024
        stat_base.st_mtime = 1600000000.0

        # 1. Different size
        stat_diff_size = MagicMock()
        stat_diff_size.st_size = 2048
        stat_diff_size.st_mtime = 1600000000.0

        # 2. Different mtime
        stat_diff_mtime = MagicMock()
        stat_diff_mtime.st_size = 1024
        stat_diff_mtime.st_mtime = 1600000001.0

        # 3. Different name (but same size/mtime)
        file_path_diff_name = "other_file.txt"

        with patch.object(Path, "stat") as mocked_stat:
            mocked_stat.return_value = stat_base
            id_base = engine._generate_doc_id(file_path)

            mocked_stat.return_value = stat_diff_size
            id_size = engine._generate_doc_id(file_path)

            mocked_stat.return_value = stat_diff_mtime
            id_mtime = engine._generate_doc_id(file_path)

            mocked_stat.return_value = stat_base
            id_name = engine._generate_doc_id(file_path_diff_name)

            assert id_base != id_size
            assert id_base != id_mtime
            assert id_base != id_name

    def test_generate_doc_id_with_actual_file(self, engine, tmp_path):
        """Integration test with an actual temporary file."""
        test_file = tmp_path / "actual_test.txt"
        test_file.write_text("Hello World")

        # Get ID
        id1 = engine._generate_doc_id(str(test_file))

        # Get ID again
        id2 = engine._generate_doc_id(str(test_file))

        assert id1 == id2

        # Modify file (size changes)
        test_file.write_text("Hello World - modified")

        id3 = engine._generate_doc_id(str(test_file))

        assert id1 != id3
