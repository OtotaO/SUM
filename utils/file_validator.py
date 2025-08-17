"""
File validation utilities for robust file handling
"""
import os
import magic
import hashlib
from typing import Optional, Tuple, BinaryIO
from werkzeug.datastructures import FileStorage
import logging

logger = logging.getLogger(__name__)

class FileValidator:
    """Validates uploaded files for security and compatibility"""
    
    # Maximum file sizes by type (in bytes)
    MAX_FILE_SIZES = {
        'pdf': 50 * 1024 * 1024,      # 50MB for PDFs
        'txt': 10 * 1024 * 1024,      # 10MB for text files
        'json': 10 * 1024 * 1024,     # 10MB for JSON
        'csv': 100 * 1024 * 1024,     # 100MB for CSV
        'md': 10 * 1024 * 1024,       # 10MB for Markdown
        'docx': 25 * 1024 * 1024,     # 25MB for Word docs
        'default': 10 * 1024 * 1024   # 10MB default
    }
    
    # Allowed MIME types with extensions
    ALLOWED_MIME_TYPES = {
        'application/pdf': ['pdf'],
        'text/plain': ['txt', 'text'],
        'text/csv': ['csv'],
        'application/json': ['json'],
        'text/markdown': ['md', 'markdown'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['docx']
    }
    
    # Magic numbers for file type verification
    FILE_SIGNATURES = {
        b'%PDF': 'pdf',
        b'PK\x03\x04': 'docx',  # Also zip, but we check extension
        b'\x89PNG': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif'
    }
    
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        
    def validate_file(self, file: FileStorage) -> Tuple[bool, Optional[str], dict]:
        """
        Comprehensive file validation
        
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {
            'filename': file.filename,
            'size': 0,
            'mime_type': None,
            'extension': None,
            'hash': None
        }
        
        # Check filename
        if not file.filename:
            return False, "No filename provided", metadata
            
        # Get file extension
        extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        metadata['extension'] = extension
        
        if not extension:
            return False, "File must have an extension", metadata
            
        # Check file size first (before reading content)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        metadata['size'] = file_size
        
        max_size = self.MAX_FILE_SIZES.get(extension, self.MAX_FILE_SIZES['default'])
        if file_size > max_size:
            return False, f"File too large. Maximum size for {extension} is {max_size / 1024 / 1024:.1f}MB", metadata
            
        if file_size == 0:
            return False, "File is empty", metadata
            
        # Read first 8KB for magic number detection
        file_header = file.read(8192)
        file.seek(0)
        
        # Verify file type using magic numbers
        detected_type = self._detect_file_type(file_header)
        if detected_type and detected_type != extension:
            return False, f"File content doesn't match extension. Detected: {detected_type}, Extension: {extension}", metadata
            
        # Check MIME type
        try:
            mime_type = self.mime.from_buffer(file_header)
            metadata['mime_type'] = mime_type
            
            # Verify MIME type is allowed
            if mime_type not in self.ALLOWED_MIME_TYPES:
                return False, f"File type not allowed: {mime_type}", metadata
                
            # Verify extension matches MIME type
            allowed_extensions = self.ALLOWED_MIME_TYPES[mime_type]
            if extension not in allowed_extensions:
                return False, f"Extension {extension} doesn't match MIME type {mime_type}", metadata
                
        except Exception as e:
            logger.error(f"Error detecting MIME type: {e}")
            return False, "Could not determine file type", metadata
            
        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(file)
        metadata['hash'] = file_hash
        
        # Additional validation for specific file types
        if extension == 'json':
            is_valid, error = self._validate_json(file)
            if not is_valid:
                return False, error, metadata
                
        elif extension == 'csv':
            is_valid, error = self._validate_csv(file)
            if not is_valid:
                return False, error, metadata
                
        return True, None, metadata
        
    def _detect_file_type(self, header: bytes) -> Optional[str]:
        """Detect file type from magic numbers"""
        for signature, file_type in self.FILE_SIGNATURES.items():
            if header.startswith(signature):
                return file_type
        return None
        
    def _calculate_hash(self, file: BinaryIO, chunk_size: int = 8192) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        file.seek(0)
        
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha256_hash.update(chunk)
            
        file.seek(0)
        return sha256_hash.hexdigest()
        
    def _validate_json(self, file: BinaryIO) -> Tuple[bool, Optional[str]]:
        """Validate JSON file structure"""
        import json
        try:
            content = file.read()
            file.seek(0)
            json.loads(content)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, f"Error reading JSON: {str(e)}"
            
    def _validate_csv(self, file: BinaryIO) -> Tuple[bool, Optional[str]]:
        """Validate CSV file structure"""
        import csv
        import io
        
        try:
            # Read first few lines to validate
            file.seek(0)
            sample = file.read(8192).decode('utf-8', errors='ignore')
            file.seek(0)
            
            # Try to parse as CSV
            reader = csv.reader(io.StringIO(sample))
            rows = list(reader)
            
            if len(rows) < 1:
                return False, "CSV file is empty"
                
            # Check if all rows have same number of columns
            if len(rows) > 1:
                col_count = len(rows[0])
                for i, row in enumerate(rows[1:], 1):
                    if len(row) != col_count:
                        return False, f"Inconsistent column count at row {i+1}"
                        
            return True, None
            
        except Exception as e:
            return False, f"Error parsing CSV: {str(e)}"
            
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        import re
        # Remove path components
        filename = os.path.basename(filename)
        # Remove special characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        return f"{name}{ext}"