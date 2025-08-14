"""
Streaming file processor for memory-efficient handling of large files
"""
import os
import io
import tempfile
from typing import Iterator, Optional, BinaryIO, Callable, Any, Tuple, List
import logging
from contextlib import contextmanager
import mmap

logger = logging.getLogger(__name__)

class StreamingFileProcessor:
    """Process large files without loading them entirely into memory"""
    
    # Chunk size for streaming (1MB)
    CHUNK_SIZE = 1024 * 1024
    
    # Memory map threshold (files larger than 10MB use mmap)
    MMAP_THRESHOLD = 10 * 1024 * 1024
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    @contextmanager
    def process_file_stream(self, file_path: str, mode: str = 'rb'):
        """
        Context manager for streaming file processing
        
        Automatically chooses between regular file reading and memory mapping
        based on file size
        """
        file_size = os.path.getsize(file_path)
        
        if file_size > self.MMAP_THRESHOLD and mode == 'rb':
            # Use memory mapping for large files
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                    yield mmapped
        else:
            # Regular file reading for smaller files
            with open(file_path, mode) as f:
                yield f
                
    def process_text_chunks(self, 
                          file_obj: BinaryIO, 
                          chunk_processor: Callable[[str, int], Any],
                          encoding: str = 'utf-8') -> Iterator[Any]:
        """
        Process text file in chunks
        
        Args:
            file_obj: File object to read from
            chunk_processor: Function to process each chunk (chunk_text, chunk_index) -> result
            encoding: Text encoding
            
        Yields:
            Results from chunk_processor
        """
        chunk_index = 0
        remainder = b''
        
        while True:
            chunk = file_obj.read(self.CHUNK_SIZE)
            if not chunk:
                # Process any remaining data
                if remainder:
                    try:
                        text = remainder.decode(encoding)
                        yield chunk_processor(text, chunk_index)
                    except Exception as e:
                        logger.error(f"Error processing final chunk: {e}")
                break
                
            # Combine with remainder from previous chunk
            chunk = remainder + chunk
            
            # Find last complete line (avoid splitting UTF-8 characters)
            last_newline = chunk.rfind(b'\n')
            if last_newline != -1:
                # Process up to last complete line
                try:
                    text = chunk[:last_newline + 1].decode(encoding)
                    yield chunk_processor(text, chunk_index)
                except UnicodeDecodeError as e:
                    logger.error(f"Unicode decode error in chunk {chunk_index}: {e}")
                    # Try with error handling
                    text = chunk[:last_newline + 1].decode(encoding, errors='replace')
                    yield chunk_processor(text, chunk_index)
                    
                remainder = chunk[last_newline + 1:]
            else:
                # No complete line, save entire chunk
                remainder = chunk
                
            chunk_index += 1
            
    def process_csv_streaming(self, 
                            file_obj: BinaryIO,
                            row_processor: Callable[[dict, int], Any],
                            has_header: bool = True) -> Iterator[Any]:
        """
        Stream process CSV files row by row
        
        Args:
            file_obj: File object to read from
            row_processor: Function to process each row (row_dict, row_index) -> result
            has_header: Whether CSV has header row
            
        Yields:
            Results from row_processor
        """
        import csv
        import io
        
        text_stream = io.TextIOWrapper(file_obj, encoding='utf-8', newline='')
        reader = csv.DictReader(text_stream) if has_header else csv.reader(text_stream)
        
        for row_index, row in enumerate(reader):
            try:
                result = row_processor(row, row_index)
                if result is not None:
                    yield result
            except Exception as e:
                logger.error(f"Error processing row {row_index}: {e}")
                continue
                
    def process_json_streaming(self,
                             file_obj: BinaryIO,
                             item_processor: Callable[[dict, int], Any]) -> Iterator[Any]:
        """
        Stream process JSON files (assumes JSON array or newline-delimited JSON)
        
        Args:
            file_obj: File object to read from
            item_processor: Function to process each JSON item
            
        Yields:
            Results from item_processor
        """
        import json
        
        # Try to detect JSON format
        first_char = file_obj.read(1)
        file_obj.seek(0)
        
        if first_char == b'[':
            # JSON array - use streaming parser
            yield from self._process_json_array_streaming(file_obj, item_processor)
        else:
            # Assume newline-delimited JSON
            yield from self._process_json_lines(file_obj, item_processor)
            
    def _process_json_array_streaming(self,
                                    file_obj: BinaryIO,
                                    item_processor: Callable[[dict, int], Any]) -> Iterator[Any]:
        """Process JSON array using streaming parser"""
        import ijson
        
        parser = ijson.items(file_obj, 'item')
        for index, item in enumerate(parser):
            try:
                result = item_processor(item, index)
                if result is not None:
                    yield result
            except Exception as e:
                logger.error(f"Error processing JSON item {index}: {e}")
                continue
                
    def _process_json_lines(self,
                          file_obj: BinaryIO,
                          item_processor: Callable[[dict, int], Any]) -> Iterator[Any]:
        """Process newline-delimited JSON"""
        import json
        
        for index, line in enumerate(file_obj):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                result = item_processor(item, index)
                if result is not None:
                    yield result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line {index}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing JSON line {index}: {e}")
                continue
                
    def save_upload_streaming(self,
                            uploaded_file,
                            destination: str,
                            validate_chunk: Optional[Callable[[bytes], bool]] = None) -> Tuple[bool, str]:
        """
        Save uploaded file using streaming to avoid memory issues
        
        Args:
            uploaded_file: Werkzeug FileStorage object
            destination: Destination file path
            validate_chunk: Optional function to validate chunks during upload
            
        Returns:
            Tuple of (success, error_message)
        """
        bytes_written = 0
        
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            with open(destination, 'wb') as f:
                while True:
                    chunk = uploaded_file.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                        
                    # Optional chunk validation (e.g., virus scanning)
                    if validate_chunk and not validate_chunk(chunk):
                        os.unlink(destination)
                        return False, "File validation failed"
                        
                    f.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Log progress for large files
                    if bytes_written % (10 * self.CHUNK_SIZE) == 0:
                        logger.info(f"Uploaded {bytes_written / 1024 / 1024:.1f}MB to {destination}")
                        
            return True, None
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            if os.path.exists(destination):
                os.unlink(destination)
            return False, str(e)
            
    @contextmanager
    def temporary_file(self, suffix: str = '', prefix: str = 'sum_temp_'):
        """
        Context manager for temporary files with automatic cleanup
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.temp_dir)
        
        try:
            os.close(temp_fd)  # Close the file descriptor
            yield temp_path
        finally:
            # Ensure cleanup even if exception occurs
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_path}: {e}")
                
    def estimate_memory_usage(self, file_path: str, processing_type: str = 'text') -> int:
        """
        Estimate memory usage for processing a file
        
        Returns:
            Estimated memory usage in bytes
        """
        file_size = os.path.getsize(file_path)
        
        # Estimation multipliers based on processing type
        multipliers = {
            'text': 2.5,      # Text processing with tokenization
            'csv': 3.0,       # CSV with parsing overhead  
            'json': 4.0,      # JSON parsing overhead
            'binary': 1.2     # Binary processing
        }
        
        multiplier = multipliers.get(processing_type, 2.0)
        return int(file_size * multiplier)