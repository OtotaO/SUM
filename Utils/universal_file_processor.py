"""
universal_file_processor.py - Universal File Processing with Intelligent Fallbacks

This module provides universal file processing capabilities that can handle
ANY file type through intelligent content extraction and fallback mechanisms.

Key Features:
- Automatic file type detection
- Multiple extraction strategies
- Graceful fallback for unknown types
- Binary file handling
- Encoding detection

Author: ototao
License: Apache License 2.0
"""

import os
import mimetypes
import chardet
import logging
from typing import Dict, Any, Optional, Tuple
import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path

# Optional imports for enhanced processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import python_docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False

try:
    import pandas as pd
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

logger = logging.getLogger(__name__)


class UniversalFileProcessor:
    """
    Universal file processor that can extract text from any file type.
    Uses multiple strategies with intelligent fallbacks.
    """
    
    def __init__(self):
        """Initialize processor with mime type mappings."""
        self.processors = {
            'text/plain': self._process_text,
            'text/html': self._process_html,
            'text/xml': self._process_xml,
            'application/json': self._process_json,
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/msword': self._process_doc,
            'text/csv': self._process_csv,
            'application/vnd.ms-excel': self._process_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel,
        }
        
        # Code file extensions
        self.code_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.go',
            '.rs', '.swift', '.kt', '.scala', '.r', '.php', '.pl', '.sh', '.bat',
            '.ps1', '.tsx', '.jsx', '.vue', '.svelte', '.lua', '.sql', '.yaml',
            '.yml', '.toml', '.ini', '.cfg', '.conf', '.json', '.xml', '.html',
            '.css', '.scss', '.sass', '.less'
        }
        
    def process_file(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """
        Process any file and extract its text content.
        
        Args:
            file_path: Path to the file
            encoding: Optional encoding override
            
        Returns:
            Dict containing extracted text and metadata
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': 'File not found',
                'text': '',
                'metadata': {}
            }
        
        file_info = self._analyze_file(file_path)
        
        # Try specific processor based on mime type
        mime_type = file_info['mime_type']
        if mime_type in self.processors:
            try:
                result = self.processors[mime_type](file_path, encoding)
                if result['success']:
                    result['metadata'].update(file_info)
                    return result
            except Exception as e:
                logger.warning(f"Specific processor failed for {mime_type}: {e}")
        
        # Fallback strategies
        return self._fallback_processing(file_path, file_info, encoding)
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze file to determine type and characteristics."""
        path = Path(file_path)
        stat = path.stat()
        
        # Guess mime type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Check if it's likely a code file
        is_code = path.suffix.lower() in self.code_extensions
        
        return {
            'filename': path.name,
            'extension': path.suffix.lower(),
            'size': stat.st_size,
            'mime_type': mime_type,
            'is_code': is_code,
            'is_binary': self._is_binary(file_path)
        }
    
    def _is_binary(self, file_path: str, sample_size: int = 8192) -> bool:
        """Check if file is binary by examining first bytes."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(sample_size)
                
            # Check for null bytes
            if b'\x00' in chunk:
                return True
            
            # Check if mostly printable
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
            non_text = chunk.translate(None, text_chars)
            
            return len(non_text) / len(chunk) > 0.3
        except:
            return True
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _process_text(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process plain text files."""
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
            
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'encoding': encoding,
                    'line_count': len(text.splitlines()),
                    'word_count': len(text.split())
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_pdf(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process PDF files."""
        if not PDF_SUPPORT:
            return self._process_text(file_path, encoding)
        
        try:
            text_parts = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text_parts.append(page.extract_text())
            
            return {
                'success': True,
                'text': '\n\n'.join(text_parts),
                'metadata': {
                    'page_count': num_pages,
                    'file_type': 'pdf'
                }
            }
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_docx(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process DOCX files."""
        if not DOCX_SUPPORT:
            return self._extract_zip_text(file_path)
        
        try:
            import python_docx
            doc = python_docx.Document(file_path)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text_parts.append(cell.text)
            
            return {
                'success': True,
                'text': '\n\n'.join(text_parts),
                'metadata': {
                    'paragraph_count': len(doc.paragraphs),
                    'table_count': len(doc.tables),
                    'file_type': 'docx'
                }
            }
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            return self._extract_zip_text(file_path)
    
    def _process_doc(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process old DOC files - fallback to text extraction."""
        # For old .doc files, we'll use a simple text extraction
        return self._process_text(file_path, 'latin-1')
    
    def _process_html(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process HTML files."""
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            if HTML_SUPPORT:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return {
                    'success': True,
                    'text': text,
                    'metadata': {
                        'title': soup.title.string if soup.title else '',
                        'file_type': 'html'
                    }
                }
            else:
                # Fallback: remove HTML tags with regex
                text = re.sub(r'<[^>]+>', '', content)
                return {
                    'success': True,
                    'text': text,
                    'metadata': {'file_type': 'html'}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_xml(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process XML files."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract all text content
            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            
            return {
                'success': True,
                'text': '\n'.join(text_parts),
                'metadata': {
                    'root_tag': root.tag,
                    'file_type': 'xml'
                }
            }
        except Exception as e:
            # Fallback to text processing
            return self._process_text(file_path, encoding)
    
    def _process_json(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process JSON files."""
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Convert to readable text
            text = json.dumps(data, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'file_type': 'json',
                    'encoding': encoding
                }
            }
        except json.JSONDecodeError:
            # Fallback to text processing
            return self._process_text(file_path, encoding)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_csv(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process CSV files."""
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        try:
            if EXCEL_SUPPORT:
                df = pd.read_csv(file_path, encoding=encoding)
                text = df.to_string()
                
                return {
                    'success': True,
                    'text': text,
                    'metadata': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'file_type': 'csv'
                    }
                }
            else:
                # Fallback to text processing
                return self._process_text(file_path, encoding)
                
        except Exception as e:
            return self._process_text(file_path, encoding)
    
    def _process_excel(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """Process Excel files."""
        if not EXCEL_SUPPORT:
            return {
                'success': False,
                'error': 'Excel support not available',
                'text': '',
                'metadata': {}
            }
        
        try:
            # Read all sheets
            xl_file = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                text_parts.append(df.to_string())
                text_parts.append("")
            
            return {
                'success': True,
                'text': '\n'.join(text_parts),
                'metadata': {
                    'sheet_count': len(xl_file.sheet_names),
                    'file_type': 'excel'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _extract_zip_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from ZIP-based formats (docx, xlsx, etc)."""
        try:
            import zipfile
            text_parts = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for name in zip_file.namelist():
                    if name.endswith('.xml'):
                        with zip_file.open(name) as xml_file:
                            content = xml_file.read().decode('utf-8', errors='ignore')
                            # Simple XML text extraction
                            text = re.sub(r'<[^>]+>', ' ', content)
                            text = re.sub(r'\s+', ' ', text).strip()
                            if text:
                                text_parts.append(text)
            
            return {
                'success': True,
                'text': '\n\n'.join(text_parts),
                'metadata': {'extraction_method': 'zip_xml'}
            }
        except:
            return {
                'success': False,
                'error': 'ZIP extraction failed',
                'text': '',
                'metadata': {}
            }
    
    def _fallback_processing(self, file_path: str, file_info: Dict[str, Any], 
                           encoding: Optional[str] = None) -> Dict[str, Any]:
        """Fallback processing for unknown file types."""
        
        # Strategy 1: If it looks like code or text, try text processing
        if file_info['is_code'] or not file_info['is_binary']:
            result = self._process_text(file_path, encoding)
            if result['success']:
                return result
        
        # Strategy 2: Try different encodings
        for enc in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    text = f.read()
                
                # Check if we got reasonable text
                if text and len([c for c in text[:1000] if c == '�']) / len(text[:1000]) < 0.1:
                    return {
                        'success': True,
                        'text': text,
                        'metadata': {
                            'encoding': enc,
                            'fallback_method': 'encoding_detection'
                        }
                    }
            except:
                continue
        
        # Strategy 3: Binary file - extract strings
        if file_info['is_binary']:
            text = self._extract_strings_from_binary(file_path)
            if text:
                return {
                    'success': True,
                    'text': text,
                    'metadata': {
                        'file_type': 'binary',
                        'extraction_method': 'string_extraction'
                    }
                }
        
        # Final fallback
        return {
            'success': False,
            'error': 'Unable to extract text from file',
            'text': '',
            'metadata': file_info
        }
    
    def _extract_strings_from_binary(self, file_path: str, min_length: int = 4) -> str:
        """Extract readable strings from binary files."""
        strings = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Extract ASCII strings
            ascii_strings = re.findall(b'[\x20-\x7E]{%d,}' % min_length, data)
            strings.extend(s.decode('ascii', errors='ignore') for s in ascii_strings)
            
            # Try to extract UTF-16 strings (common in Windows binaries)
            try:
                utf16_text = data.decode('utf-16-le', errors='ignore')
                utf16_strings = re.findall(r'[\x20-\x7E]{%d,}' % min_length, utf16_text)
                strings.extend(utf16_strings)
            except:
                pass
            
            # Filter out likely garbage
            filtered_strings = []
            for s in strings:
                # Check if string has reasonable character distribution
                if len(s) >= min_length and not s.isdigit():
                    alpha_ratio = sum(c.isalpha() for c in s) / len(s)
                    if alpha_ratio > 0.5:  # At least 50% letters
                        filtered_strings.append(s)
            
            return '\n'.join(filtered_strings[:1000])  # Limit to first 1000 strings
            
        except Exception as e:
            logger.error(f"String extraction failed: {e}")
            return ''


# Global processor instance
universal_processor = UniversalFileProcessor()


def process_any_file(file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to process any file type.
    
    Args:
        file_path: Path to the file
        encoding: Optional encoding override
        
    Returns:
        Dict with 'success', 'text', 'metadata', and optionally 'error'
    """
    return universal_processor.process_file(file_path, encoding)