"""
MultiModal Processor - Handles PDF, Images, and Audio/Video
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MultiModalProcessor:
    def __init__(self):
        self.vision_enabled = False
        self.ocr_enabled = False
        
        # Check for Tesseract
        try:
            import pytesseract
            from PIL import Image
            self.ocr_enabled = True
        except ImportError:
            logger.warning("pytesseract or PIL not found. OCR disabled.")
            
        # Check for PDF support
        try:
            import pypdf
        except ImportError:
            logger.warning("pypdf not found. PDF support limited.")

    def process_file(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Process any file type and return extracted text and metadata."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            return self._process_pdf(filepath)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self._process_image(filepath)
        elif ext in ['.mp3', '.wav', '.m4a']:
            return self._process_audio(filepath)
        elif ext in ['.mp4', '.mov', '.avi']:
            return self._process_video(filepath)
        else:
            return self._process_text(filepath)

    def _process_pdf(self, filepath: str) -> Dict[str, Any]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return {"content": text, "type": "pdf", "pages": len(reader.pages)}
        except ImportError:
            return {"error": "pypdf not installed", "type": "pdf"}
        except Exception as e:
            logger.error(f"PDF error: {e}")
            return {"error": str(e), "type": "pdf"}

    def _process_image(self, filepath: str) -> Dict[str, Any]:
        if not self.ocr_enabled:
            return {"error": "OCR not enabled (install pytesseract and Pillow)", "type": "image"}
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(filepath))
            return {"content": text, "type": "image"}
        except Exception as e:
            logger.error(f"Image error: {e}")
            return {"error": str(e), "type": "image"}

    def _process_audio(self, filepath: str) -> Dict[str, Any]:
        # Placeholder for Whisper or SpeechRecognition
        return {"content": "[Audio processing requires Whisper - Implementation Pending]", "type": "audio", "warning": "Not implemented"}

    def _process_video(self, filepath: str) -> Dict[str, Any]:
        return {"content": "[Video processing requires extensive dependencies - Implementation Pending]", "type": "video", "warning": "Not implemented"}

    def _process_text(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return {"content": f.read(), "type": "text"}
        except UnicodeDecodeError:
             try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return {"content": f.read(), "type": "text"}
             except Exception as e:
                 return {"error": str(e), "type": "text"}
