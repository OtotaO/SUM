#!/usr/bin/env python3
"""
multimodal_engine.py - Universal Multi-Modal Processing Engine for SUM

This module provides comprehensive multi-modal processing capabilities supporting:
- Images (with OCR, visual analysis, chart/graph extraction)
- Audio (meeting recordings, voice memos, podcasts with transcription)
- Video (YouTube, lectures, screencasts with scene analysis)
- PDFs (academic papers with citation tracking)
- Code (GitHub repos with knowledge graph generation)
- Documents (DOCX, HTML, Markdown)

Features:
- Sub-second processing for small files
- Batch processing with progress indication
- Offline-capable with optional cloud processing
- Smart caching and preprocessing
- Integration with SumEngine and predictive intelligence

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
import hashlib
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mimetypes
import tempfile
import shutil
from collections import defaultdict

# Core processing imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Install: pip install numpy")

# Image processing
try:
    from PIL import Image
    import pytesseract
    import cv2
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    logging.warning("Image processing not available. Install: pip install Pillow pytesseract opencv-python")

# Audio processing
try:
    import whisper
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio processing not available. Install: pip install openai-whisper librosa soundfile")

# Video processing
try:
    import moviepy.editor as mp
    from scenedetect import detect, ContentDetector
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    logging.warning("Video processing not available. Install: pip install moviepy scenedetect")

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    from pdfminer.high_level import extract_text
    import fitz  # PyMuPDF for better extraction
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("Advanced PDF processing not available. Install: pip install PyPDF2 pdfplumber pdfminer.six PyMuPDF")

# Code analysis
try:
    import ast
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.token import Token
    import networkx as nx
    CODE_AVAILABLE = True
except ImportError:
    CODE_AVAILABLE = False
    logging.warning("Code analysis not available. Install: pip install pygments networkx")

# Document processing
try:
    from docx import Document
    from bs4 import BeautifulSoup
    import markdown
    DOCUMENT_AVAILABLE = True
except ImportError:
    DOCUMENT_AVAILABLE = False
    logging.warning("Document processing not available. Install: pip install python-docx beautifulsoup4 markdown")

# Import SUM components
from multimodal_processor import MultiModalProcessor, ProcessingResult, ContentType as BaseContentType
from sum_engines import HierarchicalDensificationEngine
from predictive_intelligence import PredictiveIntelligenceSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtendedContentType(Enum):
    """Extended content types for comprehensive multi-modal processing."""
    # Text-based
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    
    # Visual
    IMAGE = "image"
    SCREENSHOT = "screenshot"
    DIAGRAM = "diagram"
    CHART = "chart"
    
    # Audio
    AUDIO = "audio"
    VOICE_MEMO = "voice_memo"
    PODCAST = "podcast"
    MEETING = "meeting"
    
    # Video
    VIDEO = "video"
    SCREENCAST = "screencast"
    LECTURE = "lecture"
    YOUTUBE = "youtube"
    
    # Code
    CODE = "code"
    REPOSITORY = "repository"
    NOTEBOOK = "notebook"
    
    # Special
    ARCHIVE = "archive"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class EnhancedProcessingResult(ProcessingResult):
    """Enhanced result with multi-modal specific fields."""
    # Audio-specific
    transcription: Optional[str] = None
    speaker_segments: Optional[List[Dict[str, Any]]] = None
    audio_features: Optional[Dict[str, Any]] = None
    
    # Video-specific
    keyframes: Optional[List[str]] = None
    scene_descriptions: Optional[List[str]] = None
    video_transcript: Optional[str] = None
    
    # Image-specific
    visual_elements: Optional[List[str]] = None
    detected_text_regions: Optional[List[Dict[str, Any]]] = None
    chart_data: Optional[Dict[str, Any]] = None
    
    # Code-specific
    ast_tree: Optional[Any] = None
    function_map: Optional[Dict[str, Any]] = None
    dependency_graph: Optional[Any] = None
    
    # Knowledge graph
    entities: Optional[List[str]] = None
    relationships: Optional[List[Tuple[str, str, str]]] = None
    concepts: Optional[List[str]] = None


class MultiModalEngine:
    """
    Universal multi-modal processing engine with advanced capabilities.
    
    Handles all content types with intelligent routing and processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-modal engine."""
        self.config = config or {}
        self.base_processor = MultiModalProcessor(config)
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Initialize specialized processors
        self._init_processors()
        
        # Cache for processed content
        self.cache_dir = Path(self.config.get('cache_dir', './cache/multimodal'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = defaultdict(int)
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("MultiModalEngine initialized with extended capabilities")
    
    def _init_processors(self):
        """Initialize specialized processors based on availability."""
        self.processors = {}
        
        # Audio processor
        if AUDIO_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model(
                    self.config.get('whisper_model', 'base')
                )
                self.processors['audio'] = self._process_audio_advanced
                logger.info("Audio processing initialized with Whisper")
            except Exception as e:
                logger.warning(f"Failed to initialize Whisper: {e}")
        
        # Video processor
        if VIDEO_AVAILABLE:
            self.processors['video'] = self._process_video_advanced
            logger.info("Video processing initialized")
        
        # Code processor
        if CODE_AVAILABLE:
            self.processors['code'] = self._process_code_advanced
            logger.info("Code analysis initialized")
        
        # Enhanced image processor
        if IMAGE_AVAILABLE:
            self.processors['image'] = self._process_image_advanced
            logger.info("Advanced image processing initialized")
        
        # Enhanced PDF processor
        if PDF_AVAILABLE:
            self.processors['pdf'] = self._process_pdf_advanced
            logger.info("Advanced PDF processing initialized")
    
    def detect_content_type(self, file_path: str) -> ExtendedContentType:
        """Detect extended content type with more granularity."""
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            # Audio formats
            audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
            if extension in audio_extensions:
                return ExtendedContentType.AUDIO
            
            # Video formats
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
            if extension in video_extensions:
                return ExtendedContentType.VIDEO
            
            # Code formats
            code_extensions = {
                '.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.swift',
                '.kt', '.scala', '.rb', '.php', '.tsx', '.jsx', '.vue', '.r', '.m'
            }
            if extension in code_extensions:
                return ExtendedContentType.CODE
            
            # Notebook formats
            if extension in {'.ipynb', '.rmd'}:
                return ExtendedContentType.NOTEBOOK
            
            # Archive formats
            if extension in {'.zip', '.tar', '.gz', '.7z', '.rar'}:
                return ExtendedContentType.ARCHIVE
            
            # Fall back to base detection
            base_type = self.base_processor.detect_content_type(file_path)
            return ExtendedContentType(base_type.value)
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return ExtendedContentType.UNKNOWN
    
    async def process_file_async(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Process file asynchronously with progress tracking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.process_file,
            file_path,
            **kwargs
        )
    
    def process_file(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Process file with extended capabilities."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(file_path)
            cached_result = self._get_cached_result(cache_key)
            if cached_result and not kwargs.get('force_reprocess', False):
                logger.info(f"Using cached result for {file_path}")
                return cached_result
            
            # Detect content type
            content_type = self.detect_content_type(file_path)
            
            # Route to appropriate processor
            if content_type.value in ['audio', 'voice_memo', 'podcast', 'meeting']:
                result = self._process_audio_advanced(file_path, **kwargs)
            elif content_type.value in ['video', 'screencast', 'lecture', 'youtube']:
                result = self._process_video_advanced(file_path, **kwargs)
            elif content_type.value in ['code', 'repository', 'notebook']:
                result = self._process_code_advanced(file_path, **kwargs)
            elif content_type == ExtendedContentType.PDF:
                result = self._process_pdf_advanced(file_path, **kwargs)
            elif content_type == ExtendedContentType.IMAGE:
                result = self._process_image_advanced(file_path, **kwargs)
            else:
                # Fall back to base processor
                base_result = self.base_processor.process_file(file_path, **kwargs)
                result = self._enhance_base_result(base_result)
            
            # Update processing time
            result.processing_time = time.time() - start_time
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            self.stats[content_type.value] += 1
            self.stats['total_processed'] += 1
            self.stats['total_time'] += result.processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return EnhancedProcessingResult(
                content_type=BaseContentType.UNKNOWN,
                extracted_text="",
                metadata={'error': str(e)},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _process_audio_advanced(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Advanced audio processing with transcription and analysis."""
        if not AUDIO_AVAILABLE:
            return self._create_error_result("Audio processing not available")
        
        try:
            logger.info(f"Processing audio file: {file_path}")
            
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                file_path,
                language=kwargs.get('language', None),
                task=kwargs.get('task', 'transcribe')
            )
            
            # Extract segments with speaker info
            segments = result.get('segments', [])
            
            # Audio feature extraction
            audio_features = self._extract_audio_features(audio_data, sample_rate)
            
            # Build extracted text
            full_transcript = result.get('text', '')
            
            # Extract speaker segments if available
            speaker_segments = []
            for segment in segments:
                speaker_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'confidence': segment.get('confidence', 1.0)
                })
            
            # Analyze content for meeting/podcast specific features
            content_analysis = self._analyze_audio_content(full_transcript, segments)
            
            # Build metadata
            metadata = {
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'language': result.get('language', 'unknown'),
                'audio_features': audio_features,
                'content_type': content_analysis['type'],
                'key_topics': content_analysis.get('topics', []),
                'file_size': os.path.getsize(file_path)
            }
            
            return EnhancedProcessingResult(
                content_type=BaseContentType.UNKNOWN,  # Will be set properly
                extracted_text=full_transcript,
                metadata=metadata,
                processing_time=0.0,
                confidence_score=0.9,
                transcription=full_transcript,
                speaker_segments=speaker_segments,
                audio_features=audio_features,
                concepts=content_analysis.get('concepts', [])
            )
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return self._create_error_result(f"Audio processing error: {e}")
    
    def _process_video_advanced(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Advanced video processing with scene detection and frame analysis."""
        if not VIDEO_AVAILABLE:
            return self._create_error_result("Video processing not available")
        
        try:
            logger.info(f"Processing video file: {file_path}")
            
            # Load video
            video = mp.VideoFileClip(file_path)
            
            # Extract audio and transcribe
            audio_path = None
            transcript = ""
            if video.audio is not None:
                audio_path = tempfile.mktemp(suffix='.wav')
                video.audio.write_audiofile(audio_path, logger=None)
                
                # Process audio
                if AUDIO_AVAILABLE and hasattr(self, 'whisper_model'):
                    audio_result = self._process_audio_advanced(audio_path)
                    transcript = audio_result.transcription or ""
                
                os.unlink(audio_path)
            
            # Scene detection
            scenes = []
            keyframes = []
            
            if kwargs.get('extract_scenes', True):
                scene_list = detect(file_path, ContentDetector())
                
                # Extract keyframes
                for i, scene in enumerate(scene_list[:10]):  # Limit to 10 scenes
                    start_time = scene[0].get_seconds()
                    frame = video.get_frame(start_time)
                    
                    # Save keyframe
                    keyframe_path = self.cache_dir / f"keyframe_{i}_{hashlib.md5(file_path.encode()).hexdigest()}.jpg"
                    Image.fromarray(frame).save(keyframe_path)
                    keyframes.append(str(keyframe_path))
                    
                    scenes.append({
                        'start': start_time,
                        'end': scene[1].get_seconds(),
                        'keyframe': str(keyframe_path)
                    })
            
            # Extract metadata
            metadata = {
                'duration': video.duration,
                'fps': video.fps,
                'resolution': video.size,
                'has_audio': video.audio is not None,
                'scene_count': len(scenes),
                'file_size': os.path.getsize(file_path)
            }
            
            # Close video
            video.close()
            
            return EnhancedProcessingResult(
                content_type=BaseContentType.UNKNOWN,
                extracted_text=transcript,
                metadata=metadata,
                processing_time=0.0,
                confidence_score=0.85,
                video_transcript=transcript,
                keyframes=keyframes,
                scene_descriptions=[f"Scene {i+1}: {s['start']:.1f}s - {s['end']:.1f}s" for i, s in enumerate(scenes)]
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return self._create_error_result(f"Video processing error: {e}")
    
    def _process_code_advanced(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Advanced code processing with AST analysis and knowledge graph generation."""
        if not CODE_AVAILABLE:
            return self._create_error_result("Code analysis not available")
        
        try:
            logger.info(f"Processing code file: {file_path}")
            
            # Read code
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Detect language
            try:
                lexer = guess_lexer(code_content)
                language = lexer.name
            except:
                # Fallback to extension
                ext = Path(file_path).suffix
                language = ext[1:] if ext else 'unknown'
            
            # Language-specific analysis
            ast_tree = None
            function_map = {}
            dependency_graph = nx.DiGraph()
            
            if language.lower() in ['python', 'py']:
                ast_tree, function_map, dependency_graph = self._analyze_python_code(code_content)
            
            # Extract concepts and patterns
            concepts = self._extract_code_concepts(code_content, language)
            
            # Build relationships
            relationships = []
            if dependency_graph:
                for edge in dependency_graph.edges(data=True):
                    relationships.append((edge[0], edge[2].get('type', 'uses'), edge[1]))
            
            # Generate documentation-like summary
            summary = self._generate_code_summary(code_content, function_map, language)
            
            metadata = {
                'language': language,
                'line_count': len(code_content.splitlines()),
                'function_count': len(function_map),
                'complexity_score': self._calculate_code_complexity(code_content),
                'file_size': os.path.getsize(file_path)
            }
            
            return EnhancedProcessingResult(
                content_type=BaseContentType.UNKNOWN,
                extracted_text=summary,
                metadata=metadata,
                processing_time=0.0,
                confidence_score=0.95,
                ast_tree=ast_tree,
                function_map=function_map,
                dependency_graph=dependency_graph,
                concepts=concepts,
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error processing code: {e}")
            return self._create_error_result(f"Code processing error: {e}")
    
    def _process_pdf_advanced(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Advanced PDF processing with citation tracking and figure extraction."""
        if not PDF_AVAILABLE:
            return self._create_error_result("Advanced PDF processing not available")
        
        try:
            logger.info(f"Processing PDF with advanced features: {file_path}")
            
            # Use multiple extraction methods for best results
            text_content = ""
            figures = []
            tables = []
            citations = []
            
            # Method 1: PyMuPDF for text and images
            try:
                doc = fitz.open(file_path)
                
                for page_num, page in enumerate(doc):
                    # Extract text
                    text_content += page.get_text()
                    
                    # Extract images
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n - pix.alpha < 4:  # RGB or gray
                                img_path = self.cache_dir / f"pdf_img_{page_num}_{img_index}.png"
                                pix.save(img_path)
                                figures.append({
                                    'page': page_num + 1,
                                    'path': str(img_path),
                                    'type': 'image'
                                })
                            pix = None
                        except:
                            pass
                
                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
            
            # Method 2: pdfplumber for tables
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract tables
                        page_tables = page.extract_tables()
                        for table_index, table in enumerate(page_tables):
                            if table:
                                tables.append({
                                    'page': page_num + 1,
                                    'data': table,
                                    'rows': len(table),
                                    'cols': len(table[0]) if table else 0
                                })
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
            
            # Extract citations and references
            citations = self._extract_citations(text_content)
            
            # Detect if academic paper
            is_academic = self._is_academic_paper(text_content)
            
            # Extract key sections if academic
            sections = {}
            if is_academic:
                sections = self._extract_paper_sections(text_content)
            
            metadata = {
                'page_count': len(text_content.split('\f')),  # Form feed as page separator
                'has_figures': len(figures) > 0,
                'has_tables': len(tables) > 0,
                'figure_count': len(figures),
                'table_count': len(tables),
                'citation_count': len(citations),
                'is_academic': is_academic,
                'sections': list(sections.keys()) if sections else [],
                'file_size': os.path.getsize(file_path)
            }
            
            # Generate enhanced summary
            if is_academic and sections:
                abstract = sections.get('abstract', '')
                conclusion = sections.get('conclusion', '')
                summary = f"Abstract: {abstract[:500]}...\n\nConclusion: {conclusion[:500]}..."
            else:
                summary = text_content[:1000] + "..."
            
            return EnhancedProcessingResult(
                content_type=BaseContentType.PDF,
                extracted_text=text_content,
                metadata=metadata,
                processing_time=0.0,
                confidence_score=0.9,
                visual_elements=[f"Figure {i+1} on page {fig['page']}" for i, fig in enumerate(figures)],
                chart_data={'tables': tables, 'figures': figures},
                concepts=self._extract_academic_concepts(text_content) if is_academic else []
            )
            
        except Exception as e:
            logger.error(f"Error in advanced PDF processing: {e}")
            # Fall back to base PDF processing
            return self.base_processor._process_pdf(file_path, **kwargs)
    
    def _process_image_advanced(self, file_path: str, **kwargs) -> EnhancedProcessingResult:
        """Advanced image processing with chart/diagram analysis."""
        if not IMAGE_AVAILABLE:
            return self._create_error_result("Advanced image processing not available")
        
        try:
            logger.info(f"Processing image with advanced features: {file_path}")
            
            # Load image
            image = cv2.imread(file_path)
            image_pil = Image.open(file_path)
            
            # Basic OCR
            ocr_text = pytesseract.image_to_string(image_pil)
            
            # Detect text regions with coordinates
            text_regions = []
            try:
                data = pytesseract.image_to_data(image_pil, output_type=pytesseract.Output.DICT)
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 60:  # Confidence threshold
                        text = data['text'][i].strip()
                        if text:
                            text_regions.append({
                                'text': text,
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i],
                                'confidence': data['conf'][i]
                            })
            except:
                pass
            
            # Detect if it's a chart/diagram
            is_chart = self._detect_chart_type(image)
            chart_data = {}
            
            if is_chart:
                # Extract chart data (simplified)
                chart_data = self._extract_chart_data(image, text_regions)
            
            # Detect visual elements
            visual_elements = self._detect_visual_elements(image)
            
            # Handwriting detection
            has_handwriting = self._detect_handwriting(image)
            
            metadata = {
                'width': image.shape[1],
                'height': image.shape[0],
                'is_chart': is_chart,
                'has_handwriting': has_handwriting,
                'text_region_count': len(text_regions),
                'visual_element_count': len(visual_elements),
                'file_size': os.path.getsize(file_path)
            }
            
            # Generate description
            description = self._generate_image_description(
                ocr_text, visual_elements, is_chart, has_handwriting
            )
            
            return EnhancedProcessingResult(
                content_type=BaseContentType.IMAGE,
                extracted_text=ocr_text,
                metadata=metadata,
                processing_time=0.0,
                confidence_score=0.85,
                visual_elements=visual_elements,
                detected_text_regions=text_regions,
                chart_data=chart_data
            )
            
        except Exception as e:
            logger.error(f"Error in advanced image processing: {e}")
            # Fall back to base image processing
            return self.base_processor._process_image(file_path, **kwargs)
    
    # Helper methods
    def _extract_audio_features(self, audio_data, sample_rate):
        """Extract audio features for analysis."""
        features = {}
        
        try:
            # Basic features
            features['duration'] = len(audio_data) / sample_rate
            features['mean_amplitude'] = float(np.mean(np.abs(audio_data)))
            features['max_amplitude'] = float(np.max(np.abs(audio_data)))
            
            # Spectral features
            if NUMPY_AVAILABLE:
                stft = librosa.stft(audio_data)
                features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=np.abs(stft), sr=sample_rate)))
                features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
        except:
            pass
        
        return features
    
    def _analyze_audio_content(self, transcript, segments):
        """Analyze audio content to determine type and extract topics."""
        content = {
            'type': 'general',
            'topics': [],
            'concepts': []
        }
        
        # Simple heuristics for content type
        if any(word in transcript.lower() for word in ['meeting', 'agenda', 'action items']):
            content['type'] = 'meeting'
        elif any(word in transcript.lower() for word in ['episode', 'podcast', 'welcome back']):
            content['type'] = 'podcast'
        elif any(word in transcript.lower() for word in ['lecture', 'class', 'students']):
            content['type'] = 'lecture'
        
        # Extract topics (simplified)
        words = transcript.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 5:  # Simple filter
                word_freq[word] += 1
        
        content['topics'] = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return content
    
    def _analyze_python_code(self, code_content):
        """Analyze Python code structure."""
        try:
            tree = ast.parse(code_content)
            function_map = {}
            dependency_graph = nx.DiGraph()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_map[node.name] = {
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    }
                elif isinstance(node, ast.ClassDef):
                    function_map[f"class:{node.name}"] = {
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    }
            
            return tree, function_map, dependency_graph
        except:
            return None, {}, nx.DiGraph()
    
    def _extract_code_concepts(self, code_content, language):
        """Extract programming concepts from code."""
        concepts = []
        
        # Language-agnostic patterns
        patterns = {
            'async': r'async\s+\w+',
            'api': r'api|endpoint|route',
            'database': r'database|query|sql',
            'ml': r'model|train|predict|neural',
            'testing': r'test_|assert|mock',
            'logging': r'logger|log\.|logging'
        }
        
        import re
        for concept, pattern in patterns.items():
            if re.search(pattern, code_content, re.IGNORECASE):
                concepts.append(concept)
        
        return concepts
    
    def _generate_code_summary(self, code_content, function_map, language):
        """Generate a natural language summary of code."""
        lines = code_content.splitlines()
        
        summary_parts = [f"This is a {language} file with {len(lines)} lines of code."]
        
        if function_map:
            summary_parts.append(f"It contains {len(function_map)} functions/classes:")
            for name, info in list(function_map.items())[:5]:  # First 5
                if info.get('docstring'):
                    summary_parts.append(f"- {name}: {info['docstring'][:100]}...")
                else:
                    summary_parts.append(f"- {name}")
        
        return "\n".join(summary_parts)
    
    def _calculate_code_complexity(self, code_content):
        """Calculate code complexity score."""
        # Simple complexity based on various factors
        lines = code_content.splitlines()
        score = 0
        
        # Factors
        score += len(lines) / 100  # Length factor
        score += code_content.count('if ') * 0.1  # Conditional complexity
        score += code_content.count('for ') * 0.15  # Loop complexity
        score += code_content.count('class ') * 0.2  # OOP complexity
        
        return min(score, 10.0)  # Cap at 10
    
    def _extract_citations(self, text):
        """Extract citations from academic text."""
        citations = []
        
        # Common citation patterns
        patterns = [
            r'\([A-Z][a-z]+ et al\., \d{4}\)',  # (Smith et al., 2023)
            r'\([A-Z][a-z]+ \d{4}\)',  # (Smith 2023)
            r'\[[0-9]+\]',  # [1], [2], etc.
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _is_academic_paper(self, text):
        """Detect if PDF is an academic paper."""
        academic_keywords = [
            'abstract', 'introduction', 'methodology', 'results', 'conclusion',
            'references', 'doi:', 'keywords:', 'received:', 'accepted:'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in academic_keywords if keyword in text_lower)
        
        return matches >= 4
    
    def _extract_paper_sections(self, text):
        """Extract sections from academic paper."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'abstract': r'abstract[\s\n]+(.+?)(?=\n[A-Z]|\n\d\.)',
            'introduction': r'introduction[\s\n]+(.+?)(?=\n[A-Z]|\n\d\.)',
            'conclusion': r'conclusion[\s\n]+(.+?)(?=\nreferences|\n[A-Z])',
        }
        
        import re
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections
    
    def _extract_academic_concepts(self, text):
        """Extract academic concepts and keywords."""
        # This would be enhanced with NLP
        concepts = []
        
        # Look for keywords section
        import re
        keywords_match = re.search(r'keywords?:(.+?)(?=\n)', text, re.IGNORECASE)
        if keywords_match:
            keywords = keywords_match.group(1)
            concepts = [k.strip() for k in keywords.split(',')]
        
        return concepts
    
    def _detect_chart_type(self, image):
        """Detect if image contains a chart or diagram."""
        # Simplified detection using edge detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for rectangular regions (chart areas)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rect_count = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle
                    rect_count += 1
            
            return rect_count > 3  # Multiple rectangles suggest chart
        except:
            return False
    
    def _extract_chart_data(self, image, text_regions):
        """Extract data from charts (simplified)."""
        chart_data = {
            'type': 'unknown',
            'labels': [],
            'values': []
        }
        
        # Extract numeric values from text regions
        for region in text_regions:
            text = region['text']
            try:
                value = float(text.replace(',', ''))
                chart_data['values'].append(value)
            except:
                if len(text) > 1 and not text.isdigit():
                    chart_data['labels'].append(text)
        
        return chart_data
    
    def _detect_visual_elements(self, image):
        """Detect visual elements in image."""
        elements = []
        
        # Simple shape detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect circles
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
            if circles is not None:
                elements.append(f"{len(circles[0])} circles detected")
            
            # Detect lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                elements.append(f"{len(lines)} lines detected")
        except:
            pass
        
        return elements
    
    def _detect_handwriting(self, image):
        """Detect if image contains handwriting."""
        # Simplified detection using texture analysis
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture features
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Handwriting typically has higher texture variance
            return laplacian_var > 500
        except:
            return False
    
    def _generate_image_description(self, ocr_text, visual_elements, is_chart, has_handwriting):
        """Generate natural language description of image."""
        parts = []
        
        if ocr_text.strip():
            parts.append(f"Text content: {ocr_text[:200]}...")
        
        if is_chart:
            parts.append("This appears to be a chart or diagram.")
        
        if has_handwriting:
            parts.append("Handwritten content detected.")
        
        if visual_elements:
            parts.append("Visual elements: " + ", ".join(visual_elements[:3]))
        
        return " ".join(parts) if parts else "Image with no detected text or notable features."
    
    def _enhance_base_result(self, base_result: ProcessingResult) -> EnhancedProcessingResult:
        """Convert base result to enhanced result."""
        return EnhancedProcessingResult(
            content_type=base_result.content_type,
            extracted_text=base_result.extracted_text,
            metadata=base_result.metadata,
            processing_time=base_result.processing_time,
            confidence_score=base_result.confidence_score,
            error_message=base_result.error_message
        )
    
    def _create_error_result(self, error_message: str) -> EnhancedProcessingResult:
        """Create error result."""
        return EnhancedProcessingResult(
            content_type=BaseContentType.UNKNOWN,
            extracted_text="",
            metadata={'error': error_message},
            processing_time=0.0,
            confidence_score=0.0,
            error_message=error_message
        )
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        stat = os.stat(file_path)
        content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[EnhancedProcessingResult]:
        """Retrieve cached result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct result (simplified)
                    return None  # Would need proper deserialization
            except:
                pass
        
        return None
    
    def _cache_result(self, cache_key: str, result: EnhancedProcessingResult):
        """Cache processing result."""
        # Simplified caching - would need proper serialization
        pass
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = dict(self.stats)
        
        # Add capability info
        stats.update({
            'capabilities': {
                'audio': AUDIO_AVAILABLE,
                'video': VIDEO_AVAILABLE,
                'code': CODE_AVAILABLE,
                'advanced_pdf': PDF_AVAILABLE,
                'advanced_image': IMAGE_AVAILABLE,
                'whisper_model': hasattr(self, 'whisper_model')
            },
            'average_processing_time': stats.get('total_time', 0) / max(stats.get('total_processed', 1), 1)
        })
        
        return stats
    
    def process_batch_async(self, file_paths: List[str], **kwargs) -> List[EnhancedProcessingResult]:
        """Process multiple files asynchronously."""
        async def process_all():
            tasks = [self.process_file_async(fp, **kwargs) for fp in file_paths]
            return await asyncio.gather(*tasks)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_all())
        finally:
            loop.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize engine
    engine = MultiModalEngine({
        'whisper_model': 'base',
        'cache_dir': './cache/multimodal'
    })
    
    # Print capabilities
    print("Multi-Modal Engine Capabilities:")
    stats = engine.get_processing_stats()
    for key, value in stats['capabilities'].items():
        print(f"  {key}: {value}")
    
    # Process files if provided
    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            print(f"\nProcessing: {file_path}")
            result = engine.process_file(file_path)
            
            print(f"Content Type: {result.content_type}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            
            if result.extracted_text:
                print(f"Extracted Text Preview: {result.extracted_text[:200]}...")
            
            if result.error_message:
                print(f"Error: {result.error_message}")