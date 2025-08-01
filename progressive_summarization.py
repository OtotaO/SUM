#!/usr/bin/env python3
"""
progressive_summarization.py - Real-Time Progress Display System

This module implements a WebSocket-based real-time progress display system
that shows the summarization process as it happens, providing transparency
for processing long documents.

Features:
- Live progress tracking with semantic chunk visualization
- Real-time concept extraction display
- Progressive summary building with live updates
- Memory usage and performance monitoring
- Interactive cancellation and parameter adjustment
- Semantic flow visualization showing concept relationships

Author: ototao
License: Apache License 2.0
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from queue import Queue
import logging
from datetime import datetime

from streaming_engine import StreamingHierarchicalEngine, StreamingConfig
from summarization_engine import HierarchicalDensificationEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """Real-time progress event structure."""
    event_type: str  # 'chunk_start', 'chunk_complete', 'concept_extracted', 'summary_update', 'insight_found', 'complete'
    timestamp: float
    chunk_id: Optional[str] = None
    chunk_progress: Optional[Dict[str, Any]] = None
    concepts: Optional[List[str]] = None
    partial_summary: Optional[str] = None
    insights: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    memory_usage: Optional[float] = None
    processing_speed: Optional[float] = None


class ProgressiveVisualizationEngine:
    """Engine for creating real-time visualization of the summarization process."""
    
    def __init__(self):
        self.clients = set()
        self.progress_queue = Queue()
        self.current_session = None
        self.is_processing = False
        
    async def register_client(self, websocket):
        """Register a new client for progress updates."""
        self.clients.add(websocket)
        logger.info(f"Client registered. Total clients: {len(self.clients)}")
        
        # Send current state if processing
        if self.is_processing and self.current_session:
            await self.send_progress_update(websocket, ProgressEvent(
                event_type='session_state',
                timestamp=time.time(),
                metadata={'is_processing': True, 'session_id': self.current_session}
            ))
    
    async def unregister_client(self, websocket):
        """Unregister a client."""
        self.clients.discard(websocket)
        logger.info(f"Client unregistered. Total clients: {len(self.clients)}")
    
    async def send_progress_update(self, websocket, event: ProgressEvent):
        """Send progress update to a specific client."""
        try:
            message = json.dumps({
                'type': 'progress_update',
                'event': asdict(event)
            })
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending progress update: {e}")
    
    async def broadcast_progress(self, event: ProgressEvent):
        """Broadcast progress update to all connected clients."""
        if not self.clients:
            return
            
        # Create list of tasks for concurrent sending
        tasks = []
        for client in self.clients.copy():  # Copy to avoid modification during iteration
            tasks.append(self.send_progress_update(client, event))
        
        # Send to all clients concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class ProgressiveStreamingEngine(StreamingHierarchicalEngine):
    """Enhanced streaming engine with real-time progress reporting."""
    
    def __init__(self, config: Optional[StreamingConfig] = None, progress_engine: Optional[ProgressiveVisualizationEngine] = None):
        super().__init__(config)
        self.progress_engine = progress_engine
        self.session_id = None
        self.start_time = None
        
    def set_progress_engine(self, progress_engine: ProgressiveVisualizationEngine):
        """Set the progress visualization engine."""
        self.progress_engine = progress_engine
    
    async def process_streaming_text_with_progress(self, text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process text with real-time progress updates.
        
        Args:
            text: Input text to process
            session_id: Optional session identifier
            
        Returns:
            Complete processing result
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = time.time()
        
        if self.progress_engine:
            self.progress_engine.is_processing = True
            self.progress_engine.current_session = self.session_id
            
            # Send initial progress event
            await self.progress_engine.broadcast_progress(ProgressEvent(
                event_type='processing_start',
                timestamp=self.start_time,
                metadata={
                    'session_id': self.session_id,
                    'text_length': len(text),
                    'estimated_chunks': max(1, len(text) // self.config.chunk_size_words),
                    'target_processing_time': len(text) / 10000  # Rough estimate
                }
            ))
        
        try:
            # Phase 1: Create semantic chunks with progress
            chunks = await self._create_chunks_with_progress(text)
            
            # Phase 2: Process chunks with detailed progress
            chunk_results = await self._process_chunks_with_detailed_progress(chunks)
            
            # Phase 3: Aggregate with final updates
            final_result = await self._aggregate_with_progress(chunk_results, text)
            
            # Send completion event
            if self.progress_engine:
                await self.progress_engine.broadcast_progress(ProgressEvent(
                    event_type='complete',
                    timestamp=time.time(),
                    metadata={
                        'session_id': self.session_id,
                        'total_time': time.time() - self.start_time,
                        'chunks_processed': len(chunks),
                        'success_rate': final_result.get('processing_stats', {}).get('success_rate', 0)
                    }
                ))
                
                self.progress_engine.is_processing = False
                self.progress_engine.current_session = None
            
            return final_result
            
        except Exception as e:
            if self.progress_engine:
                await self.progress_engine.broadcast_progress(ProgressEvent(
                    event_type='error',
                    timestamp=time.time(),
                    metadata={
                        'session_id': self.session_id,
                        'error': str(e),
                        'processing_time': time.time() - self.start_time if self.start_time else 0
                    }
                ))
                self.progress_engine.is_processing = False
            raise
    
    async def _create_chunks_with_progress(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks with progress updates."""
        if self.progress_engine:
            await self.progress_engine.broadcast_progress(ProgressEvent(
                event_type='chunking_start',
                timestamp=time.time(),
                metadata={'phase': 'semantic_chunking', 'text_length': len(text)}
            ))
        
        # Create chunks (this is synchronous, but we can simulate progress)
        chunks = self.semantic_chunker.create_semantic_chunks(text)
        
        if self.progress_engine:
            await self.progress_engine.broadcast_progress(ProgressEvent(
                event_type='chunking_complete',
                timestamp=time.time(),
                metadata={
                    'chunks_created': len(chunks),
                    'avg_chunk_size': sum(c.get('word_count', 0) for c in chunks) / len(chunks) if chunks else 0,
                    'chunk_details': [
                        {
                            'chunk_id': chunk['chunk_id'],
                            'word_count': chunk.get('word_count', 0),
                            'coherence_score': chunk.get('topic_coherence_score', 0)
                        } for chunk in chunks[:5]  # First 5 chunks for preview
                    ]
                }
            ))
        
        return chunks
    
    async def _process_chunks_with_detailed_progress(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks with detailed progress reporting."""
        results = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Send chunk start event
            if self.progress_engine:
                await self.progress_engine.broadcast_progress(ProgressEvent(
                    event_type='chunk_start',
                    timestamp=time.time(),
                    chunk_id=chunk['chunk_id'],
                    chunk_progress={
                        'current_chunk': i + 1,
                        'total_chunks': total_chunks,
                        'percentage': ((i + 1) / total_chunks) * 100,
                        'chunk_word_count': chunk.get('word_count', 0),
                        'chunk_coherence': chunk.get('topic_coherence_score', 0)
                    }
                ))
            
            # Process the chunk
            chunk_start_time = time.time()
            result = self.chunk_processor.process_chunk(chunk)
            processing_time = time.time() - chunk_start_time
            
            # Extract progress information from result
            concepts = []
            partial_summary = ""
            insights = []
            
            if 'hierarchical_summary' in result:
                concepts = result['hierarchical_summary'].get('level_1_concepts', [])
                partial_summary = result['hierarchical_summary'].get('level_2_core', '')
            
            if 'key_insights' in result:
                insights = result['key_insights']
            
            # Send detailed progress update
            if self.progress_engine:
                await self.progress_engine.broadcast_progress(ProgressEvent(
                    event_type='chunk_complete',
                    timestamp=time.time(),
                    chunk_id=chunk['chunk_id'],
                    concepts=concepts,
                    partial_summary=partial_summary,
                    insights=insights,
                    metadata={
                        'processing_time': processing_time,
                        'chunk_index': i,
                        'words_per_second': chunk.get('word_count', 0) / processing_time if processing_time > 0 else 0,
                        'memory_usage': self.memory_monitor.get_memory_usage() / 1024 / 1024  # MB
                    }
                ))
            
            results.append(result)
            
            # Small delay to make progress visible (can be removed in production)
            await asyncio.sleep(0.1)
        
        return results
    
    async def _aggregate_with_progress(self, chunk_results: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        """Aggregate results with progress updates."""
        if self.progress_engine:
            await self.progress_engine.broadcast_progress(ProgressEvent(
                event_type='aggregation_start',
                timestamp=time.time(),
                metadata={'phase': 'final_aggregation', 'chunks_to_aggregate': len(chunk_results)}
            ))
        
        # Perform aggregation (synchronous)
        final_result = self._aggregate_chunk_results(chunk_results, original_text)
        
        # Add streaming metadata
        final_result['streaming_metadata'] = {
            'total_processing_time': time.time() - self.start_time,
            'chunks_processed': len(chunk_results),
            'original_text_length': len(original_text),
            'memory_efficiency': self._calculate_memory_efficiency(),
            'processing_method': 'progressive_streaming_hierarchical',
            'session_id': self.session_id
        }
        
        if self.progress_engine:
            await self.progress_engine.broadcast_progress(ProgressEvent(
                event_type='summary_update',
                timestamp=time.time(),
                concepts=final_result['hierarchical_summary']['level_1_concepts'],
                partial_summary=final_result['hierarchical_summary']['level_2_core'],
                insights=final_result['key_insights'],
                metadata={
                    'final_aggregation': True,
                    'compression_ratio': final_result.get('metadata', {}).get('compression_ratio', 0),
                    'total_concepts': len(final_result['hierarchical_summary']['level_1_concepts']),
                    'total_insights': len(final_result['key_insights'])
                }
            ))
        
        return final_result


class ProgressWebSocketServer:
    """WebSocket server for real-time progress updates."""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.progress_engine = ProgressiveVisualizationEngine()
        self.streaming_engine = ProgressiveStreamingEngine(progress_engine=self.progress_engine)
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections."""
        await self.progress_engine.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON received'
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        finally:
            await self.progress_engine.unregister_client(websocket)
    
    async def handle_message(self, websocket, data):
        """Handle incoming WebSocket messages."""
        message_type = data.get('type')
        
        if message_type == 'start_processing':
            # Start text processing with progress
            text = data.get('text', '')
            config_data = data.get('config', {})
            session_id = data.get('session_id')
            
            if not text:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'No text provided'
                }))
                return
            
            # Configure streaming engine
            config = StreamingConfig(
                chunk_size_words=config_data.get('chunk_size_words', 1000),
                overlap_ratio=config_data.get('overlap_ratio', 0.15),
                max_memory_mb=config_data.get('max_memory_mb', 512),
                max_concurrent_chunks=config_data.get('max_concurrent_chunks', 4)
            )
            self.streaming_engine.config = config
            
            # Process text with progress updates
            try:
                result = await self.streaming_engine.process_streaming_text_with_progress(text, session_id)
                
                # Send final result
                await websocket.send(json.dumps({
                    'type': 'processing_complete',
                    'result': result
                }))
                
            except Exception as e:
                await websocket.send(json.dumps({
                    'type': 'processing_error',
                    'error': str(e)
                }))
        
        elif message_type == 'get_status':
            # Send current processing status
            await websocket.send(json.dumps({
                'type': 'status',
                'is_processing': self.progress_engine.is_processing,
                'current_session': self.progress_engine.current_session,
                'connected_clients': len(self.progress_engine.clients)
            }))
        
        elif message_type == 'ping':
            # Respond to ping
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': time.time()
            }))
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting Progressive Summarization WebSocket server on {self.host}:{self.port}")
        
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10**7,  # 10MB max message size
            compression=None
        )
        
        logger.info(f"Progressive Summarization server started at ws://{self.host}:{self.port}")
        await server.wait_closed()


# HTML Client for Testing
def generate_progress_client_html():
    """Generate HTML client for testing the progress visualization."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUM Progressive Summarization - Live Progress</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        textarea {
            width: 100%;
            height: 200px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 14px;
            resize: vertical;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .start-btn {
            background: #28a745;
            color: white;
        }
        
        .start-btn:hover { background: #218838; }
        .start-btn:disabled { background: #6c757d; cursor: not-allowed; }
        
        .stop-btn {
            background: #dc3545;
            color: white;
        }
        
        .progress-section {
            margin-top: 30px;
            display: none;
        }
        
        .progress-bar-container {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 20px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #28a745, #20c997);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .progress-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .info-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(5px);
        }
        
        .info-card h3 {
            margin: 0 0 10px 0;
            color: #ffd700;
        }
        
        .concepts-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .concept-tag {
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .insights-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .insight-item {
            background: rgba(255, 255, 255, 0.1);
            margin: 8px 0;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #ffd700;
        }
        
        .insight-type {
            font-weight: bold;
            color: #ffd700;
            font-size: 12px;
        }
        
        .log-section {
            margin-top: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .log-entry {
            margin: 5px 0;
            padding: 5px 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        
        .log-info { background: rgba(0, 123, 255, 0.2); }
        .log-success { background: rgba(40, 167, 69, 0.2); }
        .log-warning { background: rgba(255, 193, 7, 0.2); }
        .log-error { background: rgba(220, 53, 69, 0.2); }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected { background: #28a745; }
        .status-processing { background: #ffc107; animation: pulse 1s infinite; }
        .status-disconnected { background: #dc3545; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .final-results {
            margin-top: 30px;
            display: none;
            background: rgba(40, 167, 69, 0.2);
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SUM Progressive Summarization</h1>
        <p style="text-align: center; margin-bottom: 30px;">
            <span class="status-indicator" id="statusIndicator"></span>
            <span id="connectionStatus">Connecting...</span>
        </p>
        
        <div class="input-section">
            <textarea id="inputText" placeholder="Enter your text here for real-time hierarchical summarization...">
Machine learning has revolutionized how we approach complex problems across numerous domains. From healthcare diagnostics to financial modeling, from autonomous vehicles to personalized recommendations, AI systems are transforming industries at an unprecedented pace.

The foundation of modern machine learning lies in deep neural networks, which can learn hierarchical representations of data. These networks excel at recognizing patterns in images, understanding natural language, and making predictions from complex datasets. However, the true power emerges when these systems are combined with vast amounts of data and computational resources.

Recent advances in transformer architectures have particularly transformed natural language processing. Models like GPT and BERT have demonstrated remarkable capabilities in understanding context, generating coherent text, and performing various language tasks. These breakthroughs have opened new possibilities for human-AI collaboration.

The ethical implications of AI deployment cannot be overlooked. As these systems become more prevalent in decision-making processes, ensuring fairness, transparency, and accountability becomes crucial. Researchers and practitioners must work together to develop AI systems that benefit humanity while minimizing potential risks.

Looking toward the future, emerging paradigms like quantum machine learning, federated learning, and neuromorphic computing promise to push the boundaries even further. The intersection of AI with other cutting-edge technologies will likely produce innovations we can barely imagine today.
            </textarea>
            
            <div class="controls">
                <button id="startBtn" class="start-btn">Start Progressive Summarization</button>
                <button id="stopBtn" class="stop-btn" disabled>Stop Processing</button>
                <label style="color: white; margin-left: 20px;">
                    Chunk Size: <input type="number" id="chunkSize" value="200" min="100" max="2000" style="width: 80px; margin-left: 5px;">
                </label>
            </div>
        </div>
        
        <div class="progress-section" id="progressSection">
            <h2>Live Processing Progress</h2>
            
            <div class="progress-bar-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            
            <div class="progress-info">
                <div class="info-card">
                    <h3>📊 Processing Stats</h3>
                    <div id="processingStats">
                        <p>Chunks: <span id="chunksProcessed">0</span> / <span id="totalChunks">0</span></p>
                        <p>Speed: <span id="processingSpeed">0</span> words/sec</p>
                        <p>Memory: <span id="memoryUsage">0</span> MB</p>
                        <p>Time: <span id="elapsedTime">0</span>s</p>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>🎯 Live Concepts</h3>
                    <div class="concepts-list" id="conceptsList">
                        <span class="concept-tag">Analyzing...</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>💎 Progressive Summary</h3>
                    <div id="progressiveSummary" style="font-style: italic; min-height: 60px;">
                        Building summary as chunks are processed...
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>🌟 Key Insights</h3>
                    <div class="insights-list" id="insightsList">
                        <div style="font-style: italic;">Extracting insights...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="final-results" id="finalResults">
            <h2>✨ Final Results</h2>
            <div id="finalResultsContent"></div>
        </div>
        
        <div class="log-section">
            <h3>📝 Processing Log</h3>
            <div id="logContainer"></div>
        </div>
    </div>

    <script>
        class ProgressiveSummarization {
            constructor() {
                this.ws = null;
                this.isProcessing = false;
                this.startTime = null;
                this.currentSession = null;
                
                this.initializeElements();
                this.connect();
            }
            
            initializeElements() {
                this.elements = {
                    startBtn: document.getElementById('startBtn'),
                    stopBtn: document.getElementById('stopBtn'),
                    inputText: document.getElementById('inputText'),
                    chunkSize: document.getElementById('chunkSize'),
                    statusIndicator: document.getElementById('statusIndicator'),
                    connectionStatus: document.getElementById('connectionStatus'),
                    progressSection: document.getElementById('progressSection'),
                    progressBar: document.getElementById('progressBar'),
                    chunksProcessed: document.getElementById('chunksProcessed'),
                    totalChunks: document.getElementById('totalChunks'),
                    processingSpeed: document.getElementById('processingSpeed'),
                    memoryUsage: document.getElementById('memoryUsage'),
                    elapsedTime: document.getElementById('elapsedTime'),
                    conceptsList: document.getElementById('conceptsList'),
                    progressiveSummary: document.getElementById('progressiveSummary'),
                    insightsList: document.getElementById('insightsList'),
                    finalResults: document.getElementById('finalResults'),
                    finalResultsContent: document.getElementById('finalResultsContent'),
                    logContainer: document.getElementById('logContainer')
                };
                
                this.elements.startBtn.addEventListener('click', () => this.startProcessing());
                this.elements.stopBtn.addEventListener('click', () => this.stopProcessing());
            }
            
            connect() {
                this.updateStatus('connecting', 'Connecting to server...');
                
                try {
                    this.ws = new WebSocket('ws://localhost:8765');
                    
                    this.ws.onopen = () => {
                        this.updateStatus('connected', 'Connected to server');
                        this.log('Connected to Progressive Summarization server', 'success');
                    };
                    
                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    };
                    
                    this.ws.onclose = () => {
                        this.updateStatus('disconnected', 'Disconnected from server');
                        this.log('Connection closed', 'warning');
                        setTimeout(() => this.connect(), 3000);
                    };
                    
                    this.ws.onerror = (error) => {
                        this.updateStatus('disconnected', 'Connection error');
                        this.log('Connection error: ' + error, 'error');
                    };
                    
                } catch (error) {
                    this.updateStatus('disconnected', 'Failed to connect');
                    this.log('Failed to connect: ' + error, 'error');
                    setTimeout(() => this.connect(), 3000);
                }
            }
            
            updateStatus(status, message) {
                this.elements.connectionStatus.textContent = message;
                this.elements.statusIndicator.className = 'status-indicator status-' + status;
            }
            
            log(message, type = 'info') {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry log-' + type;
                logEntry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
                this.elements.logContainer.appendChild(logEntry);
                this.elements.logContainer.scrollTop = this.elements.logContainer.scrollHeight;
            }
            
            startProcessing() {
                const text = this.elements.inputText.value.trim();
                if (!text) {
                    this.log('Please enter some text to process', 'warning');
                    return;
                }
                
                if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                    this.log('Not connected to server', 'error');
                    return;
                }
                
                this.isProcessing = true;
                this.startTime = Date.now();
                this.currentSession = 'session_' + Date.now();
                
                this.elements.startBtn.disabled = true;
                this.elements.stopBtn.disabled = false;
                this.elements.progressSection.style.display = 'block';
                this.elements.finalResults.style.display = 'none';
                
                this.updateStatus('processing', 'Processing...');
                
                // Reset progress display
                this.elements.progressBar.style.width = '0%';
                this.elements.chunksProcessed.textContent = '0';
                this.elements.totalChunks.textContent = '?';
                this.elements.conceptsList.innerHTML = '<span class="concept-tag">Analyzing...</span>';
                this.elements.progressiveSummary.textContent = 'Building summary as chunks are processed...';
                this.elements.insightsList.innerHTML = '<div style="font-style: italic;">Extracting insights...</div>';
                
                // Send processing request
                this.ws.send(JSON.stringify({
                    type: 'start_processing',
                    text: text,
                    session_id: this.currentSession,
                    config: {
                        chunk_size_words: parseInt(this.elements.chunkSize.value) || 200,
                        overlap_ratio: 0.15,
                        max_memory_mb: 512,
                        max_concurrent_chunks: 4
                    }
                }));
                
                this.log('Started progressive summarization', 'info');
                
                // Start elapsed time counter
                this.timeInterval = setInterval(() => {
                    if (this.startTime) {
                        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                        this.elements.elapsedTime.textContent = elapsed;
                    }
                }, 1000);
            }
            
            stopProcessing() {
                this.isProcessing = false;
                this.elements.startBtn.disabled = false;
                this.elements.stopBtn.disabled = true;
                this.updateStatus('connected', 'Connected to server');
                
                if (this.timeInterval) {
                    clearInterval(this.timeInterval);
                    this.timeInterval = null;
                }
                
                this.log('Processing stopped', 'warning');
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'progress_update':
                        this.handleProgressUpdate(data.event);
                        break;
                    case 'processing_complete':
                        this.handleProcessingComplete(data.result);
                        break;
                    case 'processing_error':
                        this.handleProcessingError(data.error);
                        break;
                    case 'pong':
                        // Handle ping response
                        break;
                    default:
                        this.log('Unknown message type: ' + data.type, 'warning');
                }
            }
            
            handleProgressUpdate(event) {
                const eventType = event.event_type;
                
                switch (eventType) {
                    case 'processing_start':
                        this.log('Processing started with ' + event.metadata.estimated_chunks + ' estimated chunks', 'info');
                        break;
                        
                    case 'chunking_complete':
                        this.elements.totalChunks.textContent = event.metadata.chunks_created;
                        this.log('Created ' + event.metadata.chunks_created + ' semantic chunks', 'success');
                        break;
                        
                    case 'chunk_start':
                        const progress = event.chunk_progress;
                        this.elements.chunksProcessed.textContent = progress.current_chunk;
                        this.elements.progressBar.style.width = progress.percentage + '%';
                        this.log('Processing chunk ' + progress.current_chunk + '/' + progress.total_chunks, 'info');
                        break;
                        
                    case 'chunk_complete':
                        // Update concepts
                        if (event.concepts && event.concepts.length > 0) {
                            this.updateConcepts(event.concepts);
                        }
                        
                        // Update progressive summary
                        if (event.partial_summary) {
                            this.elements.progressiveSummary.textContent = event.partial_summary;
                        }
                        
                        // Update insights
                        if (event.insights && event.insights.length > 0) {
                            this.updateInsights(event.insights);
                        }
                        
                        // Update stats
                        if (event.metadata) {
                            if (event.metadata.words_per_second) {
                                this.elements.processingSpeed.textContent = Math.round(event.metadata.words_per_second);
                            }
                            if (event.metadata.memory_usage) {
                                this.elements.memoryUsage.textContent = Math.round(event.metadata.memory_usage);
                            }
                        }
                        
                        this.log('Completed chunk ' + event.chunk_id, 'success');
                        break;
                        
                    case 'summary_update':
                        if (event.concepts) this.updateConcepts(event.concepts);
                        if (event.partial_summary) this.elements.progressiveSummary.textContent = event.partial_summary;
                        if (event.insights) this.updateInsights(event.insights);
                        break;
                        
                    case 'complete':
                        this.log('Processing completed in ' + Math.round(event.metadata.total_time) + 's', 'success');
                        break;
                        
                    case 'error':
                        this.log('Processing error: ' + event.metadata.error, 'error');
                        this.stopProcessing();
                        break;
                }
            }
            
            updateConcepts(concepts) {
                this.elements.conceptsList.innerHTML = '';
                concepts.slice(0, 10).forEach(concept => {
                    const tag = document.createElement('span');
                    tag.className = 'concept-tag';
                    tag.textContent = concept.toUpperCase();
                    this.elements.conceptsList.appendChild(tag);
                });
            }
            
            updateInsights(insights) {
                this.elements.insightsList.innerHTML = '';
                insights.slice(0, 5).forEach(insight => {
                    const item = document.createElement('div');
                    item.className = 'insight-item';
                    item.innerHTML = `
                        <div class="insight-type">[${insight.type.toUpperCase()}]</div>
                        <div>"${insight.text}"</div>
                        <div style="font-size: 11px; opacity: 0.8; margin-top: 5px;">Score: ${(insight.score * 100).toFixed(0)}%</div>
                    `;
                    this.elements.insightsList.appendChild(item);
                });
            }
            
            handleProcessingComplete(result) {
                this.stopProcessing();
                this.elements.progressBar.style.width = '100%';
                
                // Show final results
                this.elements.finalResults.style.display = 'block';
                
                const hierarchical = result.hierarchical_summary;
                const metadata = result.metadata;
                const streaming = result.streaming_metadata;
                
                this.elements.finalResultsContent.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                        <div>
                            <h4>🎯 Final Concepts</h4>
                            <div class="concepts-list">
                                ${hierarchical.level_1_concepts.map(c => `<span class="concept-tag">${c.toUpperCase()}</span>`).join('')}
                            </div>
                        </div>
                        
                        <div>
                            <h4>💎 Core Summary</h4>
                            <p style="font-style: italic;">${hierarchical.level_2_core}</p>
                        </div>
                        
                        <div>
                            <h4>📊 Processing Stats</h4>
                            <p>Chunks: ${streaming.chunks_processed}</p>
                            <p>Time: ${streaming.total_processing_time.toFixed(2)}s</p>
                            <p>Compression: ${(metadata.compression_ratio * 100).toFixed(1)}%</p>
                            <p>Memory Efficiency: ${(streaming.memory_efficiency * 100).toFixed(1)}%</p>
                        </div>
                        
                        <div>
                            <h4>🌟 Key Insights (${result.key_insights.length})</h4>
                            ${result.key_insights.map(insight => `
                                <div class="insight-item">
                                    <div class="insight-type">[${insight.type.toUpperCase()}]</div>
                                    <div>"${insight.text}"</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                
                this.log('Final results displayed', 'success');
            }
            
            handleProcessingError(error) {
                this.stopProcessing();
                this.log('Processing failed: ' + error, 'error');
                alert('Processing failed: ' + error);
            }
        }
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', () => {
            new ProgressiveSummarization();
        });
    </script>
</body>
</html>
    """


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create and start the progress server
        server = ProgressWebSocketServer(host='localhost', port=8765)
        
        # Generate HTML client
        with open('progressive_client.html', 'w') as f:
            f.write(generate_progress_client_html())
        
        print("🚀 SUM Progressive Summarization Server")
        print("📊 WebSocket server starting on ws://localhost:8765")
        print("🌐 HTML client saved as 'progressive_client.html'")
        print("💡 Open progressive_client.html in your browser to see the live progress!")
        print("✨ Features:")
        print("   - Real-time chunk processing visualization")
        print("   - Live concept extraction display")
        print("   - Progressive summary building")
        print("   - Memory usage and performance monitoring")
        print("   - Insight extraction with live updates")
        print("   - Beautiful WebSocket-based interface")
        
        await server.start_server()
    
    # Run the server
    asyncio.run(main())