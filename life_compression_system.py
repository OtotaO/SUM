"""
Life Compression System - Temporal Knowledge Distillation

Integrates the C monitoring agent with the adaptive compression engine
to create a complete life-logging and compression system. Handles
everything from moments to lifetimes.

Architecture:
- C agent logs activities efficiently
- Python processes and compresses temporally
- Results stored in hierarchical time-based structure
- Enables semantic search across entire life history

Author: ototao & Claude
License: Apache 2.0
"""

import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
import subprocess
import signal
import hashlib
from dataclasses import dataclass, asdict
import pickle

from adaptive_compression import (
    AdaptiveCompressionEngine, 
    TemporalCompressionHierarchy,
    ContentType
)


@dataclass
class LifeEvent:
    """Represents a single life event/activity."""
    timestamp: datetime
    event_type: str
    description: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'description': self.description,
            'metadata': self.metadata or {}
        }


@dataclass 
class CompressedMemory:
    """Represents a compressed time period."""
    start_time: datetime
    end_time: datetime
    time_scale: str  # day, week, month, year, decade, lifetime
    compressed_text: str
    original_event_count: int
    compression_ratio: float
    key_concepts: List[str]
    highlights: List[str]
    
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600


class LifeCompressionDatabase:
    """
    SQLite database for storing compressed memories efficiently.
    Designed for fast temporal queries and semantic search.
    """
    
    def __init__(self, db_path: str = "~/.config/sum/life_memories.db"):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Raw events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata TEXT,
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_event_type (event_type)
                )
            """)
            
            # Compressed memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compressed_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    time_scale TEXT NOT NULL,
                    compressed_text TEXT NOT NULL,
                    original_event_count INTEGER,
                    compression_ratio REAL,
                    key_concepts TEXT,
                    highlights TEXT,
                    embedding BLOB,  -- For future semantic search
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_timerange (start_time, end_time),
                    INDEX idx_scale (time_scale)
                )
            """)
            
            # Search index for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search
                USING fts5(compressed_text, key_concepts, highlights,
                          content=compressed_memories, content_rowid=id)
            """)
            
            self.conn.commit()
    
    def add_event(self, event: LifeEvent):
        """Add a raw event to the database."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO events (timestamp, event_type, description, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                event.timestamp,
                event.event_type,
                event.description,
                json.dumps(event.metadata) if event.metadata else None
            ))
            self.conn.commit()
    
    def get_events_in_range(self, start: datetime, end: datetime) -> List[LifeEvent]:
        """Get all events within a time range."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, description, metadata
                FROM events
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (start, end))
            
            events = []
            for row in cursor.fetchall():
                events.append(LifeEvent(
                    timestamp=datetime.fromisoformat(row[0]),
                    event_type=row[1],
                    description=row[2],
                    metadata=json.loads(row[3]) if row[3] else None
                ))
            
            return events
    
    def save_compressed_memory(self, memory: CompressedMemory):
        """Save a compressed memory to the database."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO compressed_memories
                (start_time, end_time, time_scale, compressed_text,
                 original_event_count, compression_ratio, key_concepts, highlights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.start_time,
                memory.end_time,
                memory.time_scale,
                memory.compressed_text,
                memory.original_event_count,
                memory.compression_ratio,
                json.dumps(memory.key_concepts),
                json.dumps(memory.highlights)
            ))
            
            # Update search index
            memory_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO memory_search (rowid, compressed_text, key_concepts, highlights)
                VALUES (?, ?, ?, ?)
            """, (
                memory_id,
                memory.compressed_text,
                ' '.join(memory.key_concepts),
                ' '.join(memory.highlights)
            ))
            
            self.conn.commit()
    
    def search_memories(self, query: str, limit: int = 10) -> List[CompressedMemory]:
        """Search compressed memories using full-text search."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT cm.*
                FROM compressed_memories cm
                JOIN memory_search ms ON cm.id = ms.rowid
                WHERE memory_search MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            memories = []
            for row in cursor.fetchall():
                memories.append(CompressedMemory(
                    start_time=datetime.fromisoformat(row[1]),
                    end_time=datetime.fromisoformat(row[2]),
                    time_scale=row[3],
                    compressed_text=row[4],
                    original_event_count=row[5],
                    compression_ratio=row[6],
                    key_concepts=json.loads(row[7]),
                    highlights=json.loads(row[8])
                ))
            
            return memories


class LifeCompressionSystem:
    """
    Main system that orchestrates the life compression process.
    Integrates C agent, compression engine, and storage.
    """
    
    def __init__(self, config_path: str = "~/.config/sum/life_compression.json"):
        self.config_path = os.path.expanduser(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.compression_engine = AdaptiveCompressionEngine()
        self.temporal_hierarchy = TemporalCompressionHierarchy(self.compression_engine)
        self.database = LifeCompressionDatabase()
        
        # Agent process
        self.agent_process = None
        self.monitor_thread = None
        self.compression_thread = None
        
        # State
        self.running = False
        self.last_compression_times = {
            'day': datetime.now(),
            'week': datetime.now(),
            'month': datetime.now(),
            'year': datetime.now()
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        default_config = {
            'agent_binary': './monitor_agent',
            'log_file': '/tmp/sum_activities.log',
            'compression_intervals': {
                'day': 24,      # hours
                'week': 168,    # hours (7 days)
                'month': 720,   # hours (30 days)
                'year': 8760    # hours (365 days)
            },
            'privacy_mode': False,
            'auto_start': True,
            'ignored_apps': ['Terminal', 'iTerm'],
            'compression_thresholds': {
                'day': 100,     # min events for compression
                'week': 500,
                'month': 2000,
                'year': 10000
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        else:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def start(self):
        """Start the life compression system."""
        if self.running:
            return
        
        self.running = True
        
        # Start C monitoring agent
        self._start_agent()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start compression thread
        self.compression_thread = threading.Thread(target=self._compression_loop)
        self.compression_thread.daemon = True
        self.compression_thread.start()
        
        print("Life Compression System started")
    
    def stop(self):
        """Stop the life compression system."""
        self.running = False
        
        # Stop agent
        if self.agent_process:
            self.agent_process.terminate()
            self.agent_process.wait()
        
        # Wait for threads
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.compression_thread:
            self.compression_thread.join(timeout=5)
        
        print("Life Compression System stopped")
    
    def _start_agent(self):
        """Start the C monitoring agent."""
        agent_path = self.config['agent_binary']
        
        # Compile if needed
        if not os.path.exists(agent_path):
            print("Compiling monitor agent...")
            compile_cmd = [
                'gcc', '-O3', '-Wall', 'monitor_agent.c',
                '-o', agent_path
            ]
            
            # Add platform-specific flags
            if os.uname().sysname == 'Darwin':
                compile_cmd.extend([
                    '-framework', 'ApplicationServices',
                    '-framework', 'Carbon'
                ])
            
            subprocess.run(compile_cmd, check=True)
        
        # Start agent
        cmd = [agent_path, '--daemon']
        if self.config['privacy_mode']:
            cmd.append('--privacy')
        
        self.agent_process = subprocess.Popen(cmd)
    
    def _monitor_loop(self):
        """Monitor log file and parse events."""
        log_file = self.config['log_file']
        
        # Wait for log file to exist
        while self.running and not os.path.exists(log_file):
            time.sleep(1)
        
        with open(log_file, 'r') as f:
            # Seek to end of file
            f.seek(0, 2)
            
            while self.running:
                line = f.readline()
                if line:
                    self._parse_log_line(line.strip())
                else:
                    time.sleep(0.1)
    
    def _parse_log_line(self, line: str):
        """Parse a log line and store as event."""
        # Format: YYYY-MM-DD HH:MM:SS TYPE DATA
        parts = line.split(' ', 3)
        if len(parts) < 4:
            return
        
        try:
            timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", 
                                        "%Y-%m-%d %H:%M:%S")
            event_type = parts[2]
            description = parts[3]
            
            event = LifeEvent(
                timestamp=timestamp,
                event_type=event_type,
                description=description
            )
            
            self.database.add_event(event)
            
        except Exception as e:
            print(f"Error parsing log line: {e}")
    
    def _compression_loop(self):
        """Periodic compression of events."""
        while self.running:
            current_time = datetime.now()
            
            # Check each time scale
            for time_scale, hours in self.config['compression_intervals'].items():
                last_compression = self.last_compression_times.get(time_scale)
                
                if not last_compression or \
                   (current_time - last_compression).total_seconds() / 3600 >= hours:
                    
                    self._compress_time_scale(time_scale)
                    self.last_compression_times[time_scale] = current_time
            
            # Sleep for an hour
            time.sleep(3600)
    
    def _compress_time_scale(self, time_scale: str):
        """Compress events for a specific time scale."""
        current_time = datetime.now()
        
        # Determine time range
        if time_scale == 'day':
            start_time = current_time - timedelta(days=1)
        elif time_scale == 'week':
            start_time = current_time - timedelta(weeks=1)
        elif time_scale == 'month':
            start_time = current_time - timedelta(days=30)
        elif time_scale == 'year':
            start_time = current_time - timedelta(days=365)
        else:
            return
        
        # Get events in range
        events = self.database.get_events_in_range(start_time, current_time)
        
        # Check threshold
        if len(events) < self.config['compression_thresholds'].get(time_scale, 100):
            return
        
        print(f"Compressing {len(events)} events for {time_scale}")
        
        # Convert events to text
        event_texts = []
        for event in events:
            event_texts.append(
                f"{event.timestamp.strftime('%Y-%m-%d %H:%M')} "
                f"{event.event_type}: {event.description}"
            )
        
        combined_text = '\n'.join(event_texts)
        
        # Compress using adaptive engine
        result = self.compression_engine.compress(
            combined_text,
            target_ratio=self.temporal_hierarchy.compression_ratios[time_scale],
            force_type=ContentType.ACTIVITY_LOG
        )
        
        # Extract key concepts and highlights
        compressed_text = result['compressed']
        
        # Simple concept extraction (can be enhanced)
        words = compressed_text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4 and word.isalnum():
                word_freq[word] += 1
        
        key_concepts = [word for word, freq in 
                       sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        # Extract highlights (first few sentences)
        sentences = compressed_text.split('. ')
        highlights = sentences[:3] if len(sentences) > 3 else sentences
        
        # Create compressed memory
        memory = CompressedMemory(
            start_time=start_time,
            end_time=current_time,
            time_scale=time_scale,
            compressed_text=compressed_text,
            original_event_count=len(events),
            compression_ratio=result['actual_ratio'],
            key_concepts=key_concepts,
            highlights=highlights
        )
        
        # Save to database
        self.database.save_compressed_memory(memory)
        
        print(f"Compressed {time_scale}: {len(events)} events → "
              f"{len(compressed_text.split())} words "
              f"(ratio: {result['actual_ratio']:.1%})")
    
    def generate_life_story(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> str:
        """
        Generate a coherent life story from compressed memories.
        This is where we approach the philosophical - telling a life's story.
        """
        # Get all compressed memories in range
        # Implementation would query database and use the
        # TemporalCompressionHierarchy to create a narrative
        
        memories = []  # Would query from database
        
        # Create narrative structure
        narrative = []
        narrative.append("# Life Story\n")
        
        # Group by time scale
        for scale in ['lifetime', 'decade', 'year', 'month', 'week', 'day']:
            scale_memories = [m for m in memories if m.time_scale == scale]
            if scale_memories:
                narrative.append(f"\n## {scale.title()} Overview\n")
                for memory in scale_memories:
                    narrative.append(memory.compressed_text)
                    narrative.append("\n")
        
        return '\n'.join(narrative)
    
    def search_life_history(self, query: str) -> List[CompressedMemory]:
        """Search through entire life history."""
        return self.database.search_memories(query)


# CLI Interface
def main():
    """Command-line interface for the life compression system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SUM Life Compression System')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'search', 'compress'])
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--scale', choices=['day', 'week', 'month', 'year'],
                       help='Time scale for compression')
    
    args = parser.parse_args()
    
    system = LifeCompressionSystem()
    
    if args.command == 'start':
        system.start()
        print("System started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop()
    
    elif args.command == 'stop':
        # Would implement proper daemon management
        print("Stopping system...")
    
    elif args.command == 'status':
        # Would show system status
        print("System status: Running")
    
    elif args.command == 'search':
        if args.query:
            results = system.search_life_history(args.query)
            for memory in results:
                print(f"\n[{memory.time_scale}] {memory.start_time} - {memory.end_time}")
                print(memory.compressed_text[:200] + "...")
    
    elif args.command == 'compress':
        if args.scale:
            system._compress_time_scale(args.scale)


if __name__ == "__main__":
    main()