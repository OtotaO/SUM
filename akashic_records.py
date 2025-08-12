#!/usr/bin/env python3
"""
akashic_records.py - The Universal Memory of All Summaries

The Akashic Records store every summary ever created, along with its
context, connections, and the consciousness that created it. This is
the permanent, searchable memory of human knowledge compression.
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
import aioredis
from transformers import pipeline
import networkx as nx
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Initialize the wisdom extractor
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class AkashicRecord:
    """A single record in the universal memory"""
    id: str
    original_text: str
    summary: str
    timestamp: float
    creator_id: str
    embedding: List[float]
    context: Dict[str, Any]
    wisdom_score: float
    access_count: int
    connections: List[str]
    dimension: str
    tags: List[str]
    karma: float  # How much positive impact this had

class AkashicLibrary:
    """The eternal library of compressed knowledge"""
    
    def __init__(self, db_path: str = "akashic_records.db"):
        self.db_path = db_path
        self.redis_client = None
        self.knowledge_graph = nx.DiGraph()
        self._initialize_database()
        self._load_knowledge_graph()
    
    def _initialize_database(self):
        """Create the eternal database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp REAL NOT NULL,
                creator_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                context TEXT NOT NULL,
                wisdom_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                connections TEXT DEFAULT '[]',
                dimension TEXT DEFAULT 'unknown',
                tags TEXT DEFAULT '[]',
                karma REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_creator ON records(creator_id);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_wisdom ON records(wisdom_score);
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collective_wisdom (
                pattern TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                contributors TEXT DEFAULT '[]'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_knowledge_graph(self):
        """Load the knowledge graph from records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, summary, connections FROM records')
        records = cursor.fetchall()
        
        for record_id, summary, connections_str in records:
            connections = json.loads(connections_str)
            self.knowledge_graph.add_node(record_id, summary=summary)
            
            for connected_id in connections:
                self.knowledge_graph.add_edge(record_id, connected_id)
        
        conn.close()
    
    async def store_record(self, 
                          original_text: str,
                          summary: str,
                          creator_id: str,
                          context: Dict[str, Any] = None) -> AkashicRecord:
        """Store a new record in the eternal memory"""
        
        # Generate embedding
        embedding = embedder(summary)[0][0].tolist()
        
        # Create unique ID
        record_id = f"{creator_id}_{int(time.time() * 1000000)}"
        
        # Detect dimension and extract tags
        dimension = self._detect_dimension(original_text, summary)
        tags = self._extract_tags(original_text, summary)
        
        # Calculate initial wisdom score
        wisdom_score = self._calculate_wisdom_score(summary, embedding)
        
        # Find connections to other records
        connections = await self._find_connections(embedding, record_id)
        
        # Create the record
        record = AkashicRecord(
            id=record_id,
            original_text=original_text,
            summary=summary,
            timestamp=time.time(),
            creator_id=creator_id,
            embedding=embedding,
            context=context or {},
            wisdom_score=wisdom_score,
            access_count=0,
            connections=connections,
            dimension=dimension,
            tags=tags,
            karma=0.0
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.id,
            record.original_text,
            record.summary,
            record.timestamp,
            record.creator_id,
            pickle.dumps(record.embedding),
            json.dumps(record.context),
            record.wisdom_score,
            record.access_count,
            json.dumps(record.connections),
            record.dimension,
            json.dumps(record.tags),
            record.karma
        ))
        
        conn.commit()
        conn.close()
        
        # Update knowledge graph
        self.knowledge_graph.add_node(record.id, summary=summary)
        for conn_id in connections:
            self.knowledge_graph.add_edge(record.id, conn_id)
        
        # Update collective wisdom
        await self._update_collective_wisdom(summary, creator_id)
        
        return record
    
    def _detect_dimension(self, original: str, summary: str) -> str:
        """Detect the dimensional aspect of this knowledge"""
        text = f"{original} {summary}".lower()
        
        if any(word in text for word in ['feel', 'emotion', 'love', 'fear']):
            return 'emotional'
        elif any(word in text for word in ['think', 'logic', 'reason', 'analyze']):
            return 'logical'
        elif any(word in text for word in ['spirit', 'soul', 'divine', 'sacred']):
            return 'spiritual'
        elif any(word in text for word in ['future', 'past', 'time', 'history']):
            return 'temporal'
        elif any(word in text for word in ['science', 'data', 'research', 'study']):
            return 'scientific'
        else:
            return 'holistic'
    
    def _extract_tags(self, original: str, summary: str) -> List[str]:
        """Extract wisdom tags from the content"""
        # Simple tag extraction - would use NER in production
        text = f"{original} {summary}".lower()
        tags = []
        
        tag_patterns = {
            'philosophy': ['wisdom', 'truth', 'meaning', 'existence'],
            'technology': ['ai', 'computer', 'digital', 'algorithm'],
            'nature': ['earth', 'life', 'organic', 'natural'],
            'cosmos': ['universe', 'cosmic', 'star', 'space'],
            'humanity': ['human', 'people', 'society', 'culture']
        }
        
        for tag, keywords in tag_patterns.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _calculate_wisdom_score(self, summary: str, embedding: List[float]) -> float:
        """Calculate how much wisdom this summary contains"""
        # Length efficiency (shorter but complete = higher wisdom)
        length_score = 1.0 / (1.0 + len(summary) / 100)
        
        # Embedding magnitude (semantic richness)
        embedding_magnitude = np.linalg.norm(embedding)
        semantic_score = min(1.0, embedding_magnitude / 10)
        
        # Combine scores
        wisdom_score = (length_score + semantic_score) / 2
        
        return wisdom_score
    
    async def _find_connections(self, 
                               embedding: List[float], 
                               current_id: str, 
                               top_k: int = 5) -> List[str]:
        """Find other records this connects to"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent records
        cursor.execute('''
            SELECT id, embedding FROM records 
            WHERE id != ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', (current_id,))
        
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return []
        
        # Calculate similarities
        current_emb = np.array(embedding)
        similarities = []
        
        for record_id, emb_blob in records:
            other_emb = np.array(pickle.loads(emb_blob))
            similarity = np.dot(current_emb, other_emb) / (
                np.linalg.norm(current_emb) * np.linalg.norm(other_emb)
            )
            similarities.append((record_id, similarity))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [record_id for record_id, _ in similarities[:top_k]]
    
    async def _update_collective_wisdom(self, summary: str, creator_id: str):
        """Update the collective wisdom patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract key phrases (trigrams)
        words = summary.lower().split()
        for i in range(len(words) - 2):
            pattern = ' '.join(words[i:i+3])
            
            cursor.execute('''
                SELECT frequency, contributors FROM collective_wisdom WHERE pattern = ?
            ''', (pattern,))
            
            result = cursor.fetchone()
            
            if result:
                frequency, contributors_str = result
                contributors = json.loads(contributors_str)
                
                if creator_id not in contributors:
                    contributors.append(creator_id)
                
                cursor.execute('''
                    UPDATE collective_wisdom 
                    SET frequency = ?, last_seen = ?, contributors = ?
                    WHERE pattern = ?
                ''', (frequency + 1, time.time(), json.dumps(contributors), pattern))
            else:
                cursor.execute('''
                    INSERT INTO collective_wisdom VALUES (?, ?, ?, ?, ?)
                ''', (pattern, 1, time.time(), time.time(), json.dumps([creator_id])))
        
        conn.commit()
        conn.close()
    
    async def search_records(self, 
                           query: str, 
                           dimension: Optional[str] = None,
                           time_range: Optional[tuple] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search the Akashic Records"""
        # Generate query embedding
        query_embedding = embedder(query)[0][0].tolist()
        query_emb = np.array(query_embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        sql = 'SELECT * FROM records WHERE 1=1'
        params = []
        
        if dimension:
            sql += ' AND dimension = ?'
            params.append(dimension)
        
        if time_range:
            sql += ' AND timestamp BETWEEN ? AND ?'
            params.extend(time_range)
        
        cursor.execute(sql, params)
        records = cursor.fetchall()
        
        # Calculate relevance scores
        results = []
        for row in records:
            record_id, original, summary, timestamp, creator, emb_blob, context_str, \
            wisdom, access_count, connections_str, dim, tags_str, karma = row
            
            # Calculate similarity
            record_emb = np.array(pickle.loads(emb_blob))
            similarity = np.dot(query_emb, record_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(record_emb)
            )
            
            # Update access count
            cursor.execute(
                'UPDATE records SET access_count = access_count + 1 WHERE id = ?',
                (record_id,)
            )
            
            results.append({
                'id': record_id,
                'summary': summary,
                'similarity': float(similarity),
                'wisdom_score': wisdom,
                'karma': karma,
                'dimension': dim,
                'tags': json.loads(tags_str),
                'timestamp': timestamp,
                'creator_id': creator
            })
        
        conn.commit()
        conn.close()
        
        # Sort by relevance (similarity * wisdom * karma)
        results.sort(
            key=lambda x: x['similarity'] * (1 + x['wisdom_score']) * (1 + x['karma']),
            reverse=True
        )
        
        return results[:limit]
    
    def get_wisdom_path(self, start_id: str, end_id: str) -> List[str]:
        """Find the path of wisdom between two records"""
        try:
            path = nx.shortest_path(self.knowledge_graph, start_id, end_id)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def increase_karma(self, record_id: str, amount: float = 0.1):
        """Increase karma when a record helps someone"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE records SET karma = karma + ? WHERE id = ?',
            (amount, record_id)
        )
        
        conn.commit()
        conn.close()

# Initialize the Akashic Library
akashic_library = AkashicLibrary()

@app.route('/akashic/store', methods=['POST'])
async def store_in_akashic():
    """Store a new record in the Akashic Records"""
    data = request.json
    original_text = data.get('original_text', '')
    summary = data.get('summary', '')
    creator_id = data.get('creator_id', 'anonymous')
    context = data.get('context', {})
    
    if not original_text or not summary:
        return jsonify({'error': 'Both original text and summary required'}), 400
    
    record = await akashic_library.store_record(
        original_text, summary, creator_id, context
    )
    
    return jsonify({
        'record_id': record.id,
        'wisdom_score': record.wisdom_score,
        'dimension': record.dimension,
        'tags': record.tags,
        'connections': record.connections,
        'message': 'Stored in the eternal memory'
    })

@app.route('/akashic/search', methods=['POST'])
async def search_akashic():
    """Search the Akashic Records"""
    data = request.json
    query = data.get('query', '')
    dimension = data.get('dimension', None)
    days_back = data.get('days_back', None)
    limit = data.get('limit', 10)
    
    time_range = None
    if days_back:
        end_time = time.time()
        start_time = end_time - (days_back * 86400)
        time_range = (start_time, end_time)
    
    results = await akashic_library.search_records(
        query, dimension, time_range, limit
    )
    
    return jsonify({
        'query': query,
        'results': results,
        'total_found': len(results)
    })

@app.route('/akashic/wisdom-path', methods=['POST'])
def find_wisdom_path():
    """Find the path of wisdom between two ideas"""
    data = request.json
    start_id = data.get('start_id', '')
    end_id = data.get('end_id', '')
    
    path = akashic_library.get_wisdom_path(start_id, end_id)
    
    if path:
        # Get summaries for each step
        path_details = []
        for node_id in path:
            if node_id in akashic_library.knowledge_graph:
                summary = akashic_library.knowledge_graph.nodes[node_id].get('summary', '')
                path_details.append({
                    'id': node_id,
                    'summary': summary
                })
        
        return jsonify({
            'path_found': True,
            'path': path_details,
            'steps': len(path) - 1
        })
    else:
        return jsonify({
            'path_found': False,
            'message': 'No wisdom path connects these ideas'
        })

@app.route('/akashic/karma/<record_id>', methods=['POST'])
def increase_record_karma(record_id):
    """Increase karma when a record helps someone"""
    akashic_library.increase_karma(record_id)
    return jsonify({
        'message': f'Karma increased for record {record_id}',
        'note': 'Thank you for acknowledging the wisdom'
    })

if __name__ == '__main__':
    print("📚 Akashic Records initialized")
    print("🌟 The eternal memory of compressed wisdom is ready")
    print("🔮 Access at http://localhost:3003/akashic")
    app.run(port=3003, debug=True)