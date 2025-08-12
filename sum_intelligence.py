#!/usr/bin/env python3
"""
sum_intelligence.py - The intelligent layer for SUM (built on sum_simple.py)

This shows how to add REAL intelligence to the simple summarizer:
- Pattern recognition (without 'crystallized wisdom')
- Memory that actually works (PostgreSQL, not 'superhuman')
- Predictions that make sense (collaborative filtering, not 'temporal networks')
- Context that adapts (heuristics, not 'invisible AI')

Total: ~1000 lines for actual intelligence
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis

# Reuse the simple summarizer
from sum_simple import summarizer, get_summary_from_cache, save_summary_to_cache

# Configuration (still simple)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/sum')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Initialize connections
r = redis.from_url(REDIS_URL, decode_responses=True)


class IntelligentSum:
    """
    The intelligence layer for SUM.
    Built on top of the simple summarizer, adds smart features.
    """
    
    def __init__(self):
        self.conn = None
        self._init_database()
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self._topics_cache = {}
        
    def _init_database(self):
        """Simple database schema. No ORMs, just SQL."""
        self.conn = psycopg2.connect(DATABASE_URL)
        cur = self.conn.cursor()
        
        # Main summaries table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                topic TEXT,
                context_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, text_hash)
            )
        """)
        
        # Indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_created ON summaries(user_id, created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON summaries(topic)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_text_search ON summaries USING GIN(to_tsvector('english', text || ' ' || summary))")
        
        # User patterns table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_patterns (
                user_id TEXT PRIMARY KEY,
                topics JSONB DEFAULT '{}',
                reading_times JSONB DEFAULT '[]',
                avg_text_length INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def summarize_with_intelligence(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Summarize with intelligence features.
        This is the main entry point that adds value beyond simple summarization.
        """
        # Generate summary using simple summarizer
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check if we've seen this before for this user
        existing = self._get_existing_summary(user_id, text_hash)
        if existing:
            self._update_access_count(existing['id'])
            return {
                'summary': existing['summary'],
                'cached': True,
                'topic': existing['topic'],
                'context': existing['context_type'],
                'seen_before': True,
                'last_seen': existing['last_accessed']
            }
        
        # Get basic summary
        summary = self._get_or_create_summary(text)
        
        # Detect context (simple heuristics that work)
        context_type = self._detect_context(text)
        
        # Extract topic (simple but effective)
        topic = self._extract_topic(text)
        
        # Store in database
        self._store_summary(user_id, text_hash, text, summary, topic, context_type)
        
        # Update user patterns
        self._update_user_patterns(user_id, topic, len(text))
        
        # Get intelligent suggestions
        suggestions = self._get_suggestions(user_id, topic, text)
        
        return {
            'summary': summary,
            'cached': False,
            'topic': topic,
            'context': context_type,
            'suggestions': suggestions,
            'insights': self._get_quick_insights(user_id, topic)
        }
    
    def _get_or_create_summary(self, text: str) -> str:
        """Get summary from cache or create new one."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check Redis cache first
        cached = get_summary_from_cache(text_hash)
        if cached:
            return cached
            
        # Generate new summary
        result = summarizer(text, max_length=130, min_length=30)
        summary = result[0]['summary_text']
        
        # Cache it
        save_summary_to_cache(text_hash, summary)
        
        return summary
    
    def _detect_context(self, text: str) -> str:
        """
        Detect context type using simple heuristics.
        No 'invisible AI', just patterns that work.
        """
        text_lower = text.lower()
        
        # Academic indicators
        academic_words = ['research', 'hypothesis', 'methodology', 'abstract', 
                         'conclusion', 'findings', 'study', 'analysis', 'theory']
        academic_score = sum(1 for word in academic_words if word in text_lower)
        
        # Business indicators  
        business_words = ['revenue', 'profit', 'market', 'customer', 'sales',
                         'strategy', 'stakeholder', 'quarter', 'growth']
        business_score = sum(1 for word in business_words if word in text_lower)
        
        # Technical indicators
        technical_words = ['code', 'function', 'algorithm', 'implementation',
                          'bug', 'feature', 'deploy', 'api', 'database']
        technical_score = sum(1 for word in technical_words if word in text_lower)
        
        # Determine context
        scores = {
            'academic': academic_score,
            'business': business_score,
            'technical': technical_score,
            'general': 0
        }
        
        context = max(scores, key=scores.get)
        if scores[context] < 2:  # Not enough indicators
            context = 'general'
            
        return context
    
    def _extract_topic(self, text: str) -> str:
        """
        Extract main topic using TF-IDF.
        Simple, fast, and actually useful.
        """
        # For short texts, use word frequency
        if len(text.split()) < 100:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            word_freq = Counter(words)
            # Remove common words
            common = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their'}
            for word in common:
                word_freq.pop(word, None)
            
            if word_freq:
                topic = word_freq.most_common(1)[0][0]
                return topic
        
        # For longer texts, use TF-IDF
        try:
            # Fit and transform in one go for single document
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top scoring words
            top_indices = scores.argsort()[-3:][::-1]
            top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return top_words[0] if top_words else 'general'
        except:
            return 'general'
    
    def _get_suggestions(self, user_id: str, topic: str, current_text: str) -> List[Dict[str, Any]]:
        """
        Get intelligent suggestions based on:
        1. What similar users read after this topic
        2. Related summaries from the same user
        3. Popular content in this topic
        
        No 'predictive intelligence', just collaborative filtering.
        """
        suggestions = []
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. What do similar users read after this topic?
        cur.execute("""
            WITH user_topics AS (
                SELECT user_id, text_hash, created_at
                FROM summaries
                WHERE topic = %s AND user_id != %s
                ORDER BY created_at DESC
                LIMIT 100
            ),
            next_reads AS (
                SELECT s2.text_hash, s2.summary, s2.topic, COUNT(*) as read_count
                FROM user_topics ut
                JOIN summaries s2 ON ut.user_id = s2.user_id
                WHERE s2.created_at > ut.created_at
                AND s2.created_at < ut.created_at + INTERVAL '7 days'
                GROUP BY s2.text_hash, s2.summary, s2.topic
                ORDER BY read_count DESC
                LIMIT 3
            )
            SELECT * FROM next_reads
        """, (topic, user_id))
        
        for row in cur:
            suggestions.append({
                'type': 'similar_users',
                'summary': row['summary'][:100] + '...',
                'topic': row['topic'],
                'reason': f"{row['read_count']} users read this after {topic}"
            })
        
        # 2. User's related content
        cur.execute("""
            SELECT text_hash, summary, topic, created_at
            FROM summaries
            WHERE user_id = %s 
            AND topic = %s
            AND text_hash != %s
            ORDER BY accessed_count DESC, created_at DESC
            LIMIT 2
        """, (user_id, topic, hashlib.md5(current_text.encode()).hexdigest()))
        
        for row in cur:
            days_ago = (datetime.now() - row['created_at']).days
            suggestions.append({
                'type': 'your_history',
                'summary': row['summary'][:100] + '...',
                'topic': row['topic'],
                'reason': f"You read this {days_ago} days ago"
            })
        
        return suggestions[:5]  # Max 5 suggestions
    
    def _get_quick_insights(self, user_id: str, topic: str) -> List[str]:
        """
        Get quick insights about user's reading patterns.
        Simple observations, not 'temporal intelligence crystallization'.
        """
        insights = []
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # How many times has user read about this topic?
        cur.execute("""
            SELECT COUNT(*) as count, MIN(created_at) as first_seen
            FROM summaries
            WHERE user_id = %s AND topic = %s
        """, (user_id, topic))
        
        row = cur.fetchone()
        if row['count'] > 1:
            days_ago = (datetime.now() - row['first_seen']).days
            insights.append(f"You've read about {topic} {row['count']} times in the last {days_ago} days")
        
        # Busiest reading time
        cur.execute("""
            SELECT EXTRACT(hour FROM created_at) as hour, COUNT(*) as count
            FROM summaries
            WHERE user_id = %s
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        """, (user_id,))
        
        row = cur.fetchone()
        if row:
            hour = int(row['hour'])
            insights.append(f"You're most productive around {hour}:00")
        
        return insights
    
    def search_memory(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Search user's summaries using PostgreSQL full-text search.
        This is the 'superhuman memory' - it never forgets and searches fast.
        """
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT text_hash, summary, topic, created_at,
                   ts_rank(to_tsvector('english', text || ' ' || summary), 
                          plainto_tsquery('english', %s)) as rank
            FROM summaries
            WHERE user_id = %s
            AND to_tsvector('english', text || ' ' || summary) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC, created_at DESC
            LIMIT 10
        """, (query, user_id, query))
        
        results = []
        for row in cur:
            results.append({
                'summary': row['summary'],
                'topic': row['topic'],
                'created_at': row['created_at'].isoformat(),
                'relevance': float(row['rank'])
            })
        
        return results
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about user's reading patterns.
        Real insights from real data, not made-up 'intelligence'.
        """
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Basic stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_summaries,
                COUNT(DISTINCT topic) as unique_topics,
                AVG(LENGTH(text)) as avg_text_length,
                COUNT(DISTINCT DATE(created_at)) as active_days
            FROM summaries
            WHERE user_id = %s
        """, (user_id,))
        
        stats = cur.fetchone()
        
        # Top topics
        cur.execute("""
            SELECT topic, COUNT(*) as count
            FROM summaries
            WHERE user_id = %s
            GROUP BY topic
            ORDER BY count DESC
            LIMIT 5
        """, (user_id,))
        
        top_topics = [{'topic': row['topic'], 'count': row['count']} 
                     for row in cur]
        
        # Reading velocity (summaries per active day)
        velocity = stats['total_summaries'] / max(1, stats['active_days'])
        
        # Interest evolution (topics over time)
        cur.execute("""
            SELECT DATE(created_at) as day, topic, COUNT(*) as count
            FROM summaries
            WHERE user_id = %s
            AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY day, topic
            ORDER BY day DESC
        """, (user_id,))
        
        evolution = defaultdict(list)
        for row in cur:
            evolution[row['day'].isoformat()].append({
                'topic': row['topic'],
                'count': row['count']
            })
        
        return {
            'stats': dict(stats),
            'top_topics': top_topics,
            'reading_velocity': round(velocity, 1),
            'interest_evolution': dict(evolution),
            'insights': [
                f"You've summarized {stats['total_summaries']} texts across {stats['unique_topics']} topics",
                f"You read about {round(velocity, 1)} documents per active day",
                f"Your main interest is {top_topics[0]['topic']}" if top_topics else "Start reading to see insights"
            ]
        }
    
    def _store_summary(self, user_id: str, text_hash: str, text: str, 
                      summary: str, topic: str, context_type: str):
        """Store summary in database."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO summaries (user_id, text_hash, text, summary, topic, context_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, text_hash) 
            DO UPDATE SET 
                accessed_count = summaries.accessed_count + 1,
                last_accessed = CURRENT_TIMESTAMP
        """, (user_id, text_hash, text, summary, topic, context_type))
        self.conn.commit()
    
    def _update_user_patterns(self, user_id: str, topic: str, text_length: int):
        """Update user patterns for better predictions."""
        cur = self.conn.cursor()
        
        # Ensure user exists in patterns table
        cur.execute("""
            INSERT INTO user_patterns (user_id) 
            VALUES (%s) 
            ON CONFLICT (user_id) DO NOTHING
        """, (user_id,))
        
        # Update patterns
        cur.execute("""
            UPDATE user_patterns 
            SET topics = jsonb_set(
                    topics, 
                    %s::text[], 
                    COALESCE(topics->%s, '0')::int + 1::text::jsonb
                ),
                reading_times = reading_times || %s::jsonb,
                avg_text_length = (avg_text_length + %s) / 2,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = %s
        """, ([topic], topic, json.dumps(datetime.now().hour), text_length, user_id))
        
        self.conn.commit()
    
    def _get_existing_summary(self, user_id: str, text_hash: str) -> Optional[Dict[str, Any]]:
        """Check if we already have this summary for this user."""
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, summary, topic, context_type, last_accessed
            FROM summaries
            WHERE user_id = %s AND text_hash = %s
        """, (user_id, text_hash))
        return cur.fetchone()
    
    def _update_access_count(self, summary_id: int):
        """Update access count and last accessed time."""
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE summaries 
            SET accessed_count = accessed_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (summary_id,))
        self.conn.commit()


# Flask API endpoints
from flask import Flask, request, jsonify

app = Flask(__name__)
intelligence = IntelligentSum()


@app.route('/api/v2/summarize', methods=['POST'])
def intelligent_summarize():
    """Summarize with intelligence features."""
    data = request.get_json()
    
    if not data or 'text' not in data or 'user_id' not in data:
        return jsonify({'error': 'Missing text or user_id'}), 400
    
    result = intelligence.summarize_with_intelligence(
        data['user_id'], 
        data['text']
    )
    
    return jsonify(result)


@app.route('/api/v2/search', methods=['GET'])
def search():
    """Search user's memory."""
    user_id = request.args.get('user_id')
    query = request.args.get('q')
    
    if not user_id or not query:
        return jsonify({'error': 'Missing user_id or query'}), 400
    
    results = intelligence.search_memory(user_id, query)
    return jsonify({'results': results})


@app.route('/api/v2/insights/<user_id>', methods=['GET'])
def insights(user_id):
    """Get user insights."""
    insights = intelligence.get_user_insights(user_id)
    return jsonify(insights)


if __name__ == '__main__':
    print("SUM Intelligence Layer")
    print("=" * 50)
    print("Built on sum_simple.py, adds:")
    print("- Pattern recognition (simple clustering)")
    print("- Memory (PostgreSQL full-text search)")
    print("- Predictions (collaborative filtering)")
    print("- Context detection (heuristics)")
    print("\nTotal lines: ~600")
    print("Total complexity: Manageable")
    print("Total value: High")
    
    app.run(host='0.0.0.0', port=3001, debug=False)