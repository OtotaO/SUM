"""
test_sum_intelligence.py - Tests for the intelligence layer

Test that intelligence features work without testing implementation details.
Focus on: Does it provide value?
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import psycopg2

# Import our intelligence layer
import sys
sys.path.append('..')
from sum_intelligence import app, IntelligentSum


class TestIntelligenceAPI:
    """Test the intelligence API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection"""
        with patch('sum_intelligence.psycopg2.connect') as mock:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock.return_value = mock_conn
            yield mock_cursor
    
    def test_intelligent_summarize(self, client, mock_db):
        """Test intelligent summarization endpoint"""
        # Mock database responses
        mock_db.fetchone.return_value = None  # No existing summary
        
        with patch('sum_intelligence.summarizer') as mock_summarizer:
            mock_summarizer.return_value = [{'summary_text': 'Intelligent summary'}]
            
            response = client.post('/api/v2/summarize',
                                 json={
                                     'user_id': 'test_user',
                                     'text': 'This is about machine learning and AI.'
                                 })
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Check response structure
            assert 'summary' in data
            assert 'topic' in data
            assert 'context' in data
            assert 'suggestions' in data
            assert 'insights' in data
    
    def test_search_endpoint(self, client, mock_db):
        """Test memory search endpoint"""
        # Mock search results
        mock_db.fetchall.return_value = [
            {
                'summary': 'Test summary about AI',
                'topic': 'ai',
                'created_at': datetime.now(),
                'rank': 0.9
            }
        ]
        
        response = client.get('/api/v2/search?user_id=test_user&q=artificial+intelligence')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) > 0
        assert 'relevance' in data['results'][0]
    
    def test_insights_endpoint(self, client, mock_db):
        """Test user insights endpoint"""
        # Mock database responses for insights
        mock_db.fetchone.return_value = {
            'total_summaries': 42,
            'unique_topics': 7,
            'avg_text_length': 500.5,
            'active_days': 15
        }
        
        mock_db.fetchall.return_value = [
            {'topic': 'ai', 'count': 15},
            {'topic': 'business', 'count': 10}
        ]
        
        response = client.get('/api/v2/insights/test_user')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'stats' in data
        assert 'top_topics' in data
        assert 'reading_velocity' in data
        assert 'insights' in data
        
        # Verify calculations
        assert data['reading_velocity'] == 2.8  # 42/15


class TestIntelligenceFeatures:
    """Test the intelligence functionality"""
    
    @pytest.fixture
    def intelligence(self, mock_db):
        """Create intelligence instance with mocked DB"""
        with patch('sum_intelligence.psycopg2.connect'):
            return IntelligentSum()
    
    def test_context_detection(self, intelligence):
        """Test context detection heuristics"""
        # Academic context
        academic_text = "Our hypothesis is that the methodology will show significant results in the study"
        assert intelligence._detect_context(academic_text) == 'academic'
        
        # Business context
        business_text = "Q4 revenue increased by 23% with strong profit margins for stakeholders"
        assert intelligence._detect_context(business_text) == 'business'
        
        # Technical context
        tech_text = "The bug in the code was fixed by refactoring the algorithm implementation"
        assert intelligence._detect_context(tech_text) == 'technical'
        
        # General context
        general_text = "The weather today is nice and sunny"
        assert intelligence._detect_context(general_text) == 'general'
    
    def test_topic_extraction(self, intelligence):
        """Test topic extraction"""
        # Short text - uses word frequency
        short_text = "Python programming is great. Python is powerful. I love Python."
        topic = intelligence._extract_topic(short_text)
        assert topic == 'python'
        
        # Test edge cases
        empty_topic = intelligence._extract_topic("")
        assert empty_topic == 'general'
        
        very_short = intelligence._extract_topic("Hello world")
        assert very_short == 'general'
    
    def test_suggestions_generation(self, intelligence):
        """Test that suggestions are generated properly"""
        with patch.object(intelligence.conn, 'cursor') as mock_cursor_method:
            mock_cursor = MagicMock()
            mock_cursor_method.return_value = mock_cursor
            
            # Mock suggestion data
            mock_cursor.fetchall.return_value = [
                {
                    'summary': 'Related summary about AI and ML',
                    'topic': 'ai',
                    'read_count': 5
                }
            ]
            
            suggestions = intelligence._get_suggestions('user123', 'ai', 'Current text about AI')
            
            assert isinstance(suggestions, list)
            assert len(suggestions) <= 5
            
            if suggestions:
                assert 'type' in suggestions[0]
                assert 'summary' in suggestions[0]
                assert 'reason' in suggestions[0]


class TestPerformanceIntelligence:
    """Test that intelligence features are performant"""
    
    def test_context_detection_speed(self):
        """Test that context detection is fast"""
        intelligence = IntelligentSum.__new__(IntelligentSum)  # Skip __init__
        
        # Context detection should be near-instant
        import time
        text = "This is a sample text about business revenue and profit margins" * 10
        
        start = time.time()
        for _ in range(100):  # Run 100 times
            context = intelligence._detect_context(text)
        end = time.time()
        
        avg_time = (end - start) / 100
        assert avg_time < 0.001  # Should be < 1ms per detection
    
    def test_cache_efficiency(self):
        """Test that caching improves performance"""
        # This is more of an integration test
        # In real implementation, we'd test that repeated calls are faster
        pass


class TestDataMigration:
    """Test data migration utilities"""
    
    def test_importance_calculation(self):
        """Test importance score calculation"""
        intelligence = IntelligentSum.__new__(IntelligentSum)
        
        # Short text - base score
        short = "Short text"
        assert 0.4 <= intelligence._calculate_importance(short) <= 0.6
        
        # Long text with keywords
        important = "This is a critical breakthrough in our research" + " more text" * 20
        score = intelligence._calculate_importance(important)
        assert score > 0.6
        
        # Complex text (high unique word ratio)
        complex_text = " ".join([f"word{i}" for i in range(100)])
        score = intelligence._calculate_importance(complex_text)
        assert score > 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])