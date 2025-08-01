"""
Insight extraction service - identifies key insights and important quotes.
Focused on finding meaningful statements in text.
"""

from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize


class InsightExtractor:
    """Extracts key insights and important quotes from text."""
    
    def extract_insights(
        self, 
        text: str, 
        max_insights: int = 3, 
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find sentences that contain key insights."""
        sentences = sent_tokenize(text)
        insights = []
        
        for sentence in sentences:
            insight_score = self._score_insight_importance(sentence)
            
            if insight_score >= min_score:
                insights.append({
                    'text': sentence.strip(),
                    'score': insight_score,
                    'type': self._classify_insight_type(sentence)
                })
        
        # Sort by score and return top insights
        insights.sort(key=lambda x: x['score'], reverse=True)
        return insights[:max_insights]
    
    def _score_insight_importance(self, sentence: str) -> float:
        """Score the importance and insightfulness of a sentence."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Key insight markers
        insight_markers = [
            'important', 'significant', 'key', 'essential', 'critical', 'fundamental',
            'demonstrates', 'shows', 'reveals', 'indicates', 'suggests', 'implies'
        ]
        score += sum(0.15 for marker in insight_markers if marker in sentence_lower)
        
        # Definitive statement indicators
        definitive_indicators = [
            'therefore', 'thus', 'hence', 'consequently', 'clearly', 'evidently',
            'importantly', 'significantly', 'notably', 'particularly'
        ]
        score += sum(0.1 for indicator in definitive_indicators if indicator in sentence_lower)
        
        # Paradox patterns
        paradox_patterns = [
            ('more', 'less'), ('give', 'receive'), ('lose', 'find'),
            ('empty', 'full'), ('simple', 'complex'), ('small', 'great')
        ]
        for pattern in paradox_patterns:
            if all(word in sentence_lower for word in pattern):
                score += 0.2
        
        # Metaphorical language
        metaphor_markers = ['like', 'as if', 'mirror', 'reflection', 'symbol', 'represents']
        score += sum(0.05 for marker in metaphor_markers if marker in sentence_lower)
        
        # Definitive statements
        if sentence.strip().endswith('.') and any(verb in sentence_lower for verb in ['is', 'are', 'means', 'represents']):
            score += 0.1
        
        # Length optimization
        word_count = len(sentence.split())
        if 8 <= word_count <= 30:
            score += 0.1
        elif word_count > 40:
            score -= 0.1
        
        return min(score, 1.0)
    
    def _classify_insight_type(self, sentence: str) -> str:
        """Classify the type of insight."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['truth', 'reality', 'fact']):
            return 'truth'
        elif any(word in sentence_lower for word in ['wisdom', 'understanding', 'insight']):
            return 'wisdom'
        elif any(word in sentence_lower for word in ['purpose', 'meaning', 'why']):
            return 'purpose'
        elif any(word in sentence_lower for word in ['being', 'existence', 'consciousness']):  
            return 'existential'
        elif any(word in sentence_lower for word in ['love', 'compassion', 'kindness']):
            return 'love'
        else:
            return 'insight'