"""
Summary building service - constructs summaries from scored sentences.
Clean, focused logic for summary assembly.
"""

from typing import Dict, List, Tuple
from nltk.tokenize import word_tokenize


class SummaryBuilder:
    """Builds summaries from scored sentences."""
    
    def build_summary(
        self, 
        sentences: List[str], 
        sentence_scores: Dict[str, float], 
        sentence_tokens: Dict[str, int], 
        max_tokens: int, 
        threshold: float
    ) -> str:
        """Build a summary from scored sentences."""
        qualified_sentences = [
            (sentence, score) for sentence, score in sentence_scores.items() 
            if score > threshold
        ]
        
        sorted_sentences = sorted(qualified_sentences, key=lambda x: x[1], reverse=True)
        summary_sentences = []
        current_tokens = 0
        
        for sentence, score in sorted_sentences:
            if current_tokens + sentence_tokens[sentence] <= max_tokens:
                summary_sentences.append((sentence, sentences.index(sentence)))
                current_tokens += sentence_tokens[sentence]
                
            if current_tokens >= max_tokens:
                break
                
        if not summary_sentences and sorted_sentences:
            top_sentence, _ = sorted_sentences[0]
            if sentence_tokens[top_sentence] <= max_tokens:
                summary_sentences.append((top_sentence, sentences.index(top_sentence)))
            else:
                words = word_tokenize(top_sentence)[:max_tokens-1]
                return ' '.join(words) + '...'
                
        summary_sentences.sort(key=lambda x: x[1])
        return ' '.join(sentence for sentence, _ in summary_sentences)
    
    def build_condensed_summary(
        self, 
        sentences: List[str], 
        sentence_scores: Dict[str, float], 
        sentence_tokens: Dict[str, int], 
        max_tokens: int
    ) -> str:
        """Build a condensed summary from the highest scoring sentences."""
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_sentences = []
        current_tokens = 0
        
        for sentence, _ in sorted_sentences:
            if current_tokens + sentence_tokens[sentence] <= max_tokens:
                selected_sentences.append((sentence, sentences.index(sentence)))
                current_tokens += sentence_tokens[sentence]
            else:
                break
                
        if not selected_sentences and sorted_sentences:
            top_sentence, _ = sorted_sentences[0]
            if sentence_tokens[top_sentence] <= max_tokens:
                selected_sentences.append((top_sentence, sentences.index(top_sentence)))
            else:
                words = word_tokenize(top_sentence)[:max_tokens-1]
                return ' '.join(words) + '...'
        
        if not selected_sentences:
            return ""
            
        selected_sentences.sort(key=lambda x: x[1])
        return ' '.join(sentence for sentence, _ in selected_sentences)