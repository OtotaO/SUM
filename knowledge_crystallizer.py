"""
Knowledge Crystallization Engine - The Core of Ultimate Summarization
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from collections import defaultdict
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from optimized_summarizer import OptimizedSummarizer
import logging
import asyncio
from llm_backend import llm_backend, ModelProvider

logger = logging.getLogger(__name__)


class DensityLevel(Enum):
    """Information density levels for crystallization"""
    ESSENCE = 0.01      # Single insight (1% of original)
    TWEET = 0.02        # 280 chars worth
    ELEVATOR = 0.05     # 30-second pitch
    EXECUTIVE = 0.10    # C-suite brief
    BRIEF = 0.20        # Quick read
    STANDARD = 0.30     # Balanced summary
    DETAILED = 0.50     # Thorough coverage
    COMPREHENSIVE = 0.70  # Near-complete retention


class StylePersona(Enum):
    """Writing style personas for summaries"""
    HEMINGWAY = "hemingway"  # Terse. Direct. No fluff.
    ACADEMIC = "academic"    # Rigorous, cited, methodical
    STORYTELLER = "storyteller"  # Narrative flow, engaging
    ANALYST = "analyst"      # Data-driven, quantitative
    POET = "poet"            # Metaphorical, evocative
    EXECUTIVE = "executive"  # Action-oriented, strategic
    TEACHER = "teacher"      # Educational, scaffolded
    JOURNALIST = "journalist"  # Who, what, when, where, why
    DEVELOPER = "developer"  # Technical, precise, code-aware
    NEUTRAL = "neutral"      # Balanced, objective


@dataclass
class CrystallizationConfig:
    """Configuration for knowledge crystallization"""
    density: DensityLevel = DensityLevel.STANDARD
    style: StylePersona = StylePersona.NEUTRAL
    preserve_entities: bool = True
    preserve_numbers: bool = True
    preserve_quotes: bool = False
    interactive: bool = False
    progressive: bool = False
    user_preferences: Optional[Dict] = None


@dataclass
class CrystallizedKnowledge:
    """Output structure for crystallized knowledge"""
    essence: str  # Single most important insight
    levels: Dict[str, str]  # Different density levels
    metadata: Dict[str, Any]  # Stats and info
    interactive_elements: Optional[Dict] = None
    quality_score: float = 0.0


class KnowledgeCrystallizer:
    """
    The ultimate knowledge crystallization engine that transforms
    information into perfectly dense, stylized summaries
    """
    
    def __init__(self):
        self.style_templates = self._load_style_templates()
        self.density_algorithms = self._load_density_algorithms()
        self.quality_scorer = QualityScorer()
        self.preference_learner = PreferenceLearner()
        self.summarizer = OptimizedSummarizer()
        
    def crystallize(self, 
                   text: str, 
                   config: Optional[CrystallizationConfig] = None) -> CrystallizedKnowledge:
        """
        Transform text into crystallized knowledge at specified density and style
        """
        if not config:
            config = CrystallizationConfig()
            
        # Learn from user preferences if available
        if config.user_preferences:
            config = self.preference_learner.adapt_config(config, text)
        
        # Check if we can use LLM for better results
        available_providers = llm_backend.get_available_providers()
        use_llm = len(available_providers) > 1 or 'openai' in available_providers or 'anthropic' in available_providers
        
        if use_llm:
            # Use LLM-powered crystallization
            return self._crystallize_with_llm(text, config)
        else:
            # Fallback to traditional extraction
            return self._crystallize_traditional(text, config)
    
    def _crystallize_traditional(self, text: str, config: CrystallizationConfig) -> CrystallizedKnowledge:
        """Traditional extraction-based crystallization"""
        # Extract key information components
        components = self._extract_components(text)
        
        # Generate summaries at different density levels
        levels = {}
        for density in DensityLevel:
            levels[density.name.lower()] = self._generate_at_density(
                components, density, config.style
            )
        
        # Extract the single most important insight
        essence = self._extract_essence(components, config.style)
        
        # Calculate quality metrics
        quality_score = self.quality_scorer.score(text, levels['standard'])
        
        # Create interactive elements if requested
        interactive = None
        if config.interactive:
            interactive = self._create_interactive_elements(components, levels)
        
        # Build metadata
        metadata = {
            'original_length': len(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'key_entities': components['entities'][:10],
            'main_topics': components['topics'][:5],
            'sentiment': components['sentiment'],
            'style_applied': config.style.value,
            'density_used': config.density.value,
            'method': 'traditional'
        }
        
        return CrystallizedKnowledge(
            essence=essence,
            levels=levels,
            metadata=metadata,
            interactive_elements=interactive,
            quality_score=quality_score
        )
    
    def _crystallize_with_llm(self, text: str, config: CrystallizationConfig) -> CrystallizedKnowledge:
        """LLM-powered crystallization for superior results"""
        # Run async in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._async_crystallize_with_llm(text, config))
        finally:
            loop.close()
    
    async def _async_crystallize_with_llm(self, text: str, config: CrystallizationConfig) -> CrystallizedKnowledge:
        """Async LLM crystallization with Cascading Summarization"""
        
        levels = {}
        style_instruction = self._get_style_instruction(config.style)
        
        # 1. Generate Comprehensive (Top Level) first from original text
        # Use full text (truncated reasonably by prompt creator if needed)
        comprehensive_prompt = self._create_density_prompt(text, DensityLevel.COMPREHENSIVE, config.style)
        
        # Use concurrent execution if we want, but cascading requires dependency.
        # We start with the heaviest lifting.
        comprehensive_summary = await llm_backend.generate(
            comprehensive_prompt,
            temperature=0.3,
            max_tokens=2500
        )
        levels['comprehensive'] = comprehensive_summary
        
        # 2. Cascade down: Use the previous summary as the source for the next level
        # This ensures coherence and saves tokens.
        cascade_order = [
            DensityLevel.DETAILED,
            DensityLevel.STANDARD,
            DensityLevel.BRIEF,
            DensityLevel.EXECUTIVE,
            DensityLevel.ELEVATOR,
            DensityLevel.TWEET,
            DensityLevel.ESSENCE
        ]

        current_source_text = comprehensive_summary

        for density in cascade_order:
            # Use the previous level's output as input
            prompt = self._create_density_prompt(current_source_text, density, config.style)

            # Estimate tokens needed
            max_tokens = 1000
            if density == DensityLevel.TWEET: max_tokens = 100
            elif density == DensityLevel.ESSENCE: max_tokens = 100
            elif density == DensityLevel.ELEVATOR: max_tokens = 200
            elif density == DensityLevel.EXECUTIVE: max_tokens = 400
            elif density == DensityLevel.BRIEF: max_tokens = 800

            summary = await llm_backend.generate(
                prompt,
                temperature=0.4,
                max_tokens=max_tokens
            )

            levels[density.name.lower()] = summary

            # Update source only if the summary is substantial enough to be a source
            # Otherwise keep using the detailed one?
            # Actually, standard -> brief -> executive is a good chain.
            if len(summary) > 200:
                current_source_text = summary
        
        essence = levels.get('essence', '')
        if not essence:
             # Fallback if essence failed
             essence = levels.get('tweet', 'Summary generated.')

        # Extract components for metadata (using original text)
        components = self._extract_components(text)
        
        # Calculate quality score
        quality_score = self.quality_scorer.score(text, levels.get('standard', ''))
        
        # Build metadata
        metadata = {
            'original_length': len(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'key_entities': components['entities'][:10],
            'main_topics': components['topics'][:5],
            'sentiment': components['sentiment'],
            'style_applied': config.style.value,
            'density_used': config.density.value,
            'method': 'llm_cascade',
            'provider': llm_backend.default_provider.value
        }
        
        return CrystallizedKnowledge(
            essence=essence,
            levels=levels,
            metadata=metadata,
            interactive_elements=None,
            quality_score=quality_score
        )

    def _create_density_prompt(self, text: str, density: DensityLevel, style: StylePersona) -> str:
        """Create prompt for specific density level"""
        
        style_instruction = self._get_style_instruction(style)
        
        density_instructions = {
            DensityLevel.ESSENCE: "Summarize in exactly one sentence capturing the core message.",
            DensityLevel.TWEET: "Summarize in 280 characters or less (tweet length).",
            DensityLevel.ELEVATOR: "Summarize in 3-4 sentences (30-second elevator pitch).",
            DensityLevel.EXECUTIVE: "Summarize in one paragraph suitable for C-suite briefing.",
            DensityLevel.BRIEF: "Summarize in 2-3 paragraphs for a quick read.",
            DensityLevel.STANDARD: "Provide a balanced 3-5 paragraph summary.",
            DensityLevel.DETAILED: "Provide a thorough half-page summary covering all key points.",
            DensityLevel.COMPREHENSIVE: "Provide a comprehensive full-page summary retaining most information."
        }
        
        return f"""Summarize the following text.
Density: {density_instructions[density]}
Style: {style_instruction}

Text: {text[:5000]}...

Summary:"""
    
    def _get_style_instruction(self, style: StylePersona) -> str:
        """Get instruction for writing style"""
        
        instructions = {
            StylePersona.HEMINGWAY: "Write in Hemingway style - terse, direct sentences with no fluff.",
            StylePersona.ACADEMIC: "Write in academic style - rigorous, methodical, with evidence.",
            StylePersona.STORYTELLER: "Write with narrative flow, engaging and story-like.",
            StylePersona.ANALYST: "Focus on data, metrics, and quantitative insights.",
            StylePersona.POET: "Use metaphorical and evocative language.",
            StylePersona.EXECUTIVE: "Be action-oriented and strategic, focus on decisions.",
            StylePersona.TEACHER: "Be educational and clear, explaining concepts.",
            StylePersona.JOURNALIST: "Follow the 5 W's and H structure (who, what, when, where, why, how).",
            StylePersona.DEVELOPER: "Be technical and precise, suitable for documentation.",
            StylePersona.NEUTRAL: "Use balanced, objective language."
        }
        
        return instructions.get(style, instructions[StylePersona.NEUTRAL])
    
    def _extract_components(self, text: str) -> Dict[str, Any]:
        """Extract key components from text for crystallization"""
        sentences = sent_tokenize(text)
        
        return {
            'sentences': sentences,
            'entities': self._extract_entities(text),
            'topics': self._extract_topics(text),
            'key_phrases': self._extract_key_phrases(text),
            'sentiment': self._analyze_sentiment(text),
            'structure': self._analyze_structure(text),
            'facts': self._extract_facts(text),
            'opinions': self._extract_opinions(text),
            'actions': self._extract_actions(text),
            'numbers': self._extract_numbers(text),
            'quotes': self._extract_quotes(text)
        }
    
    def _generate_at_density(self, 
                            components: Dict,
                            density: DensityLevel,
                            style: StylePersona) -> str:
        """Generate summary at specific density level using OptimizedSummarizer"""
        # Reconstruct text from components if not available directly
        text = ' '.join(components.get('sentences', []))
        if not text:
            return ""

        # Use OptimizedSummarizer for high quality extractive summarization
        # Density value is used as ratio
        ratio = density.value
        
        # Ensure minimum ratio for very short texts or low densities
        if len(text) < 500:
            ratio = max(ratio, 0.5)

        try:
            summary = self.summarizer.summarize_ultra_fast(text, ratio=ratio)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            summary = text[:500] + "..."
        
        return summary
    
    def _extract_essence(self, components: Dict, style: StylePersona) -> str:
        """Extract the single most important insight"""
        # Find the most central sentence
        central = self._find_central_sentence(components)
        
        # Apply style to make it punchy
        if style == StylePersona.HEMINGWAY:
            return self._hemingway_punch(central)
        elif style == StylePersona.POET:
            return self._poeticize(central)
        elif style == StylePersona.EXECUTIVE:
            return self._executivize(central)
        else:
            return central
    
    def _rank_sentences(self, components: Dict) -> List[str]:
        """Rank sentences by importance using multiple signals"""
        sentences = components['sentences']
        scores = []
        
        for sentence in sentences:
            score = 0
            # Position bias (first and last paragraphs)
            position = sentences.index(sentence) / len(sentences)
            if position < 0.2 or position > 0.8:
                score += 2
            
            # Entity density
            entity_count = sum(1 for e in components['entities'] if e in sentence)
            score += entity_count * 3
            
            # Number presence
            if any(num in sentence for num in components['numbers']):
                score += 2
            
            # Key phrase presence
            phrase_count = sum(1 for p in components['key_phrases'] if p in sentence.lower())
            score += phrase_count * 2
            
            # Action words
            if any(action in sentence.lower() for action in components['actions']):
                score += 1
            
            scores.append((score, sentence))
        
        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])
        return [s[1] for s in scores]
    
    def _apply_style(self, sentences: List[str], style: StylePersona, components: Dict) -> str:
        """Apply style persona to text"""
        if style == StylePersona.HEMINGWAY:
            return self._style_hemingway(sentences)
        elif style == StylePersona.ACADEMIC:
            return self._style_academic(sentences, components)
        elif style == StylePersona.STORYTELLER:
            return self._style_storyteller(sentences, components)
        elif style == StylePersona.ANALYST:
            return self._style_analyst(sentences, components)
        elif style == StylePersona.EXECUTIVE:
            return self._style_executive(sentences, components)
        elif style == StylePersona.JOURNALIST:
            return self._style_journalist(sentences, components)
        elif style == StylePersona.DEVELOPER:
            return self._style_developer(sentences, components)
        else:
            return ' '.join(sentences)
    
    def _style_hemingway(self, sentences: List[str]) -> str:
        """Short sentences. Clear words. No adjectives."""
        styled = []
        for sentence in sentences:
            # Remove adjectives and adverbs
            simplified = re.sub(r'\b(very|really|quite|rather|extremely)\b', '', sentence)
            # Split long sentences
            if len(simplified) > 100:
                parts = simplified.split(',')
                styled.extend([p.strip() + '.' for p in parts if len(p.strip()) > 10])
            else:
                styled.append(simplified)
        return ' '.join(styled)
    
    def _style_academic(self, sentences: List[str], components: Dict) -> str:
        """Add rigor and citations"""
        styled = []
        for i, sentence in enumerate(sentences):
            if components['facts'] and any(fact in sentence for fact in components['facts'][:3]):
                sentence += f" (see evidence {i+1})"
            styled.append(sentence)
        
        # Add methodology note if applicable
        if components['structure'].get('has_methodology'):
            styled.insert(0, "The following summary synthesizes key findings using systematic analysis. ")
        
        return ' '.join(styled)
    
    def _style_storyteller(self, sentences: List[str], components: Dict) -> str:
        """Create narrative flow"""
        if not sentences:
            return ""
        
        # Add narrative connectors
        connectors = ["First, ", "Then, ", "Meanwhile, ", "Subsequently, ", "Finally, "]
        styled = []
        
        for i, sentence in enumerate(sentences):
            if i < len(connectors):
                styled.append(connectors[i] + sentence[0].lower() + sentence[1:])
            else:
                styled.append(sentence)
        
        # Add dramatic conclusion if sentiment is strong
        if components['sentiment'].get('compound', 0) > 0.5:
            styled.append("The implications are profound.")
        elif components['sentiment'].get('compound', 0) < -0.5:
            styled.append("The challenges remain significant.")
        
        return ' '.join(styled)
    
    def _style_analyst(self, sentences: List[str], components: Dict) -> str:
        """Focus on data and metrics"""
        styled = []
        
        # Prioritize sentences with numbers
        data_sentences = [s for s in sentences if any(n in s for n in components['numbers'])]
        other_sentences = [s for s in sentences if s not in data_sentences]
        
        # Lead with data
        if data_sentences:
            styled.append("Key metrics: " + ' '.join(data_sentences[:2]))
        
        # Add analysis
        if other_sentences:
            styled.append("Analysis reveals: " + ' '.join(other_sentences))
        
        # Add trend if detected
        if self._detect_trend(components['numbers']):
            styled.append("Trend indicates growth trajectory.")
        
        return ' '.join(styled)
    
    def _style_executive(self, sentences: List[str], components: Dict) -> str:
        """Action-oriented and strategic"""
        styled = []
        
        # Extract action items
        actions = [s for s in sentences if any(a in s.lower() for a in ['should', 'must', 'need', 'require', 'recommend'])]
        
        if actions:
            styled.append("ACTION REQUIRED: " + actions[0])
        
        # Key findings
        non_actions = [s for s in sentences if s not in actions]
        if non_actions:
            styled.append("KEY FINDINGS: " + ' '.join(non_actions[:2]))
        
        # Strategic implications
        if components['facts']:
            styled.append("STRATEGIC IMPACT: Based on evidence, decisive action needed.")
        
        return ' '.join(styled)
    
    def _style_journalist(self, sentences: List[str], components: Dict) -> str:
        """5 W's and H structure"""
        styled = []
        
        # Try to answer who, what, when, where, why, how
        questions = {
            'who': self._find_who(sentences, components),
            'what': self._find_what(sentences, components),
            'when': self._find_when(sentences, components),
            'where': self._find_where(sentences, components),
            'why': self._find_why(sentences),
            'how': self._find_how(sentences)
        }
        
        # Build lede
        lede_parts = []
        for q, answer in questions.items():
            if answer:
                lede_parts.append(answer)
                if len(lede_parts) >= 2:
                    break
        
        if lede_parts:
            styled.append(' '.join(lede_parts))
        
        # Add remaining context
        remaining = [s for s in sentences if not any(s in str(v) for v in questions.values() if v)]
        styled.extend(remaining[:2])
        
        return ' '.join(styled)
    
    def _style_developer(self, sentences: List[str], components: Dict) -> str:
        """Technical and precise"""
        styled = []
        
        # Look for technical terms
        tech_sentences = [s for s in sentences if any(
            term in s.lower() for term in ['api', 'function', 'method', 'class', 'variable', 
                                           'algorithm', 'data', 'system', 'process', 'implement']
        )]
        
        if tech_sentences:
            styled.append("Technical implementation: " + ' '.join(tech_sentences[:2]))
        
        # Add specifications if found
        spec_sentences = [s for s in sentences if any(n in s for n in components['numbers'])]
        if spec_sentences:
            styled.append("Specifications: " + spec_sentences[0])
        
        # Add remaining
        other = [s for s in sentences if s not in tech_sentences and s not in spec_sentences]
        if other:
            styled.append("Context: " + ' '.join(other[:1]))
        
        return ' '.join(styled)
    
    def _create_interactive_elements(self, components: Dict, levels: Dict) -> Dict:
        """Create interactive UI elements for progressive disclosure"""
        return {
            'expandable_sections': {
                'entities': {
                    'title': 'Key Entities',
                    'items': components['entities'][:10],
                    'expandable': True
                },
                'topics': {
                    'title': 'Main Topics',
                    'items': components['topics'][:5],
                    'expandable': True
                },
                'facts': {
                    'title': 'Key Facts',
                    'items': components['facts'][:10],
                    'expandable': True
                }
            },
            'density_slider': {
                'min': 0.01,
                'max': 0.70,
                'default': 0.30,
                'steps': list(levels.keys())
            },
            'style_selector': {
                'options': [s.value for s in StylePersona],
                'default': 'neutral'
            },
            'highlight_controls': {
                'entities': True,
                'numbers': True,
                'actions': False,
                'sentiment': False
            }
        }
    
    # Helper methods for extraction
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        # Simplified entity extraction - in production, use NER
        entities = []
        # Look for capitalized words that aren't sentence starters
        words = word_tokenize(text)
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                entities.append(word)
        return list(set(entities))
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics using TF-IDF or similar"""
        # Simplified topic extraction
        from collections import Counter
        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and len(w) > 4]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(10)]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract important phrases"""
        # Simplified - in production use proper keyphrase extraction
        phrases = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Look for phrases between commas or with certain patterns
            if ',' in sentence:
                parts = sentence.split(',')
                phrases.extend([p.strip() for p in parts if 10 < len(p) < 50])
        return phrases[:10]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment"""
        # Simplified sentiment - in production use VADER or transformers
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve']
        negative_words = ['bad', 'poor', 'negative', 'fail', 'problem', 'issue']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            'positive': pos_count / (pos_count + neg_count + 1),
            'negative': neg_count / (pos_count + neg_count + 1),
            'neutral': 1 - (pos_count + neg_count) / (len(word_tokenize(text)) + 1),
            'compound': (pos_count - neg_count) / (pos_count + neg_count + 1)
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure"""
        return {
            'has_sections': bool(re.search(r'\n\n.*\n\n', text)),
            'has_lists': bool(re.search(r'\n\s*[-*•]\s+', text)),
            'has_methodology': bool(re.search(r'(method|approach|process|procedure)', text.lower())),
            'has_conclusion': bool(re.search(r'(conclusion|summary|in conclusion|finally)', text.lower()))
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements"""
        facts = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Look for objective language patterns
            if (re.search(r'\b(is|are|was|were|has|have)\b', sentence) and
                not re.search(r'\b(may|might|could|should|believe|think|feel)\b', sentence)):
                facts.append(sentence)
        return facts[:10]
    
    def _extract_opinions(self, text: str) -> List[str]:
        """Extract opinion statements"""
        opinions = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Look for subjective language
            if re.search(r'\b(believe|think|feel|seems|appears|may|might|could|should)\b', sentence.lower()):
                opinions.append(sentence)
        return opinions[:10]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action items or recommendations"""
        actions = []
        action_verbs = ['implement', 'execute', 'perform', 'conduct', 'develop', 
                       'create', 'build', 'establish', 'improve', 'enhance']
        sentences = sent_tokenize(text)
        for sentence in sentences:
            for verb in action_verbs:
                if verb in sentence.lower():
                    actions.append(sentence)
        return list(set(actions))
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and statistics"""
        # Find all numbers, percentages, and numerical expressions
        numbers = re.findall(r'\b\d+\.?\d*%?\b', text)
        return numbers
    
    def _extract_quotes(self, text: str) -> List[str]:
        """Extract quoted text"""
        quotes = re.findall(r'"([^"]*)"', text)
        quotes.extend(re.findall(r"'([^']*)'", text))
        return quotes[:5]
    
    def _find_central_sentence(self, components: Dict) -> str:
        """Find the most central/important sentence"""
        if not components['sentences']:
            return ""
        
        # Score each sentence by how many key elements it contains
        scores = []
        for sentence in components['sentences']:
            score = 0
            score += sum(1 for e in components['entities'] if e in sentence) * 2
            score += sum(1 for t in components['topics'] if t in sentence.lower())
            score += len([n for n in components['numbers'] if n in sentence])
            scores.append((score, sentence))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0][1] if scores else components['sentences'][0]
    
    def _hemingway_punch(self, sentence: str) -> str:
        """Make sentence punchy like Hemingway (Safe Mode)"""
        # In traditional mode without LLM, extensive rewriting is dangerous.
        # We perform only minimal, safe simplifications.

        # Remove truly redundant qualifiers if safe
        words_to_remove = ['very', 'really', 'quite', 'extremely']
        for word in words_to_remove:
            sentence = sentence.replace(f" {word} ", " ")

        return sentence
    
    def _poeticize(self, sentence: str) -> str:
        """Add poetic flair (Safe Mode)"""
        # Without LLM, returning original is safer than naive replacement
        return sentence
    
    def _executivize(self, sentence: str) -> str:
        """Make it executive-friendly (Safe Mode)"""
        # Minimal changes
        return sentence
    
    def _detect_trend(self, numbers: List[str]) -> bool:
        """Detect if numbers show a trend"""
        if len(numbers) < 3:
            return False
        try:
            vals = [float(n.replace('%', '')) for n in numbers[:5] if n.replace('%', '').replace('.', '').isdigit()]
            if len(vals) >= 3:
                # Simple trend detection
                return vals[-1] > vals[0]
        except ValueError:
            return False
        return False
    
    def _find_who(self, sentences: List[str], components: Dict) -> Optional[str]:
        """Find WHO in the text"""
        for sentence in sentences:
            for entity in components['entities']:
                if entity in sentence and entity[0].isupper():
                    return sentence
        return None
    
    def _find_what(self, sentences: List[str], components: Dict) -> Optional[str]:
        """Find WHAT in the text"""
        for sentence in sentences:
            if any(topic in sentence.lower() for topic in components['topics'][:3]):
                return sentence
        return None
    
    def _find_when(self, sentences: List[str], components: Dict) -> Optional[str]:
        """Find WHEN in the text"""
        time_patterns = [r'\b\d{4}\b', r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', r'\b(today|yesterday|tomorrow)\b']
        for sentence in sentences:
            if any(re.search(pattern, sentence.lower()) for pattern in time_patterns):
                return sentence
        return None
    
    def _find_where(self, sentences: List[str], components: Dict) -> Optional[str]:
        """Find WHERE in the text"""
        location_indicators = ['in', 'at', 'near', 'from', 'to']
        for sentence in sentences:
            for indicator in location_indicators:
                if indicator in sentence.lower() and any(e in sentence for e in components['entities']):
                    return sentence
        return None
    
    def _find_why(self, sentences: List[str]) -> Optional[str]:
        """Find WHY in the text"""
        why_patterns = ['because', 'due to', 'reason', 'since', 'as a result']
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in why_patterns):
                return sentence
        return None
    
    def _find_how(self, sentences: List[str]) -> Optional[str]:
        """Find HOW in the text"""
        how_patterns = ['by', 'through', 'using', 'via', 'method', 'approach']
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in how_patterns):
                return sentence
        return None
    
    def _load_style_templates(self) -> Dict:
        """Load style templates"""
        return {}  # Placeholder for style templates
    
    def _load_density_algorithms(self) -> Dict:
        """Load density algorithms"""
        return {}  # Placeholder for density algorithms


class QualityScorer:
    """Score the quality of generated summaries"""
    
    def score(self, original: str, summary: str) -> float:
        """Calculate quality score from 0-100"""
        scores = {
            'coverage': self._score_coverage(original, summary),
            'coherence': self._score_coherence(summary),
            'conciseness': self._score_conciseness(original, summary),
            'accuracy': self._score_accuracy(original, summary)
        }
        
        # Weighted average
        weights = {'coverage': 0.25, 'coherence': 0.25, 'conciseness': 0.25, 'accuracy': 0.25}
        total = sum(scores[metric] * weights[metric] for metric in scores)
        
        return round(total * 100, 2)
    
    def _score_coverage(self, original: str, summary: str) -> float:
        """How much key information is retained"""
        # Simplified - in production use ROUGE or similar
        original_words = set(word_tokenize(original.lower()))
        summary_words = set(word_tokenize(summary.lower()))
        
        if not original_words:
            return 0.0
        
        coverage = len(summary_words.intersection(original_words)) / len(original_words)
        return min(coverage * 2, 1.0)  # Scale up but cap at 1.0
    
    def _score_coherence(self, summary: str) -> float:
        """How well the summary flows"""
        sentences = sent_tokenize(summary)
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words and logical flow
        transition_words = ['however', 'therefore', 'furthermore', 'additionally', 'moreover']
        score = 0.5  # Base score
        
        for word in transition_words:
            if word in summary.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_conciseness(self, original: str, summary: str) -> float:
        """How efficiently information is conveyed"""
        if len(original) == 0:
            return 0.0
        
        compression_ratio = len(summary) / len(original)
        
        # Ideal compression is around 20-30%
        if 0.2 <= compression_ratio <= 0.3:
            return 1.0
        elif compression_ratio < 0.2:
            return compression_ratio * 5  # Too short
        else:
            return max(0, 1.0 - (compression_ratio - 0.3) * 2)  # Too long
    
    def _score_accuracy(self, original: str, summary: str) -> float:
        """How accurate the summary is"""
        # Simplified - check if key entities are preserved
        original_entities = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+\b', summary))
        
        if not original_entities:
            return 1.0
        
        accuracy = len(summary_entities.intersection(original_entities)) / len(original_entities)
        return accuracy


class PreferenceLearner:
    """Learn and adapt to user preferences"""
    
    def __init__(self):
        self.user_profiles = {}
        
    def adapt_config(self, config: CrystallizationConfig, text: str) -> CrystallizationConfig:
        """Adapt configuration based on learned preferences"""
        if not config.user_preferences:
            return config
        
        user_id = config.user_preferences.get('user_id')
        if not user_id:
            return config
        
        # Get or create user profile
        profile = self.user_profiles.get(user_id, {})
        
        # Adapt based on text type
        text_type = self._detect_text_type(text)
        
        if text_type in profile:
            # Use learned preferences for this text type
            preferences = profile[text_type]
            if 'preferred_density' in preferences:
                config.density = DensityLevel[preferences['preferred_density']]
            if 'preferred_style' in preferences:
                config.style = StylePersona[preferences['preferred_style']]
        
        return config
    
    def learn_from_feedback(self, user_id: str, text_type: str, 
                           config: CrystallizationConfig, rating: float):
        """Learn from user feedback"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        if text_type not in self.user_profiles[user_id]:
            self.user_profiles[user_id][text_type] = {}
        
        profile = self.user_profiles[user_id][text_type]
        
        # Update preferences based on rating
        if rating > 4.0:  # Good rating
            profile['preferred_density'] = config.density.name
            profile['preferred_style'] = config.style.name
            profile['rating_history'] = profile.get('rating_history', []) + [rating]
    
    def _detect_text_type(self, text: str) -> str:
        """Detect the type of text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['meeting', 'agenda', 'action item']):
            return 'meeting'
        elif any(word in text_lower for word in ['research', 'study', 'findings', 'methodology']):
            return 'research'
        elif any(word in text_lower for word in ['contract', 'agreement', 'terms', 'party']):
            return 'legal'
        elif any(word in text_lower for word in ['news', 'report', 'announced', 'statement']):
            return 'news'
        elif any(word in text_lower for word in ['function', 'class', 'method', 'api']):
            return 'technical'
        else:
            return 'general'