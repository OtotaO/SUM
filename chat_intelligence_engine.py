"""
Chat Export Intelligence Engine - LLM Conversation Archaeologist

This revolutionary system mines chat exports from LLM conversations to extract
training data goldmines, focusing on context drift, error patterns, and 
correction sequences that can be used to train superior local expert models.

"Every failed conversation is a training opportunity waiting to be discovered."
- The Conversation Archaeologist's Creed

Architecture:
- Parse exports from Claude, ChatGPT, Copilot, Cursor, etc.
- Detect context drift and model failures
- Extract user corrections and redirections
- Generate high-quality training data pairs
- Compress conversations while preserving learning moments
- Train local subject matter expert models

Author: ototao & Claude
License: Apache 2.0
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter
import logging

# Import our adaptive compression system
from adaptive_compression import AdaptiveCompressionEngine, ContentType

# Configure logging
logger = logging.getLogger(__name__)


class ConversationSource(Enum):
    """Supported conversation export sources."""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    COPILOT = "github_copilot"
    CURSOR = "cursor_ai"
    GENERIC = "generic"


class InsightType(Enum):
    """Types of learning insights we can extract."""
    CONTEXT_DRIFT = "context_drift"          # Model lost track of conversation
    ERROR_CORRECTION = "error_correction"    # User corrected model mistake
    KNOWLEDGE_GAP = "knowledge_gap"          # Model lacks domain knowledge
    DIRECTION_CHANGE = "direction_change"    # User redirected conversation
    CLARIFICATION = "clarification"          # User had to explain further
    CODE_FIX = "code_fix"                    # Code correction sequence
    CONCEPT_EXPLANATION = "concept_explanation"  # User explained concept to model


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    speaker: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def word_count(self) -> int:
        return len(word_tokenize(self.content))


@dataclass
class LearningInsight:
    """Represents a learning opportunity extracted from conversation."""
    insight_type: InsightType
    context_before: str  # What led to the issue
    model_response: str  # What the model said/did wrong
    user_correction: str  # How the user corrected it
    context_after: str   # Result after correction
    confidence_score: float  # How confident we are this is valuable
    domain_tags: List[str]  # Domain-specific tags
    metadata: Dict[str, Any] = None


@dataclass
class TrainingPair:
    """High-quality training data pair extracted from insights."""
    input_text: str      # What should be fed to model
    target_output: str   # What the model should produce
    source_insight: LearningInsight
    quality_score: float
    domain: str
    metadata: Dict[str, Any] = None


class ChatExportParser:
    """Universal parser for different chat export formats."""
    
    def __init__(self):
        self.parsers = {
            ConversationSource.CLAUDE: self._parse_claude_export,
            ConversationSource.CHATGPT: self._parse_chatgpt_export,
            ConversationSource.COPILOT: self._parse_copilot_export,
            ConversationSource.CURSOR: self._parse_cursor_export,
            ConversationSource.GENERIC: self._parse_generic_export
        }
    
    def detect_format(self, file_path: str) -> ConversationSource:
        """Auto-detect the export format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Claude exports often have specific structure
        if '"type": "human"' in content and '"type": "assistant"' in content:
            return ConversationSource.CLAUDE
        
        # ChatGPT exports have conversation structure
        if '"conversations"' in content or '"title"' in content:
            return ConversationSource.CHATGPT
        
        # Look for copilot patterns
        if 'github' in content.lower() and ('completion' in content or 'suggestion' in content):
            return ConversationSource.COPILOT
        
        # Cursor AI patterns
        if 'cursor' in content.lower() or 'ai_chat' in content:
            return ConversationSource.CURSOR
        
        return ConversationSource.GENERIC
    
    def parse(self, file_path: str, source: Optional[ConversationSource] = None) -> List[ConversationTurn]:
        """Parse a chat export file into conversation turns."""
        if source is None:
            source = self.detect_format(file_path)
        
        parser_func = self.parsers.get(source, self._parse_generic_export)
        return parser_func(file_path)
    
    def _parse_claude_export(self, file_path: str) -> List[ConversationTurn]:
        """Parse Claude conversation exports."""
        turns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different Claude export formats
            if isinstance(data, list):
                # Array of messages
                for item in data:
                    if isinstance(item, dict) and 'type' in item:
                        speaker = 'user' if item['type'] == 'human' else 'assistant'
                        content = item.get('content', item.get('text', ''))
                        turns.append(ConversationTurn(speaker=speaker, content=content))
            
            elif isinstance(data, dict):
                # Single conversation object
                messages = data.get('messages', data.get('conversation', []))
                for msg in messages:
                    speaker = msg.get('role', msg.get('type', 'unknown'))
                    if speaker == 'human':
                        speaker = 'user'
                    content = msg.get('content', msg.get('text', ''))
                    turns.append(ConversationTurn(speaker=speaker, content=content))
        
        except Exception as e:
            logger.error(f"Error parsing Claude export: {e}")
            return self._parse_generic_export(file_path)
        
        return turns
    
    def _parse_chatgpt_export(self, file_path: str) -> List[ConversationTurn]:
        """Parse ChatGPT conversation exports."""
        turns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ChatGPT export structure
            conversations = data if isinstance(data, list) else [data]
            
            for conv in conversations:
                mapping = conv.get('mapping', {})
                
                # Sort by create_time or order
                sorted_items = sorted(mapping.items(), 
                                    key=lambda x: x[1].get('create_time', 0))
                
                for node_id, node in sorted_items:
                    message = node.get('message')
                    if not message:
                        continue
                    
                    author = message.get('author', {})
                    role = author.get('role', 'unknown')
                    
                    content_parts = message.get('content', {}).get('parts', [])
                    content = ' '.join(content_parts) if content_parts else ''
                    
                    if content and role in ['user', 'assistant']:
                        turns.append(ConversationTurn(speaker=role, content=content))
        
        except Exception as e:
            logger.error(f"Error parsing ChatGPT export: {e}")
            return self._parse_generic_export(file_path)
        
        return turns
    
    def _parse_copilot_export(self, file_path: str) -> List[ConversationTurn]:
        """Parse GitHub Copilot interaction exports."""
        turns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Copilot format (simplified)
            if isinstance(data, list):
                for item in data:
                    if 'prompt' in item and 'completion' in item:
                        turns.append(ConversationTurn(speaker='user', content=item['prompt']))
                        turns.append(ConversationTurn(speaker='assistant', content=item['completion']))
        
        except Exception as e:
            logger.error(f"Error parsing Copilot export: {e}")
        
        return turns
    
    def _parse_cursor_export(self, file_path: str) -> List[ConversationTurn]:
        """Parse Cursor AI chat exports."""
        # Similar to generic, but with Cursor-specific patterns
        return self._parse_generic_export(file_path)
    
    def _parse_generic_export(self, file_path: str) -> List[ConversationTurn]:
        """Generic parser for unknown formats."""
        turns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Look for common patterns
                            speaker = item.get('role', item.get('speaker', item.get('user', 'unknown')))
                            text = item.get('content', item.get('message', item.get('text', '')))
                            if text:
                                turns.append(ConversationTurn(speaker=speaker, content=text))
            
            except json.JSONDecodeError:
                # Try plain text parsing
                lines = content.split('\n')
                current_speaker = None
                current_content = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Look for speaker indicators
                    if line.startswith('User:') or line.startswith('Human:'):
                        if current_speaker and current_content:
                            turns.append(ConversationTurn(
                                speaker=current_speaker, 
                                content='\n'.join(current_content)
                            ))
                        current_speaker = 'user'
                        current_content = [line[line.find(':')+1:].strip()]
                    
                    elif line.startswith('Assistant:') or line.startswith('AI:') or line.startswith('Bot:'):
                        if current_speaker and current_content:
                            turns.append(ConversationTurn(
                                speaker=current_speaker,
                                content='\n'.join(current_content)
                            ))
                        current_speaker = 'assistant'
                        current_content = [line[line.find(':')+1:].strip()]
                    
                    else:
                        if current_speaker:
                            current_content.append(line)
                
                # Add final turn
                if current_speaker and current_content:
                    turns.append(ConversationTurn(
                        speaker=current_speaker,
                        content='\n'.join(current_content)
                    ))
        
        except Exception as e:
            logger.error(f"Error in generic parsing: {e}")
        
        return turns


class ConversationInsightExtractor:
    """Extracts learning insights from parsed conversations."""
    
    def __init__(self, compression_engine: Optional[AdaptiveCompressionEngine] = None):
        self.compression_engine = compression_engine or AdaptiveCompressionEngine()
        
        # Pattern matching for different insight types
        self.patterns = {
            InsightType.ERROR_CORRECTION: [
                r"(?i)(that's|that is|this is)\s+(wrong|incorrect|not right|mistaken)",
                r"(?i)(actually|no,|sorry,|wait,)\s+",
                r"(?i)(let me correct|fix that|correction:)",
            ],
            InsightType.CLARIFICATION: [
                r"(?i)(what i meant|to clarify|let me explain)",
                r"(?i)(i need to|let me be more specific)",
                r"(?i)(sorry for confusion|to be clear)",
            ],
            InsightType.DIRECTION_CHANGE: [
                r"(?i)(instead|rather|actually, let's)",
                r"(?i)(never mind|forget that|different approach)",
                r"(?i)(let's try|how about|what about)",
            ],
            InsightType.CODE_FIX: [
                r"```.*?```",  # Code blocks
                r"(?i)(this code|the function|the method)\s+(is wrong|has an error|doesn't work)",
                r"(?i)(fix|correct|update)\s+(this|the)\s+(code|function|method)",
            ]
        }
    
    def extract_insights(self, conversation: List[ConversationTurn]) -> List[LearningInsight]:
        """Extract all learning insights from a conversation."""
        insights = []
        
        for i in range(len(conversation) - 1):
            current_turn = conversation[i]
            next_turn = conversation[i + 1] if i + 1 < len(conversation) else None
            
            # Only analyze user corrections to assistant responses
            if current_turn.speaker == 'assistant' and next_turn and next_turn.speaker == 'user':
                insight = self._analyze_correction_pattern(
                    current_turn, next_turn, 
                    context_before=conversation[max(0, i-2):i],
                    context_after=conversation[i+2:i+4] if i+2 < len(conversation) else []
                )
                
                if insight:
                    insights.append(insight)
        
        return insights
    
    def _analyze_correction_pattern(self, assistant_turn: ConversationTurn, 
                                  user_turn: ConversationTurn,
                                  context_before: List[ConversationTurn],
                                  context_after: List[ConversationTurn]) -> Optional[LearningInsight]:
        """Analyze a potential correction pattern."""
        
        user_content = user_turn.content.lower()
        
        # Detect insight type
        insight_type = None
        confidence = 0.0
        
        for itype, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_content):
                    insight_type = itype
                    confidence += 0.3
                    break
        
        if not insight_type or confidence < 0.2:
            return None
        
        # Extract domain tags
        domain_tags = self._extract_domain_tags(assistant_turn.content + " " + user_turn.content)
        
        # Build context
        context_before_text = " ".join([turn.content for turn in context_before])
        context_after_text = " ".join([turn.content for turn in context_after])
        
        return LearningInsight(
            insight_type=insight_type,
            context_before=context_before_text,
            model_response=assistant_turn.content,
            user_correction=user_turn.content,
            context_after=context_after_text,
            confidence_score=min(1.0, confidence),
            domain_tags=domain_tags
        )
    
    def _extract_domain_tags(self, text: str) -> List[str]:
        """Extract domain-specific tags from text."""
        tags = []
        
        # Programming domains
        if re.search(r'\b(python|javascript|java|cpp|rust|go)\b', text, re.IGNORECASE):
            tags.append('programming')
        
        if re.search(r'\b(function|class|method|variable|array|object)\b', text, re.IGNORECASE):
            tags.append('coding')
        
        if re.search(r'\b(api|endpoint|request|response|json|xml)\b', text, re.IGNORECASE):
            tags.append('api_development')
        
        # Data science
        if re.search(r'\b(pandas|numpy|sklearn|tensorflow|pytorch)\b', text, re.IGNORECASE):
            tags.append('data_science')
        
        # System administration
        if re.search(r'\b(docker|kubernetes|aws|linux|server)\b', text, re.IGNORECASE):
            tags.append('devops')
        
        # General domains
        if re.search(r'\b(database|sql|query|table)\b', text, re.IGNORECASE):
            tags.append('database')
        
        return tags or ['general']


class TrainingDataGenerator:
    """Generates high-quality training data from extracted insights."""
    
    def __init__(self, compression_engine: AdaptiveCompressionEngine):
        self.compression_engine = compression_engine
    
    def generate_training_pairs(self, insights: List[LearningInsight]) -> List[TrainingPair]:
        """Generate training pairs from learning insights."""
        training_pairs = []
        
        for insight in insights:
            # Different strategies based on insight type
            if insight.insight_type == InsightType.ERROR_CORRECTION:
                pairs = self._generate_error_correction_pairs(insight)
            elif insight.insight_type == InsightType.CODE_FIX:
                pairs = self._generate_code_fix_pairs(insight)
            elif insight.insight_type == InsightType.CLARIFICATION:
                pairs = self._generate_clarification_pairs(insight)
            else:
                pairs = self._generate_generic_pairs(insight)
            
            training_pairs.extend(pairs)
        
        return training_pairs
    
    def _generate_error_correction_pairs(self, insight: LearningInsight) -> List[TrainingPair]:
        """Generate training pairs for error corrections."""
        pairs = []
        
        # Create input that includes the context and original question
        input_text = f"Context: {insight.context_before}\nUser: Please help with this."
        
        # Target should be what the user actually wanted (extracted from correction)
        target_output = self._extract_intended_response(insight.user_correction)
        
        if target_output:
            pair = TrainingPair(
                input_text=input_text,
                target_output=target_output,
                source_insight=insight,
                quality_score=insight.confidence_score,
                domain=insight.domain_tags[0] if insight.domain_tags else 'general'
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_code_fix_pairs(self, insight: LearningInsight) -> List[TrainingPair]:
        """Generate training pairs for code fixes."""
        pairs = []
        
        # Extract code blocks from model response and user correction
        model_code = self._extract_code_blocks(insight.model_response)
        corrected_code = self._extract_code_blocks(insight.user_correction)
        
        if model_code and corrected_code:
            input_text = f"Please fix this code:\n{model_code[0]}"
            target_output = f"Here's the corrected code:\n{corrected_code[0]}"
            
            pair = TrainingPair(
                input_text=input_text,
                target_output=target_output,
                source_insight=insight,
                quality_score=insight.confidence_score * 1.2,  # Code fixes are high value
                domain='programming'
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_clarification_pairs(self, insight: LearningInsight) -> List[TrainingPair]:
        """Generate training pairs for clarifications."""
        pairs = []
        
        # User had to clarify - model should have asked for clarification instead
        input_text = insight.context_before
        target_output = "I need more clarification to help you better. Could you provide more details about what specifically you're looking for?"
        
        pair = TrainingPair(
            input_text=input_text,
            target_output=target_output,
            source_insight=insight,
            quality_score=insight.confidence_score * 0.8,  # Lower value but still useful
            domain=insight.domain_tags[0] if insight.domain_tags else 'general'
        )
        pairs.append(pair)
        
        return pairs
    
    def _generate_generic_pairs(self, insight: LearningInsight) -> List[TrainingPair]:
        """Generate generic training pairs."""
        # Compress the context to essentials
        compressed_context = self.compression_engine.compress(
            insight.context_before, target_ratio=0.3
        )
        
        input_text = compressed_context['compressed']
        target_output = self._extract_intended_response(insight.user_correction)
        
        if target_output:
            pair = TrainingPair(
                input_text=input_text,
                target_output=target_output,
                source_insight=insight,
                quality_score=insight.confidence_score,
                domain=insight.domain_tags[0] if insight.domain_tags else 'general'
            )
            return [pair]
        
        return []
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown-formatted text."""
        code_blocks = re.findall(r'```(?:\w+)?\n?(.*?)\n?```', text, re.DOTALL)
        return [block.strip() for block in code_blocks if block.strip()]
    
    def _extract_intended_response(self, correction_text: str) -> Optional[str]:
        """Extract what the user actually wanted from their correction."""
        # Simple heuristic - look for statements after correction indicators
        correction_indicators = [
            "actually", "should be", "correct answer is", "what i wanted",
            "instead", "rather", "the right way"
        ]
        
        for indicator in correction_indicators:
            if indicator in correction_text.lower():
                # Extract text after the indicator
                parts = correction_text.lower().split(indicator, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # Fallback - return the whole correction
        return correction_text


class ChatIntelligenceEngine:
    """Main engine that orchestrates the entire chat intelligence pipeline."""
    
    def __init__(self):
        self.parser = ChatExportParser()
        self.compression_engine = AdaptiveCompressionEngine()
        self.insight_extractor = ConversationInsightExtractor(self.compression_engine)
        self.training_generator = TrainingDataGenerator(self.compression_engine)
        
        # Statistics
        self.stats = {
            'conversations_processed': 0,
            'insights_extracted': 0,
            'training_pairs_generated': 0,
            'domains_covered': set()
        }
    
    def process_chat_export(self, file_path: str, 
                          source: Optional[ConversationSource] = None) -> Dict[str, Any]:
        """Process a chat export file and extract training data."""
        
        logger.info(f"Processing chat export: {file_path}")
        
        # Parse conversation
        conversation = self.parser.parse(file_path, source)
        if not conversation:
            return {'error': 'Failed to parse conversation'}
        
        # Extract insights
        insights = self.insight_extractor.extract_insights(conversation)
        
        # Generate training pairs
        training_pairs = self.training_generator.generate_training_pairs(insights)
        
        # Update statistics
        self.stats['conversations_processed'] += 1
        self.stats['insights_extracted'] += len(insights)
        self.stats['training_pairs_generated'] += len(training_pairs)
        
        for insight in insights:
            self.stats['domains_covered'].update(insight.domain_tags)
        
        # Compress conversation for storage
        full_conversation = " ".join([turn.content for turn in conversation])
        compressed_conversation = self.compression_engine.compress(
            full_conversation, target_ratio=0.2
        )
        
        return {
            'conversation_turns': len(conversation),
            'insights_found': len(insights),
            'training_pairs': len(training_pairs),
            'compressed_conversation': compressed_conversation['compressed'],
            'compression_ratio': compressed_conversation['actual_ratio'],
            'domains': list(set(insight.domain_tags[0] for insight in insights if insight.domain_tags)),
            'insights': [asdict(insight) for insight in insights],
            'training_data': [asdict(pair) for pair in training_pairs]
        }
    
    def batch_process(self, directory_path: str) -> Dict[str, Any]:
        """Process all chat exports in a directory."""
        
        directory = Path(directory_path)
        results = []
        
        for file_path in directory.glob("*.json"):
            try:
                result = self.process_chat_export(str(file_path))
                results.append({
                    'file': file_path.name,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file': file_path.name,
                    'error': str(e)
                })
        
        return {
            'files_processed': len(results),
            'total_insights': sum(r['result'].get('insights_found', 0) for r in results if 'result' in r),
            'total_training_pairs': sum(r['result'].get('training_pairs', 0) for r in results if 'result' in r),
            'domains_covered': list(self.stats['domains_covered']),
            'results': results,
            'stats': self.stats
        }
    
    def export_training_dataset(self, output_path: str, domain_filter: Optional[str] = None) -> str:
        """Export collected training data as a dataset."""
        # This would save the training pairs in a format suitable for fine-tuning
        # Implementation depends on the target training framework (HuggingFace, etc.)
        pass


# Example usage and testing
if __name__ == "__main__":
    print("🚀 CHAT EXPORT INTELLIGENCE ENGINE")
    print("=" * 40)
    
    # Initialize the engine
    engine = ChatIntelligenceEngine()
    
    # Create a sample conversation for testing
    sample_conversation = [
        ConversationTurn(speaker='user', content='Can you help me write a Python function to sort a list?'),
        ConversationTurn(speaker='assistant', content='''Here's a function to sort a list:

```python
def sort_list(lst):
    return lst.sort()
```

This function sorts the list in place and returns None.'''),
        ConversationTurn(speaker='user', content='''Actually, that's not quite right. The sort() method returns None, but I want a function that returns the sorted list. It should be:

```python
def sort_list(lst):
    return sorted(lst)
```

This way it returns a new sorted list without modifying the original.'''),
        ConversationTurn(speaker='assistant', content='You\'re absolutely right! Thank you for the correction. The sorted() function returns a new sorted list, while .sort() modifies the original list in place.')
    ]
    
    # Extract insights from sample
    insights = engine.insight_extractor.extract_insights(sample_conversation)
    
    print(f"📊 Extracted {len(insights)} insights from sample conversation:")
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight.insight_type.value}")
        print(f"   Confidence: {insight.confidence_score:.2f}")
        print(f"   Domain: {', '.join(insight.domain_tags)}")
        print(f"   Correction: {insight.user_correction[:100]}...")
    
    # Generate training pairs
    training_pairs = engine.training_generator.generate_training_pairs(insights)
    
    print(f"\n🎯 Generated {len(training_pairs)} training pairs:")
    for i, pair in enumerate(training_pairs, 1):
        print(f"\n{i}. Domain: {pair.domain}")
        print(f"   Quality: {pair.quality_score:.2f}")
        print(f"   Input: {pair.input_text[:100]}...")
        print(f"   Target: {pair.target_output[:100]}...")
    
    print(f"\n✨ This is where the magic happens!")
    print(f"Every LLM conversation becomes fuel for better models! 🔥")