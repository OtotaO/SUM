#!/usr/bin/env python3
"""
Business Intelligence Engine - Meeting Processor

Transforms meeting recordings into actionable decisions, action items,
and strategic insights using SUM's progressive processing capabilities.

Key Features:
- Real-time meeting transcription and analysis
- Decision extraction and tracking
- Action item identification and assignment
- Pattern detection across multiple meetings
- Executive summary generation

Author: ototao
License: Apache License 2.0
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

from progressive_summarization import ProgressiveStreamingEngine, ProgressEvent
from streaming_engine import StreamingConfig
from summarization_engine import HierarchicalDensificationEngine


logger = logging.getLogger(__name__)


@dataclass
class MeetingParticipant:
    """Represents a meeting participant."""
    name: str
    role: str
    department: str
    speaking_time: float = 0.0
    contributions: List[str] = None


@dataclass
class Decision:
    """Represents a decision made in a meeting."""
    decision_text: str
    decision_type: str  # "approved", "rejected", "deferred", "conditional"
    participants_involved: List[str]
    impact_level: str  # "high", "medium", "low"
    follow_up_required: bool
    deadline: Optional[datetime] = None


@dataclass
class ActionItem:
    """Represents an action item from a meeting."""
    description: str
    assignee: str
    deadline: datetime
    priority: str  # "high", "medium", "low"
    status: str = "pending"
    dependencies: List[str] = None


@dataclass
class MeetingAnalysis:
    """Comprehensive meeting analysis output."""
    summary: str
    decisions: List[Decision]
    action_items: List[ActionItem]
    unresolved_questions: List[str]
    key_insights: List[str]
    participant_contributions: Dict[str, Any]
    patterns_detected: List[Dict[str, Any]]
    processing_stats: Dict[str, Any]


class MeetingProcessor:
    """
    Core engine for processing meeting recordings and extracting business intelligence.
    
    Transforms audio/video recordings into structured insights, decisions,
    action items, and strategic recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the meeting processor."""
        self.config = config or {}
        
        # Initialize SUM engines
        streaming_config = StreamingConfig(
            chunk_size_words=1500,  # Optimal for meeting transcripts
            overlap_ratio=0.25,     # More overlap for context
            max_memory_mb=512,      # Moderate memory usage
            max_concurrent_chunks=4  # Parallel processing
        )
        
        self.progressive_engine = ProgressiveStreamingEngine(streaming_config)
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Meeting-specific components
        self.meetings = []
        self.participants = {}
        self.decision_history = []
        self.action_item_tracker = {}
        
    async def process_meeting(self, meeting_data: Dict[str, Any], 
                            session_id: str = None) -> MeetingAnalysis:
        """
        Process a meeting recording and extract business intelligence.
        
        Args:
            meeting_data: Dictionary containing meeting information
                - transcript: Full meeting transcript
                - participants: List of participant information
                - metadata: Meeting metadata (date, duration, etc.)
            session_id: Optional session identifier for progress tracking
            
        Returns:
            Comprehensive meeting analysis with decisions and action items
        """
        logger.info(f"Starting meeting processing for: {meeting_data.get('title', 'Unknown Meeting')}")
        
        session_id = session_id or f"meeting_{int(time.time())}"
        
        try:
            # Phase 1: Process meeting transcript with progressive updates
            transcript_analysis = await self._process_transcript(
                meeting_data['transcript'], session_id
            )
            
            # Phase 2: Extract decisions and action items
            decisions = await self._extract_decisions(transcript_analysis, meeting_data)
            action_items = await self._extract_action_items(transcript_analysis, meeting_data)
            
            # Phase 3: Analyze participant contributions
            participant_analysis = await self._analyze_participants(
                transcript_analysis, meeting_data.get('participants', [])
            )
            
            # Phase 4: Detect patterns and insights
            pattern_analysis = await self._detect_patterns(transcript_analysis, meeting_data)
            
            # Phase 5: Generate comprehensive analysis
            meeting_analysis = await self._generate_meeting_analysis(
                transcript_analysis, decisions, action_items, 
                participant_analysis, pattern_analysis, meeting_data
            )
            
            return meeting_analysis
            
        except Exception as e:
            logger.error(f"Error processing meeting: {e}")
            raise
    
    async def _process_transcript(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """Process meeting transcript with progressive analysis."""
        logger.info("Processing meeting transcript with progressive analysis")
        
        # Process with progressive engine
        result = await self.progressive_engine.process_streaming_text_with_progress(
            transcript, f"{session_id}_transcript"
        )
        
        # Extract meeting-specific insights
        meeting_insights = self._extract_meeting_insights(transcript, result)
        
        return {
            "summarization_result": result,
            "meeting_insights": meeting_insights,
            "processing_time": time.time()
        }
    
    def _extract_meeting_insights(self, transcript: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meeting-specific insights from transcript analysis."""
        insights = {
            "meeting_type": "",
            "key_topics": [],
            "decisions_made": [],
            "action_items": [],
            "unresolved_questions": [],
            "participant_sentiment": {},
            "meeting_effectiveness": 0.0
        }
        
        # Determine meeting type
        transcript_lower = transcript.lower()
        if any(word in transcript_lower for word in ['decision', 'approve', 'reject']):
            insights["meeting_type"] = "decision_making"
        elif any(word in transcript_lower for word in ['plan', 'strategy', 'roadmap']):
            insights["meeting_type"] = "planning"
        elif any(word in transcript_lower for word in ['review', 'update', 'status']):
            insights["meeting_type"] = "status_update"
        else:
            insights["meeting_type"] = "general"
        
        # Extract key topics from hierarchical summary
        if 'hierarchical_summary' in result:
            summary = result['hierarchical_summary']
            if 'level_1_concepts' in summary:
                insights["key_topics"] = summary['level_1_concepts'][:5]
        
        # Extract decisions and action items from key insights
        if 'key_insights' in result:
            for insight in result['key_insights']:
                insight_text = insight.get('text', '').lower()
                
                # Identify decisions
                if any(word in insight_text for word in ['decided', 'approved', 'rejected', 'agreed']):
                    insights["decisions_made"].append(insight['text'])
                
                # Identify action items
                if any(word in insight_text for word in ['will', 'going to', 'need to', 'should']):
                    insights["action_items"].append(insight['text'])
                
                # Identify unresolved questions
                if '?' in insight['text'] and any(word in insight_text for word in ['unclear', 'unsure', 'need clarification']):
                    insights["unresolved_questions"].append(insight['text'])
        
        # Calculate meeting effectiveness (simplified)
        total_insights = len(result.get('key_insights', []))
        decisions_count = len(insights["decisions_made"])
        action_items_count = len(insights["action_items"])
        
        insights["meeting_effectiveness"] = min(1.0, (decisions_count + action_items_count) / max(total_insights, 1))
        
        return insights
    
    async def _extract_decisions(self, transcript_analysis: Dict[str, Any], 
                                meeting_data: Dict[str, Any]) -> List[Decision]:
        """Extract decisions made during the meeting."""
        logger.info("Extracting decisions from meeting transcript")
        
        decisions = []
        transcript = meeting_data['transcript']
        
        # Use SUM to identify decision-related content
        decision_keywords = ['decided', 'approved', 'rejected', 'agreed', 'voted', 'resolved']
        
        # Find sentences containing decision keywords
        sentences = transcript.split('.')
        decision_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in decision_keywords):
                decision_sentences.append(sentence.strip())
        
        # Process decision sentences with SUM
        if decision_sentences:
            decision_text = '. '.join(decision_sentences)
            result = self.hierarchical_engine.process_text(decision_text)
            
            # Extract decisions from insights
            if 'key_insights' in result:
                for insight in result['key_insights']:
                    decision = self._create_decision_from_insight(insight, meeting_data)
                    if decision:
                        decisions.append(decision)
        
        return decisions
    
    def _create_decision_from_insight(self, insight: Dict[str, Any], 
                                    meeting_data: Dict[str, Any]) -> Optional[Decision]:
        """Create a Decision object from an insight."""
        decision_text = insight.get('text', '')
        
        if not decision_text:
            return None
        
        # Determine decision type
        decision_text_lower = decision_text.lower()
        if any(word in decision_text_lower for word in ['approved', 'agreed', 'yes']):
            decision_type = "approved"
        elif any(word in decision_text_lower for word in ['rejected', 'denied', 'no']):
            decision_type = "rejected"
        elif any(word in decision_text_lower for word in ['deferred', 'postponed', 'later']):
            decision_type = "deferred"
        else:
            decision_type = "conditional"
        
        # Determine impact level
        if any(word in decision_text_lower for word in ['major', 'significant', 'important', 'strategic']):
            impact_level = "high"
        elif any(word in decision_text_lower for word in ['minor', 'small', 'routine']):
            impact_level = "low"
        else:
            impact_level = "medium"
        
        # Extract participants (simplified)
        participants = meeting_data.get('participants', [])
        participants_involved = [p.get('name', '') for p in participants if p.get('name')]
        
        return Decision(
            decision_text=decision_text,
            decision_type=decision_type,
            participants_involved=participants_involved,
            impact_level=impact_level,
            follow_up_required=impact_level == "high"
        )
    
    async def _extract_action_items(self, transcript_analysis: Dict[str, Any], 
                                   meeting_data: Dict[str, Any]) -> List[ActionItem]:
        """Extract action items from the meeting."""
        logger.info("Extracting action items from meeting transcript")
        
        action_items = []
        transcript = meeting_data['transcript']
        
        # Use SUM to identify action item content
        action_keywords = ['will', 'going to', 'need to', 'should', 'must', 'task', 'action']
        
        # Find sentences containing action keywords
        sentences = transcript.split('.')
        action_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in action_keywords):
                action_sentences.append(sentence.strip())
        
        # Process action sentences with SUM
        if action_sentences:
            action_text = '. '.join(action_sentences)
            result = self.hierarchical_engine.process_text(action_text)
            
            # Extract action items from insights
            if 'key_insights' in result:
                for insight in result['key_insights']:
                    action_item = self._create_action_item_from_insight(insight, meeting_data)
                    if action_item:
                        action_items.append(action_item)
        
        return action_items
    
    def _create_action_item_from_insight(self, insight: Dict[str, Any], 
                                       meeting_data: Dict[str, Any]) -> Optional[ActionItem]:
        """Create an ActionItem object from an insight."""
        action_text = insight.get('text', '')
        
        if not action_text:
            return None
        
        # Determine priority
        action_text_lower = action_text.lower()
        if any(word in action_text_lower for word in ['urgent', 'immediate', 'critical', 'asap']):
            priority = "high"
        elif any(word in action_text_lower for word in ['soon', 'quickly', 'prompt']):
            priority = "medium"
        else:
            priority = "low"
        
        # Extract assignee (simplified - would need more sophisticated NLP)
        participants = meeting_data.get('participants', [])
        assignee = participants[0].get('name', 'TBD') if participants else 'TBD'
        
        # Set deadline based on priority
        meeting_date = meeting_data.get('date', datetime.now())
        if priority == "high":
            deadline = meeting_date + timedelta(days=3)
        elif priority == "medium":
            deadline = meeting_date + timedelta(days=7)
        else:
            deadline = meeting_date + timedelta(days=14)
        
        return ActionItem(
            description=action_text,
            assignee=assignee,
            deadline=deadline,
            priority=priority
        )
    
    async def _analyze_participants(self, transcript_analysis: Dict[str, Any], 
                                   participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze participant contributions and engagement."""
        logger.info("Analyzing participant contributions")
        
        participant_analysis = {
            "contributions": {},
            "engagement_levels": {},
            "key_contributors": [],
            "participation_patterns": {}
        }
        
        # Analyze each participant's contributions
        for participant in participants:
            name = participant.get('name', 'Unknown')
            role = participant.get('role', 'Unknown')
            
            # Count mentions and contributions (simplified)
            transcript_lower = transcript_analysis.get('summarization_result', {}).get('text', '').lower()
            name_lower = name.lower()
            
            # Count mentions
            mention_count = transcript_lower.count(name_lower)
            
            # Analyze sentiment (simplified)
            positive_words = ['good', 'great', 'excellent', 'agree', 'support']
            negative_words = ['bad', 'poor', 'disagree', 'concern', 'issue']
            
            positive_count = sum(transcript_lower.count(word) for word in positive_words)
            negative_count = sum(transcript_lower.count(word) for word in negative_words)
            
            sentiment_score = (positive_count - negative_count) / max(mention_count, 1)
            
            participant_analysis["contributions"][name] = {
                "role": role,
                "mention_count": mention_count,
                "sentiment_score": sentiment_score,
                "engagement_level": "high" if mention_count > 5 else "medium" if mention_count > 2 else "low"
            }
        
        # Identify key contributors
        sorted_contributors = sorted(
            participant_analysis["contributions"].items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )
        
        participant_analysis["key_contributors"] = [
            {"name": name, "contributions": data} 
            for name, data in sorted_contributors[:3]
        ]
        
        return participant_analysis
    
    async def _detect_patterns(self, transcript_analysis: Dict[str, Any], 
                              meeting_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns across meetings and within this meeting."""
        logger.info("Detecting patterns in meeting content")
        
        patterns = []
        
        # Pattern 1: Recurring topics
        if 'meeting_insights' in transcript_analysis:
            key_topics = transcript_analysis['meeting_insights'].get('key_topics', [])
            patterns.append({
                "pattern_type": "recurring_topics",
                "description": f"Key topics discussed: {', '.join(key_topics)}",
                "frequency": "high" if len(key_topics) > 3 else "medium"
            })
        
        # Pattern 2: Decision patterns
        decisions = transcript_analysis['meeting_insights'].get('decisions_made', [])
        if len(decisions) > 2:
            patterns.append({
                "pattern_type": "decision_heavy_meeting",
                "description": f"Meeting resulted in {len(decisions)} decisions",
                "frequency": "high"
            })
        
        # Pattern 3: Action item patterns
        action_items = transcript_analysis['meeting_insights'].get('action_items', [])
        if len(action_items) > 3:
            patterns.append({
                "pattern_type": "action_oriented_meeting",
                "description": f"Meeting generated {len(action_items)} action items",
                "frequency": "high"
            })
        
        # Pattern 4: Unresolved questions
        unresolved = transcript_analysis['meeting_insights'].get('unresolved_questions', [])
        if unresolved:
            patterns.append({
                "pattern_type": "unresolved_questions",
                "description": f"Meeting left {len(unresolved)} questions unresolved",
                "frequency": "medium"
            })
        
        return patterns
    
    async def _generate_meeting_analysis(self, transcript_analysis: Dict[str, Any],
                                       decisions: List[Decision], action_items: List[ActionItem],
                                       participant_analysis: Dict[str, Any],
                                       pattern_analysis: List[Dict[str, Any]],
                                       meeting_data: Dict[str, Any]) -> MeetingAnalysis:
        """Generate comprehensive meeting analysis."""
        logger.info("Generating comprehensive meeting analysis")
        
        # Get summary from transcript analysis
        summary = transcript_analysis.get('summarization_result', {}).get('hierarchical_summary', {}).get('level_2_core', '')
        
        # Extract unresolved questions
        unresolved_questions = transcript_analysis.get('meeting_insights', {}).get('unresolved_questions', [])
        
        # Extract key insights
        key_insights = []
        if 'key_insights' in transcript_analysis.get('summarization_result', {}):
            key_insights = [insight.get('text', '') for insight in transcript_analysis['summarization_result']['key_insights']]
        
        # Calculate processing stats
        processing_stats = {
            "transcript_length": len(meeting_data.get('transcript', '')),
            "decisions_extracted": len(decisions),
            "action_items_identified": len(action_items),
            "participants_analyzed": len(participant_analysis.get('contributions', {})),
            "patterns_detected": len(pattern_analysis),
            "processing_time": transcript_analysis.get('processing_time', 0)
        }
        
        return MeetingAnalysis(
            summary=summary,
            decisions=decisions,
            action_items=action_items,
            unresolved_questions=unresolved_questions,
            key_insights=key_insights,
            participant_contributions=participant_analysis,
            patterns_detected=pattern_analysis,
            processing_stats=processing_stats
        )


# Example usage
async def main():
    """Example usage of the Meeting Processor."""
    processor = MeetingProcessor()
    
    # Example meeting data
    meeting_data = {
        "title": "Q4 Strategy Planning Meeting",
        "date": datetime.now(),
        "duration": 90,
        "participants": [
            {"name": "John Smith", "role": "CEO", "department": "Executive"},
            {"name": "Sarah Johnson", "role": "CTO", "department": "Technology"},
            {"name": "Mike Davis", "role": "CFO", "department": "Finance"}
        ],
        "transcript": """
        John: Welcome everyone to our Q4 strategy planning meeting. Let's start with the budget review.
        Sarah: I think we should increase our AI investment by 20% this quarter.
        Mike: I agree, but we need to ensure ROI metrics are clear.
        John: Good point. Let's approve the AI budget increase with quarterly ROI reviews.
        Sarah: I'll create a detailed implementation plan by next week.
        Mike: I'll set up the tracking dashboard for ROI metrics.
        """
    }
    
    # Process meeting
    analysis = await processor.process_meeting(meeting_data)
    
    print("Meeting Analysis Complete!")
    print(f"Summary: {analysis.summary}")
    print(f"Decisions: {len(analysis.decisions)}")
    print(f"Action Items: {len(analysis.action_items)}")


if __name__ == "__main__":
    asyncio.run(main())
