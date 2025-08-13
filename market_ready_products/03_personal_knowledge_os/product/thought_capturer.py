#!/usr/bin/env python3
"""
Personal Knowledge OS - Thought Capturer

Automatically captures, organizes, and connects personal thoughts,
insights, and knowledge using SUM's progressive processing capabilities.

Key Features:
- Effortless thought capture and auto-tagging
- Automatic knowledge organization and connections
- Insight generation from scattered thoughts
- Policy-based processing (private vs. shared)
- Contextual surfacing of relevant insights

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
import hashlib

from progressive_summarization import ProgressiveStreamingEngine, ProgressEvent
from streaming_engine import StreamingConfig
from summarization_engine import HierarchicalDensificationEngine


logger = logging.getLogger(__name__)


@dataclass
class Thought:
    """Represents a captured thought or insight."""
    id: str
    content: str
    timestamp: datetime
    tags: List[str]
    category: str
    importance: float  # 0.0 to 1.0
    connections: List[str]  # IDs of related thoughts
    policy: str  # "private", "shared", "meetings", "diary"
    metadata: Dict[str, Any]


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related thoughts and insights."""
    id: str
    name: str
    description: str
    thoughts: List[str]  # Thought IDs
    key_insights: List[str]
    created_at: datetime
    last_updated: datetime
    strength: float  # Connection strength within cluster


@dataclass
class PersonalInsight:
    """Represents a crystallized insight from personal knowledge."""
    id: str
    title: str
    description: str
    source_thoughts: List[str]  # Thought IDs that contributed
    confidence: float  # 0.0 to 1.0
    category: str
    created_at: datetime
    last_referenced: datetime


@dataclass
class ThoughtCapture:
    """Output from thought capture and processing."""
    thought_id: str
    processed_content: str
    extracted_tags: List[str]
    identified_category: str
    importance_score: float
    suggested_connections: List[str]
    insights_generated: List[str]
    processing_stats: Dict[str, Any]


class ThoughtCapturer:
    """
    Core engine for capturing and processing personal thoughts and insights.
    
    Automatically organizes thoughts, identifies connections, generates insights,
    and applies policy-based processing for different types of content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the thought capturer."""
        self.config = config or {}
        
        # Initialize SUM engines
        streaming_config = StreamingConfig(
            chunk_size_words=1000,  # Optimal for thought processing
            overlap_ratio=0.3,      # High overlap for context
            max_memory_mb=256,      # Lightweight processing
            max_concurrent_chunks=2  # Sequential for personal thoughts
        )
        
        self.progressive_engine = ProgressiveStreamingEngine(streaming_config)
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Personal knowledge components
        self.thoughts = {}
        self.knowledge_clusters = {}
        self.insights = {}
        self.connection_graph = {}
        
        # Policy configurations
        self.policies = {
            "private": {"auto_connect": False, "insight_generation": False, "sharing": False},
            "shared": {"auto_connect": True, "insight_generation": True, "sharing": True},
            "meetings": {"auto_connect": True, "insight_generation": True, "sharing": True},
            "diary": {"auto_connect": False, "insight_generation": True, "sharing": False}
        }
        
    async def capture_thought(self, content: str, policy: str = "shared", 
                            session_id: str = None) -> ThoughtCapture:
        """
        Capture and process a new thought or insight.
        
        Args:
            content: The thought content to capture
            policy: Processing policy ("private", "shared", "meetings", "diary")
            session_id: Optional session identifier for progress tracking
            
        Returns:
            Processed thought with tags, connections, and insights
        """
        logger.info(f"Capturing thought with policy: {policy}")
        
        session_id = session_id or f"thought_{int(time.time())}"
        
        try:
            # Phase 1: Process thought content with progressive analysis
            thought_analysis = await self._process_thought_content(content, session_id)
            
            # Phase 2: Extract tags and categorize
            tags_and_category = await self._extract_tags_and_category(thought_analysis)
            
            # Phase 3: Calculate importance and connections
            importance_and_connections = await self._calculate_importance_and_connections(
                thought_analysis, tags_and_category, policy
            )
            
            # Phase 4: Generate insights based on policy
            insights = await self._generate_insights(thought_analysis, policy)
            
            # Phase 5: Create thought capture result
            thought_capture = await self._create_thought_capture(
                content, thought_analysis, tags_and_category, 
                importance_and_connections, insights, policy
            )
            
            # Phase 6: Store thought and update knowledge base
            await self._store_thought(thought_capture, policy)
            
            return thought_capture
            
        except Exception as e:
            logger.error(f"Error capturing thought: {e}")
            raise
    
    async def _process_thought_content(self, content: str, session_id: str) -> Dict[str, Any]:
        """Process thought content with progressive analysis."""
        logger.info("Processing thought content with progressive analysis")
        
        # Process with progressive engine
        result = await self.progressive_engine.process_streaming_text_with_progress(
            content, f"{session_id}_thought"
        )
        
        # Extract thought-specific insights
        thought_insights = self._extract_thought_insights(content, result)
        
        return {
            "summarization_result": result,
            "thought_insights": thought_insights,
            "processing_time": time.time()
        }
    
    def _extract_thought_insights(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thought-specific insights from content analysis."""
        insights = {
            "thought_type": "",
            "emotional_content": False,
            "actionable_content": False,
            "reflective_content": False,
            "creative_content": False,
            "complexity_score": 0.0,
            "clarity_score": 0.0
        }
        
        content_lower = content.lower()
        
        # Determine thought type
        if any(word in content_lower for word in ['feel', 'emotion', 'happy', 'sad', 'angry']):
            insights["thought_type"] = "emotional"
            insights["emotional_content"] = True
        elif any(word in content_lower for word in ['should', 'need to', 'will', 'going to']):
            insights["thought_type"] = "actionable"
            insights["actionable_content"] = True
        elif any(word in content_lower for word in ['think', 'believe', 'realize', 'understand']):
            insights["thought_type"] = "reflective"
            insights["reflective_content"] = True
        elif any(word in content_lower for word in ['idea', 'creative', 'imagine', 'could']):
            insights["thought_type"] = "creative"
            insights["creative_content"] = True
        else:
            insights["thought_type"] = "general"
        
        # Calculate complexity and clarity scores
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        insights["complexity_score"] = min(1.0, avg_sentence_length / 20.0)  # Normalize
        insights["clarity_score"] = max(0.0, 1.0 - insights["complexity_score"])
        
        return insights
    
    async def _extract_tags_and_category(self, thought_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tags and categorize the thought."""
        logger.info("Extracting tags and categorizing thought")
        
        # Get concepts from hierarchical summary
        concepts = []
        if 'hierarchical_summary' in thought_analysis.get('summarization_result', {}):
            summary = thought_analysis['summarization_result']['hierarchical_summary']
            if 'level_1_concepts' in summary:
                concepts = summary['level_1_concepts']
        
        # Extract additional tags from key insights
        additional_tags = []
        if 'key_insights' in thought_analysis.get('summarization_result', {}):
            for insight in thought_analysis['summarization_result']['key_insights']:
                insight_text = insight.get('text', '').lower()
                # Extract key words as tags
                words = insight_text.split()
                additional_tags.extend([word for word in words if len(word) > 3])
        
        # Combine and deduplicate tags
        all_tags = list(set(concepts + additional_tags))
        
        # Determine category based on thought type and content
        thought_insights = thought_analysis.get('thought_insights', {})
        thought_type = thought_insights.get('thought_type', 'general')
        
        category_mapping = {
            "emotional": "personal",
            "actionable": "tasks",
            "reflective": "insights",
            "creative": "ideas",
            "general": "notes"
        }
        
        category = category_mapping.get(thought_type, "notes")
        
        return {
            "tags": all_tags[:10],  # Limit to top 10 tags
            "category": category,
            "thought_type": thought_type
        }
    
    async def _calculate_importance_and_connections(self, thought_analysis: Dict[str, Any],
                                                 tags_and_category: Dict[str, Any],
                                                 policy: str) -> Dict[str, Any]:
        """Calculate importance score and identify connections."""
        logger.info("Calculating importance and identifying connections")
        
        # Calculate importance based on multiple factors
        thought_insights = thought_analysis.get('thought_insights', {})
        
        # Base importance from thought type
        type_importance = {
            "emotional": 0.7,
            "actionable": 0.9,
            "reflective": 0.8,
            "creative": 0.6,
            "general": 0.5
        }
        
        base_importance = type_importance.get(thought_insights.get('thought_type', 'general'), 0.5)
        
        # Adjust based on complexity and clarity
        complexity_score = thought_insights.get('complexity_score', 0.0)
        clarity_score = thought_insights.get('clarity_score', 0.0)
        
        # More complex thoughts might be more important
        complexity_boost = complexity_score * 0.2
        
        # Clear thoughts are also important
        clarity_boost = clarity_score * 0.3
        
        # Calculate final importance
        importance = min(1.0, base_importance + complexity_boost + clarity_boost)
        
        # Find connections based on tags and content
        connections = []
        if policy in ["shared", "meetings"]:
            connections = self._find_related_thoughts(tags_and_category.get('tags', []))
        
        return {
            "importance": importance,
            "connections": connections,
            "connection_strength": len(connections) / 10.0  # Normalize
        }
    
    def _find_related_thoughts(self, tags: List[str]) -> List[str]:
        """Find thoughts related to the given tags."""
        connections = []
        
        for thought_id, thought in self.thoughts.items():
            # Check for tag overlap
            thought_tags = thought.get('tags', [])
            overlap = set(tags).intersection(set(thought_tags))
            
            if overlap:
                connections.append(thought_id)
        
        return connections[:5]  # Limit to top 5 connections
    
    async def _generate_insights(self, thought_analysis: Dict[str, Any], policy: str) -> List[str]:
        """Generate insights from the thought based on policy."""
        logger.info("Generating insights from thought")
        
        insights = []
        
        # Only generate insights for certain policies
        if policy in ["shared", "meetings", "diary"]:
            # Extract insights from key insights
            if 'key_insights' in thought_analysis.get('summarization_result', {}):
                for insight in thought_analysis['summarization_result']['key_insights']:
                    insights.append(insight.get('text', ''))
            
            # Generate additional insights based on thought type
            thought_insights = thought_analysis.get('thought_insights', {})
            thought_type = thought_insights.get('thought_type', 'general')
            
            if thought_type == "reflective":
                insights.append("This reflection shows deeper thinking about the topic")
            elif thought_type == "actionable":
                insights.append("This thought contains clear action items")
            elif thought_type == "creative":
                insights.append("This creative idea could be developed further")
        
        return insights
    
    async def _create_thought_capture(self, content: str, thought_analysis: Dict[str, Any],
                                    tags_and_category: Dict[str, Any],
                                    importance_and_connections: Dict[str, Any],
                                    insights: List[str], policy: str) -> ThoughtCapture:
        """Create the final thought capture result."""
        logger.info("Creating thought capture result")
        
        # Generate thought ID
        thought_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:8]
        
        # Calculate processing stats
        processing_stats = {
            "content_length": len(content),
            "processing_time": thought_analysis.get('processing_time', 0),
            "tags_extracted": len(tags_and_category.get('tags', [])),
            "connections_found": len(importance_and_connections.get('connections', [])),
            "insights_generated": len(insights)
        }
        
        return ThoughtCapture(
            thought_id=thought_id,
            processed_content=content,
            extracted_tags=tags_and_category.get('tags', []),
            identified_category=tags_and_category.get('category', 'notes'),
            importance_score=importance_and_connections.get('importance', 0.5),
            suggested_connections=importance_and_connections.get('connections', []),
            insights_generated=insights,
            processing_stats=processing_stats
        )
    
    async def _store_thought(self, thought_capture: ThoughtCapture, policy: str):
        """Store the thought and update knowledge base."""
        logger.info(f"Storing thought {thought_capture.thought_id}")
        
        # Create thought object
        thought = Thought(
            id=thought_capture.thought_id,
            content=thought_capture.processed_content,
            timestamp=datetime.now(),
            tags=thought_capture.extracted_tags,
            category=thought_capture.identified_category,
            importance=thought_capture.importance_score,
            connections=thought_capture.suggested_connections,
            policy=policy,
            metadata={
                "processing_stats": thought_capture.processing_stats,
                "insights": thought_capture.insights_generated
            }
        )
        
        # Store thought
        self.thoughts[thought.id] = asdict(thought)
        
        # Update connection graph
        for connection_id in thought.connections:
            if connection_id in self.thoughts:
                # Add bidirectional connection
                if 'connections' not in self.thoughts[connection_id]:
                    self.thoughts[connection_id]['connections'] = []
                self.thoughts[connection_id]['connections'].append(thought.id)
        
        # Update knowledge clusters if policy allows
        if policy in ["shared", "meetings"]:
            await self._update_knowledge_clusters(thought)
    
    async def _update_knowledge_clusters(self, thought: Thought):
        """Update knowledge clusters with new thought."""
        logger.info("Updating knowledge clusters")
        
        # Find existing clusters that match the thought's tags
        matching_clusters = []
        
        for cluster_id, cluster in self.knowledge_clusters.items():
            cluster_tags = cluster.get('tags', [])
            overlap = set(thought.tags).intersection(set(cluster_tags))
            
            if len(overlap) >= 2:  # At least 2 tags overlap
                matching_clusters.append(cluster_id)
        
        if matching_clusters:
            # Add thought to the best matching cluster
            best_cluster_id = matching_clusters[0]
            self.knowledge_clusters[best_cluster_id]['thoughts'].append(thought.id)
            self.knowledge_clusters[best_cluster_id]['last_updated'] = datetime.now()
        else:
            # Create new cluster
            cluster_id = f"cluster_{len(self.knowledge_clusters) + 1}"
            new_cluster = KnowledgeCluster(
                id=cluster_id,
                name=f"Cluster {cluster_id}",
                description=f"Cluster based on tags: {', '.join(thought.tags[:3])}",
                thoughts=[thought.id],
                key_insights=thought.metadata.get('insights', []),
                created_at=datetime.now(),
                last_updated=datetime.now(),
                strength=thought.importance
            )
            
            self.knowledge_clusters[cluster_id] = asdict(new_cluster)
    
    async def get_crystallized_knowledge(self, category: str = None) -> List[PersonalInsight]:
        """Get crystallized insights from personal knowledge."""
        logger.info("Generating crystallized knowledge")
        
        insights = []
        
        # Filter thoughts by category if specified
        relevant_thoughts = []
        for thought_id, thought in self.thoughts.items():
            if category is None or thought.get('category') == category:
                relevant_thoughts.append(thought)
        
        # Group thoughts by tags
        tag_groups = {}
        for thought in relevant_thoughts:
            for tag in thought.get('tags', []):
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(thought)
        
        # Generate insights for each tag group
        for tag, thoughts in tag_groups.items():
            if len(thoughts) >= 2:  # Need at least 2 thoughts for insight
                insight = await self._generate_insight_from_thoughts(tag, thoughts)
                if insight:
                    insights.append(insight)
        
        return insights
    
    async def _generate_insight_from_thoughts(self, tag: str, thoughts: List[Dict[str, Any]]) -> Optional[PersonalInsight]:
        """Generate an insight from a group of related thoughts."""
        # Combine thought content
        combined_content = " ".join([thought.get('content', '') for thought in thoughts])
        
        if not combined_content:
            return None
        
        # Use SUM to generate insight
        result = self.hierarchical_engine.process_text(combined_content)
        
        if 'hierarchical_summary' in result:
            summary = result['hierarchical_summary'].get('level_2_core', '')
            
            # Calculate confidence based on number of thoughts and importance
            avg_importance = sum(thought.get('importance', 0.5) for thought in thoughts) / len(thoughts)
            confidence = min(1.0, (len(thoughts) / 5.0) * avg_importance)
            
            insight = PersonalInsight(
                id=f"insight_{len(self.insights) + 1}",
                title=f"Insight about {tag}",
                description=summary,
                source_thoughts=[thought.get('id') for thought in thoughts],
                confidence=confidence,
                category=thoughts[0].get('category', 'general'),
                created_at=datetime.now(),
                last_referenced=datetime.now()
            )
            
            self.insights[insight.id] = asdict(insight)
            return insight
        
        return None
    
    async def search_thoughts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search thoughts by content and tags."""
        logger.info(f"Searching thoughts for: {query}")
        
        query_lower = query.lower()
        results = []
        
        for thought_id, thought in self.thoughts.items():
            content = thought.get('content', '').lower()
            tags = [tag.lower() for tag in thought.get('tags', [])]
            
            # Check if query matches content or tags
            if (query_lower in content or 
                any(query_lower in tag for tag in tags)):
                
                results.append({
                    "id": thought_id,
                    "content": thought.get('content', ''),
                    "tags": thought.get('tags', []),
                    "category": thought.get('category', ''),
                    "importance": thought.get('importance', 0.5),
                    "timestamp": thought.get('timestamp', ''),
                    "relevance_score": self._calculate_relevance_score(query_lower, content, tags)
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]
    
    def _calculate_relevance_score(self, query: str, content: str, tags: List[str]) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        
        # Content match
        if query in content:
            score += 0.6
        
        # Tag matches
        tag_matches = sum(1 for tag in tags if query in tag)
        score += tag_matches * 0.3
        
        # Exact matches get bonus
        if query in content or any(query == tag for tag in tags):
            score += 0.2
        
        return min(1.0, score)


# Example usage
async def main():
    """Example usage of the Thought Capturer."""
    capturer = ThoughtCapturer()
    
    # Capture some thoughts
    thoughts = [
        ("I think we should focus more on user experience in our product design", "shared"),
        ("Feeling excited about the new project direction", "diary"),
        ("Need to research more about AI ethics before the meeting tomorrow", "meetings"),
        ("The team collaboration has been really effective lately", "shared")
    ]
    
    for content, policy in thoughts:
        capture = await capturer.capture_thought(content, policy)
        print(f"Captured thought: {capture.thought_id}")
        print(f"Tags: {capture.extracted_tags}")
        print(f"Category: {capture.identified_category}")
        print(f"Importance: {capture.importance_score}")
        print("---")
    
    # Get crystallized knowledge
    insights = await capturer.get_crystallized_knowledge()
    print(f"Generated {len(insights)} insights from personal knowledge")
    
    # Search thoughts
    results = await capturer.search_thoughts("user experience")
    print(f"Found {len(results)} thoughts about user experience")


if __name__ == "__main__":
    asyncio.run(main())
