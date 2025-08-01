#!/usr/bin/env python3
"""
onepunch_bridge.py - SUM + OnePunchUpload Integration Bridge

A minimal viable integration that demonstrates the power of combining
intelligent content processing (SUM) with multi-platform publishing (OnePunchUpload).

This bridge transforms any content through SUM's processing pipeline and
formats it optimally for different platforms via OnePunchUpload.

Author: ototao
License: Apache License 2.0
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# SUM components
from summail_engine import SumMailEngine, EmailCategory
from summarization_engine import HierarchicalDensificationEngine
from ollama_manager import OllamaManager, ProcessingRequest

logger = logging.getLogger(__name__)


@dataclass
class PlatformContent:
    """Content optimized for a specific platform."""
    platform: str
    title: str
    content: str
    tags: List[str]
    metadata: Dict[str, Any]
    character_count: int
    estimated_engagement: float


@dataclass
class ContentPipeline:
    """Content processing and distribution pipeline."""
    source_content: str
    source_type: str  # 'email', 'document', 'text'
    target_platforms: List[str]
    processing_config: Dict[str, Any]
    results: List[PlatformContent]
    processing_time: float
    total_reach_multiplier: float


class PlatformOptimizer:
    """Optimizes content for different social media and publishing platforms."""
    
    PLATFORM_SPECS = {
        'twitter': {
            'max_chars': 280,
            'max_thread_length': 25,
            'optimal_hashtags': 2,
            'style': 'concise',
            'engagement_hooks': ['🧵', '👇', '💡', '🔥', '⚡']
        },
        'linkedin': {
            'max_chars': 3000,
            'optimal_length': 1300,
            'style': 'professional',
            'engagement_hooks': ['📊', '💼', '🚀', '💡', '📈'],
            'hashtag_limit': 5
        },
        'medium': {
            'max_chars': 100000,
            'optimal_length': 7500,
            'style': 'narrative',
            'reading_time_target': '5-7 min',
            'sections': True
        },
        'substack': {
            'max_chars': 50000,
            'style': 'newsletter',
            'include_cta': True,
            'personal_tone': True
        },
        'youtube_description': {
            'max_chars': 5000,
            'optimal_length': 200,
            'include_timestamps': True,
            'cta_required': True
        },
        'instagram': {
            'max_chars': 2200,
            'optimal_hashtags': 11,
            'style': 'visual-first',
            'engagement_hooks': ['📸', '✨', '💫', '🌟']
        },
        'tiktok': {
            'max_chars': 150,
            'style': 'trendy',
            'engagement_hooks': ['🎵', '💃', '🔥', '✨']
        }
    }
    
    def optimize_for_platform(self, content: str, key_points: List[str], 
                            platform: str, metadata: Dict[str, Any]) -> PlatformContent:
        """Optimize content for a specific platform."""
        specs = self.PLATFORM_SPECS.get(platform, {})
        
        if platform == 'twitter':
            return self._create_twitter_content(content, key_points, specs, metadata)
        elif platform == 'linkedin':
            return self._create_linkedin_content(content, key_points, specs, metadata)
        elif platform == 'medium':
            return self._create_medium_content(content, key_points, specs, metadata)
        else:
            return self._create_generic_content(content, key_points, platform, specs, metadata)
    
    def _create_twitter_content(self, content: str, key_points: List[str], 
                               specs: Dict, metadata: Dict) -> PlatformContent:
        """Create Twitter thread from content."""
        thread_parts = []
        hook = specs['engagement_hooks'][0]  # Use first hook
        
        # Create opening tweet
        if key_points:
            opening = f"{hook} {key_points[0][:250]}..."
            thread_parts.append(opening)
            
            # Add key points as thread
            for i, point in enumerate(key_points[1:4], 2):  # Max 4 total tweets
                tweet = f"{i}/{min(len(key_points), 4)} {point[:260]}"
                thread_parts.append(tweet)
        else:
            # Single tweet
            opening = f"{hook} {content[:250]}..."
            thread_parts.append(opening)
        
        thread_content = "\n\n---THREAD---\n\n".join(thread_parts)
        
        return PlatformContent(
            platform='twitter',
            title=f"Thread: {key_points[0][:50]}..." if key_points else "Key Insights",
            content=thread_content,
            tags=['#productivity', '#insights'],
            metadata={'thread_length': len(thread_parts), 'engagement_hook': hook},
            character_count=sum(len(part) for part in thread_parts),
            estimated_engagement=len(thread_parts) * 1.3  # Thread multiplier
        )
    
    def _create_linkedin_content(self, content: str, key_points: List[str], 
                                specs: Dict, metadata: Dict) -> PlatformContent:
        """Create LinkedIn post from content."""
        hook = specs['engagement_hooks'][0]
        
        # Professional structure
        linkedin_content = f"{hook} Key Insights:\n\n"
        
        if key_points:
            for i, point in enumerate(key_points[:5], 1):
                linkedin_content += f"{i}. {point}\n\n"
        else:
            linkedin_content += f"{content[:1000]}...\n\n"
        
        linkedin_content += "What's your experience with this? Share in the comments 👇\n\n"
        linkedin_content += "#productivity #automation #insights #business #growth"
        
        return PlatformContent(
            platform='linkedin',
            title="Professional Insights",
            content=linkedin_content,
            tags=['#productivity', '#automation', '#insights', '#business', '#growth'],
            metadata={'professional_tone': True, 'cta_included': True},
            character_count=len(linkedin_content),
            estimated_engagement=2.1  # LinkedIn professional content multiplier
        )
    
    def _create_medium_content(self, content: str, key_points: List[str], 
                             specs: Dict, metadata: Dict) -> PlatformContent:
        """Create Medium article from content."""
        # Extract title from content or key points
        title = key_points[0] if key_points else "Insights and Analysis"
        if len(title) > 60:
            title = title[:57] + "..."
        
        # Create structured article
        article_content = f"# {title}\n\n"
        
        # Introduction
        article_content += "## Introduction\n\n"
        article_content += f"{content[:500]}...\n\n"
        
        # Key insights section
        if key_points:
            article_content += "## Key Insights\n\n"
            for i, point in enumerate(key_points, 1):
                article_content += f"### {i}. {point[:100]}...\n\n"
                article_content += f"This insight highlights important aspects that deserve deeper consideration.\n\n"
        
        # Conclusion
        article_content += "## Conclusion\n\n"
        article_content += "These insights provide valuable perspective for continued growth and understanding.\n\n"
        article_content += "*What are your thoughts on these insights? I'd love to hear your perspective in the comments.*"
        
        return PlatformContent(
            platform='medium',
            title=title,
            content=article_content,
            tags=['insights', 'analysis', 'productivity'],
            metadata={'reading_time': '5 min', 'structured': True},
            character_count=len(article_content),
            estimated_engagement=1.8  # Medium long-form multiplier
        )
    
    def _create_generic_content(self, content: str, key_points: List[str], 
                               platform: str, specs: Dict, metadata: Dict) -> PlatformContent:
        """Create generic optimized content."""
        max_chars = specs.get('max_chars', 1000)
        
        if key_points:
            optimized_content = f"Key insights:\n\n"
            for point in key_points[:3]:
                optimized_content += f"• {point}\n\n"
        else:
            optimized_content = content[:max_chars-50] + "..."
        
        return PlatformContent(
            platform=platform,
            title="Content Insights",
            content=optimized_content[:max_chars],
            tags=['insights'],
            metadata=specs,
            character_count=len(optimized_content),
            estimated_engagement=1.0
        )


class OnePunchBridge:
    """
    Bridge between SUM and OnePunchUpload for intelligent content distribution.
    
    This bridge processes content through SUM's intelligence pipeline and
    formats it optimally for multi-platform publishing via OnePunchUpload.
    """
    
    def __init__(self, onepunch_api_url: str = "http://localhost:5173/api"):
        self.hierarchical_engine = HierarchicalDensificationEngine()
        self.summail_engine = SumMailEngine({'use_local_ai': True})
        self.platform_optimizer = PlatformOptimizer()
        self.onepunch_api_url = onepunch_api_url
        
        # Initialize local AI if available
        try:
            self.ollama_manager = OllamaManager()
            self.ai_available = len(self.ollama_manager.available_models) > 0
        except:
            self.ollama_manager = None
            self.ai_available = False
        
        logger.info(f"OnePunchBridge initialized (AI: {'✅' if self.ai_available else '❌'})")
    
    def process_email_to_social(self, email_content: str, 
                               target_platforms: List[str] = None) -> ContentPipeline:
        """
        Process email content and create social media posts.
        
        Perfect for newsletters, updates, and announcements.
        """
        if target_platforms is None:
            target_platforms = ['twitter', 'linkedin']
        
        start_time = datetime.now()
        
        try:
            # Process with SUM's hierarchical engine
            processing_result = self.hierarchical_engine.process_text(
                email_content,
                {
                    'max_concepts': 5,
                    'max_summary_tokens': 150,
                    'max_insights': 4
                }
            )
            
            # Extract key information
            summary = processing_result.get('hierarchical_summary', {}).get('level_2_core', '')
            key_insights = processing_result.get('key_insights', [])
            key_points = [insight.get('text', '') for insight in key_insights if insight.get('score', 0) > 0.6]
            
            # Enhance with local AI if available
            if self.ai_available and self.ollama_manager:
                try:
                    enhanced_points = self._enhance_with_ai(email_content, key_points, target_platforms)
                    if enhanced_points:
                        key_points = enhanced_points
                except Exception as e:
                    logger.warning(f"AI enhancement failed: {e}")
            
            # Optimize for each platform
            platform_content = []
            for platform in target_platforms:
                optimized = self.platform_optimizer.optimize_for_platform(
                    summary or email_content[:1000],
                    key_points,
                    platform,
                    {'source': 'email', 'processed_by': 'sum'}
                )
                platform_content.append(optimized)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate reach multiplier
            total_reach = sum(content.estimated_engagement for content in platform_content)
            
            pipeline = ContentPipeline(
                source_content=email_content[:500] + "..." if len(email_content) > 500 else email_content,
                source_type='email',
                target_platforms=target_platforms,
                processing_config={
                    'hierarchical_processing': True,
                    'local_ai_enhancement': self.ai_available,
                    'platform_optimization': True
                },
                results=platform_content,
                processing_time=processing_time,
                total_reach_multiplier=total_reach
            )
            
            logger.info(f"Processed email to {len(target_platforms)} platforms in {processing_time:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error in email processing pipeline: {e}")
            raise
    
    def process_document_to_content(self, document_path: str, 
                                  target_platforms: List[str] = None) -> ContentPipeline:
        """
        Process document and create multi-platform content.
        
        Great for research papers, reports, and long-form content.
        """
        if target_platforms is None:
            target_platforms = ['medium', 'linkedin', 'twitter']
        
        try:
            # This would integrate with multimodal_processor if available
            from multimodal_processor import MultiModalProcessor
            
            processor = MultiModalProcessor()
            result = processor.process_file(document_path)
            
            if result.error_message:
                raise ValueError(f"Document processing failed: {result.error_message}")
            
            return self.process_text_to_platforms(
                result.extracted_text,
                target_platforms,
                source_type='document'
            )
            
        except ImportError:
            logger.error("Multimodal processor not available")
            raise ValueError("Document processing requires multimodal capabilities")
    
    def process_text_to_platforms(self, text: str, 
                                 target_platforms: List[str],
                                 source_type: str = 'text') -> ContentPipeline:
        """Generic text to platform processing."""
        start_time = datetime.now()
        
        # Process with hierarchical engine
        result = self.hierarchical_engine.process_text(text, {
            'max_concepts': 7,
            'max_summary_tokens': 200,
            'max_insights': 5
        })
        
        # Extract key points
        summary = result.get('hierarchical_summary', {}).get('level_2_core', '')
        insights = result.get('key_insights', [])
        key_points = [insight.get('text', '') for insight in insights if insight.get('score', 0) > 0.5]
        
        # Optimize for platforms
        platform_content = []
        for platform in target_platforms:
            optimized = self.platform_optimizer.optimize_for_platform(
                summary or text[:1000],
                key_points,
                platform,
                {'source': source_type}
            )
            platform_content.append(optimized)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        total_reach = sum(content.estimated_engagement for content in platform_content)
        
        return ContentPipeline(
            source_content=text[:500] + "..." if len(text) > 500 else text,
            source_type=source_type,
            target_platforms=target_platforms,
            processing_config={'hierarchical_processing': True},
            results=platform_content,
            processing_time=processing_time,
            total_reach_multiplier=total_reach
        )
    
    def _enhance_with_ai(self, content: str, key_points: List[str], 
                        platforms: List[str]) -> Optional[List[str]]:
        """Enhance key points with local AI for better platform optimization."""
        if not self.ollama_manager:
            return None
        
        try:
            platform_context = ", ".join(platforms)
            prompt = f"""
            Optimize these key points for social media platforms ({platform_context}):
            
            Original points:
            {chr(10).join(f"- {point}" for point in key_points)}
            
            Make them more engaging, concise, and platform-appropriate. Return 3-5 optimized points:
            """
            
            request = ProcessingRequest(
                text=prompt,
                task_type='optimization',
                max_tokens=200,
                temperature=0.4
            )
            
            response = self.ollama_manager.process_text(request)
            
            # Extract optimized points from response
            enhanced_points = []
            for line in response.response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    point = line[1:].strip()
                    if len(point) > 10:
                        enhanced_points.append(point)
            
            return enhanced_points[:5] if enhanced_points else None
            
        except Exception as e:
            logger.warning(f"AI enhancement error: {e}")
            return None
    
    def publish_to_onepunch(self, pipeline: ContentPipeline) -> Dict[str, Any]:
        """
        Send processed content to OnePunchUpload for publishing.
        
        Note: This would require OnePunchUpload to expose a proper API.
        For now, this demonstrates the integration concept.
        """
        publish_payload = {
            'content_pipeline': asdict(pipeline),
            'platform_content': [asdict(content) for content in pipeline.results],
            'metadata': {
                'processed_by': 'sum',
                'processing_time': pipeline.processing_time,
                'source_type': pipeline.source_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # This would be the actual API call to OnePunchUpload
            # response = requests.post(f"{self.onepunch_api_url}/publish", json=publish_payload)
            
            # For now, simulate successful publishing
            logger.info(f"Would publish to OnePunchUpload: {len(pipeline.results)} platform versions")
            
            return {
                'success': True,
                'published_platforms': pipeline.target_platforms,
                'total_content_pieces': len(pipeline.results),
                'estimated_reach_multiplier': pipeline.total_reach_multiplier,
                'processing_time': pipeline.processing_time
            }
            
        except Exception as e:
            logger.error(f"Publishing to OnePunch failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_full_pipeline(self, source_content: str, source_type: str = 'text',
                           target_platforms: List[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: Process content → Optimize for platforms → Publish
        
        This is the main entry point for the SUM + OnePunch integration.
        """
        if target_platforms is None:
            target_platforms = ['twitter', 'linkedin', 'medium']
        
        try:
            # Step 1: Process content with SUM
            if source_type == 'email':
                pipeline = self.process_email_to_social(source_content, target_platforms)
            else:
                pipeline = self.process_text_to_platforms(source_content, target_platforms, source_type)
            
            # Step 2: Publish to OnePunchUpload
            publish_result = self.publish_to_onepunch(pipeline)
            
            # Step 3: Return complete result
            return {
                'pipeline': asdict(pipeline),
                'publishing': publish_result,
                'summary': {
                    'source_length': len(source_content),
                    'platforms_targeted': len(target_platforms),
                    'content_pieces_created': len(pipeline.results),
                    'total_character_count': sum(c.character_count for c in pipeline.results),
                    'estimated_reach_multiplier': pipeline.total_reach_multiplier,
                    'processing_time': pipeline.processing_time,
                    'success': publish_result.get('success', False)
                }
            }
            
        except Exception as e:
            logger.error(f"Full pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': {'processing_time': 0, 'content_pieces_created': 0}
            }


# Example usage and testing
def demo_email_to_social():
    """Demonstrate email newsletter to social media pipeline."""
    
    sample_email = """
    Subject: Weekly Tech Insights - AI Breakthrough & Market Analysis
    
    Hello subscribers,
    
    This week brought significant developments in AI and market trends that deserve your attention:
    
    1. OpenAI's latest model shows 40% improvement in reasoning tasks, potentially revolutionizing 
       how we approach complex problem-solving in business applications.
    
    2. The semiconductor market saw unexpected growth with NVIDIA reporting 25% quarter-over-quarter 
       increase, driven by enterprise AI adoption.
    
    3. Three major banks announced AI-powered fraud detection systems, reducing false positives by 60% 
       while catching 95% more actual fraud attempts.
    
    4. Remote work productivity tools are evolving rapidly, with new AI assistants helping teams 
       collaborate more effectively across time zones.
    
    Key takeaway: The convergence of AI capabilities with practical business applications is 
    accelerating faster than most predictions. Companies that adapt quickly will have significant 
    competitive advantages.
    
    What trends are you seeing in your industry? Reply and let me know!
    
    Best regards,
    Tech Weekly Team
    """
    
    bridge = OnePunchBridge()
    
    # Process newsletter into social media content
    result = bridge.create_full_pipeline(
        source_content=sample_email,
        source_type='email',
        target_platforms=['twitter', 'linkedin', 'medium']
    )
    
    print("🚀 SUM + OnePunchUpload Bridge Demo")
    print("=" * 50)
    print(f"Processing Success: {'✅' if result.get('success', True) else '❌'}")
    print(f"Content Pieces Created: {result['summary']['content_pieces_created']}")
    print(f"Platforms Targeted: {result['summary']['platforms_targeted']}")
    print(f"Processing Time: {result['summary']['processing_time']:.2f}s")
    print(f"Reach Multiplier: {result['summary']['estimated_reach_multiplier']:.1f}x")
    
    print("\n📋 Generated Content:")
    print("-" * 30)
    
    if 'pipeline' in result:
        for content in result['pipeline']['results']:
            print(f"\n🎯 {content['platform'].upper()}:")
            print(f"Title: {content['title']}")
            print(f"Length: {content['character_count']} chars")
            print(f"Content Preview: {content['content'][:200]}...")
            print(f"Tags: {', '.join(content['tags'])}")
            print(f"Engagement Score: {content['estimated_engagement']:.1f}")


if __name__ == "__main__":
    demo_email_to_social()