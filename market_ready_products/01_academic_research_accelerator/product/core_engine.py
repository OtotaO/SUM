#!/usr/bin/env python3
"""
Academic Research Accelerator - Core Engine

Transforms literature reviews from months to days using SUM's progressive
summarization and hierarchical understanding capabilities.

Key Features:
- Process 200+ papers in parallel
- Extract key findings, methodologies, conclusions
- Identify research gaps and connections
- Generate comprehensive literature reviews
- Track concept evolution across papers

Author: ototao
License: Apache License 2.0
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from progressive_summarization import ProgressiveStreamingEngine, ProgressEvent
from streaming_engine import StreamingConfig
from summarization_engine import HierarchicalDensificationEngine


logger = logging.getLogger(__name__)


@dataclass
class ResearchPaper:
    """Represents a research paper with metadata."""
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    year: int
    journal: str
    doi: str
    keywords: List[str]
    citations: int
    file_path: Optional[str] = None


@dataclass
class LiteratureReview:
    """Comprehensive literature review output."""
    summary: str
    key_findings: List[str]
    methodologies: List[str]
    research_gaps: List[str]
    connections: List[Dict[str, Any]]
    concept_evolution: Dict[str, List[Dict[str, Any]]]
    recommendations: List[str]
    processing_stats: Dict[str, Any]


class AcademicResearchAccelerator:
    """
    Core engine for accelerating academic research and literature reviews.
    
    Processes hundreds of papers in parallel, extracts key insights,
    identifies research gaps, and generates comprehensive reviews.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the academic research accelerator."""
        self.config = config or {}
        
        # Initialize SUM engines
        streaming_config = StreamingConfig(
            chunk_size_words=2000,  # Larger chunks for academic papers
            overlap_ratio=0.2,      # More overlap for context
            max_memory_mb=1024,     # More memory for large papers
            max_concurrent_chunks=8  # Parallel processing
        )
        
        self.progressive_engine = ProgressiveStreamingEngine(streaming_config)
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Research-specific components
        self.papers = []
        self.processing_results = {}
        self.literature_review = None
        
    async def process_literature_review(self, papers: List[ResearchPaper], 
                                      session_id: str = None) -> LiteratureReview:
        """
        Process a complete literature review from multiple papers.
        
        Args:
            papers: List of research papers to process
            session_id: Optional session identifier for progress tracking
            
        Returns:
            Comprehensive literature review with insights and recommendations
        """
        logger.info(f"Starting literature review processing for {len(papers)} papers")
        
        self.papers = papers
        session_id = session_id or f"lit_review_{int(time.time())}"
        
        try:
            # Phase 1: Process individual papers with progressive updates
            paper_results = await self._process_papers_parallel(papers, session_id)
            
            # Phase 2: Extract cross-paper insights and connections
            cross_paper_insights = await self._extract_cross_paper_insights(paper_results)
            
            # Phase 3: Identify research gaps and trends
            research_analysis = await self._analyze_research_landscape(paper_results)
            
            # Phase 4: Generate comprehensive literature review
            literature_review = await self._generate_literature_review(
                paper_results, cross_paper_insights, research_analysis
            )
            
            self.literature_review = literature_review
            return literature_review
            
        except Exception as e:
            logger.error(f"Error processing literature review: {e}")
            raise
    
    async def _process_papers_parallel(self, papers: List[ResearchPaper], 
                                     session_id: str) -> Dict[str, Any]:
        """Process multiple papers in parallel with progress tracking."""
        results = {}
        total_papers = len(papers)
        
        # Create tasks for parallel processing
        tasks = []
        for i, paper in enumerate(papers):
            task = self._process_single_paper(paper, i, total_papers, session_id)
            tasks.append(task)
        
        # Process papers concurrently
        paper_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        for i, result in enumerate(paper_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing paper {i}: {result}")
                results[f"paper_{i}"] = {"error": str(result)}
            else:
                results[f"paper_{i}"] = result
        
        return results
    
    async def _process_single_paper(self, paper: ResearchPaper, index: int, 
                                  total: int, session_id: str) -> Dict[str, Any]:
        """Process a single research paper with detailed analysis."""
        logger.info(f"Processing paper {index + 1}/{total}: {paper.title}")
        
        # Combine title, abstract, and full text for comprehensive analysis
        full_content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\nFull Text: {paper.full_text}"
        
        # Process with progressive engine
        result = await self.progressive_engine.process_streaming_text_with_progress(
            full_content, f"{session_id}_paper_{index}"
        )
        
        # Extract research-specific insights
        research_insights = self._extract_research_insights(paper, result)
        
        return {
            "paper_metadata": asdict(paper),
            "summarization_result": result,
            "research_insights": research_insights,
            "processing_index": index,
            "processing_time": time.time()
        }
    
    def _extract_research_insights(self, paper: ResearchPaper, 
                                 result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research-specific insights from paper analysis."""
        insights = {
            "key_findings": [],
            "methodology": "",
            "limitations": [],
            "future_work": [],
            "contributions": [],
            "research_questions": []
        }
        
        # Extract from hierarchical summary
        if 'hierarchical_summary' in result:
            summary = result['hierarchical_summary']
            
            # Extract key findings from concepts
            if 'level_1_concepts' in summary:
                insights["key_findings"] = summary['level_1_concepts'][:5]
            
            # Extract methodology from core summary
            if 'level_2_core' in summary:
                core_text = summary['level_2_core'].lower()
                if 'method' in core_text or 'approach' in core_text:
                    insights["methodology"] = summary['level_2_core']
        
        # Extract from key insights
        if 'key_insights' in result:
            for insight in result['key_insights']:
                insight_text = insight.get('text', '').lower()
                
                if any(word in insight_text for word in ['limitation', 'constraint', 'weakness']):
                    insights["limitations"].append(insight['text'])
                elif any(word in insight_text for word in ['future', 'next', 'further']):
                    insights["future_work"].append(insight['text'])
                elif any(word in insight_text for word in ['contribution', 'novel', 'advance']):
                    insights["contributions"].append(insight['text'])
                elif '?' in insight['text']:
                    insights["research_questions"].append(insight['text'])
        
        return insights
    
    async def _extract_cross_paper_insights(self, paper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights and connections across multiple papers."""
        logger.info("Extracting cross-paper insights and connections")
        
        # Collect all concepts and findings
        all_concepts = []
        all_findings = []
        all_methodologies = []
        
        for paper_id, result in paper_results.items():
            if 'research_insights' in result:
                insights = result['research_insights']
                all_concepts.extend(insights.get('key_findings', []))
                all_findings.extend(insights.get('contributions', []))
                if insights.get('methodology'):
                    all_methodologies.append(insights['methodology'])
        
        # Find common themes and connections
        connections = self._find_paper_connections(paper_results)
        
        # Identify emerging trends
        trends = self._identify_research_trends(paper_results)
        
        return {
            "common_themes": self._extract_common_themes(all_concepts),
            "methodological_trends": self._analyze_methodological_trends(all_methodologies),
            "connections": connections,
            "trends": trends
        }
    
    def _find_paper_connections(self, paper_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find connections between papers based on concepts and findings."""
        connections = []
        
        paper_ids = list(paper_results.keys())
        
        for i, paper_id_1 in enumerate(paper_ids):
            for paper_id_2 in paper_ids[i+1:]:
                paper_1 = paper_results[paper_id_1]
                paper_2 = paper_results[paper_id_2]
                
                if 'research_insights' in paper_1 and 'research_insights' in paper_2:
                    # Find overlapping concepts
                    concepts_1 = set(paper_1['research_insights'].get('key_findings', []))
                    concepts_2 = set(paper_2['research_insights'].get('key_findings', []))
                    
                    overlap = concepts_1.intersection(concepts_2)
                    
                    if overlap:
                        connections.append({
                            "paper_1": paper_1['paper_metadata']['title'],
                            "paper_2": paper_2['paper_metadata']['title'],
                            "connection_type": "concept_overlap",
                            "shared_concepts": list(overlap),
                            "strength": len(overlap) / max(len(concepts_1), len(concepts_2))
                        })
        
        return connections
    
    def _identify_research_trends(self, paper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify emerging trends in the research landscape."""
        trends = {
            "emerging_topics": [],
            "declining_topics": [],
            "methodological_shifts": [],
            "research_gaps": []
        }
        
        # Analyze by year to identify trends
        papers_by_year = {}
        for paper_id, result in paper_results.items():
            if 'paper_metadata' in result:
                year = result['paper_metadata'].get('year', 0)
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(result)
        
        # Identify emerging and declining topics
        if len(papers_by_year) > 1:
            years = sorted(papers_by_year.keys())
            recent_concepts = set()
            older_concepts = set()
            
            # Collect concepts from recent vs older papers
            for year in years[-2:]:  # Last 2 years
                for paper in papers_by_year[year]:
                    if 'research_insights' in paper:
                        recent_concepts.update(paper['research_insights'].get('key_findings', []))
            
            for year in years[:-2]:  # Older years
                for paper in papers_by_year[year]:
                    if 'research_insights' in paper:
                        older_concepts.update(paper['research_insights'].get('key_findings', []))
            
            # Find emerging topics (in recent but not older)
            trends["emerging_topics"] = list(recent_concepts - older_concepts)
            
            # Find declining topics (in older but not recent)
            trends["declining_topics"] = list(older_concepts - recent_concepts)
        
        return trends
    
    async def _analyze_research_landscape(self, paper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the overall research landscape and identify gaps."""
        logger.info("Analyzing research landscape and identifying gaps")
        
        # Collect all research questions and future work
        all_questions = []
        all_future_work = []
        all_limitations = []
        
        for paper_id, result in paper_results.items():
            if 'research_insights' in result:
                insights = result['research_insights']
                all_questions.extend(insights.get('research_questions', []))
                all_future_work.extend(insights.get('future_work', []))
                all_limitations.extend(insights.get('limitations', []))
        
        # Identify research gaps
        research_gaps = self._identify_research_gaps(all_questions, all_future_work, all_limitations)
        
        # Analyze citation patterns
        citation_analysis = self._analyze_citation_patterns(paper_results)
        
        return {
            "research_gaps": research_gaps,
            "citation_analysis": citation_analysis,
            "unanswered_questions": all_questions,
            "future_directions": all_future_work
        }
    
    def _identify_research_gaps(self, questions: List[str], 
                               future_work: List[str], 
                               limitations: List[str]) -> List[str]:
        """Identify research gaps based on questions, future work, and limitations."""
        gaps = []
        
        # Combine all text for analysis
        combined_text = " ".join(questions + future_work + limitations)
        
        # Use SUM to extract key gaps
        if combined_text:
            result = self.hierarchical_engine.process_text(combined_text)
            if 'hierarchical_summary' in result:
                gaps = result['hierarchical_summary'].get('level_1_concepts', [])
        
        return gaps[:10]  # Top 10 gaps
    
    def _analyze_citation_patterns(self, paper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze citation patterns to identify influential papers."""
        citation_data = []
        
        for paper_id, result in paper_results.items():
            if 'paper_metadata' in result:
                metadata = result['paper_metadata']
                citation_data.append({
                    "title": metadata.get('title', ''),
                    "citations": metadata.get('citations', 0),
                    "year": metadata.get('year', 0),
                    "journal": metadata.get('journal', '')
                })
        
        # Sort by citations
        citation_data.sort(key=lambda x: x['citations'], reverse=True)
        
        return {
            "most_cited": citation_data[:10],
            "citation_distribution": {
                "high_impact": len([p for p in citation_data if p['citations'] > 100]),
                "medium_impact": len([p for p in citation_data if 10 <= p['citations'] <= 100]),
                "low_impact": len([p for p in citation_data if p['citations'] < 10])
            }
        }
    
    async def _generate_literature_review(self, paper_results: Dict[str, Any],
                                        cross_paper_insights: Dict[str, Any],
                                        research_analysis: Dict[str, Any]) -> LiteratureReview:
        """Generate comprehensive literature review from all analysis."""
        logger.info("Generating comprehensive literature review")
        
        # Combine all insights for final summary
        all_content = self._combine_all_insights(paper_results, cross_paper_insights, research_analysis)
        
        # Generate final summary using SUM
        final_result = await self.progressive_engine.process_streaming_text_with_progress(
            all_content, "final_literature_review"
        )
        
        # Extract key components
        summary = final_result.get('hierarchical_summary', {}).get('level_2_core', '')
        key_findings = final_result.get('hierarchical_summary', {}).get('level_1_concepts', [])
        
        # Compile recommendations
        recommendations = self._generate_recommendations(paper_results, research_analysis)
        
        # Calculate processing stats
        processing_stats = self._calculate_processing_stats(paper_results)
        
        return LiteratureReview(
            summary=summary,
            key_findings=key_findings,
            methodologies=cross_paper_insights.get('methodological_trends', []),
            research_gaps=research_analysis.get('research_gaps', []),
            connections=cross_paper_insights.get('connections', []),
            concept_evolution=cross_paper_insights.get('trends', {}),
            recommendations=recommendations,
            processing_stats=processing_stats
        )
    
    def _combine_all_insights(self, paper_results: Dict[str, Any],
                             cross_paper_insights: Dict[str, Any],
                             research_analysis: Dict[str, Any]) -> str:
        """Combine all insights into a single text for final summarization."""
        combined_text = []
        
        # Add paper summaries
        for paper_id, result in paper_results.items():
            if 'summarization_result' in result:
                summary = result['summarization_result'].get('hierarchical_summary', {})
                if 'level_2_core' in summary:
                    combined_text.append(f"Paper: {summary['level_2_core']}")
        
        # Add cross-paper insights
        combined_text.append(f"Cross-paper themes: {cross_paper_insights.get('common_themes', [])}")
        combined_text.append(f"Research gaps: {research_analysis.get('research_gaps', [])}")
        
        return "\n\n".join(combined_text)
    
    def _generate_recommendations(self, paper_results: Dict[str, Any],
                                research_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Based on research gaps
        gaps = research_analysis.get('research_gaps', [])
        for gap in gaps[:5]:
            recommendations.append(f"Investigate: {gap}")
        
        # Based on citation patterns
        citation_analysis = research_analysis.get('citation_analysis', {})
        if citation_analysis.get('citation_distribution', {}).get('high_impact', 0) < 3:
            recommendations.append("Focus on high-impact research areas")
        
        # Based on emerging trends
        trends = research_analysis.get('trends', {})
        emerging = trends.get('emerging_topics', [])
        for topic in emerging[:3]:
            recommendations.append(f"Explore emerging topic: {topic}")
        
        return recommendations
    
    def _calculate_processing_stats(self, paper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics."""
        total_papers = len(paper_results)
        successful_papers = len([r for r in paper_results.values() if 'error' not in r])
        
        return {
            "total_papers": total_papers,
            "successful_papers": successful_papers,
            "success_rate": successful_papers / total_papers if total_papers > 0 else 0,
            "total_processing_time": sum(r.get('processing_time', 0) for r in paper_results.values()),
            "average_paper_processing_time": sum(r.get('processing_time', 0) for r in paper_results.values()) / total_papers if total_papers > 0 else 0
        }


# Example usage
async def main():
    """Example usage of the Academic Research Accelerator."""
    accelerator = AcademicResearchAccelerator()
    
    # Example papers (in real usage, these would be loaded from PDFs)
    papers = [
        ResearchPaper(
            title="Neural Network Architectures for Natural Language Processing",
            authors=["Smith, J.", "Johnson, A."],
            abstract="This paper explores novel neural network architectures...",
            full_text="Full paper content here...",
            year=2023,
            journal="Nature AI",
            doi="10.1038/ai.2023.001",
            keywords=["neural networks", "NLP", "deep learning"],
            citations=150
        )
        # Add more papers...
    ]
    
    # Process literature review
    review = await accelerator.process_literature_review(papers)
    
    print("Literature Review Complete!")
    print(f"Summary: {review.summary}")
    print(f"Key Findings: {review.key_findings}")
    print(f"Research Gaps: {review.research_gaps}")


if __name__ == "__main__":
    asyncio.run(main())
