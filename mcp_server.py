"""
MCP (Model Context Protocol) Server for SUM

Enables AI assistants to use SUM's summarization capabilities directly
through the Model Context Protocol.

This allows tools like Claude Desktop, Continue.dev, and other MCP-compatible
systems to leverage SUM for text summarization tasks.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# MCP server imports
try:
    from mcp import MCPServer, Tool, ToolResult
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Define stubs for when MCP is not available
    class MCPServer:
        pass
    class Tool:
        pass
    class ToolResult:
        pass

from summarization_engine import (
    BasicSummarizationEngine,
    AdvancedSummarizationEngine,
    HierarchicalDensificationEngine
)
from unlimited_text_processor import get_unlimited_processor
from language_detector import detect_language
from smart_cache import get_cache

logger = logging.getLogger(__name__)


class SUMMCPServer(MCPServer if MCP_AVAILABLE else object):
    """
    MCP server implementation for SUM.
    
    Provides tools for text summarization that can be used by
    AI assistants and other MCP-compatible systems.
    """
    
    def __init__(self):
        """Initialize the SUM MCP server."""
        if MCP_AVAILABLE:
            super().__init__(
                name="sum",
                version="1.0.0",
                description="Advanced text summarization tools"
            )
        
        # Initialize engines
        self.engines = {
            'basic': BasicSummarizationEngine(),
            'advanced': AdvancedSummarizationEngine(),
            'hierarchical': HierarchicalDensificationEngine()
        }
        
        self.unlimited_processor = get_unlimited_processor()
        self.cache = get_cache()
        
        # Register tools
        if MCP_AVAILABLE:
            self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        
        # Basic summarization tool
        self.register_tool(Tool(
            name="summarize",
            description="Summarize text using various models",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["basic", "advanced", "hierarchical", "unlimited"],
                        "default": "hierarchical",
                        "description": "Summarization model to use"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 500,
                        "default": 100,
                        "description": "Maximum tokens in summary"
                    }
                },
                "required": ["text"]
            },
            handler=self.summarize_text
        ))
        
        # Batch summarization tool
        self.register_tool(Tool(
            name="batch_summarize",
            description="Summarize multiple texts in batch",
            input_schema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to summarize"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["basic", "advanced", "hierarchical"],
                        "default": "basic",
                        "description": "Model to use for all texts"
                    }
                },
                "required": ["texts"]
            },
            handler=self.batch_summarize
        ))
        
        # Language detection tool
        self.register_tool(Tool(
            name="detect_language",
            description="Detect the language of text",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            },
            handler=self.detect_text_language
        ))
        
        # Hierarchical summary tool
        self.register_tool(Tool(
            name="hierarchical_summary",
            description="Generate multi-level hierarchical summary",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize hierarchically"
                    },
                    "include_insights": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include key insights extraction"
                    }
                },
                "required": ["text"]
            },
            handler=self.hierarchical_summary
        ))
        
        # Extract key concepts tool
        self.register_tool(Tool(
            name="extract_concepts",
            description="Extract key concepts and themes from text",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    },
                    "max_concepts": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum number of concepts to extract"
                    }
                },
                "required": ["text"]
            },
            handler=self.extract_concepts
        ))
        
        # Compare summaries tool
        self.register_tool(Tool(
            name="compare_summaries",
            description="Compare summaries from different models",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize with multiple models"
                    },
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["basic", "advanced", "hierarchical"]
                        },
                        "default": ["basic", "hierarchical"],
                        "description": "Models to compare"
                    }
                },
                "required": ["text"]
            },
            handler=self.compare_models
        ))
    
    async def summarize_text(self, text: str, model: str = "hierarchical", 
                            max_tokens: int = 100) -> ToolResult:
        """
        Summarize text using specified model.
        
        Args:
            text: Text to summarize
            model: Model to use
            max_tokens: Maximum tokens in summary
            
        Returns:
            ToolResult with summary
        """
        try:
            if model == "unlimited":
                # Use unlimited processor for very large texts
                result = self.unlimited_processor.process_text(
                    text,
                    {'max_summary_tokens': max_tokens}
                )
            else:
                # Use standard engine
                engine = self.engines.get(model, self.engines['basic'])
                result = engine.process_text(
                    text,
                    {'maxTokens': max_tokens}
                )
            
            if 'error' in result:
                return ToolResult(
                    success=False,
                    error=result['error']
                )
            
            # Format response
            response = {
                "summary": result.get('summary', ''),
                "condensed": result.get('sum', ''),
                "model": model,
                "language": result.get('detected_language', 'unknown'),
                "word_count": result.get('original_length', 0),
                "compression_ratio": result.get('compression_ratio', 0)
            }
            
            # Add hierarchical data if available
            if 'hierarchical_summary' in result:
                response['hierarchical'] = result['hierarchical_summary']
            
            # Add tags if available
            if 'tags' in result:
                response['key_concepts'] = result['tags']
            
            return ToolResult(
                success=True,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Error in summarize_text: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def batch_summarize(self, texts: List[str], 
                            model: str = "basic") -> ToolResult:
        """
        Summarize multiple texts in batch.
        
        Args:
            texts: List of texts to summarize
            model: Model to use
            
        Returns:
            ToolResult with summaries
        """
        try:
            engine = self.engines.get(model, self.engines['basic'])
            summaries = []
            
            for i, text in enumerate(texts):
                result = engine.process_text(text, {'maxTokens': 50})
                
                summaries.append({
                    "index": i,
                    "summary": result.get('summary', ''),
                    "language": result.get('detected_language', 'unknown'),
                    "error": result.get('error')
                })
            
            return ToolResult(
                success=True,
                data={
                    "count": len(summaries),
                    "model": model,
                    "summaries": summaries
                }
            )
            
        except Exception as e:
            logger.error(f"Error in batch_summarize: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def detect_text_language(self, text: str) -> ToolResult:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ToolResult with language information
        """
        try:
            lang_info = detect_language(text)
            
            return ToolResult(
                success=True,
                data={
                    "language": lang_info['language'],
                    "language_name": lang_info['language_name'],
                    "confidence": lang_info['confidence'],
                    "method": lang_info['method'],
                    "supported": lang_info['supported']
                }
            )
            
        except Exception as e:
            logger.error(f"Error in detect_text_language: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def hierarchical_summary(self, text: str, 
                                 include_insights: bool = True) -> ToolResult:
        """
        Generate hierarchical summary with multiple levels.
        
        Args:
            text: Text to summarize
            include_insights: Whether to include insights
            
        Returns:
            ToolResult with hierarchical summary
        """
        try:
            engine = self.engines['hierarchical']
            result = engine.process_text(text)
            
            if 'error' in result:
                return ToolResult(
                    success=False,
                    error=result['error']
                )
            
            response = {
                "hierarchical_summary": result.get('hierarchical_summary', {}),
                "metadata": result.get('metadata', {}),
                "language": result.get('detected_language', 'unknown')
            }
            
            if include_insights and 'key_insights' in result:
                response['insights'] = result['key_insights']
            
            return ToolResult(
                success=True,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Error in hierarchical_summary: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def extract_concepts(self, text: str, 
                             max_concepts: int = 10) -> ToolResult:
        """
        Extract key concepts from text.
        
        Args:
            text: Text to analyze
            max_concepts: Maximum concepts to extract
            
        Returns:
            ToolResult with concepts
        """
        try:
            # Use advanced engine for better concept extraction
            engine = self.engines['advanced']
            result = engine.process_text(text)
            
            concepts = result.get('tags', [])[:max_concepts]
            
            # Get additional metadata if available
            response = {
                "concepts": concepts,
                "count": len(concepts),
                "language": result.get('detected_language', 'unknown')
            }
            
            # Add main concept if available
            if 'meta_analysis' in result and 'main_concept' in result['meta_analysis']:
                response['main_concept'] = result['meta_analysis']['main_concept']
            
            return ToolResult(
                success=True,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Error in extract_concepts: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def compare_models(self, text: str, 
                           models: List[str] = None) -> ToolResult:
        """
        Compare summaries from different models.
        
        Args:
            text: Text to summarize
            models: Models to compare
            
        Returns:
            ToolResult with comparison
        """
        try:
            if models is None:
                models = ["basic", "hierarchical"]
            
            comparisons = {}
            
            for model in models:
                if model in self.engines:
                    engine = self.engines[model]
                    result = engine.process_text(text, {'maxTokens': 100})
                    
                    comparisons[model] = {
                        "summary": result.get('summary', ''),
                        "compression_ratio": result.get('compression_ratio', 0),
                        "processing_time": result.get('processing_time', 0),
                        "word_count": len(result.get('summary', '').split())
                    }
            
            return ToolResult(
                success=True,
                data={
                    "original_length": len(text.split()),
                    "models_compared": list(comparisons.keys()),
                    "comparisons": comparisons
                }
            )
            
        except Exception as e:
            logger.error(f"Error in compare_models: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


def create_mcp_server() -> Optional[SUMMCPServer]:
    """
    Create and configure the MCP server.
    
    Returns:
        Configured MCP server or None if MCP not available
    """
    if not MCP_AVAILABLE:
        logger.error("MCP package not installed. Install with: pip install mcp")
        return None
    
    return SUMMCPServer()


async def run_mcp_server():
    """
    Run the MCP server.
    """
    server = create_mcp_server()
    if server:
        logger.info("Starting SUM MCP server...")
        await server.run()
    else:
        logger.error("Failed to create MCP server")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    asyncio.run(run_mcp_server())