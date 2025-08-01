"""
Core SUM Engine - Optimized Architecture

Following John Carmack's principles:
- Fast: Minimal abstraction layers
- Simple: Clear interfaces with single responsibilities  
- Clear: Obvious data flow and dependencies
- Bulletproof: Robust error handling

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

# Core engine exports - clean interface
from .engine import SumEngine
from .processor import TextProcessor
from .analyzer import ContentAnalyzer

__all__ = ['SumEngine', 'TextProcessor', 'ContentAnalyzer']