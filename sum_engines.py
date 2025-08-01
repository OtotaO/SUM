"""
SUM Engines - Main entry point for all summarization engines.
Provides backward compatibility with the original module structure.
"""

# Import all engines for easy access
from application.basic_summarizer import BasicSummarizationEngine
from application.advanced_summarizer import AdvancedSummarizationEngine
from application.hierarchical_summarizer import HierarchicalDensificationEngine

# Backward compatibility aliases
SimpleSUM = BasicSummarizationEngine
BasicSUM = BasicSummarizationEngine
MagnumOpusSUM = AdvancedSummarizationEngine
AdvancedSUM = AdvancedSummarizationEngine
TrinityEngine = HierarchicalDensificationEngine

# Export all engines
__all__ = [
    'BasicSummarizationEngine',
    'AdvancedSummarizationEngine', 
    'HierarchicalDensificationEngine',
    # Aliases for backward compatibility
    'SimpleSUM',
    'BasicSUM',
    'MagnumOpusSUM',
    'AdvancedSUM',
    'TrinityEngine'
]