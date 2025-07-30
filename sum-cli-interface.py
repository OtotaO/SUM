#!/usr/bin/env python
"""
sum_cli.py - Command Line Interface for SUM Knowledge Distillation Platform

This module provides a powerful command-line interface to access all SUM
functionality, enabling scriptable text analysis, summarization, knowledge graph
generation, and temporal analysis from the terminal.

Design principles:
- Simple, intuitive interface (Torvalds/van Rossum style)
- Comprehensive documentation (Stroustrup approach)
- Efficient processing pipelines (Knuth optimization)
- Secure execution (Schneier principles)
- Extensible architecture (Fowler patterns)

Author: ototao
License: Apache License 2.0
"""

import argparse
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import glob
import yaml
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sum_cli.log')
    ]
)
logger = logging.getLogger('sum_cli')

# Import SUM components
try:
    from SUM import SimpleSUM, AdvancedSUM
    from Utils.data_loader import DataLoader
    from Models.topic_modeling import TopicModeler
    from knowledge_graph import KnowledgeGraph
    from temporal_analysis import TemporalAnalysis
except ImportError as e:
    logger.error(f"Error importing SUM components: {e}")
    logger.error("Please make sure SUM is installed or PYTHONPATH is set correctly.")
    sys.exit(1)


class SumCLI:
    """
    Command-line interface for the SUM Knowledge Distillation Platform.
    
    This class provides a comprehensive CLI for accessing all SUM functionality,
    including summarization, topic modeling, knowledge graph generation,
    and temporal analysis.
    """
    
    def __init__(self):
        """Initialize SUM CLI with parser and commands."""
        self.parser = argparse.ArgumentParser(
            description='SUM - The Ultimate Knowledge Distillation Platform',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic summarization
  sum_cli.py summarize -i input.txt -o summary.txt
  
  # Advanced summarization with entity recognition
  sum_cli.py summarize -i input.txt -o summary.json --advanced --entities
  
  # Topic modeling
  sum_cli.py topics -i research_papers/ -n 10 --algorithm nmf
  
  # Knowledge graph generation
  sum_cli.py graph -i data.json -o graph.html --entities --topics
  
  # Temporal analysis
  sum_cli.py temporal -i data/*.json --period month --report evolution_report.html
            """
        )
        
        self.subparsers = self.parser.add_subparsers(
            dest='command',
            help='Command to execute'
        )
        
        # Initialize commands
        self._init_summarize_command()
        self._init_topics_command()
        self._init_graph_command()
        self._init_temporal_command()
        self._init_batch_command()
        self._init_config_command()
        
        # Output directory for results
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _init_summarize_command(self):
        """Initialize summarize command parser."""
        parser_summarize = self.subparsers.add_parser(
            'summarize',
            help='Summarize text content'
        )
        
        # Input options
        parser_summarize.add_argument(
            '-i', '--input',
            required=True,
            help='Input file or directory path'
        )
        
        parser_summarize.add_argument(
            '-o', '--output',
            help='Output file path'
        )
        
        # Summarization options
        parser_summarize.add_argument(
            '--advanced',
            action='store_true',
            help='Use advanced summarization (AdvancedSUM)'
        )
        
        parser_summarize.add_argument(
            '--entities',
            action='store_true',
            help='Extract entities (requires --advanced)'
        )
        
        parser_summarize.add_argument(
            '--max-tokens',
            type=int,
            default=150,
            help='Maximum number of tokens in summary'
        )
        
        parser_summarize.add_argument(
            '--min-tokens',
            type=int,
            default=50,
            help='Minimum number of tokens in summary'
        )
        
        parser_summarize.add_argument(
            '--levels',
            type=str,
            default='tag,sentence,paragraph',
            help='Summary levels (comma-separated: tag,sentence,paragraph)'
        )
        
        parser_summarize.add_argument(
            '--format',
            choices=['text', 'json', 'markdown', 'html'],
            default='text',
            help='Output format'
        )
    
    def _init_topics_command(self):
        """Initialize topics command parser."""
        parser_topics = self.subparsers.add_parser(
            'topics',
            help='Extract topics from text content'
        )
        
        # Input options
        parser_topics.add_argument(
            '-i', '--input',
            required=True,
            help='Input file or directory path'
        )
        
        parser_topics.add_argument(
            '-o', '--output',
            help='Output file path'
        )
        
        # Topic modeling options
        parser_topics.add_argument(
            '-n', '--num-topics',
            type=int,
            default=5,
            help='Number of topics to extract'
        )
        
        parser_topics.add_argument(
            '--algorithm',
            choices=['lda', 'nmf', 'lsa'],
            default='lda',
            help='Topic modeling algorithm'
        )
        
        parser_topics.add_argument(
            '--top-terms',
            type=int,
            default=10,
            help='Number of top terms per topic'
        )
        
        parser_topics.add_argument(
            '--auto-optimize',
            action='store_true',
            help='Auto-optimize topic model hyperparameters'
        )
        
        parser_topics.add_argument(
            '--visualize',
            action='store_true',
            help='Generate topic visualization'
        )
        
        parser_topics.add_argument(
            '--format',
            choices=['json', 'csv', 'markdown', 'html'],
            default='json',
            help='Output format'
        )
    
    def _init_graph_command(self):
        """Initialize graph command parser."""
        parser_graph = self.subparsers.add_parser(
            'graph',
            help='Generate knowledge graph'
        )
        
        # Input options
        parser_graph.add_argument(
            '-i', '--input',
            required=True,
            help='Input file or directory path'
        )
        
        parser_graph.add_argument(
            '-o', '--output',
            help='Output file path'
        )
        
        # Knowledge graph options
        parser_graph.add_argument(
            '--entities',
            action='store_true',
            help='Extract entities for the graph'
        )
        
        parser_graph.add_argument(
            '--topics',
            action='store_true',
            help='Include topics in the graph'
        )
        
        parser_graph.add_argument(
            '--max-entities',
            type=int,
            default=100,
            help='Maximum number of entities in the graph'
        )
        
        parser_graph.add_argument(
            '--min-edge-weight',
            type=float,
            default=0.1,
            help='Minimum weight for edges to include'
        )
        
        parser_graph.add_argument(
            '--layout',
            choices=['spring', 'circular', 'kamada_kawai', 'spectral'],
            default='spring',
            help='Graph layout algorithm'
        )
        
        parser_graph.add_argument(
            '--format',
            choices=['html', 'png', 'svg', 'gexf', 'graphml', 'json', 'cytoscape'],
            default='html',
            help='Output format'
        )
        
        parser_graph.add_argument(
            '--interactive',
            action='store_true',
            help='Generate interactive visualization (HTML)'
        )
    
    def _init_temporal_command(self):
        """Initialize temporal command parser."""
        parser_temporal = self.subparsers.add_parser(
            'temporal',
            help='Analyze temporal evolution of knowledge'
        )
        
        # Input options
        parser_temporal.add_argument(
            '-i', '--input',
            required=True,
            help='Input file or directory pattern (with dates in filenames or content)'
        )
        
        parser_temporal.add_argument(
            '-o', '--output',
            help='Output file path'
        )
        
        # Date options
        parser_temporal.add_argument(
            '--date-format',
            default='%Y-%m-%d',
            help='Date format string'
        )
        
        parser_temporal.add_argument(
            '--date-field',
            default='date',
            help='Field name for date in JSON/CSV input'
        )
        
        parser_temporal.add_argument(
            '--filename-pattern',
            help='Regex pattern to extract date from filename (e.g., ".*_([0-9]{4}-[0-9]{2}-[0-9]{2}).*")'
        )
        
        # Analysis options
        parser_temporal.add_argument(
            '--period',
            choices=['day', 'week', 'month', 'year'],
            default='month',
            help='Time period granularity'
        )
        
        parser_temporal.add_argument(
            '--min-documents',
            type=int,
            default=2,
            help='Minimum documents per period'
        )
        
        parser_temporal.add_argument(
            '--smoothing',
            type=int,
            default=1,
            help='Smoothing window size (0 for no smoothing)'
        )
        
        parser_temporal.add_argument(
            '--topics',
            action='store_true',
            help='Analyze topic evolution'
        )
        
        parser_temporal.add_argument(
            '--entities',
            action='store_true',
            help='Analyze entity evolution'
        )
        
        parser_temporal.add_argument(
            '--sentiment',
            action='store_true',
            help='Analyze sentiment evolution'
        )
        
        parser_temporal.add_argument(
            '--report',
            action='store_true',
            help='Generate comprehensive evolution report'
        )
        
        parser_temporal.add_argument(
            '--format',
            choices=['html', 'json', 'markdown'],
            default='html',
            help='Output format'
        )
    
    def _init_batch_command(self):
        """Initialize batch command parser."""
        parser_batch = self.subparsers.add_parser(
            'batch',
            help='Process multiple files in batch mode'
        )
        
        # Input options
        parser_batch.add_argument(
            '-i', '--input',
            required=True,
            help='Input file pattern (glob) or directory'
        )
        
        parser_batch.add_argument(
            '-o', '--output-dir',
            required=True,
            help='Output directory'
        )
        
        # Batch options
        parser_batch.add_argument(
            '--config',
            help='Configuration file for batch processing'
        )
        
        parser_batch.add_argument(
            '--summarize',
            action='store_true',
            help='Perform summarization on each file'
        )
        
        parser_batch.add_argument(
            '--topics',
            action='store_true',
            help='Perform topic modeling on each file'
        )
        
        parser_batch.add_argument(
            '--graph',
            action='store_true',
            help='Generate knowledge graph for each file'
        )
        
        parser_batch.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of worker threads'
        )
        
        parser_batch.add_argument(
            '--format',
            choices=['json', 'text', 'markdown', 'html'],
            default='json',
            help='Output format'
        )
    
    def _init_config_command(self):
        """Initialize config command parser."""
        parser_config = self.subparsers.add_parser(
            'config',
            help='Generate or manipulate configuration'
        )
        
        # Config options
        parser_config.add_argument(
            '--generate',
            action='store_true',
            help='Generate sample configuration file'
        )
        
        parser_config.add_argument(
            '--validate',
            help='Validate configuration file'
        )
        
        parser_config.add_argument(
            '-o', '--output',
            default='sum_config.yaml',
            help='Output path for generated configuration'
        )
    
    def _load_text_input(self, input_path: str) -> str:
        """
        Load text content from file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Text content
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {input_path}: {e}")
            raise
    
    def _load_json_input(self, input_path: str) -> Dict:
        """
        Load JSON content from file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Parsed JSON content
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {input_path}: {e}")
            raise
    
    def _save_output(self, content: Any, output_path: str, format: str = 'text') -> None:
        """
        Save output content to file.
        
        Args:
            content: Content to save
            output_path: Output file path
            format: Output format
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save based on format
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
            elif format == 'markdown':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(self._format_markdown(content))
            elif format == 'html':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(self._format_html(content))
            elif format == 'csv':
                self._save_csv(content, output_path)
            else:
                # Default to text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
                
            logger.info(f"Output saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving output to {output_path}: {e}")
            raise
    
    def _format_markdown(self, content: Any) -> str:
        """
        Format content as Markdown.
        
        Args:
            content: Content to format
            
        Returns:
            Markdown formatted string
        """
        if isinstance(content, dict):
            if 'summary' in content:
                # Handle summary result
                md = "# Summary\n\n"
                
                if 'tags' in content:
                    md += "## Tags\n\n"
                    md += ", ".join(content['tags']) + "\n\n"
                
                if 'sum' in content:
                    md += "## One-Sentence Summary\n\n"
                    md += content['sum'] + "\n\n"
                
                md += "## Detailed Summary\n\n"
                md += content['summary'] + "\n\n"
                
                if 'entities' in content:
                    md += "## Entities\n\n"
                    md += "| Entity | Type | Count |\n"
                    md += "|--------|------|-------|\n"
                    
                    for entity, entity_type, count in content['entities'][:20]:  # Limit to top 20
                        md += f"| {entity} | {entity_type} | {count} |\n"
                    
                    md += "\n"
                
                if 'topics' in content:
                    md += "## Topics\n\n"
                    
                    for i, topic in enumerate(content['topics']):
                        md += f"### Topic {i+1}: {topic['label']}\n\n"
                        md += "| Term | Weight |\n"
                        md += "|------|--------|\n"
                        
                        for term_data in topic['terms']:
                            term = term_data['term']
                            weight = term_data['weight']
                            md += f"| {term} | {weight:.3f} |\n"
                        
                        md += "\n"
                
                if 'metadata' in content:
                    md += "## Metadata\n\n"
                    md += f"* Processing Time: {content['metadata'].get('processing_time', 0):.2f}s\n"
                    md += f"* Compression Ratio: {content['metadata'].get('compression_ratio', 0):.2f}\n"
                    md += f"* Original Length: {content['metadata'].get('words', 0)} words\n"
                
                return md
                
            elif 'topics' in content and isinstance(content['topics'], dict):
                # Handle topic modeling result
                md = "# Topic Modeling Results\n\n"
                
                md += f"Number of Topics: {len(content['topics'])}\n\n"
                
                for topic_id, topic_data in content['topics'].items():
                    md += f"## Topic {topic_id}: {topic_data['label']}\n\n"
                    
                    # Add coherence if available
                    if 'coherence' in topic_data:
                        md += f"Coherence: {topic_data['coherence']:.3f}\n\n"
                    
                    md += "| Term | Weight |\n"
                    md += "|------|--------|\n"
                    
                    for term, weight in topic_data['terms'].items():
                        md += f"| {term} | {weight:.3f} |\n"
                    
                    md += "\n"
                
                return md
            
            else:
                # Generic dictionary
                md = ""
                for key, value in content.items():
                    md += f"## {key}\n\n"
                    
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            md += f"### {subkey}\n\n"
                            md += f"{subvalue}\n\n"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for item_key, item_value in item.items():
                                    md += f"* **{item_key}**: {item_value}\n"
                            else:
                                md += f"* {item}\n"
                        md += "\n"
                    else:
                        md += f"{value}\n\n"
                
                return md
        
        elif isinstance(content, list):
            # Handle list
            md = ""
            for item in content:
                if isinstance(item, dict):
                    for key, value in item.items():
                        md += f"### {key}\n\n"
                        md += f"{value}\n\n"
                else:
                    md += f"* {item}\n"
            
            return md
        
        else:
            # Default to string representation
            return str(content)
    
    def _format_html(self, content: Any) -> str:
        """
        Format content as HTML.
        
        Args:
            content: Content to format
            
        Returns:
            HTML formatted string
        """
        # Convert to Markdown first then to HTML
        try:
            import markdown
            md_content = self._format_markdown(content)
            html = markdown.markdown(md_content, extensions=['tables'])
            
            # Add basic HTML structure
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>SUM Results</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    
                    h1, h2, h3, h4 {{
                        margin-top: 1.5em;
                        margin-bottom: 0.5em;
                        color: #2c3e50;
                    }}
                    
                    h1 {{
                        border-bottom: 2px solid #eee;
                        padding-bottom: 10px;
                    }}
                    
                    h2 {{
                        border-bottom: 1px solid #eee;
                        padding-bottom: 5px;
                    }}
                    
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    
                    th, td {{
                        text-align: left;
                        padding: 12px;
                        border-bottom: 1px solid #ddd;
                    }}
                    
                    th {{
                        background-color: #f8f9fa;
                        font-weight: 600;
                    }}
                    
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    
                    code {{
                        background-color: #f8f9fa;
                        padding: 2px 4px;
                        border-radius: 4px;
                    }}
                    
                    footer {{
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                        font-size: 0.9em;
                        color: #7f8c8d;
                    }}
                </style>
            </head>
            <body>
                {html}
                <footer>
                    <p>Generated by SUM - The Ultimate Knowledge Distillation Platform</p>
                </footer>
            </body>
            </html>
            """
            
        except ImportError:
            logger.warning("Python-Markdown not installed. Using basic HTML formatting.")
            
            # Basic HTML formatting
            html = "<html><head><title>SUM Results</title></head><body>"
            
            if isinstance(content, dict):
                for key, value in content.items():
                    html += f"<h2>{key}</h2>"
                    
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            html += f"<h3>{subkey}</h3><p>{subvalue}</p>"
                    elif isinstance(value, list):
                        html += "<ul>"
                        for item in value:
                            html += f"<li>{item}</li>"
                        html += "</ul>"
                    else:
                        html += f"<p>{value}</p>"
            elif isinstance(content, list):
                html += "<ul>"
                for item in content:
                    html += f"<li>{item}</li>"
                html += "</ul>"
            else:
                html += f"<p>{content}</p>"
                
            html += "</body></html>"
            return html
    
    def _save_csv(self, content: Any, output_path: str) -> None:
        """
        Save content as CSV.
        
        Args:
            content: Content to save
            output_path: Output file path
        """
        try:
            import csv
            
            if isinstance(content, dict) and 'topics' in content:
                # Format topic modeling results
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
                    # Write header
                    csv_writer.writerow(['Topic ID', 'Label', 'Term', 'Weight'])
                    
                    # Write data
                    for topic_id, topic_data in content['topics'].items():
                        for term, weight in topic_data['terms'].items():
                            csv_writer.writerow([topic_id, topic_data['label'], term, weight])
            
            elif isinstance(content, list):
                # Try to extract columns from first item
                if content and isinstance(content[0], dict):
                    keys = content[0].keys()
                    
                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=keys)
                        
                        # Write header
                        csv_writer.writeheader()
                        
                        # Write data
                        csv_writer.writerows(content)
                else:
                    # Simple list
                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        for item in content:
                            csv_writer.writerow([item])
            
            else:
                # Fallback to JSON
                logger.warning(f"Content type not suitable for CSV. Saving as JSON instead.")
                self._save_output(content, output_path, 'json')
                
        except ImportError:
            logger.warning("CSV module not available. Saving as text.")
            self._save_output(content, output_path, 'text')
    
    def _get_input_files(self, input_path: str) -> List[str]:
        """
        Get list of input files from path or pattern.
        
        Args:
            input_path: Input path or glob pattern
            
        Returns:
            List of file paths
        """
        # Check if input is a directory
        if os.path.isdir(input_path):
            # Get all files in directory
            return [os.path.join(input_path, f) for f in os.listdir(input_path) 
                   if os.path.isfile(os.path.join(input_path, f))]
        
        # Check if input is a glob pattern
        if '*' in input_path or '?' in input_path or '[' in input_path:
            return glob.glob(input_path)
        
        # Single file
        if os.path.isfile(input_path):
            return [input_path]
        
        # Invalid input
        logger.error(f"Input path not found: {input_path}")
        return []
    
    def _extract_date_from_filename(self, filename: str, pattern: str) -> Optional[str]:
        """
        Extract date from filename using regex pattern.
        
        Args:
            filename: Filename to extract date from
            pattern: Regex pattern with capture group for date
            
        Returns:
            Extracted date string or None if not found
        """
        import re
        
        try:
            match = re.search(pattern, filename)
            if match and match.groups():
                return match.group(1)
        except Exception as e:
            logger.error(f"Error extracting date from filename {filename}: {e}")
            
        return None
    
    def _extract_date_from_content(self, content: Any, date_field: str) -> Optional[str]:
        """
        Extract date from content using field name.
        
        Args:
            content: Content to extract date from
            date_field: Field name for date
            
        Returns:
            Extracted date string or None if not found
        """
        if isinstance(content, dict):
            # Direct field access
            if date_field in content:
                return str(content[date_field])
                
            # Nested field access (dot notation)
            if '.' in date_field:
                parts = date_field.split('.')
                value = content
                
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return None
                        
                return str(value)
        
        return None
    
    def _generate_sample_config(self) -> Dict:
        """
        Generate sample configuration.
        
        Returns:
            Sample configuration dictionary
        """
        return {
            'summary': {
                'engine': 'advanced',
                'max_tokens': 200,
                'min_tokens': 50,
                'levels': ['tag', 'sentence', 'paragraph'],
                'extract_entities': True,
                'extract_topics': False
            },
            'topics': {
                'algorithm': 'lda',
                'num_topics': 5,
                'top_terms': 10,
                'auto_optimize': False,
                'min_df': 2,
                'max_df': 0.95
            },
            'graph': {
                'max_entities': 100,
                'min_edge_weight': 0.2,
                'use_entities': True,
                'use_topics': True,
                'layout': 'spring',
                'interactive': True
            },
            'temporal': {
                'granularity': 'month',
                'min_documents': 2,
                'smoothing_window': 1,
                'analyze_topics': True,
                'analyze_entities': True,
                'analyze_sentiment': True,
                'generate_report': True
            },
            'batch': {
                'workers': 4,
                'recursive': False,
                'skip_errors': True,
                'output_format': 'json'
            },
            'output': {
                'directory': 'output',
                'format': 'json',
                'overwrite': True
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            # Determine file format
            ext = os.path.splitext(config_path)[1].lower()
            
            if ext == '.json':
                # Load JSON
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif ext in ['.yaml', '.yml']:
                # Load YAML
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration format: {ext}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return {}
    
    def _validate_config(self, config: Dict) -> List[str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate summary configuration
        if 'summary' in config:
            summary_config = config['summary']
            
            if not isinstance(summary_config, dict):
                errors.append("'summary' must be a dictionary")
            else:
                # Validate fields
                if 'engine' in summary_config and summary_config['engine'] not in ['simple', 'advanced']:
                    errors.append("summary.engine must be 'simple' or 'advanced'")
                    
                if 'max_tokens' in summary_config and not isinstance(summary_config['max_tokens'], int):
                    errors.append("summary.max_tokens must be an integer")
                    
                if 'min_tokens' in summary_config and not isinstance(summary_config['min_tokens'], int):
                    errors.append("summary.min_tokens must be an integer")
                    
                if 'levels' in summary_config:
                    levels = summary_config['levels']
                    if not isinstance(levels, list):
                        errors.append("summary.levels must be a list")
                    else:
                        valid_levels = ['tag', 'sentence', 'paragraph']
                        for level in levels:
                            if level not in valid_levels:
                                errors.append(f"Invalid summary level: {level}. Must be one of {valid_levels}")
        
        # Validate topics configuration
        if 'topics' in config:
            topics_config = config['topics']
            
            if not isinstance(topics_config, dict):
                errors.append("'topics' must be a dictionary")
            else:
                # Validate fields
                if 'algorithm' in topics_config and topics_config['algorithm'] not in ['lda', 'nmf', 'lsa']:
                    errors.append("topics.algorithm must be 'lda', 'nmf', or 'lsa'")
                    
                if 'num_topics' in topics_config:
                    if not isinstance(topics_config['num_topics'], int):
                        errors.append("topics.num_topics must be an integer")
                    elif topics_config['num_topics'] < 1:
                        errors.append("topics.num_topics must be at least 1")
                        
                if 'top_terms' in topics_config:
                    if not isinstance(topics_config['top_terms'], int):
                        errors.append("topics.top_terms must be an integer")
                    elif topics_config['top_terms'] < 1:
                        errors.append("topics.top_terms must be at least 1")
        
        # Validate graph configuration
        if 'graph' in config:
            graph_config = config['graph']
            
            if not isinstance(graph_config, dict):
                errors.append("'graph' must be a dictionary")
            else:
                # Validate fields
                if 'max_entities' in graph_config:
                    if not isinstance(graph_config['max_entities'], int):
                        errors.append("graph.max_entities must be an integer")
                    elif graph_config['max_entities'] < 1:
                        errors.append("graph.max_entities must be at least 1")
                        
                if 'min_edge_weight' in graph_config:
                    if not isinstance(graph_config['min_edge_weight'], (int, float)):
                        errors.append("graph.min_edge_weight must be a number")
                    elif graph_config['min_edge_weight'] < 0:
                        errors.append("graph.min_edge_weight must be non-negative")
                        
                if 'layout' in graph_config and graph_config['layout'] not in ['spring', 'circular', 'kamada_kawai', 'spectral']:
                    errors.append("graph.layout must be 'spring', 'circular', 'kamada_kawai', or 'spectral'")
        
        # Validate temporal configuration
        if 'temporal' in config:
            temporal_config = config['temporal']
            
            if not isinstance(temporal_config, dict):
                errors.append("'temporal' must be a dictionary")
            else:
                # Validate fields
                if 'granularity' in temporal_config and temporal_config['granularity'] not in ['day', 'week', 'month', 'year']:
                    errors.append("temporal.granularity must be 'day', 'week', 'month', or 'year'")
                    
                if 'min_documents' in temporal_config:
                    if not isinstance(temporal_config['min_documents'], int):
                        errors.append("temporal.min_documents must be an integer")
                    elif temporal_config['min_documents'] < 1:
                        errors.append("temporal.min_documents must be at least 1")
                        
                if 'smoothing_window' in temporal_config:
                    if not isinstance(temporal_config['smoothing_window'], int):
                        errors.append("temporal.smoothing_window must be an integer")
                    elif temporal_config['smoothing_window'] < 0:
                        errors.append("temporal.smoothing_window must be non-negative")
        
        return errors
    
    def run_summarize(self, args):
        """
        Run summarize command.
        
        Args:
            args: Command arguments
        """
        # Get input files
        input_files = self._get_input_files(args.input)
        
        if not input_files:
            logger.error(f"No input files found for path: {args.input}")
            return
        
        # Initialize summarizer
        if args.advanced:
            summarizer = AdvancedSUM(use_entities=args.entities)
            logger.info("Using advanced summarization engine")
        else:
            summarizer = SimpleSUM()
            logger.info("Using simple summarization engine")
            
            if args.entities:
                logger.warning("Entity extraction is only available with advanced summarization. Ignoring --entities.")
        
        # Create output path if not provided
        if not args.output:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename based on input
            input_name = os.path.splitext(os.path.basename(input_files[0]))[0]
            
            if args.format == 'json':
                args.output = os.path.join(output_dir, f"{input_name}_summary.json")
            elif args.format == 'markdown':
                args.output = os.path.join(output_dir, f"{input_name}_summary.md")
            elif args.format == 'html':
                args.output = os.path.join(output_dir, f"{input_name}_summary.html")
            else:
                args.output = os.path.join(output_dir, f"{input_name}_summary.txt")
        
        # Parse summary levels
        summary_levels = [level.strip() for level in args.levels.split(',')]
        
        # Process input
        if len(input_files) == 1:
            # Single file
            logger.info(f"Summarizing file: {input_files[0]}")
            
            try:
                # Load input
                text = self._load_text_input(input_files[0])
                
                # Configure summarization
                config = {
                    'max_tokens': args.max_tokens,
                    'min_tokens': args.min_tokens,
                    'summary_levels': summary_levels
                }
                
                # Summarize
                start_time = time.time()
                result = summarizer.process_text(text, config)
                processing_time = time.time() - start_time
                
                # Add processing time to result
                if 'metadata' not in result:
                    result['metadata'] = {}
                    
                result['metadata']['processing_time'] = processing_time
                
                # Log results
                logger.info(f"Summarization completed in {processing_time:.2f}s")
                
                if 'summary' in result:
                    summary_length = len(result['summary'].split())
                    original_length = result['metadata'].get('words', 0)
                    
                    logger.info(f"Summary length: {summary_length} words")
                    
                    if original_length > 0:
                        compression_ratio = summary_length / original_length
                        logger.info(f"Compression ratio: {compression_ratio:.2f}")
                
                # Save output
                self._save_output(result, args.output, args.format)
                
            except Exception as e:
                logger.error(f"Error summarizing file: {e}")
                return
                
        else:
            # Multiple files - combine text
            logger.info(f"Summarizing {len(input_files)} files")
            
            try:
                # Load and combine all files
                combined_text = ""
                for file_path in input_files:
                    try:
                        text = self._load_text_input(file_path)
                        combined_text += text + "\n\n"
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
                
                # Configure summarization
                config = {
                    'max_tokens': args.max_tokens,
                    'min_tokens': args.min_tokens,
                    'summary_levels': summary_levels
                }
                
                # Summarize
                start_time = time.time()
                result = summarizer.process_text(combined_text, config)
                processing_time = time.time() - start_time
                
                # Add processing time to result
                if 'metadata' not in result:
                    result['metadata'] = {}
                    
                result['metadata']['processing_time'] = processing_time
                
                # Log results
                logger.info(f"Summarization completed in {processing_time:.2f}s")
                
                if 'summary' in result:
                    summary_length = len(result['summary'].split())
                    original_length = result['metadata'].get('words', 0)
                    
                    logger.info(f"Summary length: {summary_length} words")
                    
                    if original_length > 0:
                        compression_ratio = summary_length / original_length
                        logger.info(f"Compression ratio: {compression_ratio:.2f}")
                
                # Save output
                self._save_output(result, args.output, args.format)
                
            except Exception as e:
                logger.error(f"Error summarizing files: {e}")
                return
    
    def run_topics(self, args):
        """
        Run topics command.
        
        Args:
            args: Command arguments
        """
        # Get input files
        input_files = self._get_input_files(args.input)
        
        if not input_files:
            logger.error(f"No input files found for path: {args.input}")
            return
        
        # Create output path if not provided
        if not args.output:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename based on input
            input_name = os.path.splitext(os.path.basename(input_files[0]))[0]
            
            if args.format == 'json':
                args.output = os.path.join(output_dir, f"{input_name}_topics.json")
            elif args.format == 'csv':
                args.output = os.path.join(output_dir, f"{input_name}_topics.csv")
            elif args.format == 'markdown':
                args.output = os.path.join(output_dir, f"{input_name}_topics.md")
            elif args.format == 'html':
                args.output = os.path.join(output_dir, f"{input_name}_topics.html")
        
        # Load documents
        documents = []
        
        for file_path in input_files:
            try:
                text = self._load_text_input(file_path)
                documents.append(text)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        if not documents:
            logger.error("No valid documents found")
            return
        
        # Initialize topic modeler
        logger.info(f"Initializing topic modeler with algorithm={args.algorithm}, num_topics={args.num_topics}")
        
        topic_modeler = TopicModeler(
            n_topics=args.num_topics,
            algorithm=args.algorithm,
            n_top_words=args.top_terms,
            auto_optimize=args.auto_optimize
        )
        
        # Extract topics
        try:
            logger.info(f"Extracting topics from {len(documents)} documents")
            
            start_time = time.time()
            topic_modeler.fit(documents)
            topics = topic_modeler.get_topics_summary()
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'topics': topics,
                'algorithm': args.algorithm,
                'num_topics': args.num_topics,
                'num_documents': len(documents),
                'processing_time': processing_time
            }
            
            # Generate visualization if requested
            if args.visualize:
                try:
                    # Generate topic visualization
                    viz_dir = os.path.join(self.output_dir, 'visualizations')
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    viz_path = os.path.join(viz_dir, f"topics_{int(time.time())}.png")
                    
                    # Simple visualization
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Set up figure
                    plt.figure(figsize=(12, 8))
                    
                    # Set up grid for multiple plots
                    n_topics = len(topics)
                    cols = min(3, n_topics)
                    rows = (n_topics + cols - 1) // cols  # Ceiling division
                    
                    # Plot each topic as word importance
                    for i, (topic_id, topic_data) in enumerate(topics.items()):
                        # Extract terms and weights
                        terms = list(topic_data['terms'].keys())[:10]  # Top 10 terms
                        weights = list(topic_data['terms'].values())[:10]
                        
                        # Sort by weight
                        term_weights = sorted(zip(terms, weights), key=lambda x: x[1], reverse=True)
                        terms = [term for term, _ in term_weights]
                        weights = [weight for _, weight in term_weights]
                        
                        # Plot
                        plt.subplot(rows, cols, i + 1)
                        y_pos = np.arange(len(terms))
                        plt.barh(y_pos, weights, align='center')
                        plt.yticks(y_pos, terms)
                        plt.xlabel('Weight')
                        plt.title(f"Topic {topic_id}: {topic_data['label']}")
                        plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(viz_path, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Topic visualization saved to {viz_path}")
                    result['visualization'] = viz_path
                    
                except Exception as e:
                    logger.error(f"Error generating topic visualization: {e}")
            
            # Save output
            self._save_output(result, args.output, args.format)
            
            logger.info(f"Topic extraction completed in {processing_time:.2f}s")
            logger.info(f"Extracted {len(topics)} topics")
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return
    
    def run_graph(self, args):
        """
        Run graph command.
        
        Args:
            args: Command arguments
        """
        # Get input files
        input_files = self._get_input_files(args.input)
        
        if not input_files:
            logger.error(f"No input files found for path: {args.input}")
            return
        
        # Create output path if not provided
        if not args.output:
            output_dir = os.path.join(self.output_dir, 'graphs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename based on input
            input_name = os.path.splitext(os.path.basename(input_files[0]))[0]
            
            if args.format == 'html' or args.interactive:
                args.output = os.path.join(output_dir, f"{input_name}_graph.html")
            elif args.format in ['png', 'svg']:
                args.output = os.path.join(output_dir, f"{input_name}_graph.{args.format}")
            else:
                args.output = os.path.join(output_dir, f"{input_name}_graph.{args.format}")
        
        # Initialize knowledge graph
        kg = KnowledgeGraph(
            output_dir=self.output_dir,
            max_entities=args.max_entities,
            min_edge_weight=args.min_edge_weight
        )
        
        # Process based on input type
        try:
            if len(input_files) == 1 and input_files[0].endswith('.json'):
                # Try to load JSON data
                try:
                    data = self._load_json_input(input_files[0])
                    
                    # Check if it's a saved knowledge graph
                    if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                        # Load directly
                        logger.info("Loading pre-built knowledge graph from JSON")
                        kg = KnowledgeGraph.load(input_files[0], output_dir=self.output_dir)
                        
                        if not kg:
                            logger.error("Failed to load knowledge graph from JSON")
                            return
                    else:
                        # Build from JSON data
                        logger.info("Building knowledge graph from JSON data")
                        
                        # Check for entity data
                        if args.entities:
                            # Extract entities
                            logger.info("Extracting entities")
                            
                            summarizer = AdvancedSUM(use_entities=True)
                            
                            if 'entries' in data and isinstance(data['entries'], list):
                                # Process each entry
                                for entry in data['entries']:
                                    if isinstance(entry, dict) and 'content' in entry:
                                        content = entry['content']
                                        result = summarizer.process_text(content)
                                        
                                        if 'entities' in result:
                                            kg.build_from_entities(result['entities'])
                            else:
                                # Process whole JSON as text
                                json_text = json.dumps(data)
                                result = summarizer.process_text(json_text)
                                
                                if 'entities' in result:
                                    kg.build_from_entities(result['entities'])
                        
                        # Check for topic data
                        if args.topics:
                            # Extract topics
                            logger.info("Extracting topics")
                            
                            # Collect text for topic modeling
                            documents = []
                            
                            if 'entries' in data and isinstance(data['entries'], list):
                                # Process each entry
                                for entry in data['entries']:
                                    if isinstance(entry, dict) and 'content' in entry:
                                        documents.append(entry['content'])
                            else:
                                # Use whole JSON as text
                                documents.append(json.dumps(data))
                            
                            # Extract topics
                            if documents:
                                topic_modeler = TopicModeler(n_topics=5)
                                topic_modeler.fit(documents)
                                topics = []
                                
                                for topic_idx in range(topic_modeler.n_topics):
                                    terms = topic_modeler.get_topic_terms(topic_idx)
                                    
                                    topic_terms = []
                                    for term, weight in terms:
                                        topic_terms.append({
                                            'term': term,
                                            'weight': float(weight)
                                        })
                                    
                                    topics.append({
                                        'id': topic_idx,
                                        'label': f"Topic {topic_idx + 1}",
                                        'terms': topic_terms
                                    })
                                
                                kg.add_topics(topics)
                except Exception as e:
                    logger.error(f"Error processing JSON file: {e}")
                    return
            else:
                # Process text files
                logger.info(f"Processing {len(input_files)} text files")
                
                # Collect all text
                all_text = ""
                for file_path in input_files:
                    try:
                        text = self._load_text_input(file_path)
                        all_text += text + "\n\n"
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
                
                if not all_text:
                    logger.error("No valid text found in input files")
                    return
                
                # Extract entities
                if args.entities:
                    logger.info("Extracting entities from text")
                    
                    summarizer = AdvancedSUM(use_entities=True)
                    result = summarizer.process_text(all_text)
                    
                    if 'entities' in result:
                        kg.build_from_entities(result['entities'])
                
                # Extract topics
                if args.topics:
                    logger.info("Extracting topics from text")
                    
                    topic_modeler = TopicModeler(n_topics=5)
                    topic_modeler.fit([all_text])
                    topics = []
                    
                    for topic_idx in range(topic_modeler.n_topics):
                        terms = topic_modeler.get_topic_terms(topic_idx)
                        
                        topic_terms = []
                        for term, weight in terms:
                            topic_terms.append({
                                'term': term,
                                'weight': float(weight)
                            })
                        
                        topics.append({
                            'id': topic_idx,
                            'label': f"Topic {topic_idx + 1}",
                            'terms': topic_terms
                        })
                    
                    kg.add_topics(topics)
            
            # Generate visualization
            if args.interactive or args.format == 'html':
                logger.info("Generating interactive HTML visualization")
                output_path = kg.generate_html_visualization(output_path=args.output)
            else:
                logger.info(f"Generating {args.format} visualization")
                
                if args.format in ['png', 'svg']:
                    # Static image
                    output_path = kg.visualize(
                        output_path=args.output,
                        layout=args.layout
                    )
                else:
                    # Export data format
                    output_path = kg.export_graph(
                        format=args.format,
                        output_path=args.output
                    )
            
            if output_path:
                logger.info(f"Knowledge graph output saved to {output_path}")
            else:
                logger.error("Failed to generate knowledge graph output")
                
        except Exception as e:
            logger.error(f"Error generating knowledge graph: {e}")
            return
    
    def run_temporal(self, args):
        """
        Run temporal command.
        
        Args:
            args: Command arguments
        """
        # Get input files
        input_files = self._get_input_files(args.input)
        
        if not input_files:
            logger.error(f"No input files found for path: {args.input}")
            return
        
        # Create output path if not provided
        if not args.output:
            output_dir = os.path.join(self.output_dir, 'temporal')
            os.makedirs(output_dir, exist_ok=True)
            
            if args.report or args.format == 'html':
                args.output = os.path.join(output_dir, f"evolution_report_{int(time.time())}.html")
            elif args.format == 'json':
                args.output = os.path.join(output_dir, f"temporal_analysis_{int(time.time())}.json")
            elif args.format == 'markdown':
                args.output = os.path.join(output_dir, f"temporal_analysis_{int(time.time())}.md")
        
        # Initialize temporal analyzer
        analyzer = TemporalAnalysis(
            output_dir=self.output_dir,
            time_granularity=args.period,
            min_documents=args.min_documents,
            smoothing_window=args.smoothing
        )
        
        # Process files
        logger.info(f"Processing {len(input_files)} files for temporal analysis")
        
        for file_path in input_files:
            try:
                # Determine file type
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.json':
                    # Process JSON file
                    data = self._load_json_input(file_path)
                    
                    # Extract date from content
                    date_str = self._extract_date_from_content(data, args.date_field)
                    
                    if not date_str and args.filename_pattern:
                        # Try to extract date from filename
                        date_str = self._extract_date_from_filename(
                            os.path.basename(file_path), args.filename_pattern)
                    
                    if not date_str:
                        logger.warning(f"Could not extract date from {file_path}, skipping")
                        continue
                    
                    # Extract text content
                    text = ""
                    
                    if 'content' in data:
                        text = data['content']
                    elif 'text' in data:
                        text = data['text']
                    elif 'entries' in data and isinstance(data['entries'], list):
                        # Combine entry content
                        for entry in data['entries']:
                            if isinstance(entry, dict) and 'content' in entry:
                                text += entry['content'] + "\n\n"
                    else:
                        # Convert whole JSON to text
                        text = json.dumps(data)
                    
                    # Add document with date
                    analyzer.add_document(text, date_str, args.date_format)
                    
                else:
                    # Process text file
                    text = self._load_text_input(file_path)
                    
                    # Try to extract date from filename
                    if args.filename_pattern:
                        date_str = self._extract_date_from_filename(
                            os.path.basename(file_path), args.filename_pattern)
                        
                        if date_str:
                            # Add document with date
                            analyzer.add_document(text, date_str, args.date_format)
                        else:
                            logger.warning(f"Could not extract date from filename {file_path}, skipping")
                    else:
                        logger.warning(f"No filename pattern provided for {file_path}, skipping")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Check if we have documents
        if not analyzer.documents:
            logger.error("No valid documents found for temporal analysis")
            return
        
        logger.info(f"Added {len(analyzer.documents)} documents to temporal analysis")
        
        # Perform analysis
        result = {}
        
        if args.report:
            # Generate comprehensive report
            logger.info("Generating comprehensive evolution report")
            
            report_path = analyzer.create_evolution_report(
                output_path=args.output,
                include_topics=args.topics,
                include_entities=args.entities,
                include_sentiment=args.sentiment
            )
            
            if report_path:
                logger.info(f"Evolution report saved to {report_path}")
                result['report_path'] = report_path
            else:
                logger.error("Failed to generate evolution report")
                
        else:
            # Perform individual analyses
            if args.topics:
                logger.info("Analyzing topic evolution")
                
                topic_evolution = analyzer.analyze_topic_evolution()
                result['topic_evolution'] = topic_evolution
                
                # Visualize
                viz_path = analyzer.visualize_topic_evolution(topic_evolution)
                if viz_path:
                    result['topic_visualization'] = viz_path
            
            if args.entities:
                logger.info("Analyzing entity evolution")
                
                entity_evolution = analyzer.analyze_entity_evolution()
                result['entity_evolution'] = entity_evolution
                
                # Visualize
                viz_path = analyzer.visualize_entity_evolution(entity_evolution)
                if viz_path:
                    result['entity_visualization'] = viz_path
            
            if args.sentiment:
                logger.info("Analyzing sentiment evolution")
                
                sentiment_evolution = analyzer.analyze_sentiment_evolution()
                result['sentiment_evolution'] = sentiment_evolution
                
                # Visualize
                viz_path = analyzer.visualize_sentiment_evolution(sentiment_evolution)
                if viz_path:
                    result['sentiment_visualization'] = viz_path
            
            # Save results
            if result and not args.report:
                self._save_output(result, args.output, args.format)
                logger.info(f"Temporal analysis results saved to {args.output}")
    
    def run_batch(self, args):
        """
        Run batch command.
        
        Args:
            args: Command arguments
        """
        # Get input files
        input_files = self._get_input_files(args.input)
        
        if not input_files:
            logger.error(f"No input files found for path: {args.input}")
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load configuration if provided
        config = {}
        if args.config:
            config = self._load_config(args.config)
            
            # Validate config
            errors = self._validate_config(config)
            if errors:
                logger.error("Configuration validation errors:")
                for error in errors:
                    logger.error(f"  - {error}")
                return
        
        # Determine operations
        operations = []
        
        if args.summarize or (not args.summarize and not args.topics and not args.graph):
            # Default to summarize if no operations specified
            operations.append('summarize')
        
        if args.topics:
            operations.append('topics')
        
        if args.graph:
            operations.append('graph')
        
        logger.info(f"Batch processing {len(input_files)} files with operations: {', '.join(operations)}")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            
            for file_path in input_files:
                future = executor.submit(
                    self._process_batch_file, 
                    file_path, 
                    args.output_dir, 
                    operations, 
                    config, 
                    args.format
                )
                futures.append(future)
            
            # Wait for completion and collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
        
        # Summarize results
        logger.info(f"Batch processing completed for {len(results)} files")
        
        # Generate batch summary
        summary = {
            'total_files': len(input_files),
            'processed_files': len(results),
            'operations': operations,
            'output_directory': args.output_dir,
            'results': results
        }
        
        # Save summary
        summary_path = os.path.join(args.output_dir, 'batch_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Batch summary saved to {summary_path}")
    
    def _process_batch_file(self, 
                          file_path: str, 
                          output_dir: str, 
                          operations: List[str], 
                          config: Dict, 
                          output_format: str) -> Dict:
        """
        Process a single file in batch mode.
        
        Args:
            file_path: Path to the file
            output_dir: Output directory
            operations: List of operations to perform
            config: Configuration dictionary
            output_format: Output format
            
        Returns:
            Result dictionary
        """
        result = {
            'file': file_path,
            'operations': {},
            'success': False,
            'error': None
        }
        
        try:
            # Load file
            text = self._load_text_input(file_path)
            
            # Generate base output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Perform operations
            for operation in operations:
                if operation == 'summarize':
                    # Summarize
                    output_path = os.path.join(output_dir, f"{base_name}_summary.{output_format}")
                    
                    # Get config
                    summarizer_config = {}
                    if 'summary' in config:
                        summarizer_config = config['summary']
                    
                    # Initialize summarizer
                    use_advanced = summarizer_config.get('engine', 'simple') == 'advanced'
                    extract_entities = summarizer_config.get('extract_entities', False)
                    
                    if use_advanced:
                        summarizer = AdvancedSUM(use_entities=extract_entities)
                    else:
                        summarizer = SimpleSUM()
                    
                    # Configure
                    summary_config = {
                        'max_tokens': summarizer_config.get('max_tokens', 150),
                        'min_tokens': summarizer_config.get('min_tokens', 50),
                        'summary_levels': summarizer_config.get('levels', ['tag', 'sentence', 'paragraph'])
                    }
                    
                    # Summarize
                    summary_result = summarizer.process_text(text, summary_config)
                    
                    # Save output
                    self._save_output(summary_result, output_path, output_format)
                    
                    # Record result
                    result['operations']['summarize'] = {
                        'output': output_path,
                        'success': True
                    }
                
                elif operation == 'topics':
                    # Topic modeling
                    output_path = os.path.join(output_dir, f"{base_name}_topics.{output_format}")
                    
                    # Get config
                    topic_config = {}
                    if 'topics' in config:
                        topic_config = config['topics']
                    
                    # Initialize topic modeler
                    topic_modeler = TopicModeler(
                        n_topics=topic_config.get('num_topics', 5),
                        algorithm=topic_config.get('algorithm', 'lda'),
                        n_top_words=topic_config.get('top_terms', 10),
                        auto_optimize=topic_config.get('auto_optimize', False)
                    )
                    
                    # Extract topics
                    topic_modeler.fit([text])
                    topics = topic_modeler.get_topics_summary()
                    
                    # Prepare result
                    topics_result = {
                        'topics': topics,
                        'algorithm': topic_config.get('algorithm', 'lda'),
                        'num_topics': topic_config.get('num_topics', 5)
                    }
                    
                    # Save output
                    self._save_output(topics_result, output_path, output_format)
                    
                    # Record result
                    result['operations']['topics'] = {
                        'output': output_path,
                        'success': True
                    }
                
                elif operation == 'graph':
                    # Knowledge graph
                    output_path = os.path.join(output_dir, f"{base_name}_graph.html")
                    
                    # Get config
                    graph_config = {}
                    if 'graph' in config:
                        graph_config = config['graph']
                    
                    # Initialize knowledge graph
                    kg = KnowledgeGraph(
                        output_dir=output_dir,
                        max_entities=graph_config.get('max_entities', 100),
                        min_edge_weight=graph_config.get('min_edge_weight', 0.1)
                    )
                    
                    # Extract entities
                    if graph_config.get('use_entities', True):
                        summarizer = AdvancedSUM(use_entities=True)
                        entity_result = summarizer.process_text(text)
                        
                        if 'entities' in entity_result:
                            kg.build_from_entities(entity_result['entities'])
                    
                    # Extract topics
                    if graph_config.get('use_topics', True):
                        topic_modeler = TopicModeler(n_topics=5)
                        topic_modeler.fit([text])
                        topics = []
                        
                        for topic_idx in range(topic_modeler.n_topics):
                            terms = topic_modeler.get_topic_terms(topic_idx)
                            
                            topic_terms = []
                            for term, weight in terms:
                                topic_terms.append({
                                    'term': term,
                                    'weight': float(weight)
                                })
                            
                            topics.append({
                                'id': topic_idx,
                                'label': f"Topic {topic_idx + 1}",
                                'terms': topic_terms
                            })
                        
                        kg.add_topics(topics)
                    
                    # Generate visualization
                    if graph_config.get('interactive', True):
                        # Interactive HTML
                        graph_path = kg.generate_html_visualization(output_path=output_path)
                    else:
                        # Static image
                        graph_path = kg.visualize(
                            output_path=output_path.replace('.html', '.png'),
                            layout=graph_config.get('layout', 'spring')
                        )
                    
                    # Record result
                    if graph_path:
                        result['operations']['graph'] = {
                            'output': graph_path,
                            'success': True
                        }
                    else:
                        result['operations']['graph'] = {
                            'success': False,
                            'error': 'Failed to generate graph'
                        }
            
            # Mark overall success
            result['success'] = all(op['success'] for op in result['operations'].values())
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            logger.error(f"Error processing file {file_path}: {e}")
            
        return result
    
    def run_config(self, args):
        """
        Run config command.
        
        Args:
            args: Command arguments
        """
        if args.generate:
            # Generate sample configuration
            config = self._generate_sample_config()
            
            try:
                # Determine format
                ext = os.path.splitext(args.output)[1].lower()
                
                if ext == '.json':
                    # Save as JSON
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                elif ext in ['.yaml', '.yml']:
                    # Save as YAML
                    import yaml
                    with open(args.output, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False)
                else:
                    # Default to YAML
                    import yaml
                    with open(args.output, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False)
                
                logger.info(f"Sample configuration saved to {args.output}")
                
            except Exception as e:
                logger.error(f"Error generating configuration: {e}")
                return
        
        if args.validate:
            # Validate configuration
            config = self._load_config(args.validate)
            
            if not config:
                logger.error(f"Failed to load configuration from {args.validate}")
                return
                
            # Validate
            errors = self._validate_config(config)
            
            if errors:
                logger.error(f"Configuration validation errors in {args.validate}:")
                for error in errors:
                    logger.error(f"  - {error}")
            else:
                logger.info(f"Configuration {args.validate} is valid")
    
    def run(self):
        """Parse arguments and run the appropriate command."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Run the appropriate command
        if args.command == 'summarize':
            self.run_summarize(args)
        elif args.command == 'topics':
            self.run_topics(args)
        elif args.command == 'graph':
            self.run_graph(args)
        elif args.command == 'temporal':
            self.run_temporal(args)
        elif args.command == 'batch':
            self.run_batch(args)
        elif args.command == 'config':
            self.run_config(args)
        else:
            self.parser.print_help()


if __name__ == '__main__':
    # Create and run CLI
    cli = SumCLI()
    cli.run()
