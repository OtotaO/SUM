#!/usr/bin/env python3
"""
sum_cli.py - Enhanced Command Line Interface for SUM

Provides a powerful CLI for all SUM operations with rich formatting,
progress indicators, and intuitive commands.

Author: SUM Development Team
License: Apache License 2.0
"""

import click
import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml
from datetime import datetime

# Rich formatting for better CLI experience
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Install 'rich' for better CLI experience: pip install rich")

# Import SUM components
from SUM import SimpleSUM, HierarchicalDensificationEngine
from invisible_ai_engine import InvisibleAI
from notes_engine import SimpleNotes
from multimodal_engine import MultiModalEngine
from config_system import ConfigManager, get_config

# Initialize console
console = Console() if RICH_AVAILABLE else None


@click.group()
@click.version_option(version='1.0.0', prog_name='SUM CLI')
@click.pass_context
def cli(ctx):
    """SUM - Intelligence Amplification System CLI
    
    Transform information into understanding with simple commands.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = get_config()


@cli.command()
@click.argument('text', required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='Input file path')
@click.option('--length', '-l', type=click.Choice(['brief', 'standard', 'detailed']), 
              default='standard', help='Summary length')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['text', 'json', 'yaml']), 
              default='text', help='Output format')
@click.pass_context
def summarize(ctx, text, file, length, output, format):
    """Summarize text or file content."""
    # Get input text
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        source = f"File: {Path(file).name}"
    elif text:
        content = text
        source = "Direct input"
    else:
        # Read from stdin
        content = sys.stdin.read()
        source = "Standard input"
    
    if not content.strip():
        click.echo("Error: No content provided", err=True)
        sys.exit(1)
    
    # Show processing message
    if RICH_AVAILABLE:
        with console.status("[bold green]Processing content...") as status:
            result = _process_summary(content, length)
    else:
        click.echo("Processing content...")
        result = _process_summary(content, length)
    
    # Format output
    if format == 'json':
        output_text = json.dumps(result, indent=2)
    elif format == 'yaml':
        output_text = yaml.dump(result, default_flow_style=False)
    else:
        output_text = _format_summary_text(result, source)
    
    # Output results
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        click.echo(f"✓ Summary saved to: {output}")
    else:
        if RICH_AVAILABLE and format == 'text':
            console.print(output_text)
        else:
            click.echo(output_text)


def _process_summary(content: str, length: str) -> Dict[str, Any]:
    """Process content and generate summary."""
    engine = HierarchicalDensificationEngine()
    
    # Map length to summary sentences
    length_map = {
        'brief': 1,
        'standard': 3,
        'detailed': 5
    }
    
    result = engine.process_text(content)
    
    # Extract relevant parts based on length
    summary_data = {
        'summary': result.get('summary', ''),
        'keywords': result.get('keywords', [])[:10],
        'insights': result.get('key_insights', [])[:3],
        'processing_time': result.get('processing_time', 0),
        'word_count': len(content.split()),
        'compression_ratio': len(content) / len(result.get('summary', 'x'))
    }
    
    if length == 'detailed':
        summary_data['hierarchical'] = result.get('hierarchical_summary', {})
    
    return summary_data


def _format_summary_text(result: Dict[str, Any], source: str) -> str:
    """Format summary result as text."""
    if RICH_AVAILABLE:
        # Create formatted output with Rich
        output = Panel(f"[bold blue]Summary Results[/bold blue]\n[dim]{source}[/dim]")
        
        # Main summary
        summary_panel = Panel(
            result['summary'],
            title="Summary",
            border_style="green"
        )
        
        # Keywords
        keywords = ", ".join(f"[cyan]{kw}[/cyan]" for kw in result['keywords'])
        
        # Insights
        insights_text = "\n".join(f"• {insight}" for insight in result['insights'])
        
        # Stats
        stats = f"""
[dim]Word Count:[/dim] {result['word_count']:,}
[dim]Compression:[/dim] {result['compression_ratio']:.1f}x
[dim]Processing:[/dim] {result['processing_time']:.2f}s
"""
        
        return f"{output}\n\n{summary_panel}\n\n[bold]Keywords:[/bold] {keywords}\n\n[bold]Insights:[/bold]\n{insights_text}\n{stats}"
    else:
        # Plain text format
        output = f"""
=== Summary Results ===
Source: {source}

Summary:
{result['summary']}

Keywords: {', '.join(result['keywords'])}

Insights:
{chr(10).join('• ' + insight for insight in result['insights'])}

Stats:
- Word Count: {result['word_count']:,}
- Compression: {result['compression_ratio']:.1f}x
- Processing: {result['processing_time']:.2f}s
"""
        return output


@cli.command()
@click.argument('content')
@click.option('--title', '-t', help='Note title')
@click.option('--policy', '-p', type=click.Choice(['diary', 'ideas', 'meeting', 'research', 'todo']),
              help='Note policy type')
@click.option('--tags', '-g', multiple=True, help='Manual tags')
@click.pass_context
def note(ctx, content, title, policy, tags):
    """Add a note with intelligent processing."""
    notes = SimpleNotes()
    
    # Add the note
    note_id = notes.note(content, title=title or "", policy=policy)
    
    # Get the note details
    recent = notes.recent(1)
    if recent:
        note_data = recent[0]
        
        if RICH_AVAILABLE:
            # Format with Rich
            table = Table(title="Note Added Successfully")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("ID", note_data['id'])
            table.add_row("Title", note_data['title'])
            table.add_row("Policy", note_data.get('policy_tag', 'general'))
            table.add_row("Auto Tags", ", ".join(note_data.get('auto_tags', [])))
            table.add_row("Importance", f"{note_data.get('importance', 0):.2f}")
            
            console.print(table)
        else:
            click.echo(f"✓ Note added: {note_id}")
            click.echo(f"  Title: {note_data['title']}")
            click.echo(f"  Policy: {note_data.get('policy_tag', 'general')}")
            click.echo(f"  Tags: {', '.join(note_data.get('auto_tags', []))}")


@cli.command()
@click.argument('query')
@click.option('--policy', '-p', help='Filter by policy type')
@click.option('--limit', '-n', default=10, help='Number of results')
@click.pass_context
def search(ctx, query, policy, limit):
    """Search notes by content, tags, or concepts."""
    notes = SimpleNotes()
    
    # Search notes
    results = notes.search(query)
    
    # Filter by policy if specified
    if policy:
        results = [r for r in results if r.get('policy_tag') == policy]
    
    # Limit results
    results = results[:limit]
    
    if not results:
        click.echo("No notes found matching your query.")
        return
    
    if RICH_AVAILABLE:
        # Create table
        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Date", style="cyan", width=12)
        table.add_column("Title", style="white", width=40)
        table.add_column("Policy", style="green", width=10)
        table.add_column("Tags", style="yellow", width=20)
        
        for result in results:
            date = datetime.fromisoformat(result['created_at']).strftime("%Y-%m-%d")
            title = result['title'][:40] + "..." if len(result['title']) > 40 else result['title']
            policy = result.get('policy_tag', 'general')
            tags = ", ".join(result.get('auto_tags', [])[:3])
            
            table.add_row(date, title, policy, tags)
        
        console.print(table)
    else:
        click.echo(f"\nSearch Results for '{query}':")
        click.echo("-" * 60)
        for i, result in enumerate(results, 1):
            date = datetime.fromisoformat(result['created_at']).strftime("%Y-%m-%d")
            click.echo(f"{i}. [{date}] {result['title']}")
            click.echo(f"   Policy: {result.get('policy_tag', 'general')}")
            click.echo(f"   Tags: {', '.join(result.get('auto_tags', []))}")


@cli.command()
@click.option('--policy', '-p', help='Filter by policy type')
@click.option('--days', '-d', default=30, help='Number of days to analyze')
@click.pass_context
def insights(ctx, policy, days):
    """Get analytical insights from your notes."""
    notes = SimpleNotes()
    
    # Get insights
    insights_data = notes.insights(policy=policy, days=days)
    
    if RICH_AVAILABLE:
        # Create formatted output
        title = f"Insights for {policy or 'all'} notes (last {days} days)"
        
        # Overview panel
        overview = f"""
[bold]Notes Analyzed:[/bold] {insights_data.get('total_notes_analyzed', 0)}
[bold]Date Range:[/bold] {insights_data.get('date_range', 'N/A')}
[bold]Policy Type:[/bold] {insights_data.get('policy_type', 'mixed')}
"""
        console.print(Panel(overview, title=title, border_style="blue"))
        
        # Key concepts
        if insights_data.get('key_concepts'):
            table = Table(title="Top Concepts")
            table.add_column("Concept", style="cyan")
            table.add_column("Frequency", style="white")
            
            for concept_data in insights_data['key_concepts'][:10]:
                table.add_row(
                    concept_data['concept'], 
                    str(concept_data['frequency'])
                )
            
            console.print(table)
        
        # Insights
        if insights_data.get('insights'):
            console.print("\n[bold]Key Insights:[/bold]")
            for insight in insights_data['insights']:
                console.print(f"• {insight}")
        
        # Emotional indicators (for diary)
        if insights_data.get('emotional_indicators'):
            console.print("\n[bold]Emotional Patterns:[/bold]")
            for emotion in insights_data['emotional_indicators']:
                console.print(f"• {emotion['emotion'].title()}: {emotion['frequency']} mentions")
    else:
        # Plain text output
        click.echo(f"\nInsights for {policy or 'all'} notes (last {days} days)")
        click.echo("=" * 60)
        click.echo(f"Notes Analyzed: {insights_data.get('total_notes_analyzed', 0)}")
        click.echo(f"Date Range: {insights_data.get('date_range', 'N/A')}")
        
        if insights_data.get('insights'):
            click.echo("\nKey Insights:")
            for insight in insights_data['insights']:
                click.echo(f"• {insight}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics and performance metrics."""
    config = ctx.obj['config']
    
    # Get various stats
    stats_data = {
        'environment': config.environment.value,
        'server': f"{config.server.host}:{config.server.port}",
        'ai_model': config.ai.default_model,
        'cache_enabled': config.performance.cache_enabled,
    }
    
    if RICH_AVAILABLE:
        # Create stats table
        table = Table(title="SUM System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in stats_data.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
        # Performance tips
        tips = Panel(
            """[bold yellow]Performance Tips:[/bold yellow]
• Enable caching for faster responses
• Use batch processing for multiple files
• Configure appropriate worker count
• Monitor memory usage for large documents""",
            border_style="yellow"
        )
        console.print(tips)
    else:
        click.echo("\nSUM System Statistics")
        click.echo("=" * 40)
        for key, value in stats_data.items():
            click.echo(f"{key.replace('_', ' ').title()}: {value}")


@cli.command()
@click.option('--check', is_flag=True, help='Check configuration')
@click.option('--generate', is_flag=True, help='Generate example config')
@click.option('--edit', is_flag=True, help='Edit configuration')
@click.pass_context
def config(ctx, check, generate, edit):
    """Manage SUM configuration."""
    config_manager = ctx.obj['config']
    
    if generate:
        from config_system import generate_example_config
        generate_example_config()
        click.echo("✓ Example configuration generated: config.example.yaml")
    
    elif check:
        config_dict = config_manager.get_config_dict()
        
        if RICH_AVAILABLE:
            # Pretty print with syntax highlighting
            yaml_str = yaml.dump(config_dict, default_flow_style=False)
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Current Configuration"))
        else:
            click.echo("\nCurrent Configuration:")
            click.echo(yaml.dump(config_dict, default_flow_style=False))
    
    elif edit:
        # Open config in default editor
        config_path = config_manager.config_path or "config.yaml"
        click.edit(filename=config_path)
        click.echo("✓ Configuration edited. Restart SUM to apply changes.")
    
    else:
        click.echo("Use --check, --generate, or --edit option")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--format', type=click.Choice(['text', 'json']), default='text')
@click.pass_context
def batch(ctx, input_path, output, format):
    """Process multiple files in batch mode."""
    input_path = Path(input_path)
    
    # Get list of files
    if input_path.is_dir():
        files = list(input_path.glob("**/*.txt")) + list(input_path.glob("**/*.md"))
    else:
        files = [input_path]
    
    if not files:
        click.echo("No files found to process.")
        return
    
    click.echo(f"Found {len(files)} files to process")
    
    # Process files
    results = []
    engine = HierarchicalDensificationEngine()
    
    with click.progressbar(files, label='Processing files') as bar:
        for file_path in bar:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                result = engine.process_text(content)
                results.append({
                    'file': str(file_path),
                    'summary': result.get('summary', ''),
                    'keywords': result.get('keywords', []),
                    'processing_time': result.get('processing_time', 0)
                })
            except Exception as e:
                click.echo(f"\nError processing {file_path}: {e}", err=True)
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            file_name = Path(result['file']).stem
            if format == 'json':
                output_file = output_path / f"{file_name}_summary.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                output_file = output_path / f"{file_name}_summary.txt"
                with open(output_file, 'w') as f:
                    f.write(f"Summary of {result['file']}\\n")
                    f.write("=" * 50 + "\\n\\n")
                    f.write(result['summary'] + "\\n\\n")
                    f.write(f"Keywords: {', '.join(result['keywords'])}\\n")
        
        click.echo(f"✓ Results saved to: {output_path}")
    else:
        # Display results
        for result in results:
            click.echo(f"\n--- {result['file']} ---")
            click.echo(result['summary'])
            click.echo(f"Keywords: {', '.join(result['keywords'])}")


@cli.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=3000, help='Server port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def serve(ctx, host, port, debug):
    """Start the SUM web server."""
    click.echo(f"Starting SUM server on {host}:{port}")
    
    # Import and run the main app
    os.environ['SUM_HOST'] = host
    os.environ['SUM_PORT'] = str(port)
    
    if debug:
        os.environ['SUM_ENV'] = 'development'
        os.environ['SUM_LOG_LEVEL'] = 'DEBUG'
    
    # Run the server
    from main import app
    app.run(host=host, port=port, debug=debug)


# Main entry point
if __name__ == '__main__':
    cli()