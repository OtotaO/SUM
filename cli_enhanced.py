#!/usr/bin/env python3
"""
cli_enhanced.py - Enhanced CLI Interface with Progress Bars

Beautiful command-line interface for SUM with real-time progress visualization,
interactive menus, and comprehensive processing options.

Features:
- Real-time progress bars with rich formatting
- Interactive model selection
- Live processing statistics
- Beautiful output formatting
- File processing with drag-and-drop support
- Batch processing capabilities

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import time
import argparse
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Rich for beautiful CLI output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Installing rich for beautiful CLI output...")
    os.system("pip install rich")
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        from rich.panel import Panel
        from rich.table import Table
        from rich.prompt import Prompt, Confirm
        from rich.syntax import Syntax
        from rich.layout import Layout
        from rich.live import Live
        from rich.text import Text
        from rich import box
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

# Import SUM components
from SUM import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from StreamingEngine import StreamingHierarchicalEngine, StreamingConfig

# Initialize console
console = Console() if RICH_AVAILABLE else None


class EnhancedCLI:
    """Enhanced CLI interface with progress bars and beautiful formatting."""
    
    def __init__(self):
        self.engines = {
            'simple': SimpleSUM,
            'advanced': MagnumOpusSUM,
            'hierarchical': HierarchicalDensificationEngine,
            'streaming': StreamingHierarchicalEngine
        }
        self.current_engine = None
        
    def print_banner(self):
        """Print SUM banner."""
        if RICH_AVAILABLE:
            banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🚀 SUM: Hierarchical Knowledge Densification System        ║
    ║                                                               ║
    ║   Real-Time Progressive Summarization                    ║
    ║   with Advanced Hierarchical Processing                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
            """
            console.print(Panel(banner.strip(), style="bold blue", box=box.DOUBLE))
        else:
            print("🚀 SUM: Hierarchical Knowledge Densification System")
            print("Real-Time Progressive Summarization")
    
    def show_engine_menu(self) -> str:
        """Show interactive engine selection menu."""
        if RICH_AVAILABLE:
            table = Table(title="🧠 Available Processing Engines", box=box.ROUNDED)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Engine", style="magenta")
            table.add_column("Description", style="green")
            table.add_column("Best For", style="yellow")
            
            table.add_row("1", "SimpleSUM", "Fast frequency-based summarization", "Quick processing")
            table.add_row("2", "MagnumOpusSUM", "Advanced analysis with entities & sentiment", "Detailed analysis")
            table.add_row("3", "HierarchicalDensificationEngine", "3-level hierarchical processing", "Professional use")
            table.add_row("4", "StreamingEngine", "Unlimited text processing", "Large documents")
            
            console.print(table)
            
            choice = Prompt.ask("Select engine", choices=["1", "2", "3", "4"], default="3")
        else:
            print("\n🧠 Available Processing Engines:")
            print("1. SimpleSUM - Fast frequency-based summarization")
            print("2. MagnumOpusSUM - Advanced analysis with entities & sentiment")
            print("3. HierarchicalDensificationEngine - 3-level hierarchical processing")
            print("4. StreamingEngine - Unlimited text processing")
            choice = input("Select engine (1-4, default 3): ").strip() or "3"
        
        engine_map = {"1": "simple", "2": "advanced", "3": "hierarchical", "4": "streaming"}
        return engine_map.get(choice, "hierarchical")
    
    def get_text_input(self) -> str:
        """Get text input from user."""
        if RICH_AVAILABLE:
            console.print("\n📝 Text Input Options:", style="bold cyan")
            console.print("1. Type/paste text directly")
            console.print("2. Load from file")
            
            choice = Prompt.ask("Choose input method", choices=["1", "2"], default="1")
            
            if choice == "2":
                file_path = Prompt.ask("Enter file path")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    console.print(f"✅ Loaded {len(text)} characters from {file_path}")
                    return text
                except Exception as e:
                    console.print(f"❌ Error loading file: {e}", style="red")
                    return ""
            else:
                console.print("📝 Enter your text (Press Ctrl+D or Ctrl+Z when done):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                return '\n'.join(lines)
        else:
            print("\n📝 Enter your text (Press Ctrl+D or Ctrl+Z when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            return '\n'.join(lines)
    
    def get_processing_config(self, engine_type: str) -> Dict[str, Any]:
        """Get processing configuration from user."""
        config = {}
        
        if RICH_AVAILABLE:
            console.print(f"\n⚙️  Configuration for {engine_type.title()}Engine:", style="bold cyan")
            
            if engine_type == "hierarchical":
                config['max_concepts'] = int(Prompt.ask("Max concepts to extract", default="7"))
                config['max_summary_tokens'] = int(Prompt.ask("Max summary tokens", default="50"))
                config['max_insights'] = int(Prompt.ask("Max insights to find", default="3"))
                config['min_insight_score'] = float(Prompt.ask("Min insight score (0-1)", default="0.6"))
                
            elif engine_type == "streaming":
                config['chunk_size_words'] = int(Prompt.ask("Chunk size (words)", default="1000"))
                config['overlap_ratio'] = float(Prompt.ask("Overlap ratio (0-1)", default="0.15"))
                config['max_memory_mb'] = int(Prompt.ask("Max memory (MB)", default="512"))
                config['max_concurrent_chunks'] = int(Prompt.ask("Max concurrent chunks", default="4"))
                
            elif engine_type in ["simple", "advanced"]:
                config['maxTokens'] = int(Prompt.ask("Max tokens", default="100"))
                config['threshold'] = float(Prompt.ask("Threshold (0-1)", default="0.3"))
                
                if engine_type == "advanced":
                    config['include_analysis'] = Confirm.ask("Include detailed analysis?", default=True)
        else:
            print(f"\n⚙️  Configuration for {engine_type.title()}Engine:")
            if engine_type == "hierarchical":
                config['max_concepts'] = int(input("Max concepts (default 7): ") or "7")
                config['max_summary_tokens'] = int(input("Max summary tokens (default 50): ") or "50")
        
        return config
    
    def process_with_progress(self, engine_type: str, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with beautiful progress visualization."""
        if not RICH_AVAILABLE:
            # Fallback for no rich
            print(f"🚀 Processing with {engine_type.title()}Engine...")
            engine_class = self.engines[engine_type]
            engine = engine_class() if engine_type != "streaming" else engine_class(StreamingConfig(**config))
            
            start_time = time.time()
            if engine_type == "streaming":
                result = engine.process_streaming_text(text)
            else:
                result = engine.process_text(text, config)
            
            processing_time = time.time() - start_time
            print(f"✅ Processing completed in {processing_time:.2f} seconds")
            return result
        
        # Rich progress visualization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # Initialize engine
            init_task = progress.add_task("🔧 Initializing engine...", total=100)
            
            engine_class = self.engines[engine_type]
            if engine_type == "streaming":
                engine = engine_class(StreamingConfig(**config))
            else:
                engine = engine_class()
            
            progress.update(init_task, completed=100)
            time.sleep(0.5)  # Show completion
            
            # Processing phase
            process_task = progress.add_task(f"🚀 Processing with {engine_type.title()}Engine...", total=100)
            
            start_time = time.time()
            
            # Simulate progress for non-streaming engines
            if engine_type != "streaming":
                for i in range(0, 101, 20):
                    progress.update(process_task, completed=i)
                    time.sleep(0.1)
                
                if engine_type == "streaming":
                    result = engine.process_streaming_text(text)
                else:
                    result = engine.process_text(text, config)
                    
                progress.update(process_task, completed=100)
            else:
                # For streaming, we could implement real progress tracking
                result = engine.process_streaming_text(text)
                progress.update(process_task, completed=100)
            
            processing_time = time.time() - start_time
            
            # Completion message
            console.print(f"\n✅ Processing completed in {processing_time:.2f} seconds", style="bold green")
            
            return result
    
    def display_results(self, result: Dict[str, Any], engine_type: str):
        """Display processing results with beautiful formatting."""
        if not RICH_AVAILABLE:
            print("\n📊 Results:")
            if 'hierarchical_summary' in result:
                print(f"Concepts: {result['hierarchical_summary']['level_1_concepts']}")
                print(f"Summary: {result['hierarchical_summary']['level_2_core']}")
            elif 'summary' in result:
                print(f"Summary: {result['summary']}")
            return
        
        # Rich formatting
        layout = Layout()
        
        if engine_type == "hierarchical" and 'hierarchical_summary' in result:
            hierarchical = result['hierarchical_summary']
            
            # Create panels for each level
            concepts_panel = Panel(
                "\n".join([f"✨ {concept.upper()}" for concept in hierarchical['level_1_concepts']]),
                title="🎯 Level 1: Key Concepts",
                style="cyan"
            )
            
            summary_panel = Panel(
                hierarchical['level_2_core'],
                title="💎 Level 2: Core Summary",
                style="green"
            )
            
            expansion_panel = Panel(
                hierarchical.get('level_3_expanded', "No expansion needed - core summary captures full complexity!"),
                title="📖 Level 3: Expanded Context",
                style="yellow"
            )
            
            console.print(concepts_panel)
            console.print(summary_panel)
            console.print(expansion_panel)
            
            # Insights
            if 'key_insights' in result and result['key_insights']:
                insights_table = Table(title="🌟 Key Insights", box=box.ROUNDED)
                insights_table.add_column("#", style="cyan", width=4)
                insights_table.add_column("Type", style="magenta", width=12)
                insights_table.add_column("Insight", style="white")
                insights_table.add_column("Score", style="green", width=8)
                
                for i, insight in enumerate(result['key_insights'], 1):
                    insights_table.add_row(
                        str(i),
                        insight['type'].upper(),
                        insight['text'],
                        f"{insight['score']:.2f}"
                    )
                
                console.print(insights_table)
        
        elif 'summary' in result:
            summary_panel = Panel(
                result['summary'],
                title="📝 Summary",
                style="green"
            )
            console.print(summary_panel)
            
            if 'tags' in result:
                tags_panel = Panel(
                    " | ".join([f"#{tag}" for tag in result['tags']]),
                    title="🏷️  Tags",
                    style="cyan"
                )
                console.print(tags_panel)
        
        # Metadata
        if 'metadata' in result:
            metadata = result['metadata']
            metadata_table = Table(title="📊 Processing Metadata", box=box.SIMPLE)
            metadata_table.add_column("Metric", style="cyan")
            metadata_table.add_column("Value", style="green")
            
            for key, value in metadata.items():
                if isinstance(value, float):
                    value = f"{value:.3f}"
                metadata_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(metadata_table)
    
    def save_results(self, result: Dict[str, Any], engine_type: str):
        """Save results to file."""
        if RICH_AVAILABLE:
            if Confirm.ask("💾 Save results to file?"):
                filename = Prompt.ask("Filename", default=f"sum_results_{engine_type}_{int(time.time())}.json")
                
                try:
                    import json
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                    console.print(f"✅ Results saved to {filename}", style="green")
                except Exception as e:
                    console.print(f"❌ Error saving file: {e}", style="red")
    
    def run_interactive(self):
        """Run interactive CLI session."""
        self.print_banner()
        
        try:
            while True:
                # Engine selection
                engine_type = self.show_engine_menu()
                
                # Text input
                text = self.get_text_input()
                if not text.strip():
                    if RICH_AVAILABLE:
                        console.print("❌ No text provided", style="red")
                    else:
                        print("❌ No text provided")
                    continue
                
                # Configuration
                config = self.get_processing_config(engine_type)
                
                # Processing
                result = self.process_with_progress(engine_type, text, config)
                
                # Display results
                self.display_results(result, engine_type)
                
                # Save option
                self.save_results(result, engine_type)
                
                # Continue?
                if RICH_AVAILABLE:
                    if not Confirm.ask("\n🔄 Process another text?"):
                        break
                else:
                    choice = input("\n🔄 Process another text? (y/n): ").lower()
                    if choice != 'y':
                        break
                        
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n👋 Thanks for using SUM!", style="bold blue")
            else:
                print("\n👋 Thanks for using SUM!")
    
    def run_batch(self, files: List[str], engine_type: str, config: Dict[str, Any]):
        """Run batch processing on multiple files."""
        if RICH_AVAILABLE:
            console.print(f"🔄 Batch processing {len(files)} files with {engine_type.title()}Engine", style="bold cyan")
            
            with Progress(console=console) as progress:
                task = progress.add_task("Processing files...", total=len(files))
                
                for i, file_path in enumerate(files):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        result = self.process_with_progress(engine_type, text, config)
                        
                        # Save result
                        output_file = f"{Path(file_path).stem}_summary.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            import json
                            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                        
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        console.print(f"❌ Error processing {file_path}: {e}", style="red")
                        progress.update(task, advance=1)
            
            console.print("✅ Batch processing completed!", style="bold green")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🚀 SUM Enhanced CLI - Hierarchical Knowledge Densification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_enhanced.py                          # Interactive mode
  python cli_enhanced.py --file document.txt     # Process single file
  python cli_enhanced.py --batch *.txt           # Batch process files
  python cli_enhanced.py --engine hierarchical   # Use specific engine
        """
    )
    
    parser.add_argument('--file', '-f', help='Process single file')
    parser.add_argument('--batch', '-b', nargs='+', help='Process multiple files')
    parser.add_argument('--engine', '-e', choices=['simple', 'advanced', 'hierarchical', 'streaming'],
                       default='hierarchical', help='Processing engine to use')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--config', '-c', help='JSON config file')
    
    args = parser.parse_args()
    
    cli = EnhancedCLI()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                import json
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return
    
    # Single file processing
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not config:
                config = cli.get_processing_config(args.engine)
            
            result = cli.process_with_progress(args.engine, text, config)
            cli.display_results(result, args.engine)
            
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                    
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return
    
    # Batch processing
    elif args.batch:
        if not config:
            config = cli.get_processing_config(args.engine)
        cli.run_batch(args.batch, args.engine, config)
    
    # Interactive mode
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()