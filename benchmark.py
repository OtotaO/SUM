#!/usr/bin/env python3
"""
benchmark.py - Performance Benchmarking Suite for SUM

Comprehensive benchmarking system to measure and compare performance
of all SUM engines with beautiful visualizations and detailed metrics.

Features:
- Multi-engine performance comparison
- Memory usage tracking
- Processing speed analysis
- Scalability testing
- Beautiful result visualization
- Export to various formats

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import time
import psutil
import statistics
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import SUM components
from summarization_engine import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from streaming_engine import StreamingHierarchicalEngine, StreamingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console() if RICH_AVAILABLE else None


@dataclass
class BenchmarkResult:
    """Benchmark result for a single test."""
    engine_name: str
    text_length: int
    word_count: int
    processing_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    tokens_per_second: float
    words_per_second: float
    compression_ratio: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.engines = {
            'SimpleSUM': SimpleSUM,
            'MagnumOpusSUM': MagnumOpusSUM, 
            'HierarchicalDensificationEngine': HierarchicalDensificationEngine,
            'StreamingHierarchicalEngine': StreamingHierarchicalEngine
        }
        self.test_texts = self._generate_test_texts()
        self.results: List[BenchmarkResult] = []
        
    def _generate_test_texts(self) -> Dict[str, str]:
        """Generate test texts of various sizes."""
        base_text = """
        Machine learning has revolutionized how we approach complex problems across numerous domains. 
        From healthcare diagnostics to financial modeling, from autonomous vehicles to personalized 
        recommendations, AI systems are transforming industries at an unprecedented pace. The foundation 
        of modern machine learning lies in deep neural networks, which can learn hierarchical 
        representations of data. These networks excel at recognizing patterns in images, understanding 
        natural language, and making predictions from complex datasets. However, the true power emerges 
        when these systems are combined with vast amounts of data and computational resources.
        
        Recent advances in transformer architectures have particularly transformed natural language 
        processing. Models like GPT and BERT have demonstrated remarkable capabilities in understanding 
        context, generating coherent text, and performing various language tasks. These breakthroughs 
        have opened new possibilities for human-AI collaboration. The ethical implications of AI 
        deployment cannot be overlooked. As these systems become more prevalent in decision-making 
        processes, ensuring fairness, transparency, and accountability becomes crucial.
        
        Looking toward the future, emerging paradigms like quantum machine learning, federated learning, 
        and neuromorphic computing promise to push the boundaries even further. The intersection of AI 
        with other cutting-edge technologies will likely produce innovations we can barely imagine today.
        Deep learning models have shown exceptional performance in computer vision tasks, from object 
        detection to medical image analysis. Convolutional neural networks have become the backbone 
        of image recognition systems, while attention mechanisms have revolutionized how models process 
        sequential data.
        """
        
        return {
            'small': base_text[:500],                    # ~500 chars
            'medium': base_text * 3,                     # ~3000 chars  
            'large': base_text * 10,                     # ~10000 chars
            'xlarge': base_text * 30,                    # ~30000 chars
            'massive': base_text * 100                   # ~100000 chars
        }
    
    def _measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_engine(self, engine_name: str, text_size: str, iterations: int = 3) -> BenchmarkResult:
        """Benchmark a single engine with given text size."""
        text = self.test_texts[text_size]
        word_count = len(text.split())
        text_length = len(text)
        
        times = []
        memory_usages = []
        peak_memories = []
        compression_ratios = []
        
        engine_class = self.engines[engine_name]
        
        for i in range(iterations):
            try:
                # Initialize engine
                if engine_name == 'StreamingHierarchicalEngine':
                    config = StreamingConfig(chunk_size_words=min(1000, word_count // 4))
                    engine = engine_class(config)
                else:
                    engine = engine_class()
                
                # Measure initial memory
                initial_memory = self._measure_memory_usage()
                
                # Process text
                start_time = time.time()
                
                if engine_name == 'StreamingHierarchicalEngine':
                    result = engine.process_streaming_text(text)
                else:
                    config = {}
                    if engine_name == 'HierarchicalDensificationEngine':
                        config = {
                            'max_concepts': 7,
                            'max_summary_tokens': min(100, word_count // 10),
                            'max_insights': 3
                        }
                    elif engine_name in ['SimpleSUM', 'MagnumOpusSUM']:
                        config = {'maxTokens': min(200, word_count // 5)}
                    
                    result = engine.process_text(text, config)
                
                processing_time = time.time() - start_time
                
                # Measure final memory
                final_memory = self._measure_memory_usage()
                peak_memory = max(initial_memory, final_memory)
                memory_usage = final_memory - initial_memory
                
                # Calculate compression ratio
                if 'hierarchical_summary' in result:
                    summary_text = result['hierarchical_summary']['level_2_core']
                elif 'summary' in result:
                    summary_text = result['summary']
                else:
                    summary_text = ""
                
                if summary_text:
                    compression_ratio = len(summary_text) / len(text)
                else:
                    compression_ratio = 0
                
                times.append(processing_time)
                memory_usages.append(max(0, memory_usage))  # Ensure non-negative
                peak_memories.append(peak_memory)
                compression_ratios.append(compression_ratio)
                
            except Exception as e:
                logger.error(f"Error benchmarking {engine_name} with {text_size}: {e}")
                return BenchmarkResult(
                    engine_name=engine_name,
                    text_length=text_length,
                    word_count=word_count,
                    processing_time=0,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    tokens_per_second=0,
                    words_per_second=0,
                    compression_ratio=0,
                    success=False,
                    error_message=str(e)
                )
        
        # Calculate averages
        avg_time = statistics.mean(times)
        avg_memory = statistics.mean(memory_usages)
        avg_peak_memory = statistics.mean(peak_memories)
        avg_compression = statistics.mean(compression_ratios)
        
        # Calculate throughput
        tokens_per_second = word_count / avg_time if avg_time > 0 else 0
        words_per_second = word_count / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            engine_name=engine_name,
            text_length=text_length,
            word_count=word_count,
            processing_time=avg_time,
            memory_usage_mb=avg_memory,
            peak_memory_mb=avg_peak_memory,
            tokens_per_second=tokens_per_second,
            words_per_second=words_per_second,
            compression_ratio=avg_compression,
            success=True
        )
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across all engines and text sizes."""
        if RICH_AVAILABLE:
            console.print("🚀 SUM Performance Benchmark Suite", style="bold cyan")
            console.print("Testing all engines across multiple text sizes...\n")
            
            total_tests = len(self.engines) * len(self.test_texts)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Running benchmarks...", total=total_tests)
                
                for engine_name in self.engines:
                    for text_size in self.test_texts:
                        progress.update(task, description=f"Testing {engine_name} - {text_size}")
                        
                        result = self.benchmark_engine(engine_name, text_size)
                        self.results.append(result)
                        
                        progress.advance(task)
                        
                        # Small delay to show progress
                        time.sleep(0.1)
        else:
            print("🚀 Running SUM Performance Benchmarks...")
            for engine_name in self.engines:
                for text_size in self.test_texts:
                    print(f"Testing {engine_name} - {text_size}")
                    result = self.benchmark_engine(engine_name, text_size)
                    self.results.append(result)
        
        return self.results
    
    def display_results(self):
        """Display benchmark results with beautiful formatting."""
        if not RICH_AVAILABLE:
            print("\n📊 Benchmark Results:")
            for result in self.results:
                if result.success:
                    print(f"{result.engine_name}: {result.processing_time:.3f}s, {result.words_per_second:.1f} words/sec")
            return
        
        # Create comprehensive results table
        table = Table(title="🏆 SUM Performance Benchmark Results", box=box.ROUNDED)
        table.add_column("Engine", style="cyan", no_wrap=True)
        table.add_column("Text Size", style="magenta")
        table.add_column("Words", style="blue", justify="right")
        table.add_column("Time (s)", style="green", justify="right")
        table.add_column("Words/sec", style="yellow", justify="right")
        table.add_column("Memory (MB)", style="red", justify="right")
        table.add_column("Compression", style="purple", justify="right")
        table.add_column("Status", style="white")
        
        # Group by text size for better readability
        text_sizes = ['small', 'medium', 'large', 'xlarge', 'massive']
        
        for text_size in text_sizes:
            size_results = [r for r in self.results if len(self.test_texts[text_size]) == r.text_length]
            
            if size_results:
                # Add separator
                table.add_row("", "", "", "", "", "", "", "")
                
                for result in size_results:
                    status = "✅ Success" if result.success else f"❌ {result.error_message[:20]}"
                    
                    table.add_row(
                        result.engine_name,
                        text_size.upper(),
                        f"{result.word_count:,}",
                        f"{result.processing_time:.3f}" if result.success else "N/A",
                        f"{result.words_per_second:.1f}" if result.success else "N/A",
                        f"{result.memory_usage_mb:.1f}" if result.success else "N/A",
                        f"{result.compression_ratio:.2f}" if result.success else "N/A",
                        status
                    )
        
        console.print(table)
        
        # Performance rankings
        self._display_performance_rankings()
        
        # Speed comparison chart
        self._display_speed_comparison()
    
    def _display_performance_rankings(self):
        """Display performance rankings."""
        if not RICH_AVAILABLE:
            return
            
        successful_results = [r for r in self.results if r.success]
        
        # Speed ranking (words per second)
        speed_ranking = sorted(successful_results, key=lambda x: x.words_per_second, reverse=True)
        
        speed_table = Table(title="🏃 Speed Rankings (Words/Second)", box=box.SIMPLE)
        speed_table.add_column("Rank", style="gold", width=6)
        speed_table.add_column("Engine", style="cyan")
        speed_table.add_column("Speed", style="green", justify="right")
        speed_table.add_column("Text Size", style="magenta")
        
        for i, result in enumerate(speed_ranking[:10], 1):
            text_size = next((size for size, text in self.test_texts.items() 
                            if len(text) == result.text_length), "unknown")
            speed_table.add_row(
                f"#{i}",
                result.engine_name,
                f"{result.words_per_second:.1f}",
                text_size.upper()
            )
        
        console.print(speed_table)
        
        # Memory efficiency ranking (lower is better)
        memory_ranking = sorted(successful_results, key=lambda x: x.memory_usage_mb)
        
        memory_table = Table(title="💾 Memory Efficiency Rankings (Lower is Better)", box=box.SIMPLE)
        memory_table.add_column("Rank", style="gold", width=6)
        memory_table.add_column("Engine", style="cyan")
        memory_table.add_column("Memory (MB)", style="red", justify="right")
        memory_table.add_column("Text Size", style="magenta")
        
        for i, result in enumerate(memory_ranking[:10], 1):
            text_size = next((size for size, text in self.test_texts.items() 
                            if len(text) == result.text_length), "unknown")
            memory_table.add_row(
                f"#{i}",
                result.engine_name,
                f"{result.memory_usage_mb:.1f}",
                text_size.upper()
            )
        
        console.print(memory_table)
    
    def _display_speed_comparison(self):
        """Display speed comparison chart."""
        if not RICH_AVAILABLE:
            return
            
        console.print("\n📈 Speed Comparison by Text Size", style="bold cyan")
        
        # Group results by engine
        engine_results = {}
        for result in self.results:
            if result.success:
                if result.engine_name not in engine_results:
                    engine_results[result.engine_name] = []
                engine_results[result.engine_name].append(result)
        
        # Create comparison table
        comparison_table = Table(title="Engine Performance Across Text Sizes", box=box.ROUNDED)
        comparison_table.add_column("Engine", style="cyan")
        comparison_table.add_column("Small", style="green", justify="right")
        comparison_table.add_column("Medium", style="blue", justify="right")
        comparison_table.add_column("Large", style="yellow", justify="right")
        comparison_table.add_column("XLarge", style="magenta", justify="right")
        comparison_table.add_column("Massive", style="red", justify="right")
        comparison_table.add_column("Avg Speed", style="white", justify="right")
        
        for engine_name, results in engine_results.items():
            # Sort by text length
            results.sort(key=lambda x: x.text_length)
            
            speeds = []
            row_data = [engine_name]
            
            text_sizes = ['small', 'medium', 'large', 'xlarge', 'massive']
            for text_size in text_sizes:
                matching_result = next((r for r in results if len(self.test_texts[text_size]) == r.text_length), None)
                if matching_result:
                    speed = matching_result.words_per_second
                    row_data.append(f"{speed:.1f}")
                    speeds.append(speed)
                else:
                    row_data.append("N/A")
            
            # Add average speed
            if speeds:
                avg_speed = statistics.mean(speeds)
                row_data.append(f"{avg_speed:.1f}")
            else:
                row_data.append("N/A")
            
            comparison_table.add_row(*row_data)
        
        console.print(comparison_table)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if not filename:
            timestamp = int(time.time())
            filename = f"sum_benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'platform': sys.platform
            },
            'test_configuration': {
                'engines_tested': list(self.engines.keys()),
                'text_sizes': {size: len(text) for size, text in self.test_texts.items()},
                'iterations_per_test': 3
            },
            'results': [result.to_dict() for result in self.results]
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            if RICH_AVAILABLE:
                console.print(f"✅ Results saved to {filename}", style="green")
            else:
                print(f"✅ Results saved to {filename}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"❌ Error saving results: {e}", style="red")
            else:
                print(f"❌ Error saving results: {e}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return "No successful benchmark results to report."
        
        # Calculate overall statistics
        avg_speed = statistics.mean([r.words_per_second for r in successful_results])
        avg_memory = statistics.mean([r.memory_usage_mb for r in successful_results])
        avg_compression = statistics.mean([r.compression_ratio for r in successful_results])
        
        fastest_result = max(successful_results, key=lambda x: x.words_per_second)
        most_efficient_memory = min(successful_results, key=lambda x: x.memory_usage_mb)
        best_compression = max(successful_results, key=lambda x: x.compression_ratio)
        
        report = f"""
🚀 SUM Performance Benchmark Report
{'=' * 50}

📊 Overall Statistics:
• Average Processing Speed: {avg_speed:.1f} words/second
• Average Memory Usage: {avg_memory:.1f} MB
• Average Compression Ratio: {avg_compression:.2f}

🏆 Performance Winners:
• Fastest Processing: {fastest_result.engine_name} ({fastest_result.words_per_second:.1f} words/sec)
• Most Memory Efficient: {most_efficient_memory.engine_name} ({most_efficient_memory.memory_usage_mb:.1f} MB)
• Best Compression: {best_compression.engine_name} ({best_compression.compression_ratio:.2f} ratio)

🧠 Engine Analysis:
"""
        
        # Add per-engine analysis
        for engine_name in self.engines:
            engine_results = [r for r in successful_results if r.engine_name == engine_name]
            if engine_results:
                avg_engine_speed = statistics.mean([r.words_per_second for r in engine_results])
                avg_engine_memory = statistics.mean([r.memory_usage_mb for r in engine_results])
                
                report += f"""
• {engine_name}:
  - Avg Speed: {avg_engine_speed:.1f} words/sec
  - Avg Memory: {avg_engine_memory:.1f} MB
  - Test Results: {len(engine_results)} successful
"""
        
        report += f"""

📈 Scalability Analysis:
• Text sizes tested: {', '.join(self.test_texts.keys())}
• Total tests run: {len(self.results)}
• Success rate: {len(successful_results) / len(self.results) * 100:.1f}%

💡 Recommendations:
• For speed: Use {fastest_result.engine_name}
• For memory efficiency: Use {most_efficient_memory.engine_name}  
• For best compression: Use {best_compression.engine_name}
• For balanced performance: Consider HierarchicalDensificationEngine

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


def main():
    """Main benchmark execution."""
    if RICH_AVAILABLE:
        console.print("🚀 SUM Performance Benchmark Suite", style="bold blue")
        console.print("Comprehensive performance testing of all SUM engines\n")
    else:
        print("🚀 SUM Performance Benchmark Suite")
        print("Comprehensive performance testing of all SUM engines\n")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Display results
    benchmark.display_results()
    
    # Generate and save report
    report = benchmark.generate_report()
    if RICH_AVAILABLE:
        console.print(Panel(report, title="📋 Performance Report", style="green"))
    else:
        print("\n📋 Performance Report:")
        print(report)
    
    # Save results
    benchmark.save_results()
    
    # Save report
    with open('sum_performance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    if RICH_AVAILABLE:
        console.print("\n✅ Benchmark completed successfully!", style="bold green")
        console.print("📄 Detailed report saved to sum_performance_report.txt")
    else:
        print("\n✅ Benchmark completed successfully!")
        print("📄 Detailed report saved to sum_performance_report.txt")


if __name__ == "__main__":
    main()