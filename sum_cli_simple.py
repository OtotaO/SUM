#!/usr/bin/env python3
"""
sum_cli_simple.py - Simple CLI for SUM API

A clean, simple command-line interface for the SUM API.
"""

import click
import requests
import json
import sys
from pathlib import Path
from typing import Optional
import time

# ANSI color codes for simple formatting
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(text, color=''):
    """Print colored text"""
    print(f"{color}{text}{Colors.END}")

def check_api(base_url):
    """Check if API is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

@click.group()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=3000, help='API port')
@click.pass_context
def cli(ctx, host, port):
    """SUM CLI - Text summarization made simple"""
    ctx.ensure_object(dict)
    ctx.obj['base_url'] = f"http://{host}:{port}"
    
    # Check API health
    if not check_api(ctx.obj['base_url']):
        print_colored("❌ Error: SUM API is not running!", Colors.RED)
        print("\nStart the API with one of these commands:")
        print("  python sum_simple.py      # Basic API")
        print("  python sum_ultimate.py    # Full features")
        sys.exit(1)

@cli.command()
@click.argument('text')
@click.option('--density', '-d', 
              type=click.Choice(['all', 'tags', 'minimal', 'short', 'medium', 'detailed']),
              default='minimal',
              help='Summary density level')
def text(text, density):
    """Summarize text directly"""
    base_url = click.get_current_context().parent.obj['base_url']
    
    print_colored("⏳ Summarizing...", Colors.YELLOW)
    
    try:
        response = requests.post(
            f"{base_url}/summarize/ultimate",
            json={"text": text, "density": density},
            timeout=30
        )
        
        if response.status_code == 200:
            display_results(response.json(), density)
        else:
            print_colored(f"❌ Error: {response.status_code}", Colors.RED)
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--density', '-d',
              type=click.Choice(['all', 'tags', 'minimal', 'short', 'medium', 'detailed']),
              default='minimal',
              help='Summary density level')
def file(file_path, density):
    """Summarize a file"""
    base_url = click.get_current_context().parent.obj['base_url']
    file_path = Path(file_path)
    
    print(f"\n{Colors.BOLD}File:{Colors.END} {file_path.name}")
    print(f"{Colors.BOLD}Size:{Colors.END} {file_path.stat().st_size / 1024:.1f} KB")
    
    print_colored("\n⏳ Processing file...", Colors.YELLOW)
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f)}
            data = {'density': density}
            
            response = requests.post(
                f"{base_url}/summarize/ultimate",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                display_results(response.json(), density)
            else:
                print_colored(f"❌ Error: {response.status_code}", Colors.RED)
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)

@cli.command()
@click.argument('input_text', required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='File to stream')
def stream(input_text, file):
    """Stream summarization with real-time progress"""
    base_url = click.get_current_context().parent.obj['base_url']
    
    # Get text from file or argument
    if file:
        with open(file, 'r') as f:
            text = f.read()
        print(f"\n{Colors.BOLD}Streaming summary of:{Colors.END} {Path(file).name}")
    elif input_text:
        text = input_text
        print(f"\n{Colors.BOLD}Streaming summary of text{Colors.END}")
    else:
        print_colored("❌ Error: Provide text or use --file", Colors.RED)
        return
    
    try:
        response = requests.post(
            f"{base_url}/summarize/stream",
            json={"text": text},
            stream=True,
            timeout=60
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        
                        if data['type'] == 'progress':
                            # Clear line and show progress
                            print(f"\r{Colors.GREEN}Progress: {data['percentage']:.1f}% "
                                  f"({data['processed_words']}/{data['total_words']} words){Colors.END}", 
                                  end='', flush=True)
                        
                        elif data['type'] == 'complete':
                            print("\n")  # New line after progress
                            print_colored("✅ Complete!", Colors.GREEN)
                            display_results({'result': data['summaries']}, 'all')
        else:
            print_colored(f"❌ Error: {response.status_code}", Colors.RED)
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)

@cli.command()
def examples():
    """Show usage examples"""
    examples_text = f"""
{Colors.BOLD}SUM CLI Examples{Colors.END}

{Colors.BLUE}Basic text summarization:{Colors.END}
  sum-cli text "Your long text here..."
  sum-cli text "Your text" --density minimal

{Colors.BLUE}File summarization:{Colors.END}
  sum-cli file document.pdf
  sum-cli file research.docx --density detailed
  sum-cli file article.txt -d tags

{Colors.BLUE}Streaming (with progress):{Colors.END}
  sum-cli stream --file book.pdf
  sum-cli stream "Very long text..."

{Colors.BLUE}Connect to remote API:{Colors.END}
  sum-cli --host api.example.com --port 8080 text "..."
"""
    print(examples_text)

def display_results(result, density):
    """Display results in a clean format"""
    if 'error' in result:
        print_colored(f"❌ Error: {result['error']}", Colors.RED)
        return
    
    data = result.get('result', {})
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print_colored("📄 SUMMARIZATION RESULTS", Colors.BLUE + Colors.BOLD)
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    if density == 'all' or density == 'tags':
        if 'tags' in data:
            print(f"{Colors.BOLD}🏷️  Tags:{Colors.END}")
            print(f"   {Colors.YELLOW}{', '.join(data['tags'])}{Colors.END}\n")
    
    if density == 'all' or density == 'minimal':
        if 'minimal' in data:
            print(f"{Colors.BOLD}📝 Minimal (one sentence):{Colors.END}")
            print(f"   {data['minimal']}\n")
    
    if density == 'all' or density == 'short':
        if 'short' in data:
            print(f"{Colors.BOLD}📄 Short:{Colors.END}")
            print(f"   {data['short']}\n")
    
    if density == 'all' or density == 'medium':
        if 'medium' in data:
            print(f"{Colors.BOLD}📃 Medium:{Colors.END}")
            print(f"   {data['medium']}\n")
    
    if density == 'all' or density == 'detailed':
        if 'detailed' in data:
            print(f"{Colors.BOLD}📚 Detailed:{Colors.END}")
            print(f"   {data['detailed']}\n")
    
    # Show single summary if returned
    if 'summary' in data:
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"   {data['summary']}\n")
    
    # Statistics
    if 'original_words' in data:
        print(f"{Colors.BOLD}📊 Statistics:{Colors.END}")
        print(f"   Original: {data['original_words']} words")
        if 'compression_ratio' in data:
            print(f"   Compression: {data['compression_ratio']:.1f}:1")
    
    if result.get('cached'):
        print(f"\n{Colors.YELLOW}(Result from cache){Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")

if __name__ == '__main__':
    cli()