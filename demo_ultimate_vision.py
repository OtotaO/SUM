#!/usr/bin/env python3
"""
demo_ultimate_vision.py - Demonstrate the COMPLETE original vision

Shows:
1. Arbitrary length summarization
2. Multiple density levels (tags -> minimal -> detailed)
3. Real-time streaming summaries
4. File processing
5. Bidirectional compression (experimental)
"""

import requests
import json
import time
from typing import Dict, Any

def demo_density_levels():
    """Show all density levels from tags to detailed"""
    print("\n🎯 DEMONSTRATING DENSITY LEVELS")
    print("=" * 60)
    
    # Sample text (could be ANY length)
    text = """
    Artificial intelligence has transformed from a science fiction concept into a practical technology 
    that shapes our daily lives. Machine learning algorithms power recommendation systems on streaming 
    platforms, natural language processing enables virtual assistants to understand our commands, and 
    computer vision helps autonomous vehicles navigate streets. The rapid advancement in AI capabilities, 
    particularly with large language models and neural networks, has opened new possibilities in healthcare, 
    education, and scientific research. However, this progress also raises important questions about ethics, 
    privacy, and the future of human work. As we stand at this technological crossroads, it's crucial to 
    develop AI systems that augment human capabilities rather than replace them, ensuring that the benefits 
    of this revolutionary technology are distributed equitably across society.
    """
    
    # Get all density levels
    response = requests.post(
        'http://localhost:3000/summarize/ultimate',
        json={'text': text, 'density': 'all'}
    )
    
    if response.status_code == 200:
        data = response.json()
        result = data['result']
        
        print(f"\n📊 Original text: {result['original_words']} words")
        print(f"🔄 Compression ratio: {result['compression_ratio']:.1f}:1")
        
        print("\n🏷️  TAGS (Ultra-minimal):")
        print(f"   {', '.join(result['tags'])}")
        
        print("\n📝 MINIMAL (The SUM - one sentence):")
        print(f"   {result['minimal']}")
        
        print("\n📄 SHORT (One paragraph):")
        print(f"   {result['short']}")
        
        print("\n📃 MEDIUM (2-3 paragraphs):")
        print(f"   {result['medium']}")
        
        print("\n📚 DETAILED (Comprehensive):")
        print(f"   {result['detailed']}")
    else:
        print(f"❌ Error: {response.text}")

def demo_streaming_summary():
    """Show real-time streaming summaries"""
    print("\n\n🌊 DEMONSTRATING STREAMING SUMMARIES")
    print("=" * 60)
    print("(Showing running total as text is processed)")
    
    # Longer text for streaming demo
    long_text = """
    The history of human civilization is a tapestry woven from countless threads of innovation, conflict, 
    and cultural exchange. From the earliest agricultural settlements in Mesopotamia to the digital age, 
    humanity has continuously adapted and evolved. The invention of writing systems allowed knowledge to 
    be preserved across generations, while the printing press democratized access to information. The 
    Industrial Revolution transformed societies from agrarian to urban, introducing both unprecedented 
    prosperity and new challenges. The 20th century saw the rise of global communication networks, space 
    exploration, and computing technology. Today, we face climate change, artificial intelligence, and 
    biotechnology as defining challenges and opportunities. Each era builds upon the achievements and 
    lessons of the past, creating an ever-more complex and interconnected world. Understanding this 
    historical continuity helps us navigate present challenges and shape a more equitable future.
    """ * 3  # Make it longer
    
    # Stream the summarization
    response = requests.post(
        'http://localhost:3000/summarize/stream',
        json={'text': long_text},
        stream=True
    )
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    
                    if data['type'] == 'progress':
                        print(f"\r🔄 Progress: {data['percentage']:.1f}% "
                              f"({data['processed_words']}/{data['total_words']} words) "
                              f"- Current summary: {data['current_summary'][:60]}...", 
                              end='', flush=True)
                    
                    elif data['type'] == 'complete':
                        print("\n\n✅ STREAMING COMPLETE!")
                        print(f"   Final summary: {data['summaries']['minimal']}")
    else:
        print(f"❌ Error: {response.text}")

def demo_file_processing():
    """Show file processing capabilities"""
    print("\n\n📁 FILE PROCESSING CAPABILITIES")
    print("=" * 60)
    
    print("Supported file types:")
    print("  ✓ PDF documents")
    print("  ✓ Word documents (.docx)")
    print("  ✓ Plain text files")
    print("  ✓ HTML pages")
    print("  ✓ Markdown files")
    print("  ✓ Any text-based format")
    
    print("\nExample API usage:")
    print("  curl -X POST localhost:3000/summarize/ultimate \\")
    print("    -F 'file=@document.pdf' \\")
    print("    -F 'density=minimal'")

def demo_bidirectional():
    """Show experimental decompression"""
    print("\n\n🔄 BIDIRECTIONAL COMPRESSION/DECOMPRESSION")
    print("=" * 60)
    
    summary = "AI transforms daily life through practical applications."
    
    print(f"Original summary: '{summary}'")
    print("\nAttempting to decompress/expand...")
    
    response = requests.post(
        'http://localhost:3000/decompress',
        json={'summary': summary, 'target_words': 100}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nExpanded text ({len(data['expanded_text'].split())} words):")
        print(f"  {data['expanded_text']}")
        print(f"\nNote: {data['note']}")
    else:
        print(f"❌ Error: {response.text}")

def show_api_examples():
    """Show API usage examples"""
    print("\n\n🚀 API USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            'description': 'Get just tags',
            'request': "POST /summarize/ultimate {'text': '...', 'density': 'tags'}"
        },
        {
            'description': 'Get minimal summary (THE SUM)',
            'request': "POST /summarize/ultimate {'text': '...', 'density': 'minimal'}"
        },
        {
            'description': 'Process a PDF file',
            'request': "POST /summarize/ultimate -F 'file=@paper.pdf' -F 'density=medium'"
        },
        {
            'description': 'Stream real-time summaries',
            'request': "POST /summarize/stream {'text': '...'} (Server-Sent Events)"
        },
        {
            'description': 'Get all density levels',
            'request': "POST /summarize/ultimate {'text': '...', 'density': 'all'}"
        }
    ]
    
    for ex in examples:
        print(f"\n📌 {ex['description']}:")
        print(f"   {ex['request']}")

def main():
    print("🌟 SUM ULTIMATE VISION DEMO")
    print("The COMPLETE original vision implemented!")
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:3000/capabilities')
        if response.status_code != 200:
            print("\n❌ Server not running! Start it with: python sum_ultimate.py")
            return
    except:
        print("\n❌ Server not running! Start it with: python sum_ultimate.py")
        return
    
    # Run all demos
    demo_density_levels()
    input("\nPress Enter to continue to streaming demo...")
    
    demo_streaming_summary()
    input("\nPress Enter to continue to file processing...")
    
    demo_file_processing()
    input("\nPress Enter to continue to bidirectional demo...")
    
    demo_bidirectional()
    
    show_api_examples()
    
    print("\n\n✨ THE VISION IS COMPLETE!")
    print("   ✓ Arbitrary length text")
    print("   ✓ Multiple density levels") 
    print("   ✓ Real-time streaming")
    print("   ✓ File processing")
    print("   ✓ Bidirectional (experimental)")
    print("\n🚀 All accessible via simple API!")

if __name__ == "__main__":
    main()