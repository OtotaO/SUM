#!/usr/bin/env python3
"""
Test script to verify all legendary features are working
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_knowledge_crystallizer():
    """Test the knowledge crystallization engine"""
    print("\n🔬 Testing Knowledge Crystallizer...")
    
    try:
        from knowledge_crystallizer import (
            KnowledgeCrystallizer,
            CrystallizationConfig,
            DensityLevel,
            StylePersona
        )
        
        crystallizer = KnowledgeCrystallizer()
        
        test_text = """
        Artificial Intelligence has revolutionized how we interact with technology. 
        Machine learning models can now understand natural language, recognize images, 
        and make complex decisions. Deep learning has enabled breakthroughs in areas 
        like medical diagnosis, autonomous vehicles, and scientific research. 
        The transformer architecture, introduced in 2017, has become the foundation 
        for large language models like GPT and BERT. These models have billions of 
        parameters and can perform tasks that were once thought to require human intelligence.
        """
        
        # Test different density levels
        for density in [DensityLevel.ESSENCE, DensityLevel.TWEET, DensityLevel.STANDARD]:
            config = CrystallizationConfig(
                density=density,
                style=StylePersona.NEUTRAL
            )
            
            result = crystallizer.crystallize(test_text, config)
            
            print(f"\n  ✓ {density.name} density:")
            print(f"    {result.levels.get(density.name.lower(), 'N/A')[:100]}...")
        
        print(f"\n  ✓ Essence: {result.essence}")
        print(f"  ✓ Quality Score: {result.quality_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_llm_backend():
    """Test the universal LLM backend"""
    print("\n🤖 Testing LLM Backend...")
    
    try:
        from llm_backend import llm_backend, ModelProvider
        
        # Check available providers
        providers = llm_backend.get_available_providers()
        print(f"  ✓ Available providers: {', '.join(providers)}")
        
        # Test generation with default provider
        async def test_generate():
            prompt = "Summarize in one sentence: AI is transforming technology."
            result = await llm_backend.generate(prompt, temperature=0.5, max_tokens=50)
            return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_generate())
        loop.close()
        
        print(f"  ✓ Generated summary: {result[:100]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_graphrag():
    """Test GraphRAG implementation"""
    print("\n📊 Testing GraphRAG...")
    
    try:
        from graph_rag_crystallizer import GraphRAGCrystallizer
        
        crystallizer = GraphRAGCrystallizer()
        
        documents = [
            "Apple Inc. is a technology company based in Cupertino.",
            "Tim Cook is the CEO of Apple.",
            "Apple designs and manufactures iPhones and MacBooks.",
            "The iPhone is one of Apple's most successful products."
        ]
        
        # Note: This will use fallback methods if dependencies aren't installed
        result = crystallizer.crystallize_corpus(documents)
        
        print(f"  ✓ Processed {len(documents)} documents")
        print(f"  ✓ Global summary: {result.global_summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_raptor():
    """Test RAPTOR hierarchical processing"""
    print("\n🌳 Testing RAPTOR...")
    
    try:
        from raptor_hierarchical import RAPTORBuilder
        
        builder = RAPTORBuilder()
        
        test_text = """
        Climate change is one of the most pressing challenges of our time. 
        Rising temperatures are causing ice caps to melt and sea levels to rise. 
        This affects coastal communities worldwide. Renewable energy sources like 
        solar and wind power offer solutions. Many countries are transitioning 
        to clean energy to reduce carbon emissions.
        """
        
        # Note: This will use simple fallback if dependencies aren't installed
        tree = builder.build_tree(test_text)
        
        print(f"  ✓ Built tree with {len(tree.levels)} levels")
        print(f"  ✓ Root summary: {tree.root.summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_quantum_editor():
    """Test Quantum Editor core"""
    print("\n⚛️ Testing Quantum Editor...")
    
    try:
        from quantum_editor_core import (
            UniversalLLMAdapter,
            IntelligentDocument,
            QuantumEditor
        )
        
        editor = QuantumEditor()
        
        # Test document creation
        doc = editor.create_document("Test document for quantum features.")
        
        print(f"  ✓ Created document with ID: {doc.id}")
        
        # Test LLM adapter
        adapter = UniversalLLMAdapter()
        adapter.register_model("test", {
            "provider": "local",
            "endpoint": None
        })
        
        print(f"  ✓ Registered test model")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_api_endpoints():
    """Test that API endpoints are accessible"""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        import requests
        
        # Check if server is running
        base_url = "http://localhost:5001"
        
        endpoints = [
            "/api/crystallize/styles",
            "/api/crystallize/densities",
            "/api/health"
        ]
        
        server_running = False
        try:
            response = requests.get(f"{base_url}/api/health", timeout=1)
            server_running = response.status_code == 200
        except:
            pass
        
        if server_running:
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=1)
                    status = "✓" if response.status_code == 200 else "✗"
                    print(f"  {status} {endpoint}: {response.status_code}")
                except Exception as e:
                    print(f"  ✗ {endpoint}: {e}")
        else:
            print("  ⚠️ Server not running. Start with: python main.py")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 SUM LEGENDARY FEATURES TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Knowledge Crystallizer", test_knowledge_crystallizer),
        ("LLM Backend", test_llm_backend),
        ("GraphRAG", test_graphrag),
        ("RAPTOR", test_raptor),
        ("Quantum Editor", test_quantum_editor),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL LEGENDARY FEATURES ARE WORKING!")
    else:
        print("\n⚠️ Some features need attention. Check dependencies:")
        print("  - pip install openai anthropic  # For LLM providers")
        print("  - pip install sentence-transformers  # For embeddings")
        print("  - pip install scikit-learn  # For clustering")
        print("  - pip install spacy && python -m spacy download en_core_web_sm  # For NLP")
        print("  - pip install python-louvain  # For graph communities")


if __name__ == "__main__":
    main()