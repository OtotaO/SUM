#!/usr/bin/env python3
"""
Dependency checker for SUM platform
Shows which features are available based on installed packages
"""
import sys
import importlib
from typing import Dict, Tuple

def check_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "Installed"
    except ImportError as e:
        return False, f"Not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_dependencies() -> Dict[str, Dict[str, any]]:
    """Check all optional dependencies"""
    
    print("🔍 Checking SUM Platform Dependencies...")
    print("=" * 50)
    
    dependencies = {
        # Core dependencies (required)
        "Core": {
            "flask": ("Flask web framework", True),
            "nltk": ("Natural Language Toolkit", True),
            "numpy": ("Numerical computing", True),
            "requests": ("HTTP library", True),
        },
        # Vector stores (optional)
        "Vector Stores": {
            "chromadb": ("ChromaDB vector database", False),
            "faiss": ("Facebook AI Similarity Search", False),
        },
        # Graph databases (optional)
        "Graph Databases": {
            "neo4j": ("Neo4j graph database", False),
            "py2neo": ("Neo4j Python driver", False),
        },
        # AI/ML (optional)
        "AI/ML Models": {
            "transformers": ("Hugging Face Transformers", False),
            "sentence_transformers": ("Sentence embeddings", False),
            "openai": ("OpenAI API", False),
            "anthropic": ("Claude API", False),
        },
        # Performance (optional)
        "Performance": {
            "redis": ("Redis caching", False),
            "celery": ("Distributed task queue", False),
            "psycopg2": ("PostgreSQL driver", False),
        },
        # Monitoring (optional)
        "Monitoring": {
            "prometheus_client": ("Prometheus metrics", False),
            "sentry_sdk": ("Error tracking", False),
        }
    }
    
    results = {}
    
    for category, modules in dependencies.items():
        print(f"\n📦 {category}")
        print("-" * 40)
        
        category_results = {}
        for module, (description, required) in modules.items():
            available, status = check_import(module)
            category_results[module] = {
                "available": available,
                "status": status,
                "description": description,
                "required": required
            }
            
            icon = "✅" if available else ("❌" if required else "⚠️")
            req_text = " (REQUIRED)" if required else ""
            print(f"{icon} {module:<20} - {description}{req_text}")
            if not available:
                print(f"   └─ {status}")
        
        results[category] = category_results
    
    return results

def check_feature_availability(deps: Dict[str, Dict[str, any]]) -> Dict[str, bool]:
    """Determine which features are available based on dependencies"""
    
    print("\n\n🚀 Feature Availability")
    print("=" * 50)
    
    features = {
        "Basic Summarization": True,  # Always available
        "File Processing": True,      # Always available
        "Web Interface": deps["Core"]["flask"]["available"],
        "Semantic Memory": (
            deps["Vector Stores"]["chromadb"]["available"] or 
            deps["Vector Stores"]["faiss"]["available"]
        ),
        "Knowledge Graphs": (
            deps["Graph Databases"]["neo4j"]["available"] or
            deps["Graph Databases"]["py2neo"]["available"]
        ),
        "Advanced Embeddings": deps["AI/ML Models"]["sentence_transformers"]["available"],
        "GPT Integration": deps["AI/ML Models"]["openai"]["available"],
        "Claude Integration": deps["AI/ML Models"]["anthropic"]["available"],
        "Distributed Processing": deps["Performance"]["celery"]["available"],
        "Caching": deps["Performance"]["redis"]["available"],
        "Production Database": deps["Performance"]["psycopg2"]["available"],
        "Monitoring": deps["Monitoring"]["prometheus_client"]["available"],
        "Error Tracking": deps["Monitoring"]["sentry_sdk"]["available"],
    }
    
    for feature, available in features.items():
        icon = "✅" if available else "❌"
        print(f"{icon} {feature}")
    
    return features

def generate_recommendations(features: Dict[str, bool]) -> None:
    """Generate recommendations based on missing features"""
    
    print("\n\n💡 Recommendations")
    print("=" * 50)
    
    if not features["Semantic Memory"]:
        print("\n🔸 For Semantic Memory features, install one of:")
        print("   pip install chromadb")
        print("   pip install faiss-cpu  # or faiss-gpu if you have CUDA")
    
    if not features["Knowledge Graphs"]:
        print("\n🔸 For Knowledge Graph features, install:")
        print("   pip install neo4j py2neo")
    
    if not features["Advanced Embeddings"]:
        print("\n🔸 For better text embeddings, install:")
        print("   pip install sentence-transformers")
    
    if not any([features["GPT Integration"], features["Claude Integration"]]):
        print("\n🔸 For AI-powered features, install:")
        print("   pip install openai      # For GPT integration")
        print("   pip install anthropic   # For Claude integration")
    
    if not features["Caching"]:
        print("\n🔸 For better performance, install:")
        print("   pip install redis")
    
    if not features["Production Database"]:
        print("\n🔸 For production deployment, install:")
        print("   pip install psycopg2-binary")

def main():
    """Main entry point"""
    # Check dependencies
    deps = check_dependencies()
    
    # Check feature availability
    features = check_feature_availability(deps)
    
    # Generate recommendations
    generate_recommendations(features)
    
    # Summary
    print("\n\n📊 Summary")
    print("=" * 50)
    
    available_count = sum(1 for v in features.values() if v)
    total_count = len(features)
    percentage = (available_count / total_count) * 100
    
    print(f"Features Available: {available_count}/{total_count} ({percentage:.0f}%)")
    
    if percentage == 100:
        print("\n🎉 All features are available! SUM is fully configured.")
    elif percentage >= 70:
        print("\n✨ Core features are available. SUM is ready for use.")
    elif percentage >= 30:
        print("\n⚠️  Basic features are available. Consider installing optional dependencies.")
    else:
        print("\n❌ Missing critical dependencies. Please install required packages.")
    
    # Exit code based on core dependencies
    core_ok = all(deps["Core"][m]["available"] for m in deps["Core"] if deps["Core"][m]["required"])
    return 0 if core_ok else 1

if __name__ == "__main__":
    sys.exit(main())