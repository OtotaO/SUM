#!/usr/bin/env python3
"""
setup_knowledge.py - Setup script for Knowledge Crystallization features

This script:
1. Checks and installs required dependencies
2. Downloads necessary language models
3. Initializes storage directories
4. Verifies the installation

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"Python {version.major}.{version.minor} detected ✓")
    return True


def install_requirements():
    """Install required packages"""
    logger.info("Installing knowledge crystallization dependencies...")
    
    # Core packages for knowledge features
    packages = [
        'sentence-transformers>=2.2.0',
        'chromadb>=0.4.0',
        'faiss-cpu>=1.7.4',
        'spacy>=3.6.0',
        'networkx>=3.1',
        'chardet>=5.2.0',
        'aiofiles>=23.2.0',
        'python-louvain>=0.16'
    ]
    
    # Optional but recommended
    optional_packages = [
        'py2neo>=2021.2.3',  # Neo4j integration
        'uvloop>=0.19.0',    # Fast event loop
    ]
    
    # Install core packages
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    # Try optional packages
    for package in optional_packages:
        try:
            logger.info(f"Installing optional: {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ])
        except subprocess.CalledProcessError:
            logger.warning(f"Optional package {package} failed to install (not critical)")
    
    return True


def download_spacy_model():
    """Download spaCy language model"""
    logger.info("Downloading spaCy language model...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
        ])
        logger.info("spaCy model downloaded ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    logger.info("Creating storage directories...")
    
    directories = [
        './semantic_memory',
        './knowledge_graph',
        './data',
        './models',
        './output',
        './uploads',
        './temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created {directory} ✓")
    
    return True


def verify_installation():
    """Verify the installation works"""
    logger.info("Verifying installation...")
    
    try:
        # Test imports
        logger.info("Testing imports...")
        
        # Semantic memory
        from memory.semantic_memory import get_semantic_memory_engine
        memory_engine = get_semantic_memory_engine()
        logger.info("Semantic memory engine initialized ✓")
        
        # Knowledge graph
        from memory.knowledge_graph import get_knowledge_graph_engine
        kg_engine = get_knowledge_graph_engine()
        logger.info("Knowledge graph engine initialized ✓")
        
        # Synthesis engine
        from application.synthesis_engine import get_synthesis_engine
        synthesis_engine = get_synthesis_engine()
        logger.info("Synthesis engine initialized ✓")
        
        # Async pipeline
        from application.async_pipeline import AsyncProcessingPipeline
        pipeline = AsyncProcessingPipeline()
        logger.info("Async pipeline initialized ✓")
        
        # Test basic functionality
        logger.info("Testing basic functionality...")
        
        # Store a test memory
        test_id = memory_engine.store_memory(
            text="Test memory for verification",
            summary="Test summary",
            metadata={"test": True}
        )
        logger.info(f"Test memory stored: {test_id} ✓")
        
        # Search for it
        results = memory_engine.search_memories("test verification", top_k=1)
        if results:
            logger.info("Memory search working ✓")
        
        # Test entity extraction
        extraction = kg_engine.extract_entities_and_relationships(
            "Apple Inc. was founded by Steve Jobs.",
            source="test"
        )
        if extraction['entities']:
            logger.info("Entity extraction working ✓")
        
        logger.info("\n✅ Knowledge crystallization setup complete!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        logger.error("Please check the error messages above")
        return False


def print_next_steps():
    """Print helpful next steps"""
    print("\n" + "="*60)
    print("🎉 Knowledge Crystallization Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the server: python main.py")
    print("2. Access the web UI: http://localhost:3000")
    print("3. Check the API docs: http://localhost:3000/api/docs")
    print("\nNew API endpoints available:")
    print("- POST /api/memory/store - Store semantic memories")
    print("- POST /api/memory/search - Search memories")
    print("- POST /api/memory/synthesize - Synthesize documents")
    print("- POST /api/knowledge/entities - Extract entities")
    print("- GET /api/memory/stats - View system statistics")
    print("\nFor more info, see KNOWLEDGE_CRYSTALLIZATION.md")
    print("="*60)


def main():
    """Main setup function"""
    print("\n🧠 Setting up Knowledge Crystallization for SUM...")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        logger.error("Failed to download spaCy model")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        logger.error("Installation verification failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()