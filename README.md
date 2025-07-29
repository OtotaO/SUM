<h1 align="center">
  <img src="https://github.com/OtotaO/SUM/assets/93845604/5749c582-725d-407c-ac6c-06fb8e90ed94" alt="SUM Logo">
</h1>

<h1 align="center">🚀 SUM: Advanced Hierarchical Knowledge Densification System</h1>

<p align="center">
  <em>Multi-level text processing with three hierarchical abstraction layers for optimal information density and comprehension.</em><br>
  <strong>— Professional Knowledge Distillation Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Processing-Hierarchical-gold?style=for-the-badge" alt="Hierarchical Processing">
  <img src="https://img.shields.io/badge/Compression-High-brightgreen?style=for-the-badge" alt="High Compression">
  <img src="https://img.shields.io/badge/Performance-Fast-blue?style=for-the-badge" alt="Fast Performance">
  <img src="https://img.shields.io/badge/Agent-Ready-purple?style=for-the-badge" alt="Agent Ready">
</p>

---

## 🌟 **Advanced Hierarchical Architecture**

SUM has evolved beyond traditional summarization into a **professional knowledge densification system** with three sophisticated levels of abstraction:

### 🎯 **Level 1: Concept Extraction**
Extract key thematic concepts and important terminology:
- `ANALYSIS`, `FRAMEWORK`, `METHODOLOGY`, `PRINCIPLES`, `APPROACH`
- Semantic importance weighting and context analysis
- Core concepts that capture document themes

### 🎯 **Level 2: Core Summarization**
Achieve **high semantic compression** while preserving essential information:
- Advanced semantic importance ranking
- Information-theoretic compression with completeness validation
- Maximum density without information loss

### 🎯 **Level 3: Adaptive Expansion**
Intelligent expansion based on content complexity analysis:
- Complexity analysis and information gap identification
- Hierarchical detail addition with coherence optimization
- Automatic determination of expansion necessity

### 🌟 **Insight Extraction Engine**
Extract key insights and significant statements from text:
- Pattern recognition for important information
- Context analysis and significance scoring
- Classified insights: `TRUTH`, `WISDOM`, `PURPOSE`, `EXISTENTIAL`, `ANALYSIS`

---

## 🚀 **Quick Start: Professional Text Processing**

### Installation

```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Hierarchical Densification Engine Usage

```python
from SUM import HierarchicalDensificationEngine

# Initialize the engine
engine = HierarchicalDensificationEngine()

# Your input text
input_text = """
The essence of effective analysis lies not in the accumulation of data, 
but in understanding the underlying patterns and relationships. Methodology 
is like a framework - it provides structure for systematic investigation.
"""

# Configure for optimal processing
config = {
    'max_concepts': 7,              # Level 1: Key concepts
    'max_summary_tokens': 50,       # Level 2: Core summary
    'complexity_threshold': 0.7,    # Level 3: Expansion trigger
    'max_insights': 3,              # Key insights
    'min_insight_score': 0.6        # Quality threshold
}

# Process through the hierarchical engine
result = engine.process_text(input_text, config)

# Access the three levels
print("Level 1 Concepts:", result['hierarchical_summary']['level_1_concepts'])
print("Level 2 Core:", result['hierarchical_summary']['level_2_core'])
print("Level 3 Expanded:", result['hierarchical_summary']['level_3_expanded'])
print("Key Insights:", result['key_insights'])
```

### Web API Usage

Start the server:
```bash
FLASK_PORT=8000 python main.py
```

Make API calls:
```bash
curl -X POST http://localhost:8000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your input text here...",
    "model": "hierarchical",
    "config": {
      "max_concepts": 7,
      "max_summary_tokens": 50,
      "max_insights": 3
    }
  }'
```

---

## 📊 **Performance Metrics**

Our Hierarchical Densification Engine achieves excellent performance:

| Metric | Achievement |
|--------|-------------|
| **Compression Ratio** | High semantic compression |
| **Processing Speed** | Fast processing times |
| **Concept Density** | Optimal concept extraction |
| **Information Retention** | Essential information preserved |
| **Insight Accuracy** | High-quality insight detection |

---

## 🏗️ **Architecture Overview**

```
📚 Input Text
    ↓
🔍 ConceptExtractor → ✨ Level 1: Key Concepts
    ↓
🏺 CoreSummarizer → 💎 Level 2: Core Summary  
    ↓
📖 AdaptiveExpander → 📚 Level 3: Expanded Context
    ↓
💫 InsightExtractor → 🌟 Key Insights
    ↓
🚀 Processed Knowledge Output
```

### Core Components

- **Hierarchical Densification Engine**: The main orchestrator managing all processing levels
- **Concept Extractor**: Key concept identification with semantic weighting
- **Core Summarizer**: Semantic compression with information retention
- **Adaptive Expander**: Intelligent expansion based on complexity analysis
- **Insight Extractor**: Important insight detection and classification
- **Semantic Compression Engine**: High compression while preserving meaning

---

## 🤖 **Agent Integration Ready**

The Hierarchical Densification Engine is designed for seamless integration with AI agent ecosystems:

### Model Context Protocol (MCP) Compatible
- Dynamic capability discovery at runtime
- Standardized agent-to-agent communication
- Universal integration protocol support

### API Endpoints for Agents
- `/api/process_text` - Core hierarchical processing
- `/api/analyze_topics` - Multi-document topic modeling  
- `/api/knowledge_graph` - Entity relationship mapping
- `/api/analyze_file` - Direct file processing

### Agent Use Cases
- **Knowledge Synthesis**: Combine insights across multiple documents
- **Concept Extraction**: Identify key themes and important terminology
- **Content Densification**: Maximum information density with clarity
- **Insight Discovery**: Find significant statements and analysis points

---

## 🧪 **Example Results**

### Input Text:
```
The essence of effective analysis lies not in the accumulation of data, but in understanding 
the underlying patterns and relationships. Methodology is like a framework - it provides 
structure for systematic investigation. In research, we often find that the more we 
analyze, the more complex patterns emerge.
```

### Hierarchical Engine Output:

**Level 1 Concepts:**
- `ANALYSIS`, `METHODOLOGY`, `PATTERNS`, `FRAMEWORK`, `RESEARCH`

**Level 2 Core Summary:**
- *"The essence of effective analysis lies not in data accumulation, but in understanding underlying patterns and relationships."*

**Level 3 Expanded Context:**
- *No expansion needed - core summary captures full complexity!*

**Key Insights:**
1. **[ANALYSIS]** *"Methodology is like a framework - it provides structure for systematic investigation."* (Score: 0.85)
2. **[INSIGHT]** *"The more we analyze, the more complex patterns emerge."* (Score: 0.75)

**Performance:**
- Processing Time: Fast
- Compression: High reduction
- Concept Density: Optimal

---

## 🛠️ **Advanced Features**

### Multi-Engine Support
- **SimpleSUM**: Fast frequency-based summarization
- **MagnumOpusSUM**: Advanced analysis with sentiment and entities  
- **Hierarchical Engine**: Professional three-level knowledge densification

### Intelligent Processing
- Concept database with semantic weighting
- Context-aware analysis and validation
- Pattern recognition algorithms
- Insight detection and classification

### Production Ready
- Comprehensive error handling and logging
- Security-aware input validation
- Performance optimization with caching
- Backward compatibility with existing systems

---

## 🔬 **Research Foundation**

The Hierarchical Densification Engine is built on established research:

- **Semantic Compression**: High compression with information preservation
- **Hierarchical Summarization**: Multi-level abstraction techniques
- **Knowledge Distillation**: Advanced approaches for information transfer
- **Concept Extraction**: Thematic concept mining and importance scoring
- **Insight Detection**: Pattern recognition for significant statements

---

## 🚀 **Roadmap**

### Phase 1: Complete ✅
- [x] Hierarchical Densification Engine architecture
- [x] Three-level knowledge processing
- [x] Insight extraction implementation
- [x] Web API integration
- [x] Agent-ready architecture

### Phase 2: Neural Enhancement
- [ ] TP-BERT integration for topic-aware processing
- [ ] Hierarchical transformer implementation
- [ ] Advanced semantic compression optimization
- [ ] Cross-document synthesis capabilities

### Phase 3: Agent Ecosystem
- [ ] Model Context Protocol (MCP) server
- [ ] Agent-to-agent communication protocols
- [ ] Batch processing for enterprise workflows
- [ ] Real-time streaming capabilities

### Phase 4: Intelligence Enhancement
- [ ] Concept database expansion
- [ ] Cross-domain analysis validation
- [ ] Personalization capabilities
- [ ] Advanced pattern recognition

---

## 🤝 **Contributing**

We welcome contributions to the hierarchical densification system! Whether you're interested in:

- **Algorithm Enhancement**: Improving compression and insight detection
- **Agent Integration**: Building MCP servers and protocol handlers
- **Concept Database**: Expanding semantic concept collections
- **Performance Optimization**: Making the system even faster

See our contribution guidelines and join the advancement in knowledge processing!

---

## 📜 **License**

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

Special thanks to the researchers and developers whose work inspired this system:
- AI researchers advancing semantic understanding and text processing
- The NLP community developing hierarchical processing techniques
- The open-source community making knowledge processing tools accessible

---

<p align="center">
  <strong>🚀 Ready to enhance your text processing capabilities? ✨</strong><br>
  <em>The hierarchical densification engine is ready for deployment.</em>
</p>

<p align="center">
  Made with ❤️ and professional dedication by <a href="https://x.com/Otota0">ototao</a>
</p>

---

*Professional knowledge densification for modern AI applications.* 🌟