# Knowledge Crystallization Features

SUM now includes advanced knowledge crystallization capabilities that transform how information is processed, stored, and synthesized. These features represent a major leap towards true knowledge management.

## 🧠 Semantic Memory System

The semantic memory system provides intelligent, vector-based storage and retrieval of knowledge:

### Features
- **Vector Embeddings**: Every piece of text is converted to high-dimensional vectors for semantic understanding
- **Similarity Search**: Find related memories based on meaning, not just keywords
- **Persistent Storage**: Knowledge persists across sessions with efficient disk-based storage
- **Multiple Backends**: Supports ChromaDB, FAISS, or simple numpy arrays

### Usage

```python
from memory.semantic_memory import get_semantic_memory_engine

# Get the memory engine
memory = get_semantic_memory_engine()

# Store a memory
memory_id = memory.store_memory(
    text="The mitochondria is the powerhouse of the cell",
    summary="Mitochondria provide cellular energy",
    metadata={"source": "biology", "importance": "high"}
)

# Search for related memories
results = memory.search_memories(
    query="cellular energy production",
    top_k=5,
    threshold=0.7
)

for memory_entry, score in results:
    print(f"Found: {memory_entry.summary} (relevance: {score:.2f})")
```

### API Endpoints

**Store Memory**
```bash
POST /api/memory/store
{
    "text": "Full text content",
    "summary": "Brief summary",
    "metadata": {"tags": ["biology", "science"]},
    "relationships": ["related_memory_id"]
}
```

**Search Memories**
```bash
POST /api/memory/search
{
    "query": "What do we know about cellular energy?",
    "top_k": 10,
    "threshold": 0.6
}
```

## 🕸️ Knowledge Graph

The knowledge graph system extracts entities and relationships, building a connected understanding:

### Features
- **Entity Extraction**: Automatically identifies people, places, organizations, and concepts
- **Relationship Mapping**: Discovers how entities relate to each other
- **Graph Navigation**: Find paths between concepts
- **Community Detection**: Identify clusters of related knowledge

### Usage

```python
from memory.knowledge_graph import get_knowledge_graph_engine

# Get the knowledge graph engine
kg = get_knowledge_graph_engine()

# Extract entities and relationships
result = kg.extract_entities_and_relationships(
    text="Apple Inc. was founded by Steve Jobs in Cupertino.",
    source="tech_history"
)

# Find paths between entities
paths = kg.find_path(
    start_entity_id="apple_inc_id",
    end_entity_id="steve_jobs_id",
    max_length=3
)

# Get context around an entity
context = kg.get_entity_context(
    entity_id="steve_jobs_id",
    depth=2  # Two hops from the entity
)
```

### API Endpoints

**Extract Entities**
```bash
POST /api/knowledge/entities
{
    "text": "Text to analyze for entities",
    "source": "document_123"
}
```

**Find Knowledge Paths**
```bash
POST /api/knowledge/path
{
    "start_entity_id": "entity_1",
    "end_entity_id": "entity_2",
    "max_length": 5
}
```

**Visualize Knowledge**
```bash
POST /api/knowledge/visualize
{
    "entity_ids": ["entity_1", "entity_2", "entity_3"],
    "show_labels": true
}
```

## 📊 Cross-Document Synthesis

The synthesis engine intelligently combines knowledge from multiple documents:

### Features
- **Contradiction Detection**: Identifies conflicting information across sources
- **Consensus Building**: Finds points of agreement
- **Concept Evolution**: Tracks how ideas develop over time
- **Confidence Scoring**: Rates the reliability of synthesized knowledge

### Usage

```python
from application.synthesis_engine import synthesize_texts

# Synthesize multiple documents
texts = [
    "AI is revolutionizing healthcare...",
    "Concerns about AI in healthcare include...",
    "Machine learning can predict diseases..."
]

result = synthesize_texts(texts, synthesis_type="comprehensive")

print(f"Unified Summary: {result['unified_summary']}")
print(f"Key Insights: {result['key_insights']}")
print(f"Contradictions: {result['contradictions']}")
print(f"Confidence: {result['confidence_score']}")
```

### Synthesis Types

1. **Comprehensive**: Full analysis with all perspectives
2. **Focused**: Highlights key concepts and their contexts
3. **Comparative**: Emphasizes differences and similarities

### API Endpoints

**Synthesize Documents**
```bash
POST /api/memory/synthesize
{
    "memory_ids": ["memory_1", "memory_2", "memory_3"],
    "synthesis_type": "comprehensive"
}
```

## ⚡ Async Processing Pipeline

High-performance asynchronous processing for scalability:

### Features
- **Concurrent Processing**: Handle multiple documents simultaneously
- **Streaming Support**: Process large files without loading into memory
- **Progress Tracking**: Real-time updates on processing status
- **Flexible Task Types**: Summarization, entity extraction, embedding generation

### Usage

```python
from application.async_pipeline import AsyncProcessingPipeline

# Create pipeline
pipeline = AsyncProcessingPipeline(max_concurrent_tasks=10)

# Process multiple items
items = [
    {"content": "Document 1 text..."},
    {"content": "Document 2 text..."},
]

results = await pipeline.batch_process(
    items,
    task_type="summarize",
    progress_callback=lambda p, c, t: print(f"{p:.1%} complete")
)
```

## 🔧 Configuration

Configure the knowledge crystallization features via environment variables:

```bash
# Semantic Memory
SUM_SEMANTIC_MEMORY_MODEL=all-MiniLM-L6-v2
SUM_SEMANTIC_MEMORY_PATH=./semantic_memory
SUM_USE_GPU=False

# Knowledge Graph
SUM_NEO4J_URI=bolt://localhost:7687
SUM_NEO4J_USER=neo4j
SUM_NEO4J_PASSWORD=your_password
SUM_KNOWLEDGE_GRAPH_PATH=./knowledge_graph
SUM_SPACY_MODEL=en_core_web_sm

# Async Processing
SUM_MAX_CONCURRENT_TASKS=10
SUM_CHUNK_SIZE=1048576  # 1MB
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### 2. Initialize the Systems

```python
# In your Python code or startup script
from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine

# Initialize engines (happens automatically on first use)
memory_engine = get_semantic_memory_engine()
kg_engine = get_knowledge_graph_engine()
```

### 3. Start Using

The knowledge crystallization features integrate seamlessly with existing SUM functionality:

1. **Every summarization** automatically stores results in semantic memory
2. **Entity extraction** happens during text processing
3. **Cross-references** are built as you process more documents
4. **Knowledge evolves** with each new piece of information

## 📈 Performance Considerations

### Memory Usage
- Embeddings: ~2KB per text chunk
- Knowledge graph: ~1KB per entity + relationships
- Recommended: 8GB RAM for datasets up to 100k documents

### Storage
- Vector database: ~10MB per 1000 documents
- Knowledge graph: ~5MB per 1000 entities
- Indexes are built incrementally

### Processing Speed
- Embedding generation: ~100 documents/second
- Entity extraction: ~50 documents/second  
- Synthesis: ~10 document sets/second

## 🔮 Future Enhancements

Planned improvements for knowledge crystallization:

1. **Active Learning**: System improves from user feedback
2. **Temporal Reasoning**: Better understanding of time-based knowledge
3. **Multi-modal Integration**: Connect text with images and audio
4. **Distributed Processing**: Scale across multiple machines
5. **Knowledge Verification**: Fact-checking against trusted sources

## 🤝 Contributing

We welcome contributions to enhance knowledge crystallization:

1. **Vector Database Backends**: Add support for Pinecone, Weaviate, Qdrant
2. **Graph Algorithms**: Implement advanced graph analysis
3. **Synthesis Methods**: New ways to merge knowledge
4. **Visualization Tools**: Better ways to explore knowledge
5. **Performance Optimizations**: Make it even faster

## 📚 Examples

### Building a Knowledge Base

```python
# Process a collection of documents
documents = load_documents("./research_papers/")

for doc in documents:
    # Summarize
    summary = summarize_text_universal(doc.text)
    
    # Store in memory with extraction
    memory_id = memory_engine.store_memory(
        text=doc.text,
        summary=summary['medium'],
        metadata={'source': doc.filename}
    )
    
    # Extract knowledge
    kg_engine.extract_entities_and_relationships(
        doc.text,
        source=memory_id
    )

# Now query your knowledge base
insights = memory_engine.search_memories(
    "What are the key findings about climate change?",
    top_k=20
)

# Synthesize the findings
synthesis = synthesis_engine.synthesize_documents(
    [insight[0] for insight in insights],
    synthesis_type="comprehensive"
)
```

### Real-time Knowledge Updates

```python
# Stream processing for live data
async def process_news_stream(news_source):
    async for article in news_source:
        # Process asynchronously
        task = await pipeline.process_task({
            'content': article.text,
            'task_type': 'extract_entities',
            'metadata': {'source': 'news', 'timestamp': article.date}
        })
        
        # Update knowledge graph in real-time
        if task.status == 'completed':
            update_knowledge_dashboard(task.result)
```

---

The knowledge crystallization features in SUM represent a paradigm shift from simple summarization to true knowledge management. Start crystallizing your knowledge today!