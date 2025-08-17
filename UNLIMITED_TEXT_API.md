# Unlimited Text Processing API

## Overview

SUM can now process texts of ANY length - from 1 byte to 1 terabyte and beyond. The system automatically detects text size and applies the appropriate processing strategy.

## Processing Strategies

### Automatic Size Detection

| Text Size | Strategy | Description |
|-----------|----------|-------------|
| < 10KB | Direct | Process entire text at once |
| 10KB - 10MB | Smart Chunking | Overlapping chunks with context preservation |
| 10MB - 1GB | Memory-Mapped Streaming | Efficient processing without loading entire file |
| > 1GB | Hierarchical Streaming | Multi-level processing with batch summarization |

## API Endpoints

### 1. Standard Processing with Unlimited Support

**POST** `/api/process_text`

```json
{
  "text": "Your text content...",
  "model": "unlimited",
  "config": {
    "max_summary_tokens": 500,
    "overlap_ratio": 0.1,
    "chunk_size": 100000
  }
}
```

### 2. Dedicated Unlimited Processing

**POST** `/api/process_unlimited`

#### Option 1: Direct Text
```json
{
  "text": "Very long text content...",
  "config": {
    "max_summary_tokens": 1000,
    "enable_chunk_summaries": true
  }
}
```

#### Option 2: File Path
```json
{
  "file_path": "/path/to/massive/document.txt",
  "config": {
    "max_memory_mb": 512,
    "hierarchical_levels": 3
  }
}
```

#### Option 3: File Upload
```bash
curl -X POST http://localhost:5001/api/process_unlimited \
  -F "file=@huge_document.txt" \
  -F 'config={"max_summary_tokens": 500}'
```

## Response Format

```json
{
  "summary": "Final summary of entire document...",
  "hierarchical_summary": {
    "level_1_essence": "Core message in 1-2 sentences",
    "level_2_core": "Key points summary",
    "level_3_expanded": "Detailed summary with main themes",
    "level_4_comprehensive": "Full summary with examples"
  },
  "key_concepts": ["concept1", "concept2", ...],
  "total_word_count": 5000000,
  "processing_method": "hierarchical_streaming",
  "chunks_processed": 150,
  "batches_processed": 8,
  "processing_time": 45.2,
  "chunk_summaries": [
    {
      "chunk_id": 0,
      "summary": "Summary of first chunk...",
      "word_count": 50000
    }
  ],
  "metadata": {
    "chunk_size": 100000,
    "overlap_ratio": 0.1,
    "hierarchical_levels": 2
  }
}
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_summary_tokens` | int | 500 | Maximum tokens in final summary |
| `chunk_size` | int | auto | Characters per chunk (auto-calculated if not set) |
| `overlap_ratio` | float | 0.1 | Overlap between chunks (0.1 = 10%) |
| `max_memory_mb` | int | 512 | Maximum memory usage |
| `enable_chunk_summaries` | bool | false | Include individual chunk summaries |
| `hierarchical_levels` | int | auto | Levels of hierarchical summarization |

## Examples

### Summarize a 100MB Research Paper
```python
import requests

response = requests.post('http://localhost:5001/api/process_unlimited',
    json={
        'file_path': '/data/research_paper.pdf',
        'config': {
            'max_summary_tokens': 1000,
            'enable_chunk_summaries': True
        }
    }
)

result = response.json()
print(f"Processed {result['total_word_count']:,} words in {result['processing_time']}s")
print(f"Summary: {result['summary']}")
```

### Process Streaming Text
```python
# For extremely large texts, use file upload
with open('massive_book.txt', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/api/process_unlimited',
        files={'file': f},
        data={'config': '{"max_summary_tokens": 2000}'}
    )
```

### Batch Process Multiple Large Files
```python
# Use the mass processing endpoint for multiple files
response = requests.post('http://localhost:5001/api/mass/upload',
    files=[
        ('files', open('doc1.txt', 'rb')),
        ('files', open('doc2.txt', 'rb')),
        ('files', open('doc3.txt', 'rb'))
    ]
)
```

## Performance Guidelines

1. **Memory Usage**: The system uses at most 512MB RAM regardless of file size
2. **Processing Speed**: Approximately 1GB per minute on standard hardware
3. **Chunk Size**: Automatically optimized based on text size
4. **Accuracy**: Context preservation through overlapping chunks ensures high accuracy

## Best Practices

1. For files > 100MB, use file path or upload instead of JSON text
2. Enable chunk summaries for research/analysis purposes
3. Adjust overlap_ratio higher (0.2-0.3) for technical documents
4. Use hierarchical processing for documents > 1GB
5. Monitor the processing_method in response to understand how your text was processed

## Error Handling

```json
{
  "error": "Text processing failed",
  "details": "Insufficient memory for processing",
  "suggestions": [
    "Reduce chunk_size",
    "Increase max_memory_mb",
    "Use file path instead of direct text"
  ]
}
```