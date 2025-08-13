# SUM API Documentation

## Overview

The SUM Knowledge Crystallization System provides a comprehensive REST API for text summarization, knowledge management, and intelligent document processing.

## Base URL

```
http://localhost:3000/api
```

## Authentication

Currently, the API does not require authentication. In production, implement API key authentication.

## Common Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": 1234567890
}
```

Error responses:

```json
{
  "error": true,
  "message": "Error description",
  "error_id": "ERR_1234567890",
  "details": { ... }
}
```

## Endpoints

### 1. Text Summarization

#### Basic Summarization

**POST** `/api/summarize`

Summarize text with multiple density options.

**Request:**
```json
{
  "text": "Your long text here...",
  "density": "medium"  // Options: minimal, short, medium, detailed, all
}
```

**Response:**
```json
{
  "summary": "Generated summary text",
  "original_words": 500,
  "compression_ratio": 5.2,
  "processing_time": 0.235
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence is transforming how we live and work...",
    "density": "short"
  }'
```

#### Streaming Summarization

**POST** `/api/stream/summarize`

Get real-time progress updates for large text summarization.

**Request:**
```json
{
  "text": "Very large text content...",
  "density": "all",
  "store_memory": true,
  "extract_entities": true
}
```

**Response:** Server-Sent Events stream

**Example:**
```javascript
const eventSource = new EventSource('/api/stream/summarize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: largeText })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
};
```

### 2. File Processing

#### Process File

**POST** `/api/process-file`

Process and summarize uploaded files.

**Request:** Multipart form data
- `file`: File to process (any format)
- `density`: Summarization density (optional)

**Response:**
```json
{
  "filename": "document.pdf",
  "file_size": 1234567,
  "text_length": 5000,
  "summary": {
    "minimal": "One sentence summary",
    "short": "Three sentence summary",
    "medium": "Five sentence summary"
  },
  "metadata": {
    "file_type": "pdf",
    "pages": 10
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/process-file \
  -F "file=@document.pdf" \
  -F "density=all"
```

### 3. Semantic Memory

#### Store Memory

**POST** `/api/memory/store`

Store text in semantic memory for later retrieval.

**Request:**
```json
{
  "text": "Content to remember",
  "summary": "Brief summary",
  "metadata": {
    "source": "user_upload",
    "category": "technical"
  },
  "relationships": ["related_id_1", "related_id_2"]
}
```

**Response:**
```json
{
  "memory_id": "mem_abc123",
  "stored": true,
  "embedding_generated": true
}
```

#### Search Memory

**POST** `/api/memory/search`

Search semantic memory by similarity.

**Request:**
```json
{
  "query": "search terms",
  "top_k": 10,
  "threshold": 0.7,
  "filters": {
    "category": "technical"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "memory_id": "mem_abc123",
      "text": "Stored content",
      "summary": "Brief summary",
      "similarity": 0.89,
      "metadata": { ... }
    }
  ],
  "total_results": 10
}
```

### 4. Knowledge Synthesis

#### Synthesize Documents

**POST** `/api/memory/synthesize`

Synthesize knowledge from multiple documents.

**Request:**
```json
{
  "memory_ids": ["mem_1", "mem_2", "mem_3"],
  "synthesis_type": "comprehensive",
  "min_consensus": 0.6
}
```

**Response:**
```json
{
  "unified_summary": "Synthesized knowledge from all documents",
  "key_insights": [
    "Central themes revolve around: AI, automation",
    "Strong agreement found on 5 key points"
  ],
  "contradictions": [
    {
      "concept": "implementation approach",
      "doc1_context": "Approach A is better",
      "doc2_context": "Approach B is better"
    }
  ],
  "consensus_points": [
    "AI will transform industries",
    "Human oversight remains critical"
  ],
  "confidence_score": 0.85
}
```

### 5. Feedback System

#### Submit Feedback

**POST** `/api/feedback/submit`

Submit user feedback on generated summaries.

**Request:**
```json
{
  "content_hash": "hash_of_original_text",
  "summary_type": "medium",
  "rating": 5,
  "helpful": true,
  "comment": "Very accurate summary"
}
```

**Response:**
```json
{
  "feedback_id": "fb_xyz789",
  "stored": true,
  "preferences_updated": true
}
```

### 6. Health & Monitoring

#### Health Check

**GET** `/api/health`

Basic health check.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "SUM Knowledge Crystallization System"
}
```

#### Detailed Health

**GET** `/api/health/detailed`

Comprehensive system health information.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "components": {
    "semantic_memory": {
      "status": "healthy",
      "message": "Semantic memory operational"
    },
    "knowledge_graph": {
      "status": "healthy",
      "backend": "networkx"
    }
  },
  "resources": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "memory_available_mb": 2048
  },
  "performance": {
    "average_response_time_ms": 125,
    "p95_response_time_ms": 250,
    "request_count": 1500
  }
}
```

#### Metrics (Prometheus Format)

**GET** `/api/metrics`

Prometheus-compatible metrics endpoint.

**Response:**
```
# HELP sum_uptime_seconds Time since service start
# TYPE sum_uptime_seconds counter
sum_uptime_seconds 3600.00

# HELP sum_cpu_usage_percent CPU usage percentage
# TYPE sum_cpu_usage_percent gauge
sum_cpu_usage_percent 25.50
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- `/api/summarize`: 100 requests per minute
- `/api/stream/*`: 20 requests per minute
- `/api/process-file`: 30 requests per minute
- `/api/memory/*`: 60 requests per minute

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 413 | Payload Too Large - File or text too large |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Component unhealthy |

## Best Practices

1. **Batch Processing**: Use `/api/stream/batch` for multiple documents
2. **Large Files**: Use streaming endpoints for files >10MB
3. **Memory Storage**: Store important summaries for later synthesis
4. **Feedback**: Submit ratings to improve summary quality
5. **Error Handling**: Implement exponential backoff on errors

## SDK Examples

### Python

```python
import requests

class SUMClient:
    def __init__(self, base_url="http://localhost:3000/api"):
        self.base_url = base_url
    
    def summarize(self, text, density="medium"):
        response = requests.post(
            f"{self.base_url}/summarize",
            json={"text": text, "density": density}
        )
        return response.json()
    
    def search_memory(self, query, top_k=10):
        response = requests.post(
            f"{self.base_url}/memory/search",
            json={"query": query, "top_k": top_k}
        )
        return response.json()

# Usage
client = SUMClient()
result = client.summarize("Your text here...", density="short")
print(result["summary"])
```

### JavaScript

```javascript
class SUMClient {
  constructor(baseUrl = 'http://localhost:3000/api') {
    this.baseUrl = baseUrl;
  }
  
  async summarize(text, density = 'medium') {
    const response = await fetch(`${this.baseUrl}/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, density })
    });
    return response.json();
  }
  
  streamSummarize(text, onProgress) {
    const eventSource = new EventSource(`${this.baseUrl}/stream/summarize`);
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onProgress(data);
    };
    return eventSource;
  }
}

// Usage
const client = new SUMClient();
const result = await client.summarize('Your text here...', 'short');
console.log(result.summary);
```

## Changelog

### Version 2.0.0 (Current)
- Added semantic memory storage
- Implemented knowledge synthesis
- Real-time streaming endpoints
- Feedback system
- Health monitoring

### Version 1.0.0
- Initial release
- Basic summarization
- File processing