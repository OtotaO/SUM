# SUM API Documentation

## Overview

The SUM API provides programmatic access to all intelligence amplification features through both REST endpoints and WebSocket connections. All endpoints return JSON responses and accept JSON request bodies.

## Base URL

```
http://localhost:3000/api
```

## Authentication

Currently, the API does not require authentication for local usage. Production deployments should implement appropriate authentication mechanisms.

## Core Endpoints

### 1. Text Processing

#### Simple Summarization
```http
POST /api/process_text
Content-Type: application/json

{
  "text": "Your content here",
  "options": {
    "summary_length": 3,
    "include_keywords": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "summary": "Concise summary of the content",
  "keywords": ["key", "words", "extracted"],
  "processing_time": 0.234
}
```

#### Hierarchical Summarization
```http
POST /api/hierarchical_summary
Content-Type: application/json

{
  "text": "Your content here",
  "levels": 3
}
```

**Response:**
```json
{
  "success": true,
  "hierarchical_summary": {
    "level_1_keywords": ["core", "concepts"],
    "level_2_summary": "Key points extracted...",
    "level_3_detailed": "Comprehensive analysis..."
  },
  "insights": ["Insight 1", "Insight 2"],
  "confidence_score": 0.92
}
```

### 2. Invisible AI Processing

#### Zero-Configuration Processing
```http
POST /api/invisible_ai/process
Content-Type: application/json

{
  "content": "Any type of content",
  "context": {
    "user_intent": "research",
    "time_available": "5_minutes"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "primary_output": "Processed content",
    "detected_type": "academic_paper",
    "processing_approach": "deep_analysis",
    "additional_insights": ["Pattern 1", "Pattern 2"]
  }
}
```

### 3. Temporal Intelligence

#### Track Concept Evolution
```http
POST /api/temporal/track_concept
Content-Type: application/json

{
  "concept": "machine learning",
  "content": "New content about ML",
  "timestamp": "2024-01-28T10:00:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "evolution": {
    "first_seen": "2024-01-01T08:00:00Z",
    "total_mentions": 42,
    "understanding_depth": 0.87,
    "related_concepts": ["AI", "neural networks"],
    "evolution_stage": "deepening"
  }
}
```

#### Get Temporal Insights
```http
GET /api/temporal/insights?days=30
```

**Response:**
```json
{
  "success": true,
  "insights": {
    "evolving_concepts": ["concept1", "concept2"],
    "breakthrough_moments": [
      {
        "timestamp": "2024-01-15T14:30:00Z",
        "concepts_connected": ["A", "B"],
        "insight": "Major connection discovered"
      }
    ],
    "seasonal_patterns": {
      "morning": ["focus", "planning"],
      "evening": ["reflection", "creative"]
    }
  }
}
```

### 4. Collaborative Intelligence

#### Create Knowledge Cluster
```http
POST /api/collaborative/create_cluster
Content-Type: application/json

{
  "name": "AI Research Team",
  "description": "Collaborative space for AI research",
  "privacy_level": "team"
}
```

**Response:**
```json
{
  "success": true,
  "cluster": {
    "id": "cluster_1234567890",
    "name": "AI Research Team",
    "created_at": "2024-01-28T10:00:00Z",
    "join_code": "ABC123"
  }
}
```

#### Add Contribution
```http
POST /api/collaborative/contribute
Content-Type: application/json

{
  "cluster_id": "cluster_1234567890",
  "content": "New insight about transformers",
  "content_type": "text"
}
```

### 5. Notes System

#### Add Note
```http
POST /api/notes/add
Content-Type: application/json

{
  "content": "Meeting notes about project X",
  "title": "Project X Planning",
  "policy": "meeting"
}
```

**Response:**
```json
{
  "success": true,
  "note": {
    "id": "note_1234567890",
    "title": "Project X Planning",
    "policy_tag": "meeting",
    "auto_tags": ["project", "planning"],
    "importance": 0.75
  }
}
```

#### Get Insights Without Distillation
```http
GET /api/notes/insights?policy=diary&days=30
```

**Response:**
```json
{
  "success": true,
  "insights": {
    "total_notes_analyzed": 45,
    "patterns": ["Morning productivity", "Weekly cycles"],
    "emotional_indicators": [
      {"emotion": "excited", "frequency": 12},
      {"emotion": "focused", "frequency": 8}
    ],
    "key_concepts": [
      {"concept": "growth", "frequency": 15}
    ]
  }
}
```

### 6. File Processing

#### Process PDF
```http
POST /api/process_file
Content-Type: multipart/form-data

file: [PDF file]
options: {"extract_images": true, "create_outline": true}
```

#### Process Image
```http
POST /api/process_image
Content-Type: multipart/form-data

image: [Image file]
options: {"ocr": true, "extract_text": true}
```

## WebSocket Endpoints

### Progressive Summarization
```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.send(JSON.stringify({
  type: 'start_summarization',
  text: 'Large text content...',
  options: {
    chunk_size: 1000,
    update_frequency: 'realtime'
  }
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    console.log(`Progress: ${data.percentage}%`);
  } else if (data.type === 'partial_result') {
    console.log('Partial summary:', data.summary);
  }
};
```

### Live Collaboration
```javascript
const ws = new WebSocket('ws://localhost:3000/ws/collaborate');

ws.send(JSON.stringify({
  type: 'join_cluster',
  cluster_id: 'cluster_1234567890',
  user_id: 'user_123'
}));
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "error_code": "ValidationError",
    "message": "Invalid input format",
    "severity": "low",
    "details": {
      "field": "text",
      "requirement": "minimum 10 characters"
    },
    "timestamp": "2024-01-28T10:00:00Z"
  }
}
```

### Common Error Codes

| Code | Description | Severity |
|------|-------------|----------|
| ValidationError | Invalid input data | low |
| ProcessingError | Error during processing | medium |
| ResourceError | Required resource unavailable | high |
| ConfigurationError | System misconfiguration | high |
| NetworkError | Network connectivity issue | medium |
| AIModelError | AI model failure | high |

## Rate Limiting

Local deployments have no rate limits. Production deployments should implement:
- 100 requests per minute for text processing
- 10 requests per minute for file processing
- 1000 WebSocket messages per minute

## Examples

### Python Client
```python
import requests

# Simple summarization
response = requests.post('http://localhost:3000/api/process_text', 
    json={'text': 'Your long text here'})
result = response.json()
print(result['summary'])

# Invisible AI processing
response = requests.post('http://localhost:3000/api/invisible_ai/process',
    json={
        'content': 'Research paper content',
        'context': {'user_intent': 'quick_summary'}
    })
```

### JavaScript Client
```javascript
// Using fetch API
const summarize = async (text) => {
  const response = await fetch('http://localhost:3000/api/process_text', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  return await response.json();
};

// WebSocket for real-time
const connectWebSocket = () => {
  const ws = new WebSocket('ws://localhost:8765');
  ws.onopen = () => console.log('Connected');
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleRealtimeUpdate(data);
  };
};
```

### cURL Examples
```bash
# Simple text processing
curl -X POST http://localhost:3000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here"}'

# Upload and process file
curl -X POST http://localhost:3000/api/process_file \
  -F "file=@document.pdf" \
  -F 'options={"extract_images": true}'
```

## Best Practices

1. **Batch Operations**: When processing multiple items, use batch endpoints when available
2. **Streaming**: For large content, use WebSocket endpoints for real-time progress
3. **Error Recovery**: Implement exponential backoff for retries
4. **Context Preservation**: Include relevant context in requests for better AI processing
5. **Resource Management**: Close WebSocket connections when not in use

## SDK Support

Official SDKs are planned for:
- Python (`pip install sum-sdk`)
- JavaScript/TypeScript (`npm install @sum/sdk`)
- Go (`go get github.com/sum/sdk-go`)

## Support

For API support and questions:
- GitHub Issues: https://github.com/OtotaO/SUM/issues
- Documentation: https://sum-ai.com/docs
- Community: https://discord.gg/sum-community