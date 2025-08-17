# API Authentication Guide

## Overview

SUM now supports API key authentication for enhanced security, rate limiting, and usage tracking. API keys provide:

- **Higher Rate Limits**: Authenticated requests get better limits
- **Usage Analytics**: Track your API usage over time
- **Access Control**: Permission-based endpoint access
- **Security**: Secure key-based authentication

## Getting Started

### 1. Generate an API Key

Use the CLI tool to create your first API key:

```bash
python manage_api_keys.py create "My Application" --permissions=read,summarize
```

This will output:
```
✓ API Key created successfully!

Key ID: aBcDeFgHiJkLmNoP
API Key: sum_1234567890abcdefghijklmnopqrstuvwxyz
Name: My Application
Permissions: read, summarize
Rate Limit: 60 requests/minute
Daily Limit: 10000 requests/day

⚠️  Save this API key securely - it cannot be retrieved again!
```

### 2. Using Your API Key

Include your API key in requests using the `X-API-Key` header:

```bash
curl -X POST http://localhost:5001/api/process_text \
  -H "X-API-Key: sum_1234567890abcdefghijklmnopqrstuvwxyz" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "model": "hierarchical"}'
```

Or as a query parameter:

```bash
curl "http://localhost:5001/api/process_text?api_key=sum_1234567890abcdefghijklmnopqrstuvwxyz" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

## Rate Limits

### Without API Key (Public)
- `/api/process_text`: 20 requests/minute
- `/api/analyze_topics`: 10 requests/minute
- `/api/process_unlimited`: Not available

### With API Key (Default)
- Standard endpoints: 60 requests/minute
- Daily limit: 10,000 requests/day
- Access to all endpoints including unlimited processing

### Custom Limits
Create keys with custom limits:

```bash
python manage_api_keys.py create "High Volume App" \
  --rate-limit=1000 \
  --daily-limit=100000
```

## Permissions

Available permissions:
- `read`: Access to basic endpoints
- `summarize`: Access to summarization endpoints
- `write`: Access to write operations
- `admin`: Full access including key management

## Managing API Keys

### List All Keys
```bash
python manage_api_keys.py list
```

### Revoke a Key
```bash
python manage_api_keys.py revoke <key_id>
```

### View Usage Statistics
```bash
python manage_api_keys.py stats <key_id> --days=30
```

## API Endpoints

### Authentication Endpoints

#### Validate Key
```http
GET /api/auth/validate
X-API-Key: your_api_key
```

Response:
```json
{
  "valid": true,
  "key_id": "aBcDeFgHiJkLmNoP",
  "name": "My Application",
  "permissions": ["read", "summarize"],
  "rate_limit": 60,
  "daily_limit": 10000
}
```

#### Get Usage Stats
```http
GET /api/auth/usage?days=7
X-API-Key: your_api_key
```

Response:
```json
{
  "total_requests": 1523,
  "endpoints": {
    "/api/process_text": 1200,
    "/api/process_unlimited": 323
  },
  "avg_response_time": 0.234,
  "error_rate": 0.5,
  "period_days": 7,
  "limits": {
    "rate_limit": 60,
    "daily_limit": 10000
  }
}
```

### Admin Endpoints (Requires admin permission)

#### Create API Key
```http
POST /api/auth/keys
X-API-Key: admin_key
Content-Type: application/json

{
  "name": "New Application",
  "permissions": ["read", "summarize"],
  "rate_limit": 100,
  "daily_limit": 50000
}
```

#### List All Keys
```http
GET /api/auth/keys
X-API-Key: admin_key
```

#### Revoke Key
```http
DELETE /api/auth/keys/<key_id>
X-API-Key: admin_key
```

## Error Responses

### Missing API Key
```json
{
  "error": "API key required"
}
```
Status: 401 Unauthorized

### Invalid API Key
```json
{
  "error": "Invalid API key"
}
```
Status: 401 Unauthorized

### Rate Limit Exceeded
```json
{
  "error": "Rate limit exceeded",
  "limit": "60 requests per minute"
}
```
Status: 429 Too Many Requests

### Daily Limit Exceeded
```json
{
  "error": "Daily limit exceeded",
  "limit": "10000 requests per day"
}
```
Status: 429 Too Many Requests

### Insufficient Permissions
```json
{
  "error": "Insufficient permissions",
  "required": ["admin"]
}
```
Status: 403 Forbidden

## Best Practices

1. **Secure Storage**: Never commit API keys to version control
2. **Use Environment Variables**: Store keys in environment variables
3. **Rotate Keys**: Regularly rotate API keys for security
4. **Monitor Usage**: Check usage stats to detect anomalies
5. **Least Privilege**: Create keys with minimal required permissions

## Example Integration

### Python
```python
import requests

API_KEY = "sum_your_api_key_here"
BASE_URL = "http://localhost:5001"

def summarize_text(text):
    response = requests.post(
        f"{BASE_URL}/api/process_text",
        headers={"X-API-Key": API_KEY},
        json={"text": text, "model": "hierarchical"}
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print("Rate limit exceeded. Please wait.")
    else:
        print(f"Error: {response.json()}")
```

### JavaScript
```javascript
const API_KEY = 'sum_your_api_key_here';
const BASE_URL = 'http://localhost:5001';

async function summarizeText(text) {
  const response = await fetch(`${BASE_URL}/api/process_text`, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text, model: 'hierarchical' })
  });
  
  if (response.ok) {
    return await response.json();
  } else if (response.status === 429) {
    console.error('Rate limit exceeded');
  } else {
    console.error('Error:', await response.json());
  }
}
```

## Initial Setup

When you first run SUM, an admin API key is automatically created and saved to `admin_api_key.txt`. Use this key to create additional keys for your applications.

```bash
# View the admin key
cat admin_api_key.txt

# Create your first application key
python manage_api_keys.py create "My App" --save-to-file
```

## Security Considerations

1. API keys are hashed before storage - we cannot retrieve lost keys
2. Each key has independent rate limits and usage tracking
3. Keys can be revoked instantly if compromised
4. All API usage is logged for security auditing
5. HTTPS is recommended for production deployments