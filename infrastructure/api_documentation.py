"""
API Documentation Generator for SUM Platform
============================================

Automatic API documentation generation with:
- OpenAPI/Swagger specification
- Interactive documentation UI
- Request/response examples
- Authentication documentation
- Webhook documentation
- Rate limiting information

Based on industry standards like OpenAPI 3.0 and Swagger UI.

Author: ototao
License: Apache License 2.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from enum import Enum
import inspect
import re

logger = logging.getLogger(__name__)


@dataclass
class APIParameter:
    """API parameter definition"""
    name: str
    param_type: str  # query, path, body, header
    data_type: str
    required: bool = False
    description: str = ""
    default: Any = None
    example: Any = None
    enum: List[Any] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


@dataclass
class APIResponse:
    """API response definition"""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Optional[Dict[str, Any]] = None
    examples: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: str
    summary: str
    description: str = ""
    operation_id: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[APIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[APIResponse] = field(default_factory=list)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    deprecated: bool = False
    rate_limit: Optional[str] = None
    examples: Dict[str, Any] = field(default_factory=dict)


class OpenAPIGenerator:
    """
    Generates OpenAPI 3.0 specification for the SUM API.
    """
    
    def __init__(self,
                 title: str = "SUM Platform API",
                 version: str = "1.0.0",
                 description: str = ""):
        """
        Initialize OpenAPI generator.
        
        Args:
            title: API title
            version: API version
            description: API description
        """
        self.title = title
        self.version = version
        self.description = description
        
        self.endpoints: List[APIEndpoint] = []
        self.schemas: Dict[str, Any] = {}
        self.security_schemes: Dict[str, Any] = {}
        self.tags: List[Dict[str, str]] = []
        
        # Initialize with default configuration
        self._init_default_config()
    
    def _init_default_config(self):
        """Initialize default configuration"""
        # Security schemes
        self.security_schemes = {
            "ApiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        # Tags
        self.tags = [
            {
                "name": "Summarization",
                "description": "Text summarization endpoints"
            },
            {
                "name": "Knowledge",
                "description": "Knowledge OS operations"
            },
            {
                "name": "Temporal",
                "description": "Temporal intelligence features"
            },
            {
                "name": "Memory",
                "description": "Superhuman memory operations"
            },
            {
                "name": "Webhooks",
                "description": "Webhook management"
            },
            {
                "name": "System",
                "description": "System health and monitoring"
            }
        ]
        
        # Common schemas
        self.schemas = {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "integer"},
                    "timestamp": {"type": "string", "format": "date-time"}
                },
                "required": ["error", "message"]
            },
            "Success": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {"type": "object"}
                }
            },
            "PaginationParams": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1, "default": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    "sort": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
                }
            }
        }
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add an endpoint to the specification"""
        self.endpoints.append(endpoint)
    
    def generate_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
                "contact": {
                    "name": "SUM Support",
                    "email": "support@sum-platform.com",
                    "url": "https://sum-platform.com"
                },
                "license": {
                    "name": "Apache 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:3000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.sum-platform.com",
                    "description": "Production server"
                }
            ],
            "tags": self.tags,
            "paths": self._generate_paths(),
            "components": {
                "schemas": self.schemas,
                "securitySchemes": self.security_schemes
            },
            "security": [
                {"BearerAuth": []},
                {"ApiKey": []}
            ]
        }
        
        return spec
    
    def _generate_paths(self) -> Dict[str, Any]:
        """Generate paths section"""
        paths = {}
        
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "operationId": endpoint.operation_id or f"{endpoint.method}_{endpoint.path.replace('/', '_')}",
                "tags": endpoint.tags,
                "parameters": self._generate_parameters(endpoint.parameters),
                "responses": self._generate_responses(endpoint.responses)
            }
            
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            if endpoint.security:
                operation["security"] = endpoint.security
            
            if endpoint.deprecated:
                operation["deprecated"] = True
            
            if endpoint.rate_limit:
                operation["x-rate-limit"] = endpoint.rate_limit
            
            paths[endpoint.path][endpoint.method.lower()] = operation
        
        return paths
    
    def _generate_parameters(self, parameters: List[APIParameter]) -> List[Dict[str, Any]]:
        """Generate parameters section"""
        params = []
        
        for param in parameters:
            param_spec = {
                "name": param.name,
                "in": param.param_type,
                "required": param.required,
                "description": param.description,
                "schema": {
                    "type": param.data_type
                }
            }
            
            if param.default is not None:
                param_spec["schema"]["default"] = param.default
            
            if param.example is not None:
                param_spec["example"] = param.example
            
            if param.enum:
                param_spec["schema"]["enum"] = param.enum
            
            if param.min_value is not None:
                param_spec["schema"]["minimum"] = param.min_value
            
            if param.max_value is not None:
                param_spec["schema"]["maximum"] = param.max_value
            
            if param.pattern:
                param_spec["schema"]["pattern"] = param.pattern
            
            params.append(param_spec)
        
        return params
    
    def _generate_responses(self, responses: List[APIResponse]) -> Dict[str, Any]:
        """Generate responses section"""
        response_spec = {}
        
        for response in responses:
            resp = {
                "description": response.description
            }
            
            if response.schema:
                resp["content"] = {
                    response.content_type: {
                        "schema": response.schema
                    }
                }
            
            if response.examples:
                if "content" not in resp:
                    resp["content"] = {response.content_type: {}}
                resp["content"][response.content_type]["examples"] = response.examples
            
            if response.headers:
                resp["headers"] = {
                    name: {"schema": {"type": "string"}, "description": desc}
                    for name, desc in response.headers.items()
                }
            
            response_spec[str(response.status_code)] = resp
        
        return response_spec


def generate_sum_api_documentation() -> Dict[str, Any]:
    """
    Generate complete API documentation for SUM platform.
    """
    generator = OpenAPIGenerator(
        title="SUM Platform API",
        version="2.0.0",
        description="""
        # SUM Platform API Documentation
        
        The SUM (Summarization, Understanding, Memory) Platform provides a comprehensive 
        suite of AI-powered text processing capabilities including:
        
        - **Intelligent Summarization** - Multiple algorithms with auto-selection
        - **Knowledge Operating System** - Capture and process thoughts effortlessly
        - **Temporal Intelligence** - Track how knowledge evolves over time
        - **Superhuman Memory** - Perfect recall with pattern recognition
        - **Predictive Intelligence** - Anticipate information needs
        - **Webhook Integration** - Real-time event notifications
        
        ## Authentication
        
        The API supports two authentication methods:
        1. **API Key**: Include `X-API-Key` header in requests
        2. **Bearer Token**: Include `Authorization: Bearer <token>` header
        
        ## Rate Limiting
        
        - Default: 30 requests per minute
        - Premium: 100 requests per minute
        - Enterprise: Custom limits
        
        Rate limit headers are included in all responses:
        - `X-RateLimit-Limit`: Maximum requests allowed
        - `X-RateLimit-Remaining`: Requests remaining
        - `X-RateLimit-Reset`: Time when limit resets
        
        ## Webhooks
        
        Subscribe to real-time events using webhooks. All webhook payloads are signed
        with HMAC-SHA256 for security verification.
        """
    )
    
    # Add summarization endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/summarize",
        method="POST",
        summary="Summarize text",
        description="Generate intelligent summary of provided text using optimal algorithm",
        tags=["Summarization"],
        parameters=[],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to summarize",
                                "minLength": 1,
                                "maxLength": 100000
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum summary length in words",
                                "minimum": 10,
                                "maximum": 1000,
                                "default": 100
                            },
                            "algorithm": {
                                "type": "string",
                                "description": "Algorithm to use",
                                "enum": ["auto", "fast", "quality", "hierarchical"],
                                "default": "auto"
                            }
                        },
                        "required": ["text"]
                    },
                    "examples": {
                        "simple": {
                            "value": {
                                "text": "Long article text here...",
                                "max_length": 50
                            }
                        },
                        "advanced": {
                            "value": {
                                "text": "Complex document...",
                                "max_length": 200,
                                "algorithm": "quality"
                            }
                        }
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Successful summarization",
                schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "concepts": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "algorithm_used": {"type": "string"},
                                "processing_time": {"type": "number"},
                                "word_count": {"type": "integer"},
                                "compression_ratio": {"type": "number"}
                            }
                        }
                    }
                }
            ),
            APIResponse(
                status_code=400,
                description="Invalid request",
                schema={"$ref": "#/components/schemas/Error"}
            ),
            APIResponse(
                status_code=429,
                description="Rate limit exceeded",
                schema={"$ref": "#/components/schemas/Error"}
            )
        ],
        rate_limit="30 per minute"
    ))
    
    # Add batch summarization endpoint
    generator.add_endpoint(APIEndpoint(
        path="/api/summarize/batch",
        method="POST",
        summary="Batch summarize documents",
        description="Process multiple documents in a single request",
        tags=["Summarization"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "text": {"type": "string"}
                                    },
                                    "required": ["id", "text"]
                                },
                                "minItems": 1,
                                "maxItems": 100
                            },
                            "max_length": {"type": "integer", "default": 100}
                        },
                        "required": ["documents"]
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Batch processing complete",
                schema={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "keywords": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        },
                        "processing_time": {"type": "number"},
                        "total_documents": {"type": "integer"}
                    }
                }
            )
        ]
    ))
    
    # Add Knowledge OS endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/knowledge/capture",
        method="POST",
        summary="Capture a thought",
        description="Capture and process a thought in the Knowledge OS",
        tags=["Knowledge"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Thought to capture"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata"
                            }
                        },
                        "required": ["thought"]
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Thought captured successfully",
                schema={
                    "type": "object",
                    "properties": {
                        "thought_id": {"type": "string"},
                        "processed": {"type": "boolean"},
                        "concepts": {"type": "array", "items": {"type": "string"}},
                        "connections": {"type": "integer"}
                    }
                }
            )
        ]
    ))
    
    # Add Temporal Intelligence endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/temporal/analyze",
        method="POST",
        summary="Analyze temporal patterns",
        description="Analyze temporal patterns in knowledge evolution",
        tags=["Temporal"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "timeframe": {
                                "type": "string",
                                "description": "Timeframe to analyze",
                                "enum": ["24h", "7d", "30d", "90d", "1y", "all"],
                                "default": "30d"
                            },
                            "concepts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific concepts to track"
                            }
                        }
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Temporal analysis complete",
                schema={
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "description": {"type": "string"},
                                    "significance": {"type": "number"}
                                }
                            }
                        },
                        "breakthroughs": {"type": "array", "items": {"type": "object"}},
                        "evolution_graph": {"type": "object"}
                    }
                }
            )
        ]
    ))
    
    # Add Memory endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/memory/store",
        method="POST",
        summary="Store memory",
        description="Store information in superhuman memory system",
        tags=["Memory"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to memorize"
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": ["semantic", "episodic", "procedural"],
                                "default": "semantic"
                            },
                            "importance": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.5
                            }
                        },
                        "required": ["content"]
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Memory stored successfully",
                schema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "connections_formed": {"type": "integer"},
                        "patterns_detected": {"type": "array", "items": {"type": "string"}}
                    }
                }
            )
        ]
    ))
    
    # Add Webhook management endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/webhooks",
        method="POST",
        summary="Register webhook",
        description="Register a new webhook endpoint",
        tags=["Webhooks"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "Webhook endpoint URL"
                            },
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "document.summarized",
                                        "thought.captured",
                                        "insight.generated",
                                        "pattern.detected"
                                    ]
                                },
                                "description": "Events to subscribe to"
                            },
                            "secret": {
                                "type": "string",
                                "description": "Secret for HMAC signature"
                            }
                        },
                        "required": ["url", "events", "secret"]
                    }
                }
            }
        },
        responses=[
            APIResponse(
                status_code=200,
                description="Webhook registered successfully",
                schema={
                    "type": "object",
                    "properties": {
                        "webhook_id": {"type": "string"},
                        "url": {"type": "string"},
                        "events": {"type": "array", "items": {"type": "string"}},
                        "created_at": {"type": "string", "format": "date-time"}
                    }
                }
            )
        ]
    ))
    
    # Add System endpoints
    generator.add_endpoint(APIEndpoint(
        path="/api/health",
        method="GET",
        summary="Health check",
        description="Check system health and status",
        tags=["System"],
        parameters=[],
        responses=[
            APIResponse(
                status_code=200,
                description="System healthy",
                schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "version": {"type": "string"},
                        "uptime": {"type": "number"},
                        "components": {
                            "type": "object",
                            "properties": {
                                "database": {"type": "string"},
                                "cache": {"type": "string"},
                                "ai_models": {"type": "string"}
                            }
                        }
                    }
                }
            )
        ]
    ))
    
    generator.add_endpoint(APIEndpoint(
        path="/api/metrics",
        method="GET",
        summary="Get metrics",
        description="Get system performance metrics",
        tags=["System"],
        parameters=[
            APIParameter(
                name="period",
                param_type="query",
                data_type="string",
                description="Time period for metrics",
                enum=["1m", "5m", "15m", "1h", "24h"],
                default="5m"
            )
        ],
        responses=[
            APIResponse(
                status_code=200,
                description="Metrics retrieved",
                schema={
                    "type": "object",
                    "properties": {
                        "requests": {
                            "type": "object",
                            "properties": {
                                "total": {"type": "integer"},
                                "success": {"type": "integer"},
                                "error": {"type": "integer"},
                                "rate_per_second": {"type": "number"}
                            }
                        },
                        "latency": {
                            "type": "object",
                            "properties": {
                                "p50": {"type": "number"},
                                "p95": {"type": "number"},
                                "p99": {"type": "number"}
                            }
                        },
                        "resources": {
                            "type": "object",
                            "properties": {
                                "cpu_percent": {"type": "number"},
                                "memory_mb": {"type": "number"},
                                "disk_gb": {"type": "number"}
                            }
                        }
                    }
                }
            )
        ]
    ))
    
    return generator.generate_spec()


def generate_html_documentation(spec: Dict[str, Any]) -> str:
    """
    Generate HTML documentation with Swagger UI.
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} - API Documentation</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.14.0/swagger-ui.css">
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
            }}
            .header p {{
                margin: 0.5rem 0 0 0;
                opacity: 0.9;
            }}
            #swagger-ui {{
                margin: 2rem auto;
                max-width: 1200px;
            }}
            .information-container {{
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 2rem;
            }}
            .info-section {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }}
            .info-section h2 {{
                color: #343a40;
                margin-top: 0;
            }}
            .webhook-signature {{
                background: #282c34;
                color: #abb2bf;
                padding: 1rem;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Version {version}</p>
        </div>
        
        <div class="information-container">
            <div class="info-section">
                <h2>🚀 Getting Started</h2>
                <p>Welcome to the SUM Platform API! This documentation provides everything you need to integrate with our AI-powered text processing services.</p>
                <h3>Quick Start:</h3>
                <ol>
                    <li>Get your API key from the dashboard</li>
                    <li>Include the key in your request headers: <code>X-API-Key: your-key-here</code></li>
                    <li>Make your first request to <code>/api/summarize</code></li>
                </ol>
            </div>
            
            <div class="info-section">
                <h2>🔐 Webhook Security</h2>
                <p>All webhook payloads are signed with HMAC-SHA256. Verify the signature to ensure the request is from SUM:</p>
                <div class="webhook-signature">
import hmac<br>
import hashlib<br>
import json<br><br>

def verify_webhook(secret, signature, payload):<br>
&nbsp;&nbsp;&nbsp;&nbsp;expected = hmac.new(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;secret.encode('utf-8'),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;json.dumps(payload).encode('utf-8'),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hashlib.sha256<br>
&nbsp;&nbsp;&nbsp;&nbsp;).hexdigest()<br>
&nbsp;&nbsp;&nbsp;&nbsp;return f"sha256={{expected}}" == signature
                </div>
            </div>
        </div>
        
        <div id="swagger-ui"></div>
        
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.14.0/swagger-ui-bundle.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.14.0/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {{
                const spec = {spec_json};
                
                window.ui = SwaggerUIBundle({{
                    spec: spec,
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout",
                    defaultModelsExpandDepth: 1,
                    defaultModelExpandDepth: 1,
                    docExpansion: "list",
                    filter: true,
                    showExtensions: true,
                    showCommonExtensions: true,
                    tryItOutEnabled: true
                }});
            }};
        </script>
    </body>
    </html>
    """
    
    return html_template.format(
        title=spec["info"]["title"],
        version=spec["info"]["version"],
        spec_json=json.dumps(spec)
    )


# Example usage
if __name__ == "__main__":
    # Generate API specification
    spec = generate_sum_api_documentation()
    
    # Save as JSON
    with open("api_spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    
    print("API specification saved to api_spec.json")
    
    # Generate HTML documentation
    html = generate_html_documentation(spec)
    
    # Save HTML
    with open("api_documentation.html", "w") as f:
        f.write(html)
    
    print("HTML documentation saved to api_documentation.html")
    
    # Print summary
    print(f"\nAPI Documentation Summary:")
    print(f"- Title: {spec['info']['title']}")
    print(f"- Version: {spec['info']['version']}")
    print(f"- Endpoints: {len(spec['paths'])}")
    print(f"- Tags: {len(spec['tags'])}")
