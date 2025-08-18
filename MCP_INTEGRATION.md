# MCP (Model Context Protocol) Integration Guide

## Overview

SUM now supports the Model Context Protocol (MCP), allowing AI assistants like Claude Desktop, Continue.dev, and other MCP-compatible tools to use SUM's powerful summarization capabilities directly.

## What is MCP?

Model Context Protocol is a standard that enables AI assistants to interact with external tools and services. By implementing MCP, SUM becomes a tool that AI assistants can use to:

- Summarize documents and text
- Extract key concepts
- Detect languages
- Generate hierarchical summaries
- Compare different summarization models

## Installation

### 1. Install MCP Dependencies

```bash
pip install mcp
```

### 2. Configure Your AI Assistant

#### For Claude Desktop

1. Open Claude Desktop settings
2. Go to Developer → MCP Servers
3. Add SUM server configuration:

```json
{
  "sum": {
    "command": "python",
    "args": ["/path/to/SUM/mcp_server.py"],
    "env": {
      "PYTHONPATH": "/path/to/SUM"
    }
  }
}
```

#### For Continue.dev

1. Open Continue settings
2. Add to MCP servers:

```json
{
  "mcpServers": {
    "sum": {
      "command": "python",
      "args": ["/path/to/SUM/mcp_server.py"]
    }
  }
}
```

## Available Tools

### 1. summarize
Summarize text using various AI models.

**Parameters:**
- `text` (required): Text to summarize
- `model`: Choose from "basic", "advanced", "hierarchical", "unlimited" (default: "hierarchical")
- `max_tokens`: Maximum tokens in summary (10-500, default: 100)

**Example:**
```
Use the summarize tool to condense this article about climate change...
```

### 2. batch_summarize
Summarize multiple texts efficiently.

**Parameters:**
- `texts` (required): Array of texts to summarize
- `model`: Model to use for all texts

**Example:**
```
Use batch_summarize to process these 5 research abstracts...
```

### 3. detect_language
Detect the language of any text.

**Parameters:**
- `text` (required): Text to analyze

**Example:**
```
What language is this text written in: "Bonjour le monde"?
```

### 4. hierarchical_summary
Generate multi-level summaries from essence to comprehensive.

**Parameters:**
- `text` (required): Text to summarize
- `include_insights`: Include key insights extraction (default: true)

**Example:**
```
Create a hierarchical summary of this research paper...
```

### 5. extract_concepts
Extract key concepts and themes from text.

**Parameters:**
- `text` (required): Text to analyze
- `max_concepts`: Maximum concepts to extract (5-20, default: 10)

**Example:**
```
Extract the main concepts from this technical documentation...
```

### 6. compare_summaries
Compare summaries from different models.

**Parameters:**
- `text` (required): Text to summarize
- `models`: Array of models to compare

**Example:**
```
Compare how different models summarize this news article...
```

## Usage Examples

### With Claude Desktop

Once configured, you can ask Claude:

```
"Summarize this research paper using the hierarchical model"
"Extract the key concepts from this document"
"What language is this text written in?"
"Compare basic vs advanced summaries of this article"
```

Claude will automatically use the SUM MCP server to process your requests.

### With Continue.dev

In your code editor with Continue:

```
@sum Can you summarize this README file?
@sum Extract concepts from the selected code comments
```

### Programmatic Usage

You can also use the MCP server programmatically:

```python
from mcp import MCPClient

client = MCPClient("sum")
result = await client.call_tool("summarize", {
    "text": "Your long text here...",
    "model": "hierarchical",
    "max_tokens": 150
})
```

## Advanced Features

### Caching
The MCP server automatically uses SUM's smart caching system for improved performance on repeated requests.

### Language Support
All tools automatically detect and handle multiple languages including:
- English, Spanish, French, German, Italian
- Portuguese, Russian, Dutch, Swedish, Polish
- Turkish, Indonesian, and more

### Unlimited Text Processing
The "unlimited" model can handle texts of any size through intelligent chunking and streaming.

## Troubleshooting

### Server Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path is correct in configuration
- Verify MCP is installed: `pip install mcp`

### Tools Not Available
- Restart your AI assistant after configuration
- Check server logs for errors
- Ensure the mcp_server.py file is executable

### Performance Issues
- The first request may be slower due to model initialization
- Subsequent requests use caching for faster responses
- Consider using "basic" model for simple tasks

## Development

To add custom tools to the MCP server:

1. Edit `mcp_server.py`
2. Add new tool registration in `_register_tools()`
3. Implement the tool handler method
4. Update `mcp_config.json` with tool schema

Example:
```python
self.register_tool(Tool(
    name="custom_summary",
    description="Custom summarization logic",
    input_schema={...},
    handler=self.custom_summary_handler
))
```

## Security

- The MCP server runs locally by default
- No data is sent to external services
- API keys are not required for MCP usage
- All processing happens on your machine

## Support

For issues or questions:
- Check the [GitHub Issues](https://github.com/OtotaO/SUM/issues)
- Review server logs for detailed error messages
- Ensure you're using the latest version of SUM

The MCP integration makes SUM's powerful summarization capabilities available to any AI assistant, enabling more intelligent and context-aware document processing workflows.