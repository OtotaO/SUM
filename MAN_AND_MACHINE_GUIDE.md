# Man and Machine: The Symbiotic Guide to SUM

SUM is designed not just for human users but also for AI agents (Machines). This guide explains how to leverage the full power of SUM for both.

## 🤖 For Machine (AI Agents & MCP)

SUM implements the **Model Context Protocol (MCP)**, allowing AI assistants (like Claude, Cursor, Continue.dev) to natively use SUM as a tool.

### New Capabilities
We have expanded the toolset to include "Extrapolation" - the ability to grow knowledge.

#### 1. `extrapolate`
Turn a simple seed into a full article or essay.
- **Usage**: "Expand this concept into a professional article."
- **Parameters**: `seed`, `format` (article, essay), `style`, `tone`, `length`.

#### 2. `generate_book`
Architect and write a complete book from a single concept.
- **Usage**: "Write a book about 'The Future of AI' in an academic style."
- **Parameters**: `concept`, `style`, `tone`.

#### 3. `summarize` & `hierarchical_summary`
Distill knowledge into its essence.
- **Usage**: "Give me a hierarchical summary of this text."

### Integration
To connect your AI agent to SUM:
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

## 👨‍💻 For Man (Developers & Users)

SUM is built to be **Robust**. It works when the internet is down, when APIs fail, and when you need privacy.

### 1. Robust Local AI (Ollama)
SUM detects if you have **Ollama** running locally (`localhost:11434`).
- **No API Keys needed**: Just run `ollama serve`.
- **Privacy**: Your data never leaves your machine.
- **Fallback**: If OpenAI/Anthropic APIs fail, SUM automatically degrades to Local AI, and then to NLTK-based processing if necessary.

### 2. Universal LLM Backend
Our backend (`llm_backend.py`) is intelligent:
1.  Checks for `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
2.  Checks for local `Ollama` instance.
3.  Checks for `HuggingFace` transformers.
4.  Falls back to `LocalProvider` (NLTK) for pure algorithmic summarization.

### 3. CLI & API
You are not limited to the Web UI.
- **CLI**: `python sum_cli_simple.py "your text"`
- **API**: The Flask server exposes robust endpoints at `http://localhost:5001/api`.

## 🛡️ Robustness Architecture

We believe in **Graceful Degradation**.
- **Network Failure?** -> Switch to Local AI.
- **Local AI Down?** -> Switch to NLTK algorithms.
- **Memory Full?** -> Switch to Streaming processing.
- **File Too Big?** -> Switch to "Unlimited" chunked processing.

SUM is the engine that keeps running, ensuring you always have access to knowledge distillation and expansion.
