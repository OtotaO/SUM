# SUM - The Universal Knowledge Engine

> **From Atomic Tags to Infinite Books. Bi-Directional Knowledge Transformation.**

SUM is the world's first **bi-directional** text engine. It doesn't just summarize; it can also extrapolate. It allows you to move fluidly across the spectrum of knowledge density, from the atomic level (Tags) to the universal level (Books).

## Key Capabilities

### 1. Distill (Summarize)
Turn massive documents into their essence.
- **Tags**: Extract the atomic concepts.
- **SUM**: Generate a minimal "elevator pitch" summary.
- **Thought**: Create coherent paragraphs that capture the core message.
- **Context**: Produce article-length summaries that preserve nuance.

### 2. Expand (Extrapolate)
Turn simple seeds into comprehensive content.
- **Deepen**: Expand a sentence into a full essay.
- **Create**: Turn a single concept tag into a 5-chapter book.
- **Architect**: Automatically generate Table of Contents and fill them with content.

### 3. The Universal Slider
Our revolutionary UI features a single slider that controls knowledge density.
- **Left Side (0-2)**: Contraction. Reduces information entropy.
- **Right Side (3-5)**: Expansion. Increases information entropy.

## Features

- **Recursive Book Generation**: Automatically architects a book structure and writes it chapter by chapter using a two-phase "Blueprint -> Draft" process.
- **Streaming Intelligence**: Watch as the system "thinks" and generates content in real-time with transparent system logs.
- **Unlimited Context**: Process files from 1KB to 1TB using intelligent memory mapping (mmap) and streaming chunkers.
- **Legendary Intelligence**: Includes GraphRAG and RAPTOR implementations with robust "Light Mode" fallbacks for environments without heavy ML libraries.
- **Smart Caching**: Instant results for previously processed concepts.
- **Live Markdown**: Beautifully formatted output with headers, bolding, and structure.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Run the Universal Engine
python main.py
```

### Usage

1. **Open** `http://localhost:5001`
2. **Choose your input**: Paste text or upload a file.
3. **Slide** the Universal Spectrum Slider:
   - Slide **Left** to distill down to tags.
   - Slide **Right** to expand into a book.
4. **Watch** the transformation happen in real-time.

## Technology Stack

- **Backend**: Python, Flask, Universal LLM Backend (OpenAI/Ollama/Local)
- **Engine**: Hierarchical Densification Engine (for Summarization) & Recursive Extrapolation Engine (for Expansion)
- **Frontend**: SSE (Server-Sent Events) for real-time streaming, dynamic CSS variables.
- **Intelligence Architecture**: 
  - **Core**: Extractive & Abstractive summarization via NLTK & LLMs.
  - **Advanced**: GraphRAG (Graph-based retrieval) & RAPTOR (Recursive tree summarization).
  - **Multi-Agent**: Prototype orchestration system simulating 10+ specialized roles.

## Deployment

This is a server-side Python application using Flask. It requires a runtime environment and cannot be deployed to static hosting (like Firebase Hosting or GitHub Pages).

**Recommended Deployment Options:**
- **Google Cloud Run**: Ideal for containerized Python apps.
- **AWS App Runner**: Fully managed container application service.
- **Heroku / Railway / Render**: Simple PaaS deployment.
- **VPS (DigitalOcean/Linode)**: Run with `gunicorn` or `systemd`.

## Configuration

Set your API keys in `.env` or environment variables:

```bash
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
# OR use local models (Ollama) automatically if installed!
```

## Contributing

We welcome visionaries who want to help map the entire spectrum of human knowledge.

1. Fork it
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache 2.0 - Built for the future of Man and Machine.

---

<p align="center">
  <strong>SUM - Distill the Universe. Expand the Atom.</strong>
</p>