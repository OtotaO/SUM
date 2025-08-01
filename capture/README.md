# 🎯 SUM Zero-Friction Capture System

Revolutionary capture system that transforms SUM from "another great tool" into "the future of human-computer cognitive collaboration." This system makes capturing and processing thoughts as natural as breathing.

## ✨ Features

### 🎯 Zero-Friction Capture Everywhere

1. **Global Hotkey System** - Press `Ctrl+Shift+T` anywhere on desktop for instant text capture
2. **Browser Extension** - One-click capture from any webpage with context awareness
3. **Beautiful UI** - Sub-100ms popup with progress indication
4. **Background Processing** - Powered by optimized SumEngine for sub-second results
5. **Cross-Platform** - Works on Windows, macOS, and Linux

### 🚀 Revolutionary Performance

- **Sub-second processing** for most content
- **Context-aware summarization** based on source and content type
- **Intelligent algorithm selection** (fast/quality/hierarchical)
- **Beautiful progress indication** that doesn't interrupt flow
- **Production-ready** with proper error handling

## 🔧 Installation

### Prerequisites

```bash
# Install Python dependencies
cd /path/to/SUM/capture
pip install -r requirements.txt
```

### Global Hotkey System

The global hotkey system requires the `keyboard` library:

```bash
pip install keyboard
```

**Note**: On Linux, you may need to run with sudo for global hotkey access, or configure udev rules for your user.

### Browser Extension

1. **Chrome/Edge**:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `browser_extension` folder

2. **Firefox**:
   - Go to `about:debugging`
   - Click "This Firefox"
   - Click "Load Temporary Add-on"
   - Select the `manifest.json` file

## 🚀 Quick Start

### Start All Services

```bash
# Start the complete capture system
python -m capture.launcher

# Or from the SUM root directory
python -m capture.launcher
```

This starts:
- Global hotkey system (Ctrl+Shift+T)
- HTTP API server (localhost:8000)
- Background capture processing

### Global Hotkey Usage

1. **Press `Ctrl+Shift+T`** anywhere on your desktop
2. **Beautiful popup appears** in under 100ms
3. **Paste or type text** (clipboard auto-pastes if available)
4. **Press `Ctrl+Enter`** or click "Capture & Summarize"
5. **Get instant results** with summary and keywords

### Browser Extension Usage

1. **Select text** on any webpage
2. **Floating button appears** with capture options
3. **Click capture** or use `Ctrl+Shift+S` shortcut
4. **Get context-aware summary** based on page type

### API Usage

```python
import requests

# Capture text via API
response = requests.post('http://localhost:8000/api/capture', json={
    'text': 'Your text to summarize here...',
    'source': 'api',
    'context': {
        'source_app': 'my_app',
        'user_id': 'user123'
    }
})

result = response.json()
print(f"Summary: {result['summary']}")
print(f"Keywords: {result['keywords']}")
```

## 🎯 Architecture

### Core Components

```
capture/
├── capture_engine.py     # Core processing engine
├── global_hotkey.py      # Cross-platform hotkey system  
├── api_server.py         # HTTP API for external clients
├── launcher.py           # Unified service launcher
└── browser_extension/    # Chrome/Firefox extension
    ├── manifest.json
    ├── background.js
    ├── content.js
    ├── popup.html
    ├── popup.css
    └── popup.js
```

### Processing Flow

1. **Capture Request** → Various sources (hotkey, browser, API)
2. **Context Analysis** → Intelligent source and content type detection  
3. **Algorithm Selection** → Fast/Quality/Hierarchical based on content
4. **SumEngine Processing** → Optimized summarization with the bulletproof engine
5. **Result Delivery** → Callbacks, UI updates, or API responses

## ⚙️ Configuration

### Command Line Options

```bash
# Start with custom settings
python -m capture.launcher --port 9000 --host 0.0.0.0

# Disable global hotkey (API only)
python -m capture.launcher --no-hotkey

# API server only
python -m capture.launcher --api-only

# Custom log level
python -m capture.launcher --log-level DEBUG
```

### Hotkey Configuration

The global hotkey can be customized by editing `hotkey_config.json`:

```json
{
  "hotkey": "ctrl+shift+t",
  "enabled": true,
  "auto_paste_clipboard": true,
  "popup_timeout": 30
}
```

### Browser Extension Settings

Access settings through the extension popup or configure programmatically:

```javascript
// Update extension settings
chrome.storage.sync.set({
  showFloatingButton: true,
  summarizationLevel: 'quality',
  maxSummaryLength: 150
});
```

## 🔌 API Reference

### POST /api/capture

Synchronous text capture and processing.

**Request:**
```json
{
  "text": "Text to process",
  "source": "browser_extension",
  "context": {
    "url": "https://example.com",
    "title": "Page Title", 
    "capture_type": "selection"
  }
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "summary": "Generated summary",
  "keywords": ["key", "words"],
  "concepts": ["concept1", "concept2"],
  "processing_time": 0.234,
  "algorithm_used": "quality",
  "confidence_score": 0.92
}
```

### POST /api/capture/async

Asynchronous processing - returns immediately.

### GET /api/capture/status/{request_id}

Check status of async request.

### GET /api/stats

Get system statistics.

### GET /health

Health check endpoint.

## 🎨 UI Components

### Global Hotkey Popup

- **Modern Design** - Gradient backgrounds with blur effects
- **Instant Response** - Sub-100ms appearance time
- **Smart Features** - Auto-paste clipboard, word count, progress indication
- **Keyboard Shortcuts** - Ctrl+Enter to capture, Esc to cancel

### Browser Extension

- **Floating Button** - Appears on text selection with smooth animations
- **Context Menu** - Right-click options for capture and insights
- **Popup Interface** - Statistics, recent captures, and settings
- **Progress Indication** - Beautiful loading states and notifications

## 🔧 Development

### Running Tests

```bash
# Run capture system tests
python -m pytest capture/ -v

# Test specific components
python -m pytest capture/test_capture_engine.py
python -m pytest capture/test_global_hotkey.py
```

### Development Mode

```bash
# Start with debug logging
python -m capture.launcher --log-level DEBUG

# Monitor capture engine performance
curl http://localhost:8000/api/stats | python -m json.tool
```

### Browser Extension Development

1. Make changes to extension files
2. Go to `chrome://extensions/`
3. Click reload button for SUM extension
4. Test changes on webpages

## 🚀 Advanced Usage

### Custom Capture Sources

```python
from capture import capture_engine, CaptureSource

# Create custom capture source
result_id = capture_engine.capture_text(
    text="Custom text",
    source=CaptureSource.API_WEBHOOK,
    context={
        'custom_field': 'value',
        'processing_priority': 'high'
    },
    callback=lambda result: print(f"Done: {result.summary}")
)
```

### Integration with Other Tools

```python
# Slack bot integration
@app.route('/slack/capture', methods=['POST'])
def slack_capture():
    text = request.form.get('text')
    user_id = request.form.get('user_id')
    
    result_id = capture_engine.capture_text(
        text=text,
        source=CaptureSource.API_WEBHOOK,
        context={'slack_user': user_id}
    )
    
    return jsonify({'response_type': 'in_channel'})
```

## 🎯 Roadmap

- [x] Global hotkey system with beautiful popup
- [x] Browser extension for Chrome/Firefox  
- [x] HTTP API with async support
- [x] Context-aware processing
- [ ] Mobile app with voice capture
- [ ] Email integration (Gmail/Outlook)
- [ ] Slack/Discord bots
- [ ] Obsidian/Notion integrations
- [ ] OCR for document capture

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.