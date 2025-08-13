# Clipboard Integration Guide

SUM provides powerful clipboard integration capabilities that allow you to instantly capture and summarize text from anywhere on your system.

## Features

### 1. Global Hotkey Capture (Ctrl+Shift+T)
- **Instant Access**: Press `Ctrl+Shift+T` anywhere on your system to open the capture popup
- **Sub-100ms Response**: The popup appears almost instantly (<100ms)
- **Auto-Paste**: Automatically pastes clipboard content if available
- **Beautiful UI**: Modern, minimal interface with dark theme

### 2. Browser Extension
Located in `/capture/browser_extension/`, the extension provides:
- One-click capture from any webpage
- Automatic text extraction
- Direct integration with SUM API

### 3. Zero-Friction Capture
The capture system is designed for minimal interaction:
- Auto-detects clipboard content
- Pre-fills the capture window
- Single keyboard shortcut to process (`Ctrl+Enter`)
- `Esc` to cancel quickly

## Installation

### Global Hotkey System
```bash
# Install keyboard library
pip install keyboard

# Run the global hotkey daemon
python -m capture.global_hotkey
```

### Browser Extension
1. Open your browser's extension management page
2. Enable Developer Mode
3. Click "Load unpacked" and select `/capture/browser_extension/`

## Usage

### Quick Capture Workflow
1. Copy any text to your clipboard
2. Press `Ctrl+Shift+T` from anywhere
3. The capture popup appears with your text pre-filled
4. Press `Ctrl+Enter` to summarize
5. Results appear in your browser

### Keyboard Shortcuts
- `Ctrl+Shift+T`: Open capture popup (global)
- `Ctrl+Enter`: Process and summarize
- `Esc`: Cancel and close
- `Ctrl+V`: Paste (if not auto-pasted)

### Advanced Features

#### Custom Hotkey Configuration
Edit `capture/hotkey_config.json`:
```json
{
  "hotkey": "ctrl+shift+t",
  "enabled": true,
  "auto_paste_clipboard": true,
  "popup_timeout": 30
}
```

#### Programmatic Access
```python
from capture.capture_engine import capture_engine, CaptureSource

# Capture text directly
request_id = capture_engine.capture_text(
    text="Your text here",
    source=CaptureSource.CLIPBOARD,
    callback=lambda result: print(f"Summary: {result.summary}")
)
```

## System Requirements

### Windows
- No additional requirements
- Works with all Windows 10/11 versions

### macOS
- May require accessibility permissions
- Grant permissions when prompted

### Linux
- Requires X11 or Wayland
- May need to run with sudo for global hotkeys

## Troubleshooting

### Hotkey Not Working
1. Check if another application is using the same hotkey
2. Ensure the hotkey daemon is running
3. On macOS/Linux, check permissions

### Clipboard Not Auto-Pasting
1. Ensure clipboard contains text (not images/files)
2. Check `auto_paste_clipboard` is `true` in config
3. Some applications may block clipboard access

### Performance Issues
1. Close other heavy applications
2. Check system resources
3. Disable animations in config if needed

## API Integration

The clipboard system integrates with SUM's API:

```bash
# Direct API call with clipboard content
curl -X POST http://localhost:3000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{"text": "CLIPBOARD_CONTENT_HERE", "source": "clipboard"}'
```

## Security Considerations

1. **Privacy**: Clipboard content is never stored permanently
2. **Local Processing**: All processing happens locally by default
3. **Secure Communication**: Uses HTTPS when configured
4. **No External Sharing**: Clipboard data stays on your machine

## Tips for Productivity

1. **Research Workflow**: Copy interesting paragraphs and summarize instantly
2. **Meeting Notes**: Capture and condense meeting transcripts
3. **Email Summaries**: Quickly summarize long email threads
4. **Code Documentation**: Extract and summarize code comments
5. **News Articles**: Get quick summaries of news articles

## Future Enhancements

- Smart clipboard monitoring (auto-summarize when copying)
- Multi-monitor support for popup positioning
- Clipboard history with summaries
- Integration with system notifications
- Mobile app support via shared clipboard