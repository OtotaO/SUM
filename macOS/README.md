# 🍎 SUM for macOS

Native Mac app for the world's most advanced text intelligence platform.

## Features

### 🚀 Native Performance
- Built with SwiftUI for buttery-smooth 120Hz ProMotion
- Apple Silicon optimized
- Instant launch (<0.5s)
- Low memory footprint (<100MB)

### 🎯 System Integration
- **Services Menu**: Summarize from any app
- **Share Extension**: Share to SUM from anywhere
- **Spotlight**: Search and summarize instantly
- **Shortcuts**: Automate with Shortcuts app
- **Menu Bar**: Always accessible quick capture

### 💎 Mac-Specific Features
- Beautiful native UI that matches macOS
- Keyboard shortcuts for everything
- Touch Bar support (Intel Macs)
- Stage Manager compatible
- Universal Control ready

### 🔒 Privacy First
- All processing on-device by default
- Your data never leaves your Mac
- Bring your own API keys
- No tracking, no analytics

## Installation

### Option 1: Download DMG (Easiest)
1. Download `SUM-2.0.0.dmg` from Releases
2. Open the DMG file
3. Drag SUM to Applications
4. Launch from Applications or Spotlight

### Option 2: Build from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/SUM.git
cd SUM/macOS

# Make build script executable
chmod +x build.sh

# Build the app
./build.sh

# Open the app
open build/SUM.app
```

### Option 3: Homebrew
```bash
brew install --cask sum-app
```

## System Requirements

- macOS 13.0 (Ventura) or later
- Apple Silicon (M1/M2/M3) or Intel processor
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space
- Python 3.9+ (for AI features)

## Quick Start

### First Launch
1. Open SUM from Applications
2. Grant necessary permissions (if prompted)
3. Choose your AI model preference
4. Start writing!

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Document | ⌘N |
| Open | ⌘O |
| Save | ⌘S |
| Summarize | ⌘⇧S |
| Crystallize | ⌘K |
| Brainstorm | ⌘B |
| Toggle Intelligence Panel | ⌘⇧I |
| Preferences | ⌘, |
| Quick Capture (Global) | ⌥⌘S |

### Using Services Menu
1. Select text in any app
2. Right-click → Services → "Summarize with SUM"
3. View summary in notification

### Menu Bar App
1. Click the ✨ icon in menu bar
2. Type or paste text
3. Hit ⌘S to summarize instantly

## Configuration

### AI Models
SUM supports multiple AI providers. Configure in Preferences → AI Models:

- **Local Models** (No API key needed):
  - Llama 2/3
  - Mistral
  - Phi

- **Cloud Models** (BYO API key):
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)
  - Cohere

### Preferences

Access via ⌘, or SUM → Preferences:

- **General**: Default density, style, behavior
- **AI Models**: Configure providers and API keys
- **Privacy**: Data handling, local-only mode
- **Shortcuts**: Customize keyboard shortcuts
- **Advanced**: Debug options, cache settings

## Features

### Live Summary
- Adjust density slider (1% - 100%)
- Updates in real-time as you type
- Multiple style personas
- Export at any density

### Intelligent Suggestions
- Context-aware completions
- Brainstorming ideas
- Evidence and citations
- Connection discovery

### Document Types
- Plain text (.txt)
- Markdown (.md)
- Rich Text (.rtf)
- PDF documents
- Word documents (.docx)
- Web pages (via Share)

### Export Options
- PDF with formatting
- Markdown
- Plain text
- HTML
- Multiple density levels

## Troubleshooting

### App won't open
- Check System Preferences → Security & Privacy
- May need to right-click → Open first time
- Ensure macOS 13.0+

### Python features not working
```bash
# Install Python dependencies
pip3 install -r requirements.txt
```

### Performance issues
- Check Activity Monitor for high CPU
- Clear cache: ~/Library/Caches/com.opensource.sum
- Reset preferences: Hold ⌥ while launching

### Services menu not showing
1. System Preferences → Keyboard → Shortcuts → Services
2. Enable "Summarize with SUM"
3. Restart the app

## Development

### Building from Xcode
1. Open `SumApp.xcodeproj`
2. Select your development team
3. Build and run (⌘R)

### Architecture
```
SumApp/
├── SumApp.swift          # Main app entry
├── PythonBridge.swift    # Python integration
├── Views/                # SwiftUI views
├── Models/               # Data models
├── Services/             # System services
└── Resources/            # Assets, Python modules
```

### Contributing
See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Testing
```bash
swift test
```

## Privacy

SUM for macOS is designed with privacy first:

- **No telemetry**: Zero tracking or analytics
- **Local by default**: Use without internet
- **Your keys**: BYO API keys, we never see them
- **Open source**: Audit the code yourself

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/SUM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SUM/discussions)
- **Wiki**: [Documentation](https://github.com/yourusername/SUM/wiki)

## Credits

Built with ❤️ by the open source community.

Special thanks to:
- SwiftUI for the beautiful native UI
- PythonKit for seamless Python integration
- All our contributors

---

**SUM for macOS** - Making every Mac a knowledge crystallization machine.

*The text intelligence platform that feels like Apple built it.*