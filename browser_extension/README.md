# SUM Browser Extension

Instantly summarize any text on the web with SUM - the world's most powerful summarization tool.

## Features

- **Instant Summarization**: Select any text and summarize with one click
- **Full Page Summary**: Summarize entire articles and web pages
- **Keyboard Shortcuts**: Quick access with customizable hotkeys
- **Smart Caching**: Lightning-fast repeated summaries
- **Multiple Models**: Choose from simple, advanced, hierarchical, and more
- **History Tracking**: Access your recent summaries anytime
- **Dark Mode**: Automatic theme switching based on system preferences

## Installation

### Chrome / Edge / Brave

1. Clone the repository or download the extension folder
2. Open Chrome and navigate to `chrome://extensions`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the `browser_extension/chrome` folder
5. The SUM icon should appear in your browser toolbar

### Firefox

1. Clone the repository or download the extension folder
2. Open Firefox and navigate to `about:debugging`
3. Click "This Firefox" in the sidebar
4. Click "Load Temporary Add-on"
5. Select the `manifest.json` file in `browser_extension/firefox`

## Setup

1. **Start the SUM API server**:
   ```bash
   cd SUM
   python app.py
   ```

2. **Configure the extension**:
   - Click the SUM extension icon
   - Click "Settings" at the bottom
   - Verify the API URL (default: `http://localhost:5001`)
   - Optionally add an API key for higher rate limits

## Usage

### Summarize Selected Text
1. Select any text on a webpage
2. Click the floating "Summarize" button that appears
   OR
3. Right-click and select "Summarize with SUM"
   OR
4. Press `Ctrl+Shift+S` (Windows/Linux) or `Cmd+Shift+S` (Mac)

### Summarize Entire Page
1. Click the SUM extension icon
2. Click "Summarize Page"
   OR
3. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)

### Manual Text Input
1. Click the SUM extension icon
2. Paste or type text in the input area
3. Click "Summarize"

## Features in Detail

### Summary Display
- **Main Summary**: Concise overview of the content
- **Hierarchical Levels**: Essence, core, and expanded summaries
- **Key Concepts**: Automatically extracted tags and themes
- **Metadata**: Model used, compression ratio, cache status

### Keyboard Shortcuts
- `Ctrl+Shift+S`: Summarize selected text
- `Ctrl+Shift+P`: Summarize entire page
- Customize shortcuts in Chrome: `chrome://extensions/shortcuts`

### Settings
- **API Configuration**: Set custom API URL and authentication
- **Model Selection**: Choose default summarization model
- **Summary Length**: Adjust token limits (50-500)
- **Caching**: Enable/disable result caching
- **Data Management**: Clear cache, history, export data

### API Key Benefits
With an API key:
- 60 requests/minute (vs 20 without)
- Access to unlimited text processing
- Priority processing
- Usage analytics

## Development

### Project Structure
```
browser_extension/
├── chrome/
│   ├── manifest.json      # Extension manifest (v3)
│   ├── src/
│   │   ├── background.js  # Service worker
│   │   ├── content.js     # Content script
│   │   ├── popup.html/js  # Extension popup
│   │   └── options.html/js # Settings page
│   └── icons/            # Extension icons
├── firefox/              # Firefox-specific files
└── edge/                # Edge-specific files
```

### Building for Production

1. **Generate icons** (16x16, 32x32, 48x48, 128x128 PNG):
   ```bash
   # Use any image editor to create icons
   # Save to chrome/icons/ directory
   ```

2. **Package the extension**:
   ```bash
   # Chrome
   cd browser_extension/chrome
   zip -r sum-chrome-extension.zip . -x "*.DS_Store"
   
   # Firefox
   cd browser_extension/firefox
   zip -r sum-firefox-extension.zip . -x "*.DS_Store"
   ```

### Testing
1. Load the extension in developer mode
2. Test on various websites
3. Check console for errors
4. Verify all features work correctly

## Troubleshooting

### Extension not connecting to API
- Ensure SUM API is running (`python app.py`)
- Check API URL in settings (default: `http://localhost:5001`)
- Verify no firewall blocking local connections

### Summaries not appearing
- Check if content script is injected (some sites block extensions)
- Verify API key is valid (if using one)
- Check browser console for errors

### Keyboard shortcuts not working
- Some websites may override shortcuts
- Check shortcut configuration in browser settings
- Try using the context menu instead

## Privacy

The SUM browser extension:
- Only sends selected text to your configured API server
- Stores summaries locally in browser storage
- Does not collect any personal information
- All data stays on your machine (when using local API)

## Contributing

To contribute to the browser extension:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly on multiple browsers
4. Submit a pull request

## License

Apache License 2.0 - See LICENSE file for details