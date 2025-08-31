# 🚀 SUM Quick Start - 60 Seconds to Magic

## 🎯 One Command Setup

```bash
python setup.py
```

That's it! The setup wizard will:
- ✅ Install all dependencies
- ✅ Configure your environment  
- ✅ Set up legendary features (optional)
- ✅ Create shortcuts
- ✅ Test everything works

## 🏃 Start Using SUM

### Option 1: Web Interface (Easiest)
```bash
./run.sh
# Open browser to http://localhost:5001
```

### Option 2: API
```python
import requests

response = requests.post('http://localhost:5001/api/crystallize', 
    json={
        'text': 'Your text here',
        'density': 'tweet',  # essence, tweet, summary, standard, detailed
        'style': 'hemingway'  # academic, executive, storyteller, etc.
    })

print(response.json()['result'])
```

### Option 3: Command Line
```bash
curl -X POST http://localhost:5001/api/process_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "model": "hierarchical"}'
```

### Option 4: Browser Extension
1. Open Chrome/Firefox
2. Go to Extensions → Developer Mode
3. Load Unpacked → Select `browser_extension/chrome`
4. Select any text on any website → Click "Summarize"

### Option 5: macOS App (Mac only)
```bash
open macOS/SumApp.xcodeproj
# Press ⌘R to build and run
```

## 🎨 Density Levels

Choose your information density:

| Level | Ratio | Use Case |
|-------|-------|----------|
| `essence` | 1% | Single most important sentence |
| `tweet` | 5% | Twitter-length summary |
| `elevator` | 10% | 30-second pitch |
| `summary` | 20% | Executive summary |
| `standard` | 30% | Default balanced summary |
| `detailed` | 40% | Comprehensive overview |
| `thorough` | 50% | In-depth analysis |
| `complete` | 70% | Near-complete retention |

## 🎭 Style Personas

Transform your summary style:

- **hemingway**: Short. Clear. Direct.
- **academic**: Scholarly and precise
- **executive**: Business-focused insights
- **storyteller**: Narrative flow
- **technical**: Engineering precision
- **journalist**: News-style reporting
- **educator**: Teaching clarity
- **poet**: Beautiful language
- **scientist**: Data and evidence
- **philosopher**: Deep thinking

## 🌟 Legendary Features

### Knowledge Crystallization
```python
# Ultra-dense knowledge extraction
response = requests.post('http://localhost:5001/api/crystallize',
    json={'text': long_document, 'density': 'essence'})
```

### GraphRAG Analysis
```python
# Corpus-level understanding
response = requests.post('http://localhost:5001/api/graphrag/analyze',
    json={'documents': [doc1, doc2, doc3]})
```

### RAPTOR Trees
```python
# Hierarchical summarization
response = requests.post('http://localhost:5001/api/raptor/build',
    json={'text': document})
```

## 💡 Pro Tips

1. **Add API Keys** for enhanced AI:
   ```bash
   # Edit .env file
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Process Large Files**:
   ```bash
   curl -X POST http://localhost:5001/api/process_unlimited \
     -F "file=@massive_document.pdf"
   ```

3. **Batch Processing**:
   ```python
   # Process 1000 documents at once
   response = requests.post('http://localhost:5001/api/mass/upload',
       files=[('files', open(f, 'rb')) for f in file_list])
   ```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Server won't start | Run `python setup.py` |
| Missing features | Install legendary: `pip install -r requirements-legendary.txt` |
| Slow performance | Add API keys in `.env` |
| Browser extension not working | Check console → Load unpacked extension |

## 🎉 You're Ready!

Start summarizing with:
```bash
./run.sh
```

Welcome to the future of text summarization! 🚀