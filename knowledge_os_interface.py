#!/usr/bin/env python3
"""
knowledge_os_interface.py - Beautiful Web Interface for Knowledge Operating System

A thoughtful, prose-driven interface that makes capturing and organizing thoughts
feel like having a conversation with a wise friend. Every interaction is designed
to encourage deeper thinking and provide profound insights.

Author: ototao
License: Apache License 2.0
"""

from flask import Flask, render_template_string, request, jsonify, session
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import threading
import time

# Core Knowledge OS
from knowledge_os import (
    KnowledgeOperatingSystem, 
    Thought, 
    KnowledgeCluster,
    CaptureSession
)

# Configure beautiful logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KnowledgeOSInterface')

# Initialize Flask app with beautiful design
app = Flask(__name__)
app.secret_key = 'knowledge-os-session-key'

# Global Knowledge OS instance
knowledge_os = None

def init_knowledge_os():
    """Initialize the Knowledge OS system."""
    global knowledge_os
    if knowledge_os is None:
        knowledge_os = KnowledgeOperatingSystem()
        logger.info("Knowledge OS initialized for web interface")

# Beautiful HTML template with embedded CSS and JavaScript
KNOWLEDGE_OS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Operating System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #fefefe;
            --bg-secondary: #f8f9fa;
            --text-primary: #2d3748;
            --text-secondary: #718096;
            --text-accent: #4a5568;
            --border-light: #e2e8f0;
            --border-focus: #667eea;
            --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --gradient-subtle: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-prose: 'Crimson Text', Georgia, serif;
        }

        [data-theme="dark"] {
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --text-accent: #e2e8f0;
            --border-light: #4a5568;
            --border-focus: #667eea;
        }

        body {
            font-family: var(--font-body);
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
            transition: all 0.3s ease;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
            min-height: 100vh;
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            background: var(--gradient-subtle);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header .subtitle {
            font-family: var(--font-prose);
            font-size: 1.2rem;
            color: var(--text-secondary);
            font-style: italic;
        }

        .main-interface {
            display: grid;
            gap: 2rem;
            grid-template-columns: 1fr;
        }

        @media (min-width: 768px) {
            .main-interface {
                grid-template-columns: 2fr 1fr;
            }
        }

        .capture-section {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow-soft);
            border: 1px solid var(--border-light);
        }

        .prompt-display {
            font-family: var(--font-prose);
            font-size: 1.1rem;
            color: var(--text-accent);
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: var(--bg-primary);
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }

        .thought-input {
            width: 100%;
            min-height: 120px;
            padding: 1rem;
            border: 2px solid var(--border-light);
            border-radius: 12px;
            font-family: var(--font-prose);
            font-size: 1.1rem;
            line-height: 1.6;
            background: var(--bg-primary);
            color: var(--text-primary);
            resize: vertical;
            transition: all 0.3s ease;
        }

        .thought-input:focus {
            outline: none;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .thought-input::placeholder {
            color: var(--text-secondary);
            font-style: italic;
        }

        .capture-actions {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary {
            background: var(--gradient-subtle);
            color: white;
            box-shadow: var(--shadow-soft);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-medium);
        }

        .btn-secondary {
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border-light);
        }

        .btn-secondary:hover {
            background: var(--border-light);
        }

        .insights-panel {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow-soft);
            border: 1px solid var(--border-light);
        }

        .panel-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .recent-thoughts {
            margin-bottom: 2rem;
        }

        .thought-card {
            background: var(--bg-primary);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-light);
            transition: all 0.2s ease;
        }

        .thought-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-soft);
        }

        .thought-meta {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .thought-content {
            font-family: var(--font-prose);
            line-height: 1.6;
        }

        .thought-tags {
            margin-top: 0.5rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .tag {
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .system-insights {
            background: var(--bg-primary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-light);
        }

        .insight-metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border-light);
        }

        .insight-metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-weight: 500;
            color: var(--text-accent);
        }

        .metric-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        .beautiful-summary {
            font-family: var(--font-prose);
            font-size: 1rem;
            line-height: 1.7;
            color: var(--text-accent);
            font-style: italic;
            padding: 1rem;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 8px;
            margin-top: 1rem;
        }

        .theme-toggle {
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-light);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: var(--shadow-soft);
            transition: all 0.2s ease;
        }

        .theme-toggle:hover {
            transform: scale(1.1);
        }

        .loading {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .notification {
            position: fixed;
            top: 2rem;
            left: 50%;
            transform: translateX(-50%);
            background: var(--gradient-subtle);
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            box-shadow: var(--shadow-medium);
            z-index: 1000;
            opacity: 0;
            transition: all 0.3s ease;
        }

        .notification.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    </style>
</head>
<body>
    <div class="theme-toggle" onclick="toggleTheme()">
        <span id="theme-icon">🌙</span>
    </div>

    <div class="container">
        <header class="header">
            <h1>Knowledge Operating System</h1>
            <p class="subtitle">Where thoughts become wisdom</p>
        </header>

        <main class="main-interface">
            <section class="capture-section">
                <div class="prompt-display" id="capture-prompt">
                    What's on your mind?
                </div>

                <textarea 
                    id="thought-input" 
                    class="thought-input" 
                    placeholder="Share your thoughts here... Let your mind wander and capture whatever emerges."
                    onkeydown="handleKeyPress(event)"
                ></textarea>

                <div class="capture-actions">
                    <button class="btn btn-primary" onclick="captureThought()">
                        <span id="capture-text">Capture Thought</span>
                    </button>
                    <button class="btn btn-secondary" onclick="getNewPrompt()">
                        New Prompt
                    </button>
                    <button class="btn btn-secondary" onclick="showInsights()">
                        Show Insights
                    </button>
                </div>
            </section>

            <aside class="insights-panel">
                <h2 class="panel-title">
                    <span>💭</span> Recent Thoughts
                </h2>
                
                <div class="recent-thoughts" id="recent-thoughts">
                    <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                        Begin capturing thoughts to see them here...
                    </p>
                </div>

                <h3 class="panel-title" style="font-size: 1.1rem; margin-top: 2rem;">
                    <span>🧠</span> Intelligence
                </h3>
                
                <div class="system-insights" id="system-insights">
                    <div class="insight-metric">
                        <span class="metric-label">Thoughts Captured</span>
                        <span class="metric-value" id="thought-count">0</span>
                    </div>
                    <div class="insight-metric">
                        <span class="metric-label">Concepts Discovered</span>
                        <span class="metric-value" id="concept-count">0</span>
                    </div>
                    <div class="insight-metric">
                        <span class="metric-label">Connections Made</span>
                        <span class="metric-value" id="connection-count">0</span>
                    </div>
                </div>
            </aside>
        </main>
    </div>

    <div class="notification" id="notification"></div>

    <script>
        let currentTheme = localStorage.getItem('theme') || 'light';
        
        // Initialize theme
        if (currentTheme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            document.getElementById('theme-icon').textContent = '☀️';
        }

        function toggleTheme() {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', currentTheme === 'dark' ? 'dark' : null);
            document.getElementById('theme-icon').textContent = currentTheme === 'dark' ? '☀️' : '🌙';
            localStorage.setItem('theme', currentTheme);
        }

        function showNotification(message, duration = 3000) {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, duration);
        }

        function handleKeyPress(event) {
            // Capture thought with Cmd/Ctrl + Enter
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault();
                captureThought();
            }
        }

        async function captureThought() {
            const input = document.getElementById('thought-input');
            const content = input.value.trim();
            
            if (!content) {
                showNotification('Please share a thought first...');
                return;
            }

            // Show loading state
            const captureBtn = document.querySelector('.btn-primary');
            const originalText = captureBtn.innerHTML;
            captureBtn.innerHTML = '<div class="loading"><div class="spinner"></div>Capturing...</div>';
            captureBtn.disabled = true;

            try {
                const response = await fetch('/api/capture', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ content: content })
                });

                const result = await response.json();

                if (result.success) {
                    input.value = '';
                    showNotification('✨ Thought captured beautifully');
                    
                    // Update interface
                    await updateRecentThoughts();
                    await updateSystemInsights();
                    await getNewPrompt();
                    
                } else {
                    showNotification('Failed to capture thought. Please try again.');
                }
            } catch (error) {
                console.error('Error capturing thought:', error);
                showNotification('Connection error. Please check your connection.');
            } finally {
                captureBtn.innerHTML = originalText;
                captureBtn.disabled = false;
            }
        }

        async function getNewPrompt() {
            try {
                const response = await fetch('/api/prompt');
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('capture-prompt').textContent = result.prompt;
                }
            } catch (error) {
                console.error('Error getting new prompt:', error);
            }
        }

        async function updateRecentThoughts() {
            try {
                const response = await fetch('/api/recent-thoughts');
                const result = await response.json();
                
                if (result.success) {
                    const container = document.getElementById('recent-thoughts');
                    
                    if (result.thoughts.length === 0) {
                        container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">Begin capturing thoughts to see them here...</p>';
                        return;
                    }
                    
                    container.innerHTML = result.thoughts.map(thought => `
                        <div class="thought-card fade-in">
                            <div class="thought-meta">
                                ${formatDate(thought.timestamp)} • ${thought.word_count} words
                            </div>
                            <div class="thought-content">
                                ${thought.content.length > 150 ? thought.content.substring(0, 150) + '...' : thought.content}
                            </div>
                            ${thought.tags.length > 0 ? `
                                <div class="thought-tags">
                                    ${thought.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error updating recent thoughts:', error);
            }
        }

        async function updateSystemInsights() {
            try {
                const response = await fetch('/api/insights');
                const result = await response.json();
                
                if (result.success) {
                    const insights = result.insights;
                    
                    document.getElementById('thought-count').textContent = insights.thinking_patterns?.total_thoughts || 0;
                    document.getElementById('concept-count').textContent = insights.intelligence_summary?.concepts_tracked || 0;
                    document.getElementById('connection-count').textContent = insights.intelligence_summary?.connection_patterns || 0;
                    
                    // Add beautiful summary if available
                    if (insights.beautiful_summary) {
                        const existingSummary = document.querySelector('.beautiful-summary');
                        if (existingSummary) {
                            existingSummary.remove();
                        }
                        
                        const summaryElement = document.createElement('div');
                        summaryElement.className = 'beautiful-summary fade-in';
                        summaryElement.textContent = insights.beautiful_summary;
                        document.getElementById('system-insights').appendChild(summaryElement);
                    }
                }
            } catch (error) {
                console.error('Error updating system insights:', error);
            }
        }

        async function showInsights() {
            await updateSystemInsights();
            showNotification('🧠 Insights updated');
        }

        function formatDate(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diffInMinutes = Math.floor((now - date) / (1000 * 60));
            
            if (diffInMinutes < 1) return 'Just now';
            if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
            if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
            
            return date.toLocaleDateString();
        }

        // Initialize interface
        document.addEventListener('DOMContentLoaded', () => {
            updateRecentThoughts();
            updateSystemInsights();
            getNewPrompt();
            
            // Focus on input
            document.getElementById('thought-input').focus();
        });

        // Auto-refresh every 30 seconds
        setInterval(() => {
            updateRecentThoughts();
            updateSystemInsights();
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main Knowledge OS interface."""
    init_knowledge_os()
    return render_template_string(KNOWLEDGE_OS_TEMPLATE)

@app.route('/api/capture', methods=['POST'])
def capture_thought():
    """Capture a new thought."""
    try:
        init_knowledge_os()
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})
        
        # Capture the thought
        thought = knowledge_os.capture_thought(content)
        
        if thought:
            return jsonify({
                'success': True,
                'thought_id': thought.id,
                'message': 'Thought captured successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to capture thought'})
            
    except Exception as e:
        logger.error(f"Error capturing thought: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/prompt')
def get_capture_prompt():
    """Get a contextual capture prompt."""
    try:
        init_knowledge_os()
        prompt = knowledge_os.get_capture_prompt()
        return jsonify({'success': True, 'prompt': prompt})
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recent-thoughts')
def get_recent_thoughts():
    """Get recent thoughts for display."""
    try:
        init_knowledge_os()
        thoughts = knowledge_os.get_recent_thoughts(limit=5)
        
        thoughts_data = []
        for thought in thoughts:
            thoughts_data.append({
                'id': thought.id,
                'content': thought.content,
                'timestamp': thought.timestamp.isoformat(),
                'tags': thought.tags,
                'concepts': thought.concepts,
                'importance': thought.importance,
                'word_count': thought.word_count,
                'connections': len(thought.connections)
            })
        
        return jsonify({'success': True, 'thoughts': thoughts_data})
    except Exception as e:
        logger.error(f"Error getting recent thoughts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/insights')
def get_system_insights():
    """Get system insights and analytics."""
    try:
        init_knowledge_os()
        insights = knowledge_os.get_system_insights()
        return jsonify({'success': True, 'insights': insights})
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search')
def search_thoughts():
    """Search thoughts by query."""
    try:
        init_knowledge_os()
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'No search query provided'})
        
        results = knowledge_os.search_thoughts(query)
        
        search_results = []
        for thought in results[:10]:  # Limit to 10 results
            search_results.append({
                'id': thought.id,
                'content': thought.content,
                'timestamp': thought.timestamp.isoformat(),
                'importance': thought.importance,
                'tags': thought.tags,
                'concepts': thought.concepts
            })
        
        return jsonify({
            'success': True, 
            'results': search_results,
            'count': len(search_results)
        })
    except Exception as e:
        logger.error(f"Error searching thoughts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/densify')
def check_densification():
    """Check for densification opportunities."""
    try:
        init_knowledge_os()
        opportunities = knowledge_os.check_densification_opportunities()
        
        opportunities_data = []
        for opp in opportunities:
            opportunities_data.append({
                'concept': opp['concept'],
                'thought_count': len(opp['thoughts']),
                'analysis': opp['analysis'],
                'suggestion': opp['analysis']['suggestion']
            })
        
        return jsonify({
            'success': True, 
            'opportunities': opportunities_data,
            'count': len(opportunities_data)
        })
    except Exception as e:
        logger.error(f"Error checking densification: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/densify/<concept>', methods=['POST'])
def densify_concept(concept):
    """Densify thoughts for a specific concept."""
    try:
        init_knowledge_os()
        cluster = knowledge_os.densify_concept(concept)
        
        if cluster:
            return jsonify({
                'success': True,
                'cluster': {
                    'id': cluster.id,
                    'name': cluster.name,
                    'summary': cluster.summary,
                    'key_insights': cluster.key_insights,
                    'compression_ratio': cluster.compression_ratio,
                    'original_word_count': cluster.original_word_count,
                    'compressed_word_count': cluster.compressed_word_count
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Unable to densify concept'})
            
    except Exception as e:
        logger.error(f"Error densifying concept: {e}")
        return jsonify({'success': False, 'error': str(e)})

def run_knowledge_os_interface(host='localhost', port=5001, debug=True):
    """Run the Knowledge OS web interface."""
    print(f"""
🧠 Knowledge Operating System Interface Starting...

   Access your Knowledge OS at: http://{host}:{port}
   
   Features:
   • Intuitive thought capture with prose-like experience
   • Real-time background intelligence processing
   • Beautiful insights and pattern recognition
   • Contextual prompts that adapt to your thinking
   • Automatic concept clustering and densification
   
   Press Ctrl+C to stop the server
   """)
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n✨ Knowledge OS interface stopped gracefully")

if __name__ == '__main__':
    run_knowledge_os_interface()