#!/usr/bin/env python3
"""
summail_interface.py - Web Interface and API for SumMail

Provides a beautiful web interface and RESTful API for the SumMail email
compression system, with real-time processing, digest generation, and
intelligent email management.

Features:
- OAuth2 authentication for secure email access
- Real-time email processing with progress updates
- Interactive digest viewer with filtering
- Software update tracking dashboard
- Action item management
- Privacy-focused local processing

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
from flask_cors import CORS
import imaplib
from pathlib import Path

# Import SumMail components
from summail_engine import (
    SumMailEngine, EmailCategory, Priority, EmailDigest,
    CompressedEmail, SoftwareInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'summail-secret-key-change-in-production')
CORS(app)

# Global storage (in production, use proper database)
user_engines = {}  # user_id -> SumMailEngine
processing_status = {}  # user_id -> processing status
digest_cache = {}  # user_id -> cached digests

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)


def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def get_user_engine() -> Optional[SumMailEngine]:
    """Get SumMail engine for current user."""
    user_id = session.get('user_id')
    if user_id and user_id in user_engines:
        return user_engines[user_id]
    return None


@app.route('/')
def index():
    """Main SumMail interface."""
    return render_template('summail.html')


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user with email credentials."""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        imap_server = data.get('imap_server')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Create user engine
        user_id = email  # In production, use proper user ID
        engine = SumMailEngine({'use_local_ai': True})
        
        # Connect to email account
        if engine.connect_email_account(email, password, imap_server):
            user_engines[user_id] = engine
            session['user_id'] = user_id
            session['email'] = email
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'email': email
            })
        else:
            return jsonify({'error': 'Failed to connect to email account'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """Logout user and cleanup."""
    user_id = session.get('user_id')
    
    # Cleanup user data
    if user_id in user_engines:
        del user_engines[user_id]
    if user_id in processing_status:
        del processing_status[user_id]
    if user_id in digest_cache:
        del digest_cache[user_id]
    
    session.clear()
    return jsonify({'success': True})


@app.route('/api/emails/process', methods=['POST'])
@login_required
def process_emails():
    """Process emails for compression."""
    engine = get_user_engine()
    if not engine:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    try:
        data = request.get_json()
        folder = data.get('folder', 'INBOX')
        limit = data.get('limit', 100)
        start_date = data.get('start_date')
        categories = data.get('categories', [])
        
        # Start background processing
        user_id = session['user_id']
        processing_status[user_id] = {
            'status': 'processing',
            'processed': 0,
            'total': 0,
            'start_time': time.time()
        }
        
        # Submit processing task
        future = executor.submit(
            _process_emails_background,
            engine, user_id, folder, limit, start_date, categories
        )
        
        return jsonify({
            'status': 'processing',
            'message': 'Email processing started'
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500


def _process_emails_background(engine: SumMailEngine, user_id: str, 
                              folder: str, limit: int, 
                              start_date: Optional[str], 
                              categories: List[str]):
    """Background email processing task."""
    try:
        # Select folder
        engine.imap.select(folder)
        
        # Build search criteria
        criteria = []
        if start_date:
            date_obj = datetime.fromisoformat(start_date)
            criteria.append(f'SINCE {date_obj.strftime("%d-%b-%Y")}')
        
        search_string = ' '.join(criteria) if criteria else 'ALL'
        
        # Search emails
        _, message_ids = engine.imap.search(None, search_string)
        message_ids = message_ids[0].split()[-limit:]  # Get last N emails
        
        processing_status[user_id]['total'] = len(message_ids)
        
        # Process each email
        processed_count = 0
        for msg_id in message_ids:
            try:
                # Fetch email
                _, msg_data = engine.imap.fetch(msg_id, '(RFC822)')
                email_data = msg_data[0][1]
                
                # Process email
                compressed = engine.process_email(email_data)
                
                # Update progress
                processed_count += 1
                processing_status[user_id]['processed'] = processed_count
                
            except Exception as e:
                logger.error(f"Error processing email {msg_id}: {e}")
        
        # Mark as complete
        processing_status[user_id]['status'] = 'completed'
        processing_status[user_id]['end_time'] = time.time()
        
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        processing_status[user_id]['status'] = 'error'
        processing_status[user_id]['error'] = str(e)


@app.route('/api/emails/status')
@login_required
def get_processing_status():
    """Get email processing status."""
    user_id = session['user_id']
    status = processing_status.get(user_id, {'status': 'idle'})
    
    # Calculate progress percentage
    if status.get('total', 0) > 0:
        status['progress'] = (status.get('processed', 0) / status['total']) * 100
    else:
        status['progress'] = 0
    
    return jsonify(status)


@app.route('/api/digest/generate', methods=['POST'])
@login_required
def generate_digest():
    """Generate email digest for specified period."""
    engine = get_user_engine()
    if not engine:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Parse date range
        start_date = datetime.fromisoformat(data.get('start_date', 
                                                    (datetime.now() - timedelta(days=7)).isoformat()))
        end_date = datetime.fromisoformat(data.get('end_date', 
                                                  datetime.now().isoformat()))
        
        # Parse categories
        categories = None
        if data.get('categories'):
            categories = [EmailCategory[cat.upper()] for cat in data['categories']]
        
        # Generate digest
        digest = engine.generate_digest(start_date, end_date, categories)
        
        # Convert to JSON-serializable format
        digest_data = {
            'period_start': digest.period_start.isoformat(),
            'period_end': digest.period_end.isoformat(),
            'total_emails': digest.total_emails,
            'categories_breakdown': {
                cat.value: count for cat, count in digest.categories_breakdown.items()
            },
            'software_updates': [
                {
                    'name': update.name,
                    'current_version': update.current_version,
                    'latest_version': update.latest_version,
                    'release_date': update.release_date.isoformat() if update.release_date else None
                }
                for update in digest.software_updates
            ],
            'action_items': digest.action_items,
            'compressed_newsletters': digest.compressed_newsletters,
            'important_threads': digest.important_threads,
            'financial_summary': digest.financial_summary
        }
        
        # Cache digest
        user_id = session['user_id']
        if user_id not in digest_cache:
            digest_cache[user_id] = {}
        
        cache_key = f"{start_date.date()}_{end_date.date()}"
        digest_cache[user_id][cache_key] = digest_data
        
        return jsonify(digest_data)
        
    except Exception as e:
        logger.error(f"Digest generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/emails/search', methods=['POST'])
@login_required
def search_emails():
    """Search processed emails."""
    engine = get_user_engine()
    if not engine:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = data.get('category')
        priority = data.get('priority')
        sender = data.get('sender')
        
        results = []
        
        for msg_id, email in engine.processed_emails.items():
            # Apply filters
            if category and email.metadata.category.value != category:
                continue
            if priority and email.metadata.priority.value != priority:
                continue
            if sender and sender.lower() not in email.metadata.sender.lower():
                continue
            
            # Search in content
            if query:
                query_lower = query.lower()
                if (query_lower not in email.metadata.subject.lower() and
                    query_lower not in email.summary.lower() and
                    not any(query_lower in point.lower() for point in email.key_points)):
                    continue
            
            # Add to results
            results.append({
                'message_id': msg_id,
                'subject': email.metadata.subject,
                'sender': email.metadata.sender,
                'date': email.metadata.date.isoformat(),
                'category': email.metadata.category.value,
                'priority': email.metadata.priority.value,
                'summary': email.summary,
                'key_points': email.key_points,
                'compression_ratio': email.compression_ratio
            })
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'results': results[:50],  # Limit to 50 results
            'total': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/software/updates')
@login_required
def get_software_updates():
    """Get tracked software updates."""
    engine = get_user_engine()
    if not engine:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    software_list = []
    
    for name, info in engine.software_tracker.items():
        software_list.append({
            'name': info.name,
            'current_version': info.current_version,
            'latest_version': info.latest_version,
            'release_date': info.release_date.isoformat() if info.release_date else None,
            'update_available': info.latest_version and info.latest_version != info.current_version
        })
    
    # Sort by update availability and name
    software_list.sort(key=lambda x: (not x['update_available'], x['name']))
    
    return jsonify({
        'software': software_list,
        'total': len(software_list)
    })


@app.route('/api/stats')
@login_required
def get_statistics():
    """Get email processing statistics."""
    engine = get_user_engine()
    if not engine:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    # Calculate statistics
    total_emails = len(engine.processed_emails)
    
    if total_emails == 0:
        return jsonify({
            'total_emails': 0,
            'categories': {},
            'priorities': {},
            'compression': {}
        })
    
    # Category distribution
    categories = {}
    priorities = {}
    compression_ratios = []
    
    for email in engine.processed_emails.values():
        # Categories
        cat = email.metadata.category.value
        categories[cat] = categories.get(cat, 0) + 1
        
        # Priorities
        pri = email.metadata.priority.value
        priorities[pri] = priorities.get(pri, 0) + 1
        
        # Compression
        compression_ratios.append(email.compression_ratio)
    
    # Calculate average compression
    avg_compression = sum(compression_ratios) / len(compression_ratios)
    
    # Find most compressed emails
    sorted_emails = sorted(
        engine.processed_emails.values(),
        key=lambda x: x.compression_ratio,
        reverse=True
    )[:5]
    
    most_compressed = [
        {
            'subject': email.metadata.subject,
            'sender': email.metadata.sender,
            'compression_ratio': email.compression_ratio,
            'original_size': int(email.metadata.size),
            'compressed_size': int(email.metadata.size * (1 - email.compression_ratio))
        }
        for email in sorted_emails
    ]
    
    return jsonify({
        'total_emails': total_emails,
        'categories': categories,
        'priorities': priorities,
        'compression': {
            'average_ratio': avg_compression,
            'total_saved': sum((e.metadata.size * e.compression_ratio) 
                             for e in engine.processed_emails.values()),
            'most_compressed': most_compressed
        },
        'action_items': sum(len(e.metadata.action_items) 
                          for e in engine.processed_emails.values()),
        'software_tracked': len(engine.software_tracker)
    })


@app.route('/api/emails/stream')
@login_required
def stream_emails():
    """Stream email processing updates via Server-Sent Events."""
    user_id = session['user_id']
    
    def generate():
        """Generate SSE events."""
        while True:
            # Get current status
            status = processing_status.get(user_id, {'status': 'idle'})
            
            # Send status update
            data = json.dumps(status)
            yield f"data: {data}\n\n"
            
            # Stop if processing is complete or errored
            if status.get('status') in ['completed', 'error', 'idle']:
                break
            
            time.sleep(1)  # Update every second
    
    return Response(generate(), mimetype='text/event-stream')


# HTML Template for summail.html
SUMMAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SumMail - Intelligent Email Compression</title>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f3f4f6;
            --white: #ffffff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: var(--dark);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #6b7280;
            font-size: 1.1rem;
        }
        
        .login-form {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
        }
        
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: #5558d9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.3);
        }
        
        .dashboard {
            display: none;
        }
        
        .dashboard.active {
            display: block;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        }
        
        .stat-card h3 {
            font-size: 2rem;
            color: var(--primary);
            margin-bottom: 5px;
        }
        
        .stat-card p {
            color: #6b7280;
            font-size: 0.9rem;
        }
        
        .digest-container {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .category-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .badge-newsletter {
            background: #e0e7ff;
            color: #4338ca;
        }
        
        .badge-software {
            background: #d1fae5;
            color: #047857;
        }
        
        .badge-financial {
            background: #fee2e2;
            color: #dc2626;
        }
        
        .badge-personal {
            background: #fef3c7;
            color: #d97706;
        }
        
        .email-item {
            padding: 20px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        
        .email-item:hover {
            background: #f9fafb;
            border-color: var(--primary);
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.3s ease;
        }
        
        .action-item {
            background: #fef3c7;
            border-left: 4px solid var(--warning);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .software-update {
            background: #d1fae5;
            border-left: 4px solid var(--success);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header fade-in">
            <h1>📧 SumMail</h1>
            <p>Intelligent Email Compression - Cut through the noise, focus on what matters</p>
        </div>
        
        <div id="loginForm" class="login-form fade-in">
            <h2>Connect Your Email</h2>
            <form id="emailLogin">
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input type="email" id="email" required placeholder="your@email.com">
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" required placeholder="Your email password">
                </div>
                <div class="form-group">
                    <label for="imapServer">IMAP Server (optional)</label>
                    <input type="text" id="imapServer" placeholder="Auto-detected">
                </div>
                <button type="submit" class="btn btn-primary">
                    Connect & Process
                </button>
            </form>
        </div>
        
        <div id="dashboard" class="dashboard">
            <div class="stats-grid fade-in">
                <div class="stat-card">
                    <h3 id="totalEmails">0</h3>
                    <p>Emails Processed</p>
                </div>
                <div class="stat-card">
                    <h3 id="compressionRatio">0%</h3>
                    <p>Average Compression</p>
                </div>
                <div class="stat-card">
                    <h3 id="actionItems">0</h3>
                    <p>Action Items</p>
                </div>
                <div class="stat-card">
                    <h3 id="softwareUpdates">0</h3>
                    <p>Software Updates</p>
                </div>
            </div>
            
            <div class="digest-container fade-in">
                <h2>Email Digest</h2>
                
                <div class="progress-bar" id="progressBar" style="display: none;">
                    <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                </div>
                
                <div class="category-badges" id="categoryBadges"></div>
                
                <div id="digestContent">
                    <div id="actionItemsList"></div>
                    <div id="softwareUpdatesList"></div>
                    <div id="emailsList"></div>
                </div>
                
                <button class="btn btn-primary" onclick="generateDigest()">
                    Generate New Digest
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let eventSource;
        
        // Login handler
        document.getElementById('emailLogin').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const imapServer = document.getElementById('imapServer').value;
            
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password, imap_server: imapServer})
                });
                
                if (response.ok) {
                    document.getElementById('loginForm').style.display = 'none';
                    document.getElementById('dashboard').classList.add('active');
                    
                    // Start processing emails
                    processEmails();
                } else {
                    alert('Failed to connect to email account');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // Process emails
        async function processEmails() {
            const response = await fetch('/api/emails/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder: 'INBOX',
                    limit: 100
                })
            });
            
            if (response.ok) {
                // Start monitoring progress
                monitorProgress();
            }
        }
        
        // Monitor processing progress
        function monitorProgress() {
            eventSource = new EventSource('/api/emails/stream');
            
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');
            
            progressBar.style.display = 'block';
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.progress) {
                    progressFill.style.width = data.progress + '%';
                }
                
                if (data.status === 'completed') {
                    progressBar.style.display = 'none';
                    eventSource.close();
                    
                    // Generate digest
                    generateDigest();
                    
                    // Update stats
                    updateStats();
                }
            };
        }
        
        // Generate digest
        async function generateDigest() {
            const response = await fetch('/api/digest/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            
            if (response.ok) {
                const digest = await response.json();
                displayDigest(digest);
            }
        }
        
        // Display digest
        function displayDigest(digest) {
            // Update category badges
            const categoryBadges = document.getElementById('categoryBadges');
            categoryBadges.innerHTML = '';
            
            for (const [category, count] of Object.entries(digest.categories_breakdown)) {
                const badge = document.createElement('span');
                badge.className = `badge badge-${category}`;
                badge.textContent = `${category}: ${count}`;
                categoryBadges.appendChild(badge);
            }
            
            // Display action items
            const actionItemsList = document.getElementById('actionItemsList');
            actionItemsList.innerHTML = '<h3>Action Items</h3>';
            
            digest.action_items.forEach(item => {
                const div = document.createElement('div');
                div.className = 'action-item';
                div.innerHTML = `
                    <strong>${item.action}</strong><br>
                    <small>From: ${item.source} | Priority: ${item.priority}</small>
                `;
                actionItemsList.appendChild(div);
            });
            
            // Display software updates
            const softwareUpdatesList = document.getElementById('softwareUpdatesList');
            softwareUpdatesList.innerHTML = '<h3>Software Updates</h3>';
            
            digest.software_updates.forEach(update => {
                const div = document.createElement('div');
                div.className = 'software-update';
                div.innerHTML = `
                    <strong>${update.name}</strong><br>
                    Current: ${update.current_version} → Latest: ${update.latest_version || 'Up to date'}
                `;
                softwareUpdatesList.appendChild(div);
            });
        }
        
        // Update statistics
        async function updateStats() {
            const response = await fetch('/api/stats');
            
            if (response.ok) {
                const stats = await response.json();
                
                document.getElementById('totalEmails').textContent = stats.total_emails;
                document.getElementById('compressionRatio').textContent = 
                    Math.round(stats.compression.average_ratio * 100) + '%';
                document.getElementById('actionItems').textContent = stats.action_items;
                document.getElementById('softwareUpdates').textContent = stats.software_tracked;
            }
        }
    </script>
</body>
</html>
"""

# Create templates directory and save template
if __name__ == "__main__":
    # Ensure templates directory exists
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Save template
    with open(templates_dir / "summail.html", "w") as f:
        f.write(SUMMAIL_TEMPLATE)
    
    print("🚀 SumMail Interface Ready!")
    print("\nFeatures:")
    print("- Secure email authentication")
    print("- Real-time processing updates")
    print("- Interactive digest viewer")
    print("- Software update tracking")
    print("- Action item management")
    print("\nStart with: python summail_interface.py")
    
    # Run the app
    app.run(debug=True, port=5001)