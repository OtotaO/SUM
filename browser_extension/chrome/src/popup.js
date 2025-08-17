/**
 * SUM Browser Extension - Popup Script
 * Handles popup UI interactions and communication with background script
 */

// DOM elements
const elements = {
  textInput: document.getElementById('textInput'),
  summarizeTextBtn: document.getElementById('summarizeTextBtn'),
  clearTextBtn: document.getElementById('clearTextBtn'),
  summarizeSelectionBtn: document.getElementById('summarizeSelectionBtn'),
  summarizePageBtn: document.getElementById('summarizePageBtn'),
  summarySection: document.getElementById('summarySection'),
  summaryContent: document.getElementById('summaryContent'),
  copySummaryBtn: document.getElementById('copySummaryBtn'),
  newSummaryBtn: document.getElementById('newSummaryBtn'),
  recentList: document.getElementById('recentList'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  settingsLink: document.getElementById('settingsLink'),
  helpLink: document.getElementById('helpLink'),
  connectionStatus: document.getElementById('connectionStatus')
};

// State
let currentSummary = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  checkConnection();
  loadRecentSummaries();
  setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
  // Text input
  elements.textInput.addEventListener('input', () => {
    const hasText = elements.textInput.value.trim().length > 0;
    elements.summarizeTextBtn.disabled = !hasText;
    elements.clearTextBtn.disabled = !hasText;
  });
  
  // Summarize text button
  elements.summarizeTextBtn.addEventListener('click', () => {
    const text = elements.textInput.value.trim();
    if (text) {
      summarizeText(text, 'manual');
    }
  });
  
  // Clear text button
  elements.clearTextBtn.addEventListener('click', () => {
    elements.textInput.value = '';
    elements.summarizeTextBtn.disabled = true;
    elements.clearTextBtn.disabled = true;
    elements.textInput.focus();
  });
  
  // Page action buttons
  elements.summarizeSelectionBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, (response) => {
      if (response && response.text) {
        summarizeText(response.text, 'selection');
      } else {
        showError('No text selected. Please select some text and try again.');
      }
    });
  });
  
  elements.summarizePageBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' }, (response) => {
      if (response && response.content) {
        summarizeText(response.content, 'page');
      } else {
        showError('Unable to extract page content.');
      }
    });
  });
  
  // Summary actions
  elements.copySummaryBtn.addEventListener('click', () => {
    if (currentSummary) {
      const summaryText = formatSummaryForCopy(currentSummary);
      navigator.clipboard.writeText(summaryText).then(() => {
        elements.copySummaryBtn.textContent = 'Copied!';
        setTimeout(() => {
          elements.copySummaryBtn.textContent = 'Copy';
        }, 2000);
      });
    }
  });
  
  elements.newSummaryBtn.addEventListener('click', () => {
    hideSummary();
    elements.textInput.value = '';
    elements.textInput.focus();
  });
  
  // Clear history
  elements.clearHistoryBtn.addEventListener('click', async () => {
    if (confirm('Clear all summary history?')) {
      await chrome.storage.local.set({ summaryHistory: [] });
      loadRecentSummaries();
    }
  });
  
  // Links
  elements.settingsLink.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.runtime.openOptionsPage();
  });
  
  elements.helpLink.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.tabs.create({ url: 'https://github.com/OtotaO/SUM#browser-extension' });
  });
}

// Summarize text
async function summarizeText(text, source) {
  showLoading();
  
  try {
    const response = await chrome.runtime.sendMessage({
      action: 'summarize',
      text,
      type: source
    });
    
    if (response.error) {
      showError(response.error);
    } else {
      showSummary(response);
      if (source === 'manual') {
        elements.textInput.value = '';
        elements.summarizeTextBtn.disabled = true;
        elements.clearTextBtn.disabled = true;
      }
    }
  } catch (error) {
    showError('Failed to summarize text. Please try again.');
  }
}

// Show loading state
function showLoading() {
  elements.summarySection.classList.remove('hidden');
  elements.summaryContent.innerHTML = '<div class="loading"></div> Summarizing...';
  elements.copySummaryBtn.disabled = true;
}

// Show summary
function showSummary(summary) {
  currentSummary = summary;
  elements.summarySection.classList.remove('hidden');
  
  let content = `<p>${summary.summary || summary.sum}</p>`;
  
  if (summary.tags && summary.tags.length > 0) {
    content += '<div style="margin-top: 12px;"><strong>Key concepts:</strong> ' + 
               summary.tags.join(', ') + '</div>';
  }
  
  if (summary.compression_ratio) {
    content += `<div style="margin-top: 12px; font-size: 12px; color: #64748b;">
                Compression: ${summary.compression_ratio.toFixed(1)}x | 
                Model: ${summary.model || 'unknown'}
                ${summary.cached ? ' | Cached' : ''}
                </div>`;
  }
  
  elements.summaryContent.innerHTML = content;
  elements.copySummaryBtn.disabled = false;
  
  loadRecentSummaries();
}

// Show error
function showError(message) {
  elements.summarySection.classList.remove('hidden');
  elements.summaryContent.innerHTML = `
    <div style="color: #dc2626; display: flex; align-items: center; gap: 8px;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      ${message}
    </div>
  `;
  elements.copySummaryBtn.disabled = true;
}

// Hide summary
function hideSummary() {
  elements.summarySection.classList.add('hidden');
  currentSummary = null;
}

// Load recent summaries
async function loadRecentSummaries() {
  const history = await chrome.storage.local.get(['summaryHistory']);
  const summaries = history.summaryHistory || [];
  
  if (summaries.length === 0) {
    elements.recentList.innerHTML = '<p class="empty-state">No recent summaries</p>';
    return;
  }
  
  elements.recentList.innerHTML = summaries.slice(0, 5).map(item => `
    <div class="recent-item" data-id="${item.id}">
      <div class="recent-item-time">${formatTime(item.timestamp)}</div>
      <div class="recent-item-text">${item.summary}</div>
    </div>
  `).join('');
  
  // Add click handlers
  elements.recentList.querySelectorAll('.recent-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = parseInt(item.dataset.id);
      const summary = summaries.find(s => s.id === id);
      if (summary) {
        showSummary(summary.fullSummary);
      }
    });
  });
}

// Format time
function formatTime(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now - date;
  
  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + ' min ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + ' hours ago';
  return date.toLocaleDateString();
}

// Format summary for copying
function formatSummaryForCopy(summary) {
  let text = summary.summary || summary.sum;
  
  if (summary.tags && summary.tags.length > 0) {
    text += '\n\nKey concepts: ' + summary.tags.join(', ');
  }
  
  text += '\n\n[Summarized with SUM]';
  
  return text;
}

// Check connection to API
async function checkConnection() {
  try {
    const settings = await chrome.storage.sync.get(['apiUrl']);
    const apiUrl = settings.apiUrl || 'http://localhost:5001';
    
    const response = await fetch(`${apiUrl}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });
    
    if (response.ok) {
      setConnectionStatus('connected', 'Connected');
    } else {
      setConnectionStatus('error', 'API Error');
    }
  } catch (error) {
    setConnectionStatus('error', 'Not Connected');
  }
}

// Set connection status
function setConnectionStatus(status, text) {
  const statusDot = elements.connectionStatus.querySelector('.status-dot');
  const statusText = elements.connectionStatus.querySelector('.status-text');
  
  statusDot.className = 'status-dot ' + status;
  statusText.textContent = text;
}