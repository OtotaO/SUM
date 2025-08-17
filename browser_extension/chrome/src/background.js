/**
 * SUM Browser Extension - Background Service Worker
 * Handles API communication, context menus, and extension commands
 */

// Configuration
const CONFIG = {
  API_URL: 'http://localhost:5001',
  MAX_TEXT_LENGTH: 50000,
  DEFAULT_MODEL: 'hierarchical',
  CACHE_TTL: 3600000 // 1 hour
};

// Simple in-memory cache
const summaryCache = new Map();

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
  // Create context menu items
  chrome.contextMenus.create({
    id: 'sum-selection',
    title: 'Summarize with SUM',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'sum-page',
    title: 'Summarize Page with SUM',
    contexts: ['page']
  });
  
  // Set default settings
  chrome.storage.sync.get(['apiKey', 'apiUrl', 'model'], (result) => {
    if (!result.apiUrl) {
      chrome.storage.sync.set({
        apiUrl: CONFIG.API_URL,
        model: CONFIG.DEFAULT_MODEL
      });
    }
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'sum-selection') {
    const text = info.selectionText;
    if (text) {
      await summarizeText(text, tab.id, 'selection');
    }
  } else if (info.menuItemId === 'sum-page') {
    // Get page content from content script
    chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' }, async (response) => {
      if (response && response.content) {
        await summarizeText(response.content, tab.id, 'page');
      }
    });
  }
});

// Handle extension commands (keyboard shortcuts)
chrome.commands.onCommand.addListener(async (command) => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  if (command === 'summarize-selection') {
    chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
      if (response && response.text) {
        await summarizeText(response.text, tab.id, 'selection');
      }
    });
  } else if (command === 'summarize-page') {
    chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' }, async (response) => {
      if (response && response.content) {
        await summarizeText(response.content, tab.id, 'page');
      }
    });
  }
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'summarize') {
    summarizeText(request.text, sender.tab?.id || null, request.type || 'manual')
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep channel open for async response
  } else if (request.action === 'getSummaryHistory') {
    getSummaryHistory().then(sendResponse);
    return true;
  }
});

// Main summarization function
async function summarizeText(text, tabId, source) {
  try {
    // Check cache first
    const cacheKey = `${text.substring(0, 100)}_${text.length}`;
    const cached = summaryCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_TTL) {
      return showSummary(cached.summary, tabId, source);
    }
    
    // Get settings
    const settings = await chrome.storage.sync.get(['apiKey', 'apiUrl', 'model']);
    const apiUrl = settings.apiUrl || CONFIG.API_URL;
    const model = settings.model || CONFIG.DEFAULT_MODEL;
    
    // Prepare request
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (settings.apiKey) {
      headers['X-API-Key'] = settings.apiKey;
    }
    
    // Truncate if too long
    if (text.length > CONFIG.MAX_TEXT_LENGTH) {
      text = text.substring(0, CONFIG.MAX_TEXT_LENGTH) + '...';
    }
    
    // Call SUM API
    const response = await fetch(`${apiUrl}/api/process_text`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        text,
        model,
        config: {
          maxTokens: 150,
          use_cache: true
        }
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || `HTTP ${response.status}`);
    }
    
    const result = await response.json();
    
    // Cache the result
    summaryCache.set(cacheKey, {
      summary: result,
      timestamp: Date.now()
    });
    
    // Clean old cache entries
    if (summaryCache.size > 100) {
      const entries = Array.from(summaryCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      for (let i = 0; i < 50; i++) {
        summaryCache.delete(entries[i][0]);
      }
    }
    
    // Save to history
    await saveSummaryToHistory(text, result, source);
    
    return showSummary(result, tabId, source);
    
  } catch (error) {
    console.error('Summarization error:', error);
    
    if (tabId) {
      chrome.tabs.sendMessage(tabId, {
        action: 'showError',
        error: error.message
      });
    }
    
    return { error: error.message };
  }
}

// Show summary in content script
function showSummary(summary, tabId, source) {
  if (tabId) {
    chrome.tabs.sendMessage(tabId, {
      action: 'showSummary',
      summary,
      source
    });
  }
  
  return summary;
}

// Save summary to history
async function saveSummaryToHistory(originalText, summary, source) {
  const history = await chrome.storage.local.get(['summaryHistory']);
  const summaries = history.summaryHistory || [];
  
  summaries.unshift({
    id: Date.now(),
    timestamp: new Date().toISOString(),
    originalText: originalText.substring(0, 200) + (originalText.length > 200 ? '...' : ''),
    summary: summary.summary || summary.sum,
    fullSummary: summary,
    source,
    url: (await chrome.tabs.query({ active: true, currentWindow: true }))[0]?.url
  });
  
  // Keep only last 50 summaries
  if (summaries.length > 50) {
    summaries.splice(50);
  }
  
  await chrome.storage.local.set({ summaryHistory: summaries });
}

// Get summary history
async function getSummaryHistory() {
  const history = await chrome.storage.local.get(['summaryHistory']);
  return history.summaryHistory || [];
}