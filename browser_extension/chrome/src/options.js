/**
 * SUM Browser Extension - Options Page Script
 * Handles settings management and UI interactions
 */

// Default settings
const DEFAULT_SETTINGS = {
  apiUrl: 'http://localhost:5001',
  apiKey: '',
  model: 'hierarchical',
  maxTokens: 150,
  useCache: true
};

// DOM elements
const elements = {
  apiUrl: document.getElementById('apiUrl'),
  apiKey: document.getElementById('apiKey'),
  model: document.getElementById('model'),
  maxTokens: document.getElementById('maxTokens'),
  maxTokensValue: document.getElementById('maxTokensValue'),
  useCache: document.getElementById('useCache'),
  testConnectionBtn: document.getElementById('testConnectionBtn'),
  connectionResult: document.getElementById('connectionResult'),
  configureShortcutsBtn: document.getElementById('configureShortcutsBtn'),
  cacheCount: document.getElementById('cacheCount'),
  historyCount: document.getElementById('historyCount'),
  clearCacheBtn: document.getElementById('clearCacheBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  exportDataBtn: document.getElementById('exportDataBtn'),
  saveBtn: document.getElementById('saveBtn'),
  resetBtn: document.getElementById('resetBtn')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  loadDataStats();
  setupEventListeners();
});

// Load settings
async function loadSettings() {
  const settings = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  
  // Apply settings with defaults
  elements.apiUrl.value = settings.apiUrl || DEFAULT_SETTINGS.apiUrl;
  elements.apiKey.value = settings.apiKey || DEFAULT_SETTINGS.apiKey;
  elements.model.value = settings.model || DEFAULT_SETTINGS.model;
  elements.maxTokens.value = settings.maxTokens || DEFAULT_SETTINGS.maxTokens;
  elements.maxTokensValue.textContent = elements.maxTokens.value;
  elements.useCache.checked = settings.useCache !== false;
}

// Load data statistics
async function loadDataStats() {
  // Get cache count (simplified - actual implementation would query background script)
  const cacheData = await chrome.storage.local.get(['summaryCache']);
  const cacheCount = cacheData.summaryCache ? Object.keys(cacheData.summaryCache).length : 0;
  elements.cacheCount.textContent = cacheCount;
  
  // Get history count
  const historyData = await chrome.storage.local.get(['summaryHistory']);
  const historyCount = historyData.summaryHistory ? historyData.summaryHistory.length : 0;
  elements.historyCount.textContent = historyCount;
}

// Setup event listeners
function setupEventListeners() {
  // Range input
  elements.maxTokens.addEventListener('input', () => {
    elements.maxTokensValue.textContent = elements.maxTokens.value;
  });
  
  // Test connection
  elements.testConnectionBtn.addEventListener('click', testConnection);
  
  // Configure shortcuts
  elements.configureShortcutsBtn.addEventListener('click', () => {
    chrome.tabs.create({ url: 'chrome://extensions/shortcuts' });
  });
  
  // Clear cache
  elements.clearCacheBtn.addEventListener('click', async () => {
    if (confirm('Clear all cached summaries?')) {
      await chrome.storage.local.remove(['summaryCache']);
      showMessage('Cache cleared successfully');
      loadDataStats();
    }
  });
  
  // Clear history
  elements.clearHistoryBtn.addEventListener('click', async () => {
    if (confirm('Clear all summary history?')) {
      await chrome.storage.local.set({ summaryHistory: [] });
      showMessage('History cleared successfully');
      loadDataStats();
    }
  });
  
  // Export data
  elements.exportDataBtn.addEventListener('click', exportData);
  
  // Save settings
  elements.saveBtn.addEventListener('click', saveSettings);
  
  // Reset settings
  elements.resetBtn.addEventListener('click', async () => {
    if (confirm('Reset all settings to defaults?')) {
      await chrome.storage.sync.clear();
      await loadSettings();
      showMessage('Settings reset to defaults');
    }
  });
}

// Test API connection
async function testConnection() {
  const apiUrl = elements.apiUrl.value.trim();
  const apiKey = elements.apiKey.value.trim();
  
  if (!apiUrl) {
    showConnectionResult('error', 'Please enter an API URL');
    return;
  }
  
  elements.testConnectionBtn.disabled = true;
  elements.testConnectionBtn.textContent = 'Testing...';
  
  try {
    const headers = {};
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    
    const response = await fetch(`${apiUrl}/api/health`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000)
    });
    
    if (response.ok) {
      const data = await response.json();
      
      // Test API key if provided
      if (apiKey) {
        const authResponse = await fetch(`${apiUrl}/api/auth/validate`, {
          method: 'GET',
          headers: { 'X-API-Key': apiKey }
        });
        
        if (authResponse.ok) {
          const authData = await authResponse.json();
          showConnectionResult('success', 
            `Connected! API v${data.version || '1.0'} | Key valid: ${authData.name}`);
        } else {
          showConnectionResult('error', 'Connected but API key is invalid');
        }
      } else {
        showConnectionResult('success', `Connected! API v${data.version || '1.0'}`);
      }
    } else {
      showConnectionResult('error', `Connection failed: HTTP ${response.status}`);
    }
  } catch (error) {
    showConnectionResult('error', `Connection failed: ${error.message}`);
  } finally {
    elements.testConnectionBtn.disabled = false;
    elements.testConnectionBtn.textContent = 'Test Connection';
  }
}

// Show connection result
function showConnectionResult(type, message) {
  elements.connectionResult.className = `connection-result ${type}`;
  elements.connectionResult.textContent = message;
  
  setTimeout(() => {
    elements.connectionResult.className = 'connection-result';
  }, 5000);
}

// Save settings
async function saveSettings() {
  const settings = {
    apiUrl: elements.apiUrl.value.trim(),
    apiKey: elements.apiKey.value.trim(),
    model: elements.model.value,
    maxTokens: parseInt(elements.maxTokens.value),
    useCache: elements.useCache.checked
  };
  
  try {
    await chrome.storage.sync.set(settings);
    showMessage('Settings saved successfully');
  } catch (error) {
    showMessage('Failed to save settings', 'error');
  }
}

// Export data
async function exportData() {
  try {
    const data = {
      settings: await chrome.storage.sync.get(),
      history: await chrome.storage.local.get(['summaryHistory']),
      exportDate: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `sum-extension-data-${Date.now()}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showMessage('Data exported successfully');
  } catch (error) {
    showMessage('Failed to export data', 'error');
  }
}

// Show message
function showMessage(text, type = 'success') {
  const message = document.createElement('div');
  message.className = 'success-message';
  message.textContent = text;
  
  if (type === 'error') {
    message.style.background = '#dc2626';
  }
  
  document.body.appendChild(message);
  
  setTimeout(() => {
    message.remove();
  }, 3000);
}