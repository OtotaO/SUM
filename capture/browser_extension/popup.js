/**
 * SUM Browser Extension - Popup JavaScript
 * 
 * Handles the popup interface, settings, and communication
 * with background scripts and content scripts.
 * 
 * Author: ototao (optimized with Claude Code)
 * License: Apache License 2.0
 */

class SumPopup {
  constructor() {
    this.elements = {};
    this.currentTab = null;
    this.settings = {};
    this.captures = [];
    this.stats = {};
    
    this.init();
  }
  
  async init() {
    // Get DOM elements
    this.getElements();
    
    // Set up event listeners
    this.setupEventListeners();
    
    // Load data
    await this.loadCurrentTab();
    await this.loadSettings();
    await this.loadCaptures();
    await this.loadStats();
    
    // Update UI
    this.updateUI();
    
    console.log('SUM Popup initialized');
  }
  
  getElements() {
    // Main action buttons
    this.elements.captureSelection = document.getElementById('captureSelection');
    this.elements.capturePage = document.getElementById('capturePage');
    
    // Status indicator
    this.elements.statusIndicator = document.getElementById('statusIndicator');
    this.elements.statusText = this.elements.statusIndicator.querySelector('.status-text');
    this.elements.statusDot = this.elements.statusIndicator.querySelector('.status-dot');
    
    // Recent captures
    this.elements.capturesList = document.getElementById('capturesList');
    
    // Statistics
    this.elements.totalCaptures = document.getElementById('totalCaptures');
    this.elements.avgProcessingTime = document.getElementById('avgProcessingTime');
    this.elements.todayCaptures = document.getElementById('todayCaptures');
    
    // Settings
    this.elements.settingsBtn = document.getElementById('settingsBtn');
    this.elements.helpBtn = document.getElementById('helpBtn');
    this.elements.settingsPanel = document.getElementById('settingsPanel');
    this.elements.closeSettings = document.getElementById('closeSettings');
    this.elements.saveSettings = document.getElementById('saveSettings');
    
    // Settings controls
    this.elements.autoCapture = document.getElementById('autoCapture');
    this.elements.showFloatingButton = document.getElementById('showFloatingButton');
    this.elements.summarizationLevel = document.getElementById('summarizationLevel');
    this.elements.maxSummaryLength = document.getElementById('maxSummaryLength');
    this.elements.maxSummaryLengthValue = document.getElementById('maxSummaryLengthValue');
    
    // Loading and messages
    this.elements.loadingOverlay = document.getElementById('loadingOverlay');
    this.elements.successMessage = document.getElementById('successMessage');
  }
  
  setupEventListeners() {
    // Main action buttons
    this.elements.captureSelection.addEventListener('click', () => this.captureSelection());
    this.elements.capturePage.addEventListener('click', () => this.capturePage());
    
    // Settings
    this.elements.settingsBtn.addEventListener('click', () => this.showSettings());
    this.elements.helpBtn.addEventListener('click', () => this.showHelp());
    this.elements.closeSettings.addEventListener('click', () => this.hideSettings());
    this.elements.saveSettings.addEventListener('click', () => this.saveSettings());
    
    // Settings controls
    this.elements.maxSummaryLength.addEventListener('input', (e) => {
      this.elements.maxSummaryLengthValue.textContent = e.target.value;
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.shiftKey && e.key === 'S') {
          e.preventDefault();
          this.captureSelection();
        } else if (e.shiftKey && e.key === 'A') {
          e.preventDefault();
          this.capturePage();
        }
      }
    });
    
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
    });
  }
  
  async loadCurrentTab() {
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      this.currentTab = tabs[0];
    } catch (error) {
      console.error('Failed to load current tab:', error);
    }
  }
  
  async loadSettings() {
    try {
      const result = await chrome.storage.sync.get([
        'enabled',
        'autoCapture', 
        'showFloatingButton',
        'summarizationLevel',
        'maxSummaryLength'
      ]);
      
      this.settings = {
        enabled: result.enabled !== false,
        autoCapture: result.autoCapture || false,
        showFloatingButton: result.showFloatingButton !== false,
        summarizationLevel: result.summarizationLevel || 'quality',
        maxSummaryLength: result.maxSummaryLength || 150
      };
      
      // Update settings UI
      this.elements.autoCapture.checked = this.settings.autoCapture;
      this.elements.showFloatingButton.checked = this.settings.showFloatingButton;
      this.elements.summarizationLevel.value = this.settings.summarizationLevel;
      this.elements.maxSummaryLength.value = this.settings.maxSummaryLength;
      this.elements.maxSummaryLengthValue.textContent = this.settings.maxSummaryLength;
      
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }
  
  async loadCaptures() {
    try {
      const result = await chrome.storage.local.get(['recentCaptures']);
      this.captures = result.recentCaptures || [];
      
      // Sort by timestamp (newest first)
      this.captures.sort((a, b) => b.timestamp - a.timestamp);
      
      // Keep only recent captures (last 10)
      this.captures = this.captures.slice(0, 10);
      
      this.updateCapturesUI();
      
    } catch (error) {
      console.error('Failed to load captures:', error);
    }
  }
  
  async loadStats() {
    try {
      const result = await chrome.storage.local.get(['captureStats']);
      this.stats = result.captureStats || {
        totalCaptures: 0,
        totalProcessingTime: 0,
        capturesThisWeek: 0,
        capturesThisMonth: 0
      };
      
      this.updateStatsUI();
      
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  }
  
  updateUI() {
    // Update status based on extension state
    if (this.settings.enabled) {
      this.updateStatus('Ready', 'ready');
    } else {
      this.updateStatus('Disabled', 'disabled');
    }
    
    // Update button states
    this.elements.captureSelection.disabled = !this.settings.enabled;
    this.elements.capturePage.disabled = !this.settings.enabled;
  }
  
  updateStatus(text, status) {
    this.elements.statusText.textContent = text;
    
    // Update status dot color
    const colors = {
      ready: '#4caf50',
      processing: '#ff9800', 
      disabled: '#9e9e9e',
      error: '#f44336'
    };
    
    this.elements.statusDot.style.background = colors[status] || colors.ready;
  }
  
  updateCapturesUI() {
    const capturesList = this.elements.capturesList;
    
    if (this.captures.length === 0) {
      capturesList.innerHTML = `
        <div class="capture-empty">
          No captures yet. Select text and click "Capture Selection" to get started!
        </div>
      `;
      return;
    }
    
    capturesList.innerHTML = this.captures.map(capture => `
      <div class="capture-item" data-capture-id="${capture.id}">
        <div class="capture-summary">${this.truncateText(capture.summary, 100)}</div>
        <div class="capture-meta">
          <span class="capture-source">${this.formatSource(capture.source)}</span>
          <span class="capture-time">${this.formatTime(capture.timestamp)}</span>
        </div>
      </div>
    `).join('');
    
    // Add click listeners to capture items
    capturesList.querySelectorAll('.capture-item').forEach(item => {
      item.addEventListener('click', () => {
        const captureId = item.dataset.captureId;
        this.showCaptureDetails(captureId);
      });
    });
  }
  
  updateStatsUI() {
    this.elements.totalCaptures.textContent = this.stats.totalCaptures || 0;
    
    // Calculate average processing time
    const avgTime = this.stats.totalCaptures > 0 
      ? (this.stats.totalProcessingTime / this.stats.totalCaptures) * 1000
      : 0;
    this.elements.avgProcessingTime.textContent = `${Math.round(avgTime)}ms`;
    
    // Calculate today's captures
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const todayCaptures = this.captures.filter(capture => 
      new Date(capture.timestamp) >= today
    ).length;
    this.elements.todayCaptures.textContent = todayCaptures;
  }
  
  async captureSelection() {
    if (!this.settings.enabled) {
      this.showError('Extension is disabled');
      return;
    }
    
    try {
      this.showLoading('Capturing selection...');
      this.updateStatus('Processing', 'processing');
      
      // Get selected text from active tab
      const response = await chrome.tabs.sendMessage(this.currentTab.id, {
        type: 'GET_SELECTION'
      });
      
      if (!response.selection || response.selection.trim().length === 0) {
        this.hideLoading();
        this.updateStatus('Ready', 'ready');
        this.showError('No text selected. Please select some text and try again.');
        return;
      }
      
      // Send capture request to background script
      const result = await chrome.runtime.sendMessage({
        type: 'CAPTURE_TEXT',
        data: {
          text: response.selection,
          source: 'browser_extension',
          context: {
            url: this.currentTab.url,
            title: this.currentTab.title,
            captureType: 'selection',
            selectionLength: response.selection.length
          }
        }
      });
      
      if (result.success) {
        this.handleCaptureSuccess(result.data);
      } else {
        this.showError(result.error);
      }
      
    } catch (error) {
      console.error('Capture selection failed:', error);
      this.showError('Failed to capture selection: ' + error.message);
    } finally {
      this.hideLoading();
      this.updateStatus('Ready', 'ready');
    }
  }
  
  async capturePage() {
    if (!this.settings.enabled) {
      this.showError('Extension is disabled');
      return;
    }
    
    try {
      this.showLoading('Capturing page...');
      this.updateStatus('Processing', 'processing');
      
      // Send capture page request to background script
      const result = await chrome.runtime.sendMessage({
        type: 'CAPTURE_PAGE'
      });
      
      if (result.success) {
        this.handleCaptureSuccess(result.data);
      } else {
        this.showError(result.error);
      }
      
    } catch (error) {
      console.error('Capture page failed:', error);
      this.showError('Failed to capture page: ' + error.message);
    } finally {
      this.hideLoading();
      this.updateStatus('Ready', 'ready');
    }
  }
  
  handleCaptureSuccess(data) {
    // Add to captures list
    const capture = {
      id: Date.now().toString(),
      summary: data.summary,
      keywords: data.keywords,
      source: data.source || 'browser_extension',
      timestamp: Date.now(),
      processingTime: data.processing_time,
      url: this.currentTab.url,
      title: this.currentTab.title
    };
    
    this.captures.unshift(capture);
    this.captures = this.captures.slice(0, 10); // Keep only 10 recent
    
    // Update stats
    this.stats.totalCaptures = (this.stats.totalCaptures || 0) + 1;
    this.stats.totalProcessingTime = (this.stats.totalProcessingTime || 0) + (data.processing_time || 0);
    
    // Save to storage
    chrome.storage.local.set({
      recentCaptures: this.captures,
      captureStats: this.stats
    });
    
    // Update UI
    this.updateCapturesUI();
    this.updateStatsUI();
    
    // Show success message
    this.showSuccess(`Captured in ${Math.round((data.processing_time || 0) * 1000)}ms!`);
  }
  
  handleMessage(message, sender, sendResponse) {
    switch (message.type) {
      case 'CAPTURE_COMPLETE':
        this.handleCaptureSuccess(message.data);
        break;
        
      case 'UPDATE_STATUS':
        this.updateStatus(message.status, message.type);
        break;
        
      default:
        console.warn('Unknown message type:', message.type);
    }
  }
  
  showSettings() {
    this.elements.settingsPanel.classList.add('active');
  }
  
  hideSettings() {
    this.elements.settingsPanel.classList.remove('active');
  }
  
  async saveSettings() {
    try {
      // Get current settings from UI
      const newSettings = {
        enabled: true, // Always enabled for now
        autoCapture: this.elements.autoCapture.checked,
        showFloatingButton: this.elements.showFloatingButton.checked,
        summarizationLevel: this.elements.summarizationLevel.value,
        maxSummaryLength: parseInt(this.elements.maxSummaryLength.value)
      };
      
      // Save to chrome storage
      await chrome.storage.sync.set(newSettings);
      
      // Update local settings
      this.settings = newSettings;
      
      // Update content script settings
      if (this.currentTab) {
        chrome.tabs.sendMessage(this.currentTab.id, {
          type: 'UPDATE_SETTINGS',
          settings: newSettings
        }).catch(() => {
          // Content script might not be loaded, ignore error
        });
      }
      
      // Show success and hide settings
      this.showSuccess('Settings saved successfully!');
      this.hideSettings();
      
      // Update UI
      this.updateUI();
      
    } catch (error) {
      console.error('Failed to save settings:', error);
      this.showError('Failed to save settings');
    }
  }
  
  showHelp() {
    // Open help page in new tab
    chrome.tabs.create({
      url: 'https://github.com/your-repo/sum-extension#readme'
    });
  }
  
  showCaptureDetails(captureId) {
    const capture = this.captures.find(c => c.id === captureId);
    if (!capture) return;
    
    // Create and show modal with capture details
    // For now, just copy summary to clipboard
    navigator.clipboard.writeText(capture.summary).then(() => {
      this.showSuccess('Summary copied to clipboard!');
    }).catch(() => {
      this.showError('Failed to copy to clipboard');
    });
  }
  
  showLoading(message = 'Loading...') {
    this.elements.loadingOverlay.querySelector('.loading-text').textContent = message;
    this.elements.loadingOverlay.classList.add('active');
  }
  
  hideLoading() {
    this.elements.loadingOverlay.classList.remove('active');
  }
  
  showSuccess(message) {
    const successEl = this.elements.successMessage;
    successEl.querySelector('.success-text').textContent = message;
    successEl.classList.add('active');
    
    setTimeout(() => {
      successEl.classList.remove('active');
    }, 3000);
  }
  
  showError(message) {
    // Create error message element (similar to success)
    const errorEl = document.createElement('div');
    errorEl.className = 'error-message active';
    errorEl.innerHTML = `
      <div class="error-content">
        <span class="error-icon">❌</span>
        <span class="error-text">${message}</span>
      </div>
    `;
    
    // Add styles for error message
    errorEl.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: #f44336;
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      z-index: 3000;
      display: flex;
      align-items: center;
      gap: 8px;
    `;
    
    document.body.appendChild(errorEl);
    
    setTimeout(() => {
      errorEl.remove();
    }, 4000);
  }
  
  // Utility functions
  truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }
  
  formatSource(source) {
    const sourceMap = {
      'browser_extension': 'Web',
      'global_hotkey': 'Hotkey',
      'mobile_voice': 'Voice',
      'email': 'Email'
    };
    return sourceMap[source] || source;
  }
  
  formatTime(timestamp) {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new SumPopup();
});