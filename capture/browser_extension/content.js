/**
 * SUM Browser Extension - Content Script
 * 
 * Handles text selection, context detection, and UI injection
 * for seamless webpage capture experience.
 * 
 * Author: ototao (optimized with Claude Code)
 * License: Apache License 2.0
 */

(function() {
  'use strict';
  
  // Prevent multiple injections
  if (window.sumExtensionLoaded) {
    return;
  }
  window.sumExtensionLoaded = true;
  
  // Configuration
  const CONFIG = {
    minSelectionLength: 10,
    showFloatingButton: true,
    floatingButtonDelay: 500,
    maxTooltipLength: 100
  };
  
  // State
  let currentSelection = '';
  let floatingButton = null;
  let selectionTimeout = null;
  let isProcessing = false;
  
  // Initialize content script
  init();
  
  /**
   * Initialize the content script
   */
  function init() {
    console.log('SUM Extension content script loaded');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load user preferences
    loadUserPreferences();
    
    // Inject custom styles
    injectStyles();
  }
  
  /**
   * Set up event listeners for text selection and interactions
   */
  function setupEventListeners() {
    // Text selection events
    document.addEventListener('mouseup', handleTextSelection);
    document.addEventListener('keyup', handleTextSelection);
    document.addEventListener('selectionchange', handleSelectionChange);
    
    // Floating button events
    document.addEventListener('click', handleDocumentClick);
    
    // Keyboard shortcuts (handled by background script, but we can provide feedback)
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener(handleMessage);
  }
  
  /**
   * Handle text selection events
   */
  function handleTextSelection(event) {
    // Small delay to ensure selection is complete
    clearTimeout(selectionTimeout);
    selectionTimeout = setTimeout(() => {
      const selection = window.getSelection();
      const selectedText = selection.toString().trim();
      
      if (selectedText && selectedText.length >= CONFIG.minSelectionLength) {
        currentSelection = selectedText;
        
        if (CONFIG.showFloatingButton) {
          showFloatingButton(selection);
        }
        
        // Analytics: track selection
        trackSelection(selectedText);
      } else {
        hideFloatingButton();
        currentSelection = '';
      }
    }, 100);
  }
  
  /**
   * Handle selection change events
   */
  function handleSelectionChange() {
    const selection = window.getSelection();
    if (selection.rangeCount === 0) {
      hideFloatingButton();
      currentSelection = '';
    }
  }
  
  /**
   * Handle document clicks to hide floating button
   */
  function handleDocumentClick(event) {
    if (floatingButton && !floatingButton.contains(event.target)) {
      const selection = window.getSelection();
      if (selection.rangeCount === 0 || selection.toString().trim().length === 0) {
        hideFloatingButton();
      }
    }
  }
  
  /**
   * Handle keyboard shortcuts for user feedback
   */
  function handleKeyboardShortcuts(event) {
    // Ctrl+Shift+S or Cmd+Shift+S
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'S') {
      event.preventDefault();
      
      if (currentSelection) {
        captureSelection();
      } else {
        showTemporaryMessage('Please select some text first');
      }
    }
    
    // Ctrl+Shift+A or Cmd+Shift+A
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'A') {
      event.preventDefault();
      capturePage();
    }
  }
  
  /**
   * Handle messages from background script
   */
  function handleMessage(message, sender, sendResponse) {
    switch (message.type) {
      case 'GET_SELECTION':
        sendResponse({ selection: currentSelection });
        break;
        
      case 'CAPTURE_COMPLETE':
        handleCaptureComplete(message.data);
        sendResponse({ success: true });
        break;
        
      case 'SHOW_PROCESSING':
        showProcessingIndicator();
        sendResponse({ success: true });
        break;
        
      default:
        sendResponse({ success: false, error: 'Unknown message type' });
    }
  }
  
  /**
   * Show floating capture button near selection
   */
  function showFloatingButton(selection) {
    hideFloatingButton();
    
    // Get selection position
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // Create floating button
    floatingButton = createFloatingButton();
    
    // Position button
    const buttonRect = floatingButton.getBoundingClientRect();
    const x = Math.min(
      rect.left + (rect.width - buttonRect.width) / 2,
      window.innerWidth - buttonRect.width - 10
    );
    const y = rect.top - buttonRect.height - 10;
    
    floatingButton.style.left = `${Math.max(10, x)}px`;
    floatingButton.style.top = `${Math.max(10, y + window.scrollY)}px`;
    
    // Add to document
    document.body.appendChild(floatingButton);
    
    // Animate in
    setTimeout(() => {
      floatingButton.classList.add('sum-floating-button-visible');
    }, 10);
  }
  
  /**
   * Create floating capture button
   */
  function createFloatingButton() {
    const button = document.createElement('div');
    button.className = 'sum-floating-button';
    button.innerHTML = `
      <div class="sum-floating-button-content">
        <span class="sum-floating-button-icon">✨</span>
        <span class="sum-floating-button-text">Capture & Summarize</span>
        <div class="sum-floating-button-actions">
          <button class="sum-action-btn sum-capture-btn" title="Capture Selection">
            <span>📝</span>
          </button>
          <button class="sum-action-btn sum-insights-btn" title="Quick Insights">
            <span>🔍</span>
          </button>
          <button class="sum-action-btn sum-quotes-btn" title="Extract Quotes">
            <span>💬</span>
          </button>
        </div>
      </div>
    `;
    
    // Add event listeners
    button.querySelector('.sum-capture-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      captureSelection();
    });
    
    button.querySelector('.sum-insights-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      captureInsights();
    });
    
    button.querySelector('.sum-quotes-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      extractQuotes();
    });
    
    return button;
  }
  
  /**
   * Hide floating button
   */
  function hideFloatingButton() {
    if (floatingButton) {
      floatingButton.classList.remove('sum-floating-button-visible');
      setTimeout(() => {
        if (floatingButton && floatingButton.parentNode) {
          floatingButton.parentNode.removeChild(floatingButton);
        }
        floatingButton = null;
      }, 200);
    }
  }
  
  /**
   * Capture selected text
   */
  async function captureSelection() {
    if (!currentSelection || isProcessing) return;
    
    isProcessing = true;
    hideFloatingButton();
    showProcessingIndicator();
    
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'CAPTURE_TEXT',
        data: {
          text: currentSelection,
          source: 'browser_extension',
          context: {
            url: window.location.href,
            title: document.title,
            captureType: 'selection',
            selectionLength: currentSelection.length,
            pageType: detectPageType()
          }
        }
      });
      
      if (response.success) {
        showCaptureSuccess(response.data);
      } else {
        showCaptureError(response.error);
      }
      
    } catch (error) {
      console.error('Capture failed:', error);
      showCaptureError(error.message);
    } finally {
      isProcessing = false;
      hideProcessingIndicator();
    }
  }
  
  /**
   * Capture page insights
   */
  async function captureInsights() {
    if (!currentSelection || isProcessing) return;
    
    isProcessing = true;
    hideFloatingButton();
    showProcessingIndicator();
    
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'CAPTURE_TEXT',
        data: {
          text: currentSelection,
          source: 'browser_extension',
          context: {
            url: window.location.href,
            title: document.title,
            captureType: 'insights',
            processingMode: 'fast_insights',
            pageType: detectPageType()
          }
        }
      });
      
      if (response.success) {
        showInsightsResult(response.data);
      } else {
        showCaptureError(response.error);
      }
      
    } catch (error) {
      console.error('Insights extraction failed:', error);
      showCaptureError(error.message);
    } finally {
      isProcessing = false;
      hideProcessingIndicator();
    }
  }
  
  /**
   * Extract quotes from selection
   */
  async function extractQuotes() {
    if (!currentSelection || isProcessing) return;
    
    isProcessing = true;
    hideFloatingButton();
    showProcessingIndicator();
    
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'CAPTURE_TEXT',
        data: {
          text: currentSelection,
          source: 'browser_extension',
          context: {
            url: window.location.href,
            title: document.title,
            captureType: 'quotes',
            processingMode: 'extract_quotes',
            pageType: detectPageType()
          }
        }
      });
      
      if (response.success) {
        showQuotesResult(response.data);
      } else {
        showCaptureError(response.error);
      }
      
    } catch (error) {
      console.error('Quote extraction failed:', error);
      showCaptureError(error.message);
    } finally {
      isProcessing = false;
      hideProcessingIndicator();
    }
  }
  
  /**
   * Capture entire page
   */
  async function capturePage() {
    if (isProcessing) return;
    
    isProcessing = true;
    showProcessingIndicator();
    
    try {
      // This will trigger the background script to capture the page
      await chrome.runtime.sendMessage({
        type: 'CAPTURE_PAGE'
      });
      
    } catch (error) {
      console.error('Page capture failed:', error);
      showCaptureError(error.message);
    } finally {
      isProcessing = false;
      hideProcessingIndicator();
    }
  }
  
  /**
   * Detect page type for context-aware processing
   */
  function detectPageType() {
    const url = window.location.href.toLowerCase();
    const title = document.title.toLowerCase();
    
    // Check meta tags
    const description = document.querySelector('meta[name="description"]')?.content?.toLowerCase() || '';
    const keywords = document.querySelector('meta[name="keywords"]')?.content?.toLowerCase() || '';
    
    // Article detection
    if (document.querySelector('article') || 
        url.includes('/article/') || 
        title.includes('article') ||
        document.querySelector('[itemtype*="Article"]')) {
      return 'article';
    }
    
    // Blog post detection
    if (url.includes('/blog/') || 
        url.includes('/post/') || 
        title.includes('blog') ||
        document.querySelector('.blog-post, .post-content')) {
      return 'blog_post';
    }
    
    // News detection
    if (url.includes('news') || 
        title.includes('news') || 
        description.includes('news') ||
        document.querySelector('.news-article, .article-content')) {
      return 'news';
    }
    
    // Documentation detection
    if (url.includes('docs.') || 
        url.includes('/docs/') || 
        title.includes('documentation') ||
        document.querySelector('.documentation, .docs')) {
      return 'documentation';
    }
    
    // Research paper detection
    if (url.includes('arxiv.org') || 
        url.includes('scholar.google') || 
        title.includes('research') ||
        document.querySelector('.paper, .abstract')) {
      return 'research_paper';
    }
    
    return 'webpage';
  }
  
  /**
   * Show processing indicator
   */
  function showProcessingIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'sum-processing-indicator';
    indicator.className = 'sum-processing-indicator';
    indicator.innerHTML = `
      <div class="sum-processing-content">
        <div class="sum-spinner"></div>
        <span>Processing with SUM...</span>
      </div>
    `;
    
    document.body.appendChild(indicator);
    
    setTimeout(() => {
      indicator.classList.add('sum-processing-visible');
    }, 10);
  }
  
  /**
   * Hide processing indicator
   */
  function hideProcessingIndicator() {
    const indicator = document.getElementById('sum-processing-indicator');
    if (indicator) {
      indicator.classList.remove('sum-processing-visible');
      setTimeout(() => {
        if (indicator.parentNode) {
          indicator.parentNode.removeChild(indicator);
        }
      }, 200);
    }
  }
  
  /**
   * Show capture success message
   */
  function showCaptureSuccess(data) {
    showTemporaryMessage(
      `✅ Captured! Summary: ${data.summary?.substring(0, CONFIG.maxTooltipLength)}...`,
      'success',
      5000
    );
  }
  
  /**
   * Show insights result
   */
  function showInsightsResult(data) {
    const insights = data.keywords?.join(', ') || 'Key insights extracted';
    showTemporaryMessage(
      `🔍 Insights: ${insights.substring(0, CONFIG.maxTooltipLength)}...`,
      'info',
      5000
    );
  }
  
  /**
   * Show quotes result
   */
  function showQuotesResult(data) {
    showTemporaryMessage(
      `💬 Quotes extracted successfully!`,
      'success',
      3000
    );
  }
  
  /**
   * Show capture error message
   */
  function showCaptureError(error) {
    showTemporaryMessage(
      `❌ Capture failed: ${error}`,
      'error',
      4000
    );
  }
  
  /**
   * Show temporary message to user
   */
  function showTemporaryMessage(message, type = 'info', duration = 3000) {
    const messageEl = document.createElement('div');
    messageEl.className = `sum-temp-message sum-temp-message-${type}`;
    messageEl.textContent = message;
    
    document.body.appendChild(messageEl);
    
    setTimeout(() => {
      messageEl.classList.add('sum-temp-message-visible');
    }, 10);
    
    setTimeout(() => {
      messageEl.classList.remove('sum-temp-message-visible');
      setTimeout(() => {
        if (messageEl.parentNode) {
          messageEl.parentNode.removeChild(messageEl);
        }
      }, 200);
    }, duration);
  }
  
  /**
   * Track selection for analytics
   */
  function trackSelection(text) {
    // Simple analytics tracking
    const stats = {
      timestamp: Date.now(),
      textLength: text.length,
      wordCount: text.split(' ').length,
      url: window.location.href,
      pageType: detectPageType()
    };
    
    // Store in local storage for extension popup
    chrome.storage.local.get(['selectionStats'], (result) => {
      const existingStats = result.selectionStats || [];
      existingStats.push(stats);
      
      // Keep only last 100 selections
      if (existingStats.length > 100) {
        existingStats.splice(0, existingStats.length - 100);
      }
      
      chrome.storage.local.set({ selectionStats: existingStats });
    });
  }
  
  /**
   * Load user preferences
   */
  function loadUserPreferences() {
    chrome.storage.sync.get([
      'showFloatingButton',
      'minSelectionLength',
      'floatingButtonDelay'
    ], (result) => {
      if (result.showFloatingButton !== undefined) {
        CONFIG.showFloatingButton = result.showFloatingButton;
      }
      if (result.minSelectionLength !== undefined) {
        CONFIG.minSelectionLength = result.minSelectionLength;
      }
      if (result.floatingButtonDelay !== undefined) {
        CONFIG.floatingButtonDelay = result.floatingButtonDelay;
      }
    });
  }
  
  /**
   * Inject custom styles for the extension UI
   */
  function injectStyles() {
    const styleId = 'sum-extension-styles';
    if (document.getElementById(styleId)) {
      return; // Styles already injected
    }
    
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      /* SUM Extension Styles */
      .sum-floating-button {
        position: absolute;
        z-index: 999999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.2s ease;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        cursor: pointer;
        user-select: none;
      }
      
      .sum-floating-button-visible {
        opacity: 1;
        transform: translateY(0);
      }
      
      .sum-floating-button-content {
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: white;
        font-size: 14px;
        font-weight: 500;
      }
      
      .sum-floating-button-icon {
        font-size: 16px;
      }
      
      .sum-floating-button-text {
        white-space: nowrap;
      }
      
      .sum-floating-button-actions {
        display: flex;
        gap: 4px;
        margin-left: 8px;
        padding-left: 8px;
        border-left: 1px solid rgba(255, 255, 255, 0.3);
      }
      
      .sum-action-btn {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 6px;
        padding: 4px 8px;
        color: white;
        cursor: pointer;
        transition: background 0.2s ease;
        font-size: 12px;
      }
      
      .sum-action-btn:hover {
        background: rgba(255, 255, 255, 0.3);
      }
      
      .sum-processing-indicator {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999999;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
      }
      
      .sum-processing-visible {
        opacity: 1;
        transform: translateX(0);
      }
      
      .sum-processing-content {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .sum-spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-top: 2px solid white;
        border-radius: 50%;
        animation: sum-spin 1s linear infinite;
      }
      
      @keyframes sum-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      
      .sum-temp-message {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999999;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        font-weight: 500;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
        max-width: 400px;
        word-wrap: break-word;
      }
      
      .sum-temp-message-visible {
        opacity: 1;
        transform: translateY(0);
      }
      
      .sum-temp-message-success {
        background: #4caf50;
        color: white;
      }
      
      .sum-temp-message-error {
        background: #f44336;
        color: white;
      }
      
      .sum-temp-message-info {
        background: #2196f3;
        color: white;
      }
    `;
    
    document.head.appendChild(style);
  }
  
})();