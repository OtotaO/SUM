/**
 * SUM Browser Extension - Background Service Worker
 * 
 * Handles context menus, keyboard shortcuts, and communication
 * with the SUM capture system.
 * 
 * Author: ototao (optimized with Claude Code)
 * License: Apache License 2.0
 */

// Configuration
const CONFIG = {
  sumApiUrl: 'http://localhost:8000/api/capture',
  maxRetries: 3,
  retryDelay: 1000,
  timeout: 30000
};

// Install event - setup context menus
chrome.runtime.onInstalled.addListener(() => {
  console.log('SUM Extension installed');
  
  // Create context menus
  createContextMenus();
  
  // Set default settings
  chrome.storage.sync.set({
    enabled: true,
    autoCapture: false,
    summarizationLevel: 'quality',
    maxSummaryLength: 150
  });
});

/**
 * Create context menu items for text capture
 */
function createContextMenus() {
  // Remove existing menus
  chrome.contextMenus.removeAll(() => {
    // Main capture menu
    chrome.contextMenus.create({
      id: 'sum-capture-selection',
      title: '✨ Capture & Summarize Selection',
      contexts: ['selection']
    });
    
    chrome.contextMenus.create({
      id: 'sum-capture-page',
      title: '📄 Capture & Summarize Page',
      contexts: ['page']
    });
    
    chrome.contextMenus.create({
      id: 'sum-separator',
      type: 'separator',
      contexts: ['selection', 'page']
    });
    
    chrome.contextMenus.create({
      id: 'sum-quick-insights',
      title: '🔍 Quick Insights',
      contexts: ['selection']
    });
    
    chrome.contextMenus.create({
      id: 'sum-extract-quotes',
      title: '💬 Extract Key Quotes',
      contexts: ['selection']
    });
  });
}

/**
 * Handle context menu clicks
 */
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  try {
    switch (info.menuItemId) {
      case 'sum-capture-selection':
        await handleCaptureSelection(info, tab);
        break;
        
      case 'sum-capture-page':
        await handleCapturePage(info, tab);
        break;
        
      case 'sum-quick-insights':
        await handleQuickInsights(info, tab);
        break;
        
      case 'sum-extract-quotes':
        await handleExtractQuotes(info, tab);
        break;
    }
  } catch (error) {
    console.error('Context menu action failed:', error);
    showNotification('Capture failed', error.message, 'error');
  }
});

/**
 * Handle keyboard shortcuts
 */
chrome.commands.onCommand.addListener(async (command, tab) => {
  try {
    switch (command) {
      case 'capture-selection':
        await handleCaptureSelection({}, tab);
        break;
        
      case 'capture-page':
        await handleCapturePage({}, tab);
        break;
    }
  } catch (error) {
    console.error('Keyboard shortcut failed:', error);
    showNotification('Capture failed', error.message, 'error');
  }
});

/**
 * Handle messages from content scripts and popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {
        case 'CAPTURE_TEXT':
          const result = await captureText(message.data);
          sendResponse({ success: true, data: result });
          break;
          
        case 'GET_PAGE_CONTEXT':
          const context = await getPageContext(sender.tab);
          sendResponse({ success: true, data: context });
          break;
          
        case 'SHOW_NOTIFICATION':
          showNotification(message.title, message.message, message.type);
          sendResponse({ success: true });
          break;
          
        default:
          sendResponse({ success: false, error: 'Unknown message type' });
      }
    } catch (error) {
      console.error('Message handler error:', error);
      sendResponse({ success: false, error: error.message });
    }
  })();
  
  return true; // Keep message channel open for async response
});

/**
 * Handle text selection capture
 */
async function handleCaptureSelection(info, tab) {
  const text = info.selectionText || await getSelectedText(tab);
  
  if (!text || text.trim().length === 0) {
    showNotification('No text selected', 'Please select some text to capture', 'warning');
    return;
  }
  
  const context = await getPageContext(tab);
  
  const captureData = {
    text: text,
    source: 'browser_extension',
    context: {
      ...context,
      captureType: 'selection',
      selectionLength: text.length
    }
  };
  
  await captureText(captureData);
}

/**
 * Handle full page capture
 */
async function handleCapturePage(info, tab) {
  // Inject content script to extract page text
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: extractPageContent
  });
  
  const pageContent = results[0]?.result;
  
  if (!pageContent || pageContent.text.trim().length === 0) {
    showNotification('No content found', 'Could not extract text from this page', 'warning');
    return;
  }
  
  const context = await getPageContext(tab);
  
  const captureData = {
    text: pageContent.text,
    source: 'browser_extension',
    context: {
      ...context,
      captureType: 'full_page',
      extractedElements: pageContent.elements,
      wordCount: pageContent.text.split(' ').length
    }
  };
  
  await captureText(captureData);
}

/**
 * Handle quick insights extraction
 */
async function handleQuickInsights(info, tab) {
  const text = info.selectionText || await getSelectedText(tab);
  
  if (!text) {
    showNotification('No text selected', 'Please select some text for insights', 'warning');
    return;
  }
  
  const context = await getPageContext(tab);
  
  const captureData = {
    text: text,
    source: 'browser_extension',
    context: {
      ...context,
      captureType: 'insights',
      processingMode: 'fast_insights'
    }
  };
  
  await captureText(captureData);
}

/**
 * Handle quote extraction
 */
async function handleExtractQuotes(info, tab) {
  const text = info.selectionText || await getSelectedText(tab);
  
  if (!text) {
    showNotification('No text selected', 'Please select some text to extract quotes from', 'warning');
    return;
  }
  
  const context = await getPageContext(tab);
  
  const captureData = {
    text: text,
    source: 'browser_extension',
    context: {
      ...context,
      captureType: 'quotes',
      processingMode: 'extract_quotes'
    }
  };
  
  await captureText(captureData);
}

/**
 * Get selected text from active tab
 */
async function getSelectedText(tab) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: () => window.getSelection().toString()
    });
    
    return results[0]?.result || '';
  } catch (error) {
    console.error('Failed to get selected text:', error);
    return '';
  }
}

/**
 * Extract page content for analysis
 */
function extractPageContent() {
  // This function runs in the page context
  const content = {
    text: '',
    elements: []
  };
  
  // Extract main content based on semantic elements
  const contentSelectors = [
    'article',
    'main',
    '[role="main"]',
    '.content',
    '.post-content',
    '.entry-content',
    '.article-content'
  ];
  
  let mainContent = null;
  
  // Try to find main content container
  for (const selector of contentSelectors) {
    const element = document.querySelector(selector);
    if (element) {
      mainContent = element;
      break;
    }
  }
  
  // Fallback to body if no main content found
  if (!mainContent) {
    mainContent = document.body;
  }
  
  // Extract text while preserving structure
  const walker = document.createTreeWalker(
    mainContent,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: function(node) {
        // Skip script and style elements
        const parent = node.parentElement;
        if (parent && ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        
        // Skip hidden elements
        if (parent && (parent.style.display === 'none' || parent.style.visibility === 'hidden')) {
          return NodeFilter.FILTER_REJECT;
        }
        
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );
  
  const textNodes = [];
  let node;
  while (node = walker.nextNode()) {
    const text = node.textContent.trim();
    if (text.length > 0) {
      textNodes.push(text);
    }
  }
  
  content.text = textNodes.join(' ').replace(/\\s+/g, ' ').trim();
  
  // Extract semantic elements
  content.elements = extractSemanticElements(mainContent);
  
  return content;
}

/**
 * Extract semantic elements from content
 */
function extractSemanticElements(container) {
  const elements = [];
  
  // Extract headings
  const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
  headings.forEach(heading => {
    elements.push({
      type: 'heading',
      level: parseInt(heading.tagName.charAt(1)),
      text: heading.textContent.trim()
    });
  });
  
  // Extract quotes
  const quotes = container.querySelectorAll('blockquote, q');
  quotes.forEach(quote => {
    elements.push({
      type: 'quote',
      text: quote.textContent.trim()
    });
  });
  
  // Extract lists
  const lists = container.querySelectorAll('ul, ol');
  lists.forEach(list => {
    const items = Array.from(list.querySelectorAll('li')).map(li => li.textContent.trim());
    elements.push({
      type: 'list',
      items: items
    });
  });
  
  return elements;
}

/**
 * Get page context information
 */
async function getPageContext(tab) {
  return {
    url: tab.url,
    title: tab.title,
    timestamp: Date.now(),
    pageType: classifyPageType(tab.url, tab.title),
    domain: new URL(tab.url).hostname
  };
}

/**
 * Classify page type for context-aware processing
 */
function classifyPageType(url, title) {
  const urlLower = url.toLowerCase();
  const titleLower = title.toLowerCase();
  
  // Email clients
  if (urlLower.includes('gmail.com') || urlLower.includes('outlook.') || urlLower.includes('mail.')) {
    return 'email';
  }
  
  // Social media
  if (urlLower.includes('twitter.com') || urlLower.includes('linkedin.com') || urlLower.includes('facebook.com')) {
    return 'social_media';
  }
  
  // News sites
  if (urlLower.includes('news') || titleLower.includes('news') || 
      urlLower.match(/\\.(com|org|net)\\/\\d{4}\\/\\d{2}\\/\\d{2}\\//)) {
    return 'news_article';
  }
  
  // Documentation
  if (urlLower.includes('docs.') || urlLower.includes('documentation') || 
      titleLower.includes('documentation') || titleLower.includes('api')) {
    return 'documentation';
  }
  
  // Research papers
  if (urlLower.includes('arxiv.org') || urlLower.includes('scholar.google') || 
      titleLower.includes('research') || titleLower.includes('paper')) {
    return 'research_paper';
  }
  
  // Blog posts
  if (urlLower.includes('blog') || titleLower.includes('blog') || 
      urlLower.match(/\\.(com|net|org)\\/\\d{4}\\/\\d{2}\\//)) {
    return 'blog_post';
  }
  
  return 'webpage';
}

/**
 * Send capture request to SUM API
 */
async function captureText(captureData) {
  try {
    showNotification('Processing...', 'Capturing and summarizing your text', 'info');
    
    const response = await fetchWithRetry(CONFIG.sumApiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(captureData)
    });
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Show success notification
    showNotification(
      'Capture Complete!', 
      `Processed in ${result.processing_time?.toFixed(2) || 0}s`, 
      'success'
    );
    
    // Store result for popup access
    await chrome.storage.local.set({
      lastCapture: {
        ...result,
        timestamp: Date.now(),
        originalData: captureData
      }
    });
    
    return result;
    
  } catch (error) {
    console.error('Capture failed:', error);
    showNotification('Capture Failed', error.message, 'error');
    throw error;
  }
}

/**
 * Fetch with retry logic
 */
async function fetchWithRetry(url, options, retries = CONFIG.maxRetries) {
  for (let i = 0; i < retries; i++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.timeout);
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response;
      
    } catch (error) {
      if (i === retries - 1) throw error;
      
      console.warn(`Request failed, retrying... (${i + 1}/${retries})`);
      await new Promise(resolve => setTimeout(resolve, CONFIG.retryDelay * (i + 1)));
    }
  }
}

/**
 * Show notification to user
 */
function showNotification(title, message, type = 'info') {
  const icons = {
    info: 'icons/icon48.png',
    success: 'icons/icon48.png',
    warning: 'icons/icon48.png',
    error: 'icons/icon48.png'
  };
  
  chrome.notifications.create({
    type: 'basic',
    iconUrl: icons[type] || icons.info,
    title: title,
    message: message
  });
}