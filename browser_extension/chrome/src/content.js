/**
 * SUM Browser Extension - Content Script
 * Handles page interaction, text extraction, and summary display
 */

// Summary display container
let summaryContainer = null;

// Initialize content script
(function() {
  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.action) {
      case 'getSelection':
        const selectedText = window.getSelection().toString();
        sendResponse({ text: selectedText });
        break;
        
      case 'getPageContent':
        const content = extractPageContent();
        sendResponse({ content });
        break;
        
      case 'showSummary':
        showSummaryOverlay(request.summary, request.source);
        break;
        
      case 'showError':
        showErrorMessage(request.error);
        break;
    }
  });
  
  // Add floating summarize button for selected text
  let selectionButton = null;
  
  document.addEventListener('mouseup', (e) => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    
    if (text.length > 50) {
      showSelectionButton(e.pageX, e.pageY, text);
    } else {
      hideSelectionButton();
    }
  });
  
  document.addEventListener('mousedown', (e) => {
    if (!e.target.closest('.sum-selection-button')) {
      hideSelectionButton();
    }
  });
  
  function showSelectionButton(x, y, text) {
    if (!selectionButton) {
      selectionButton = document.createElement('div');
      selectionButton.className = 'sum-selection-button';
      selectionButton.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
        <span>Summarize</span>
      `;
      document.body.appendChild(selectionButton);
      
      selectionButton.addEventListener('click', () => {
        chrome.runtime.sendMessage({
          action: 'summarize',
          text: text,
          type: 'selection'
        });
        hideSelectionButton();
      });
    }
    
    selectionButton.style.left = `${x + 10}px`;
    selectionButton.style.top = `${y - 40}px`;
    selectionButton.style.display = 'flex';
  }
  
  function hideSelectionButton() {
    if (selectionButton) {
      selectionButton.style.display = 'none';
    }
  }
})();

// Extract main content from page
function extractPageContent() {
  // Remove script and style elements
  const clonedDoc = document.cloneNode(true);
  const scripts = clonedDoc.querySelectorAll('script, style, noscript');
  scripts.forEach(el => el.remove());
  
  // Try to find main content areas
  const contentSelectors = [
    'main',
    'article',
    '[role="main"]',
    '.content',
    '#content',
    '.post',
    '.article-body',
    '.entry-content'
  ];
  
  let content = '';
  
  for (const selector of contentSelectors) {
    const element = document.querySelector(selector);
    if (element) {
      content = element.innerText;
      break;
    }
  }
  
  // Fallback to body content
  if (!content) {
    content = document.body.innerText;
  }
  
  // Clean up whitespace
  content = content.replace(/\s+/g, ' ').trim();
  
  // Add page title
  const title = document.title;
  if (title) {
    content = `${title}\n\n${content}`;
  }
  
  return content;
}

// Show summary overlay
function showSummaryOverlay(summary, source) {
  // Remove existing container
  if (summaryContainer) {
    summaryContainer.remove();
  }
  
  // Create container
  summaryContainer = document.createElement('div');
  summaryContainer.className = 'sum-summary-container';
  
  // Build summary content
  const hierarchical = summary.hierarchical_summary;
  const tags = summary.tags || [];
  
  summaryContainer.innerHTML = `
    <div class="sum-summary-header">
      <h3>SUM Summary</h3>
      <div class="sum-summary-actions">
        <button class="sum-copy-btn" title="Copy summary">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
        </button>
        <button class="sum-close-btn" title="Close">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
    </div>
    <div class="sum-summary-content">
      <div class="sum-summary-main">
        <h4>Summary</h4>
        <p>${summary.summary || summary.sum}</p>
      </div>
      ${hierarchical ? `
        <div class="sum-summary-hierarchical">
          <h4>Detailed Summary</h4>
          ${hierarchical.level_1_essence ? `<p><strong>Essence:</strong> ${hierarchical.level_1_essence}</p>` : ''}
          ${hierarchical.level_2_core ? `<p><strong>Core:</strong> ${hierarchical.level_2_core}</p>` : ''}
          ${hierarchical.level_3_expanded ? `<p><strong>Expanded:</strong> ${hierarchical.level_3_expanded}</p>` : ''}
        </div>
      ` : ''}
      ${tags.length > 0 ? `
        <div class="sum-summary-tags">
          <h4>Key Concepts</h4>
          <div class="sum-tags">
            ${tags.map(tag => `<span class="sum-tag">${tag}</span>`).join('')}
          </div>
        </div>
      ` : ''}
      <div class="sum-summary-meta">
        <span>Model: ${summary.model || 'unknown'}</span>
        <span>Compression: ${summary.compression_ratio ? `${summary.compression_ratio.toFixed(1)}x` : 'N/A'}</span>
        ${summary.cached ? '<span class="sum-cached">Cached</span>' : ''}
      </div>
    </div>
  `;
  
  document.body.appendChild(summaryContainer);
  
  // Add event listeners
  const closeBtn = summaryContainer.querySelector('.sum-close-btn');
  closeBtn.addEventListener('click', () => {
    summaryContainer.remove();
    summaryContainer = null;
  });
  
  const copyBtn = summaryContainer.querySelector('.sum-copy-btn');
  copyBtn.addEventListener('click', () => {
    const summaryText = formatSummaryForCopy(summary);
    navigator.clipboard.writeText(summaryText).then(() => {
      copyBtn.innerHTML = '✓';
      setTimeout(() => {
        copyBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
        `;
      }, 2000);
    });
  });
  
  // Make draggable
  makeDraggable(summaryContainer);
}

// Format summary for copying
function formatSummaryForCopy(summary) {
  let text = `SUM Summary\n\n${summary.summary || summary.sum}\n`;
  
  if (summary.hierarchical_summary) {
    const h = summary.hierarchical_summary;
    text += '\n--- Detailed Summary ---\n';
    if (h.level_1_essence) text += `Essence: ${h.level_1_essence}\n`;
    if (h.level_2_core) text += `Core: ${h.level_2_core}\n`;
    if (h.level_3_expanded) text += `Expanded: ${h.level_3_expanded}\n`;
  }
  
  if (summary.tags && summary.tags.length > 0) {
    text += `\nKey Concepts: ${summary.tags.join(', ')}\n`;
  }
  
  text += `\n[Summarized with SUM - ${new Date().toLocaleString()}]`;
  
  return text;
}

// Show error message
function showErrorMessage(error) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'sum-error-message';
  errorDiv.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    <span>${error}</span>
  `;
  
  document.body.appendChild(errorDiv);
  
  setTimeout(() => {
    errorDiv.remove();
  }, 5000);
}

// Make element draggable
function makeDraggable(element) {
  let isDragging = false;
  let currentX;
  let currentY;
  let initialX;
  let initialY;
  let xOffset = 0;
  let yOffset = 0;
  
  const header = element.querySelector('.sum-summary-header');
  
  header.addEventListener('mousedown', dragStart);
  
  function dragStart(e) {
    initialX = e.clientX - xOffset;
    initialY = e.clientY - yOffset;
    
    if (e.target.closest('.sum-summary-actions')) {
      return;
    }
    
    isDragging = true;
    header.style.cursor = 'grabbing';
  }
  
  document.addEventListener('mousemove', drag);
  document.addEventListener('mouseup', dragEnd);
  
  function drag(e) {
    if (!isDragging) return;
    
    e.preventDefault();
    currentX = e.clientX - initialX;
    currentY = e.clientY - initialY;
    
    xOffset = currentX;
    yOffset = currentY;
    
    element.style.transform = `translate(${currentX}px, ${currentY}px)`;
  }
  
  function dragEnd(e) {
    initialX = currentX;
    initialY = currentY;
    
    isDragging = false;
    header.style.cursor = 'grab';
  }
}