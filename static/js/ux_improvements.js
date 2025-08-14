/**
 * UX Improvements for SUM Platform
 * 
 * Enhances user experience with:
 * - Better form validation
 * - Progress indicators
 * - Dark mode support
 * - Copy functionality
 * - Character counting
 * - Tooltips
 */

// Initialize UX improvements when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeValidation();
    initializeProgressIndicators();
    initializeDarkMode();
    initializeCopyButtons();
    initializeCharacterCount();
    initializeTooltips();
    initializeMobileMenu();
});

// 1. Replace alert() with inline validation
function initializeValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const isValid = validateForm(this);
            if (!isValid) {
                e.preventDefault();
            }
        });
    });
}

function validateForm(form) {
    let isValid = true;
    const errors = [];
    
    // Clear previous errors
    form.querySelectorAll('.error-message').forEach(el => el.remove());
    form.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
    
    // Validate text input
    const textInput = form.querySelector('textarea[name="text"]');
    if (textInput && !textInput.value.trim()) {
        showFieldError(textInput, 'Please enter some text to summarize');
        errors.push('Text is required');
        isValid = false;
    } else if (textInput && textInput.value.trim().length < 10) {
        showFieldError(textInput, 'Text must be at least 10 characters long');
        errors.push('Text too short');
        isValid = false;
    }
    
    // Validate file input
    const fileInput = form.querySelector('input[type="file"]');
    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        if (file.size > maxSize) {
            showFieldError(fileInput, `File size must be less than ${formatBytes(maxSize)}`);
            errors.push('File too large');
            isValid = false;
        }
    }
    
    return isValid;
}

function showFieldError(field, message) {
    field.classList.add('error');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.cssText = 'color: #e74c3c; font-size: 14px; margin-top: 5px;';
    field.parentNode.insertBefore(errorDiv, field.nextSibling);
}

// 2. Enhanced progress indicators
function initializeProgressIndicators() {
    // Override existing showLoading function if it exists
    window.showLoading = function(containerId, message = 'Processing...') {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const steps = [
            'Analyzing content...',
            'Extracting key concepts...',
            'Generating summary...',
            'Finalizing results...'
        ];
        
        let currentStep = 0;
        container.innerHTML = `
            <div class="progress-container">
                <div class="progress-spinner"></div>
                <div class="progress-text">${steps[currentStep]}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 25%"></div>
                </div>
                <button class="cancel-btn" onclick="cancelOperation()">Cancel</button>
            </div>
        `;
        
        // Animate through steps
        const interval = setInterval(() => {
            currentStep++;
            if (currentStep < steps.length) {
                container.querySelector('.progress-text').textContent = steps[currentStep];
                container.querySelector('.progress-fill').style.width = `${(currentStep + 1) * 25}%`;
            } else {
                clearInterval(interval);
            }
        }, 2000);
        
        container.dataset.progressInterval = interval;
    };
    
    window.hideLoading = function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Clear interval if exists
        if (container.dataset.progressInterval) {
            clearInterval(parseInt(container.dataset.progressInterval));
        }
        
        container.innerHTML = '';
    };
}

// 3. Dark mode toggle
function initializeDarkMode() {
    // Check for saved preference or system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = savedTheme === 'dark' || (!savedTheme && systemPrefersDark);
    
    // Add dark mode toggle button
    const header = document.querySelector('header nav');
    if (header) {
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'theme-toggle';
        toggleBtn.innerHTML = isDark ? '☀️' : '🌙';
        toggleBtn.title = 'Toggle dark mode';
        toggleBtn.onclick = toggleDarkMode;
        header.appendChild(toggleBtn);
    }
    
    // Apply initial theme
    if (isDark) {
        document.body.classList.add('dark-mode');
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    
    // Save preference
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    
    // Update toggle button
    const toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
        toggleBtn.innerHTML = isDark ? '☀️' : '🌙';
    }
}

// 4. Copy to clipboard functionality
function initializeCopyButtons() {
    // Add copy buttons to result containers
    const addCopyButton = (container) => {
        if (container.querySelector('.copy-btn')) return;
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = '📋 Copy';
        copyBtn.title = 'Copy to clipboard';
        
        copyBtn.onclick = function() {
            const text = container.querySelector('p, pre')?.textContent || '';
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.innerHTML = '✅ Copied!';
                setTimeout(() => {
                    copyBtn.innerHTML = '📋 Copy';
                }, 2000);
            }).catch(err => {
                console.error('Copy failed:', err);
                copyBtn.innerHTML = '❌ Failed';
            });
        };
        
        container.style.position = 'relative';
        container.appendChild(copyBtn);
    };
    
    // Watch for new results
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1 && (node.classList?.contains('result') || node.id?.includes('result'))) {
                    addCopyButton(node);
                }
            });
        });
    });
    
    // Observe result containers
    document.querySelectorAll('#textResult, #condensedResult, #fileResult').forEach(container => {
        observer.observe(container, { childList: true });
    });
}

// 5. Character/word count
function initializeCharacterCount() {
    const textareas = document.querySelectorAll('textarea');
    
    textareas.forEach(textarea => {
        const counter = document.createElement('div');
        counter.className = 'char-counter';
        counter.style.cssText = 'text-align: right; font-size: 12px; color: #666; margin-top: 5px;';
        textarea.parentNode.insertBefore(counter, textarea.nextSibling);
        
        const updateCount = () => {
            const text = textarea.value;
            const chars = text.length;
            const words = text.trim().split(/\s+/).filter(w => w.length > 0).length;
            counter.textContent = `${chars} characters | ${words} words`;
            
            // Add warning if too long
            if (chars > 50000) {
                counter.style.color = '#e74c3c';
                counter.textContent += ' (may take longer to process)';
            } else {
                counter.style.color = '#666';
            }
        };
        
        textarea.addEventListener('input', updateCount);
        updateCount();
    });
}

// 6. Tooltips for better guidance
function initializeTooltips() {
    const tooltips = {
        'model-simple': 'Fast, efficient summarization for general content',
        'model-advanced': 'Advanced AI model for complex, nuanced content',
        'max-length': 'Maximum length of the summary in words',
        'condensed-ratio': 'How much to compress (0.3 = 30% of original)'
    };
    
    // Add tooltips to form elements
    Object.entries(tooltips).forEach(([selector, text]) => {
        const elements = document.querySelectorAll(`[name="${selector}"], #${selector}, .${selector}`);
        elements.forEach(el => {
            el.title = text;
            
            // Add question mark icon
            const helpIcon = document.createElement('span');
            helpIcon.className = 'help-icon';
            helpIcon.innerHTML = ' ℹ️';
            helpIcon.title = text;
            helpIcon.style.cssText = 'cursor: help; font-size: 14px;';
            
            if (el.parentNode.tagName === 'LABEL') {
                el.parentNode.appendChild(helpIcon);
            } else {
                el.parentNode.insertBefore(helpIcon, el.nextSibling);
            }
        });
    });
}

// 7. Mobile menu
function initializeMobileMenu() {
    const nav = document.querySelector('header nav');
    if (!nav) return;
    
    // Create hamburger button
    const hamburger = document.createElement('button');
    hamburger.className = 'hamburger';
    hamburger.innerHTML = `
        <span></span>
        <span></span>
        <span></span>
    `;
    hamburger.style.display = 'none';
    
    nav.parentNode.insertBefore(hamburger, nav);
    
    // Toggle menu
    hamburger.onclick = () => {
        nav.classList.toggle('mobile-open');
        hamburger.classList.toggle('active');
    };
    
    // Close menu on link click
    nav.querySelectorAll('a').forEach(link => {
        link.onclick = () => {
            nav.classList.remove('mobile-open');
            hamburger.classList.remove('active');
        };
    });
}

// Helper functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function cancelOperation() {
    // This would need to be connected to actual cancel functionality
    console.log('Cancel operation requested');
    hideLoading('loadingIndicator');
    hideLoading('fileLoadingIndicator');
}