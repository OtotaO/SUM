/**
 * Interactive Onboarding for SUM Platform
 * Helps new users understand the interface and features
 */

class OnboardingTour {
    constructor() {
        this.steps = [
            {
                element: '#textInput',
                title: 'Enter Your Text',
                content: 'Paste or type the text you want to summarize here. We can handle documents up to 50,000 characters.',
                position: 'bottom'
            },
            {
                element: 'select[name="model"]',
                title: 'Choose Your Model',
                content: 'SimpleSUM is fast and efficient for general content. MagnumOpusSUM provides deeper analysis for complex texts.',
                position: 'top'
            },
            {
                element: '#maxLength',
                title: 'Set Summary Length',
                content: 'Control how detailed your summary should be. Lower values create more concise summaries.',
                position: 'top'
            },
            {
                element: 'button[type="submit"]',
                title: 'Generate Summary',
                content: 'Click here to process your text and generate an intelligent summary.',
                position: 'top'
            },
            {
                element: '#file-upload',
                title: 'File Upload',
                content: 'You can also upload PDF, Word, and other document formats for processing.',
                position: 'top'
            }
        ];
        
        this.currentStep = 0;
        this.overlay = null;
        this.tooltip = null;
    }
    
    start() {
        // Check if user has seen tour
        if (localStorage.getItem('onboarding_completed')) {
            return;
        }
        
        // Show welcome modal
        this.showWelcomeModal();
    }
    
    showWelcomeModal() {
        const modal = document.createElement('div');
        modal.className = 'onboarding-modal';
        modal.innerHTML = `
            <div class="onboarding-content">
                <h2>Welcome to SUM! 👋</h2>
                <p>SUM helps you distill knowledge from any text into clear, concise summaries.</p>
                <p>Would you like a quick tour of the features?</p>
                <div class="onboarding-actions">
                    <button class="btn btn-primary" onclick="onboardingTour.startTour()">Yes, show me around</button>
                    <button class="btn btn-secondary" onclick="onboardingTour.skipTour()">Skip tour</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    startTour() {
        // Remove welcome modal
        document.querySelector('.onboarding-modal')?.remove();
        
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'onboarding-overlay';
        document.body.appendChild(this.overlay);
        
        // Create tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'onboarding-tooltip';
        document.body.appendChild(this.tooltip);
        
        // Start first step
        this.showStep(0);
    }
    
    showStep(index) {
        if (index >= this.steps.length) {
            this.completeTour();
            return;
        }
        
        this.currentStep = index;
        const step = this.steps[index];
        const element = document.querySelector(step.element);
        
        if (!element) {
            // Skip to next step if element not found
            this.showStep(index + 1);
            return;
        }
        
        // Highlight element
        this.highlightElement(element);
        
        // Position and show tooltip
        this.positionTooltip(element, step);
        
        // Update tooltip content
        this.tooltip.innerHTML = `
            <div class="tooltip-header">
                <h3>${step.title}</h3>
                <button class="close-btn" onclick="onboardingTour.skipTour()">×</button>
            </div>
            <div class="tooltip-content">
                <p>${step.content}</p>
            </div>
            <div class="tooltip-footer">
                <span class="step-indicator">Step ${index + 1} of ${this.steps.length}</span>
                <div class="tooltip-actions">
                    ${index > 0 ? '<button class="btn btn-secondary" onclick="onboardingTour.prevStep()">Previous</button>' : ''}
                    <button class="btn btn-primary" onclick="onboardingTour.nextStep()">
                        ${index < this.steps.length - 1 ? 'Next' : 'Finish'}
                    </button>
                </div>
            </div>
        `;
        
        // Scroll element into view
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    highlightElement(element) {
        // Remove previous highlights
        document.querySelectorAll('.onboarding-highlight').forEach(el => {
            el.classList.remove('onboarding-highlight');
        });
        
        // Add highlight to current element
        element.classList.add('onboarding-highlight');
        
        // Update overlay to create spotlight effect
        const rect = element.getBoundingClientRect();
        this.overlay.style.clipPath = `polygon(
            0 0,
            0 100%,
            ${rect.left - 10}px 100%,
            ${rect.left - 10}px ${rect.top - 10}px,
            ${rect.right + 10}px ${rect.top - 10}px,
            ${rect.right + 10}px ${rect.bottom + 10}px,
            ${rect.left - 10}px ${rect.bottom + 10}px,
            ${rect.left - 10}px 100%,
            100% 100%,
            100% 0
        )`;
    }
    
    positionTooltip(element, step) {
        const rect = element.getBoundingClientRect();
        const tooltipRect = this.tooltip.getBoundingClientRect();
        
        let top, left;
        
        switch (step.position) {
            case 'top':
                top = rect.top - tooltipRect.height - 20;
                left = rect.left + (rect.width - tooltipRect.width) / 2;
                break;
            case 'bottom':
                top = rect.bottom + 20;
                left = rect.left + (rect.width - tooltipRect.width) / 2;
                break;
            case 'left':
                top = rect.top + (rect.height - tooltipRect.height) / 2;
                left = rect.left - tooltipRect.width - 20;
                break;
            case 'right':
                top = rect.top + (rect.height - tooltipRect.height) / 2;
                left = rect.right + 20;
                break;
        }
        
        // Ensure tooltip stays within viewport
        top = Math.max(10, Math.min(top, window.innerHeight - tooltipRect.height - 10));
        left = Math.max(10, Math.min(left, window.innerWidth - tooltipRect.width - 10));
        
        this.tooltip.style.top = `${top}px`;
        this.tooltip.style.left = `${left}px`;
    }
    
    nextStep() {
        this.showStep(this.currentStep + 1);
    }
    
    prevStep() {
        this.showStep(this.currentStep - 1);
    }
    
    skipTour() {
        this.completeTour();
    }
    
    completeTour() {
        // Mark as completed
        localStorage.setItem('onboarding_completed', 'true');
        
        // Clean up
        this.overlay?.remove();
        this.tooltip?.remove();
        document.querySelector('.onboarding-modal')?.remove();
        document.querySelectorAll('.onboarding-highlight').forEach(el => {
            el.classList.remove('onboarding-highlight');
        });
        
        // Show completion message
        this.showCompletionMessage();
    }
    
    showCompletionMessage() {
        const message = document.createElement('div');
        message.className = 'onboarding-complete';
        message.innerHTML = `
            <div class="complete-content">
                <h3>🎉 Tour Complete!</h3>
                <p>You're ready to start using SUM. Try pasting some text to see it in action!</p>
                <button class="btn btn-primary" onclick="this.parentElement.parentElement.remove()">Got it!</button>
            </div>
        `;
        
        document.body.appendChild(message);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            message.remove();
        }, 5000);
    }
    
    reset() {
        // Reset onboarding status
        localStorage.removeItem('onboarding_completed');
    }
}

// Initialize onboarding
const onboardingTour = new OnboardingTour();

// Start onboarding when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for page to settle
    setTimeout(() => {
        onboardingTour.start();
    }, 1000);
});

// Add help button to trigger tour manually
document.addEventListener('DOMContentLoaded', () => {
    const helpBtn = document.createElement('button');
    helpBtn.className = 'help-button';
    helpBtn.innerHTML = '?';
    helpBtn.title = 'Start tour';
    helpBtn.onclick = () => {
        onboardingTour.reset();
        onboardingTour.start();
    };
    
    document.body.appendChild(helpBtn);
});