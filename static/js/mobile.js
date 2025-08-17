// Mobile functionality for SUM

document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const mainNav = document.querySelector('.main-nav');
    
    if (mobileMenuToggle && mainNav) {
        mobileMenuToggle.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';
            
            // Toggle menu
            this.setAttribute('aria-expanded', !isExpanded);
            mainNav.classList.toggle('is-open');
            this.classList.toggle('is-active');
            
            // Prevent body scroll when menu is open
            document.body.classList.toggle('menu-open');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!mobileMenuToggle.contains(event.target) && 
                !mainNav.contains(event.target) && 
                mainNav.classList.contains('is-open')) {
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                mainNav.classList.remove('is-open');
                mobileMenuToggle.classList.remove('is-active');
                document.body.classList.remove('menu-open');
            }
        });
        
        // Close menu when clicking on a nav link
        const navLinks = mainNav.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                mainNav.classList.remove('is-open');
                mobileMenuToggle.classList.remove('is-active');
                document.body.classList.remove('menu-open');
            });
        });
    }
    
    // Improve touch handling for range sliders
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        // Add touch event listeners for better mobile experience
        let isDragging = false;
        
        input.addEventListener('touchstart', function(e) {
            isDragging = true;
        });
        
        input.addEventListener('touchend', function(e) {
            isDragging = false;
        });
        
        input.addEventListener('touchmove', function(e) {
            if (isDragging) {
                e.preventDefault(); // Prevent page scroll while adjusting slider
            }
        });
    });
    
    // Improve file input on mobile
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                // Update file hint with selected file name
                const fileHint = this.parentElement.querySelector('.file-hint');
                if (fileHint) {
                    fileHint.textContent = `Selected: ${fileName}`;
                }
            }
        });
    });
    
    // Handle viewport height changes (mobile keyboard)
    let vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    
    window.addEventListener('resize', () => {
        // Update viewport height variable
        let vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Detect if user is on mobile
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (isMobile) {
        document.body.classList.add('is-mobile');
    }
    
    // Handle orientation changes
    window.addEventListener('orientationchange', function() {
        // Force a re-render of certain elements
        setTimeout(function() {
            window.scrollTo(0, window.scrollY + 1);
            window.scrollTo(0, window.scrollY - 1);
        }, 500);
    });
    
    // Improve textarea behavior on mobile
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        // Auto-resize based on content
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
        
        // Prevent zoom on focus (iOS)
        textarea.addEventListener('focus', function() {
            if (isMobile) {
                document.querySelector('meta[name="viewport"]').setAttribute(
                    'content', 
                    'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0'
                );
            }
        });
        
        textarea.addEventListener('blur', function() {
            if (isMobile) {
                document.querySelector('meta[name="viewport"]').setAttribute(
                    'content', 
                    'width=device-width, initial-scale=1.0'
                );
            }
        });
    });
});