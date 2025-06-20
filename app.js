// Dynamic Pricing Presentation JavaScript
class PresentationApp {
    constructor() {
        this.currentSlide = 0;
        this.slides = document.querySelectorAll('.slide');
        this.totalSlides = this.slides.length;
        this.navLinks = document.querySelectorAll('.nav-link');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.slideCounter = document.getElementById('slideCounter');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateSlideCounter();
        this.updateNavigationState();
        this.setupCodeToggle();
    }
    
    setupEventListeners() {
        // Navigation buttons
        this.prevBtn.addEventListener('click', () => this.previousSlide());
        this.nextBtn.addEventListener('click', () => this.nextSlide());
        
        // Sidebar navigation
        this.navLinks.forEach((link, index) => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.goToSlide(index);
            });
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') {
                e.preventDefault();
                this.nextSlide();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                this.previousSlide();
            }
        });
        
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            });
        });
    }
    
    setupCodeToggle() {
        const codeToggleButtons = document.querySelectorAll('.code-toggle');
        codeToggleButtons.forEach(button => {
            button.addEventListener('click', async function () {
                const target = button.getAttribute('data-target');
                const file = button.getAttribute('data-file');
                const codeBlock = document.getElementById(target);
                const pre = codeBlock.querySelector('pre code');
                if (codeBlock.classList.contains('hidden')) {
                    if (!pre.textContent.trim() && file) {
                        try {
                            const response = await fetch(file);
                            if (response.ok) {
                                const code = await response.text();
                                pre.textContent = code;
                                if (window.Prism) {
                                    Prism.highlightElement(pre);
                                }
                            } else {
                                pre.textContent = 'Could not load file: ' + file;
                            }
                        } catch (e) {
                            pre.textContent = 'Error loading file: ' + file;
                        }
                    }
                    codeBlock.classList.remove('hidden');
                } else {
                    codeBlock.classList.add('hidden');
                }
            });
        });
    }
    
    goToSlide(slideIndex) {
        if (slideIndex >= 0 && slideIndex < this.totalSlides) {
            // Hide current slide
            this.slides[this.currentSlide].classList.remove('active');
            this.navLinks[this.currentSlide].classList.remove('active');
            
            // Show new slide
            this.currentSlide = slideIndex;
            this.slides[this.currentSlide].classList.add('active');
            this.navLinks[this.currentSlide].classList.add('active');
            
            this.updateSlideCounter();
            this.updateNavigationState();
            
            // Scroll to top of new slide
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }
    
    nextSlide() {
        if (this.currentSlide < this.totalSlides - 1) {
            this.goToSlide(this.currentSlide + 1);
        }
    }
    
    previousSlide() {
        if (this.currentSlide > 0) {
            this.goToSlide(this.currentSlide - 1);
        }
    }
    
    updateSlideCounter() {
        this.slideCounter.textContent = `${this.currentSlide + 1} / ${this.totalSlides}`;
    }
    
    updateNavigationState() {
        // Update previous button
        this.prevBtn.disabled = this.currentSlide === 0;
        
        // Update next button
        this.nextBtn.disabled = this.currentSlide === this.totalSlides - 1;
    }
}

// Additional utility functions
function toggleCode(targetId) {
    const codeBlock = document.getElementById(targetId);
    if (codeBlock) {
        if (codeBlock.classList.contains('hidden')) {
            codeBlock.classList.remove('hidden');
            // Trigger syntax highlighting if Prism is available
            if (window.Prism) {
                window.Prism.highlightAllUnder(codeBlock);
            }
        } else {
            codeBlock.classList.add('hidden');
        }
    }
}

// Smooth scrolling utility
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Animation utility for slide transitions
function animateSlideTransition(fromSlide, toSlide, direction = 'forward') {
    const translateValue = direction === 'forward' ? '-100%' : '100%';
    
    // Animate out current slide
    fromSlide.style.transform = `translateX(${translateValue})`;
    fromSlide.style.opacity = '0';
    
    // After animation, hide the slide
    setTimeout(() => {
        fromSlide.classList.remove('active');
        fromSlide.style.transform = '';
        fromSlide.style.opacity = '';
    }, 500);
    
    // Show and animate in new slide
    setTimeout(() => {
        toSlide.classList.add('active');
        toSlide.style.transform = direction === 'forward' ? 'translateX(100%)' : 'translateX(-100%)';
        toSlide.style.opacity = '0';
        
        // Trigger reflow
        toSlide.offsetHeight;
        
        // Animate to final position
        toSlide.style.transform = 'translateX(0)';
        toSlide.style.opacity = '1';
        
        // Clean up styles after animation
        setTimeout(() => {
            toSlide.style.transform = '';
            toSlide.style.opacity = '';
        }, 500);
    }, 100);
}

// Code syntax highlighting enhancement
function enhanceCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // Add line numbers if not already present
        if (!block.querySelector('.line-numbers')) {
            const lines = block.textContent.split('\n');
            const lineNumbers = lines.map((_, index) => index + 1).join('\n');
            
            const lineNumbersElement = document.createElement('span');
            lineNumbersElement.className = 'line-numbers';
            lineNumbersElement.textContent = lineNumbers;
            
            const container = block.parentElement;
            container.style.position = 'relative';
            container.style.paddingLeft = '3em';
            
            lineNumbersElement.style.position = 'absolute';
            lineNumbersElement.style.left = '0';
            lineNumbersElement.style.top = '20px';
            lineNumbersElement.style.color = '#6a737d';
            lineNumbersElement.style.fontSize = '12px';
            lineNumbersElement.style.lineHeight = '1.6';
            lineNumbersElement.style.paddingRight = '1em';
            lineNumbersElement.style.textAlign = 'right';
            lineNumbersElement.style.userSelect = 'none';
            lineNumbersElement.style.width = '2.5em';
            
            container.appendChild(lineNumbersElement);
        }
    });
}

// Mobile navigation toggle
function setupMobileNavigation() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    // Create mobile menu button
    const mobileMenuBtn = document.createElement('button');
    mobileMenuBtn.innerHTML = '☰';
    mobileMenuBtn.className = 'mobile-menu-btn';
    mobileMenuBtn.style.cssText = `
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1001;
        background: var(--color-primary);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        display: none;
    `;
    
    document.body.appendChild(mobileMenuBtn);
    
    // Show mobile menu button on small screens
    function checkScreenSize() {
        if (window.innerWidth <= 768) {
            mobileMenuBtn.style.display = 'block';
        } else {
            mobileMenuBtn.style.display = 'none';
            sidebar.classList.remove('open');
        }
    }
    
    // Toggle sidebar on mobile
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });
    
    // Close sidebar when clicking on main content on mobile
    mainContent.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    });
    
    // Check screen size on load and resize
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
}

// Performance optimization: Lazy load images
function setupLazyLoading() {
    const images = document.querySelectorAll('img[src]');
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.style.transition = 'opacity 0.3s';
                    if (img.complete) {
                        img.style.opacity = '1';
                    } else {
                        img.style.opacity = '0';
                        img.onload = () => {
                            img.style.opacity = '1';
                        };
                    }
                    imageObserver.unobserve(img);
                }
            });
        });
        images.forEach(img => imageObserver.observe(img));
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize main presentation app
    const app = new PresentationApp();
    
    // Setup additional features
    setupMobileNavigation();
    setupLazyLoading();
    
    // Enhance code blocks after a short delay to ensure Prism is loaded
    setTimeout(() => {
        enhanceCodeBlocks();
        
        // Trigger initial syntax highlighting
        if (window.Prism) {
            window.Prism.highlightAll();
        }
    }, 100);
    
    // Add smooth scrolling to all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                smoothScrollTo(target);
            }
        });
    });
    
    // Add keyboard shortcuts info
    console.log('Keyboard shortcuts:');
    console.log('→ or Space: Next slide');
    console.log('←: Previous slide');
    console.log('Number keys 0-9: Jump to slide');
    
    // Add number key navigation
    document.addEventListener('keydown', (e) => {
        const key = parseInt(e.key);
        if (!isNaN(key) && key >= 0 && key <= 9) {
            app.goToSlide(key);
        }
    });
});

// Export for global use
window.PresentationApp = PresentationApp;
window.toggleCode = toggleCode;
window.toggleCode = toggleCode;