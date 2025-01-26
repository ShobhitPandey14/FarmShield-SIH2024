// script.js

// Add a fade-in animation for the header content
document.addEventListener('DOMContentLoaded', function () {
    const headerContent = document.querySelector('.header-content');
    headerContent.style.opacity = 0;
    headerContent.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        headerContent.style.transition = 'opacity 1s, transform 1s';
        headerContent.style.opacity = 1;
        headerContent.style.transform = 'translateY(0)';
    }, 500);
});

// Button animation when clicked
const learnMoreBtn = document.getElementById('learn-more-btn');
learnMoreBtn.addEventListener('click', function () {
    this.style.transform = 'scale(1.3)';
    this.style.transition = 'transform 0.2s';
    setTimeout(() => {
        this.style.transform = 'scale(1)';
    }, 200);
});
