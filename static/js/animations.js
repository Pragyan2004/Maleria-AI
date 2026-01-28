// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-fadeInUp');
        }
    });
}, observerOptions);

document.addEventListener('DOMContentLoaded', function () {
    const cards = document.querySelectorAll('.card, .feature-card, .stats-card');
    cards.forEach(card => {
        observer.observe(card);
    });
});

function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.innerHTML = '';

    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

// Initial typewriter call removed to prevent interference with page-specific scripts

function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => {
            bar.style.width = width;
        }, 500);
    });
}