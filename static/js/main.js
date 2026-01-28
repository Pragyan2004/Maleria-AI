document.getElementById('mobile-menu-button')?.addEventListener('click', function () {
    const menu = document.getElementById('mobile-menu');
    menu.classList.toggle('hidden');
});
document.addEventListener('DOMContentLoaded', function () {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

document.getElementById('file-upload')?.addEventListener('change', function (e) {
    const fileName = e.target.files[0]?.name;
    if (fileName) {
        document.getElementById('file-name').textContent = fileName;
    }
});

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Code copied to clipboard!', 'success');
    });
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');

    // Define styles based on type
    let bgClass, icon;
    if (type === 'success') {
        bgClass = 'bg-green-500';
        icon = '<i class="fas fa-check-circle mr-2"></i>';
    } else if (type === 'error') {
        bgClass = 'bg-red-500';
        icon = '<i class="fas fa-exclamation-circle mr-2"></i>';
    } else {
        bgClass = 'bg-blue-500';
        icon = '<i class="fas fa-info-circle mr-2"></i>';
    }

    // Set class (Added z-50 to ensure visibility over navbar)
    notification.className = `fixed top-20 right-4 p-4 rounded-lg shadow-xl text-white ${bgClass} transform translate-x-full transition-transform duration-300 z-50 flex items-center notification-toast`;

    // Use innerHTML to include icon
    notification.innerHTML = `${icon} <span>${message}</span>`;

    // Add custom style for standard CSS fallback if needed
    if (type === 'error') notification.style.backgroundColor = '#ef4444';
    if (type === 'success') notification.style.backgroundColor = '#22c55e';
    if (type === 'info') notification.style.backgroundColor = '#3b82f6';
    notification.style.zIndex = '9999'; // Force top

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);

    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000); // Increased duration slightly
}

function submitAnalysis(form) {
    const formData = new FormData(form);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showNotification(data.error, 'error');
            } else {
                document.getElementById('result-container').innerHTML = data.result;
                document.getElementById('code-container').textContent = data.code;
                document.getElementById('data-preview').innerHTML = data.preview;
                showNotification('Analysis completed successfully!', 'success');
            }
        })
        .catch(error => {
            showNotification('An error occurred: ' + error, 'error');
        });

    return false;
}