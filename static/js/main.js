// Main JavaScript for Malaria Detection App

// Global variables
let currentImage = null;
let cameraStream = null;

// Initialize detection page
function initDetectionPage() {
    setupFileUpload();
    setupCamera();
    setupEventListeners();
}

// Setup file upload functionality
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');

    // Click to upload
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        } else {
            showAlert('Please drop an image file', 'danger');
        }
    });
}

// Handle file selection
function handleFileSelect(file) {
    // Check file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'danger');
        return;
    }

    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showAlert('Please upload a valid image file (JPEG, PNG, GIF)', 'danger');
        return;
    }

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = {
            file: file,
            dataUrl: e.target.result
        };

        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('previewSection').classList.remove('d-none');
        document.getElementById('uploadArea').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Setup camera functionality
function setupCamera() {
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraFeed = document.getElementById('cameraFeed');

    startCameraBtn.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
            });

            cameraVideo.srcObject = cameraStream;
            cameraFeed.classList.remove('d-none');
            startCameraBtn.classList.add('d-none');

        } catch (error) {
            console.error('Error accessing camera:', error);
            showAlert('Could not access camera. Please check permissions.', 'danger');
        }
    });

    stopCameraBtn.addEventListener('click', () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        cameraFeed.classList.add('d-none');
        startCameraBtn.classList.remove('d-none');
    });

    captureBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = cameraVideo.videoWidth;
        canvas.height = cameraVideo.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0);

        canvas.toBlob((blob) => {
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            handleFileSelect(file);

            // Stop camera after capture
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }

            cameraFeed.classList.add('d-none');
            startCameraBtn.classList.remove('d-none');
        }, 'image/jpeg', 0.9);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', analyzeImage);

    // Change image button
    document.getElementById('changeImageBtn').addEventListener('click', () => {
        document.getElementById('previewSection').classList.add('d-none');
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('resultsSection').classList.add('d-none');
        currentImage = null;
    });
}

// Analyze image using AJAX
async function analyzeImage() {
    if (!currentImage) {
        showAlert('Please select an image first', 'warning');
        return;
    }

    const loadingIndicator = document.getElementById('loadingIndicator');
    const previewSection = document.getElementById('previewSection');

    // Show loading
    loadingIndicator.classList.remove('d-none');
    previewSection.classList.add('d-none');

    try {
        // Create FormData
        const formData = new FormData();
        const modelSelect = document.getElementById('modelSelect');
        const selectedModel = modelSelect ? modelSelect.value : 'CNN';

        formData.append('file', currentImage.file);
        formData.append('model', selectedModel);

        // Send to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // Redirect to results page
        window.location.href = '/results';

    } catch (error) {
        console.error('Analysis error:', error);
        showAlert(`Analysis failed: ${error.message}`, 'danger');

        // Reset UI
        loadingIndicator.classList.add('d-none');
        previewSection.classList.remove('d-none');
    }
}

// Display analysis results
function displayResults(result) {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');

    // Hide loading, show results
    loadingIndicator.classList.add('d-none');
    resultsSection.classList.remove('d-none');

    // Update result text
    document.getElementById('resultText').textContent = result.result;
    document.getElementById('resultText').className = 'display-4 mb-3 ' +
        (result.result.includes('Parasitized') ? 'text-danger' : 'text-success');

    // Update confidence
    document.getElementById('confidenceValue').textContent = result.confidence;
    document.getElementById('confidenceBar').style.width = `${result.confidence}%`;

    // Update probabilities
    document.getElementById('parasitizedProb').textContent = `${result.parasitized_prob}%`;
    document.getElementById('uninfectedProb').textContent = `${result.uninfected_prob}%`;

    document.getElementById('parasitizedBar').style.width = `${result.parasitized_prob}%`;
    document.getElementById('uninfectedBar').style.width = `${result.uninfected_prob}%`;

    // Show appropriate info
    if (result.result.includes('Parasitized')) {
        document.getElementById('parasitizedInfo').classList.remove('d-none');
        document.getElementById('uninfectedInfo').classList.add('d-none');
    } else {
        document.getElementById('uninfectedInfo').classList.remove('d-none');
        document.getElementById('parasitizedInfo').classList.add('d-none');
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show alert message
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-dismissible');
    existingAlerts.forEach(alert => alert.remove());

    // Create new alert
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    // Insert at top of main content
    const mainContent = document.querySelector('main .container');
    if (mainContent) {
        mainContent.insertAdjacentHTML('afterbegin', alertHtml);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = document.querySelector('.alert-dismissible');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    // Check if we're on detection page
    if (document.getElementById('uploadArea')) {
        initDetectionPage();
    }

    // Add fade-in animation to all cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});