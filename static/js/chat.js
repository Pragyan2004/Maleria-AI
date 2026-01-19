// Chatbot JavaScript

let chatHistory = [];

// Initialize chatbot
function initChatbot() {
    loadChatHistory();
    setupEventListeners();
}

// Setup event listeners
function setupEventListeners() {
    // Send button
    document.getElementById('messageInput').addEventListener('keypress', handleKeyPress);
    
    // Clear chat button
    document.getElementById('clearChatBtn').addEventListener('click', clearChat);
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send message to chatbot
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    messageInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send to server
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Add bot response to chat
        addMessageToChat(data.response, 'bot', data.timestamp);
        
        // Save to history
        saveChatHistory();
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

// Add message to chat UI
function addMessageToChat(message, sender, timestamp = null) {
    const chatContainer = document.getElementById('chatContainer');
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // Format message content
    let content = message;
    if (sender === 'bot') {
        content = `<strong>Malaria AI Assistant:</strong><br>${formatMessage(message)}`;
    }
    
    // Add timestamp
    const time = timestamp || getCurrentTime();
    
    messageDiv.innerHTML = `
        <div>${content}</div>
        <div class="message-time">${time}</div>
    `;
    
    // Add to chat
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Add to history
    if (sender === 'user') {
        chatHistory.push({ role: 'user', content: message, time: time });
    } else {
        chatHistory.push({ role: 'assistant', content: message, time: time });
    }
}

// Format message with line breaks and lists
function formatMessage(text) {
    // Convert markdown-style lists to HTML
    let formatted = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return formatted;
}

// Show typing indicator
function showTypingIndicator() {
    const chatContainer = document.getElementById('chatContainer');
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <strong>Malaria AI Assistant:</strong>
        <div class="typing mt-2">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <div class="message-time">typing...</div>
    `;
    
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Clear chat
async function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        try {
            await fetch('/clear_chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            // Clear local chat
            chatHistory = [];
            const chatContainer = document.getElementById('chatContainer');
            
            // Keep only the initial message
            const initialMessage = chatContainer.querySelector('.bot-message');
            chatContainer.innerHTML = '';
            chatContainer.appendChild(initialMessage);
            
            // Show success message
            showChatAlert('Chat cleared successfully!', 'success');
            
        } catch (error) {
            console.error('Error clearing chat:', error);
            showChatAlert('Error clearing chat', 'danger');
        }
    }
}

// Quick question handler
function askQuestion(element) {
    const question = element.textContent;
    document.getElementById('messageInput').value = question;
    sendMessage();
}

// Load chat history from session
function loadChatHistory() {
    // This would typically load from server/session
    // For now, we'll start fresh each time
    chatHistory = [];
}

// Save chat history
function saveChatHistory() {
    // In a real app, this would save to server/database
    // For now, we just keep it in memory
    console.log('Chat history saved locally:', chatHistory.length, 'messages');
}

// Get current time in HH:MM format
function getCurrentTime() {
    const now = new Date();
    return now.getHours().toString().padStart(2, '0') + ':' + 
           now.getMinutes().toString().padStart(2, '0');
}

// Show chat alert
function showChatAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.parentNode.insertBefore(alertDiv, chatContainer.nextSibling);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 3000);
}

// Load prevention tips
async function loadPreventionTips() {
    try {
        const response = await fetch('/get_prevention_tips');
        const data = await response.json();
        
        if (data.tips) {
            // Could display tips in a modal or sidebar
            console.log('Prevention tips loaded:', data.tips);
        }
    } catch (error) {
        console.error('Error loading tips:', error);
    }
}

// Load symptoms info
async function loadSymptomsInfo() {
    try {
        const response = await fetch('/get_symptoms');
        const data = await response.json();
        
        if (data.common) {
            console.log('Symptoms loaded:', data);
        }
    } catch (error) {
        console.error('Error loading symptoms:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('chatContainer')) {
        initChatbot();
        
        // Load additional data
        loadPreventionTips();
        loadSymptomsInfo();
    }
});