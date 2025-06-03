// Global application state and utilities
class LLMCockpit {
    constructor() {
        this.socket = null;
        this.gpuGauge = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.settings = this.loadSettings();
        
        this.init();
    }
    
    init() {
        this.setupSocketIO();
        this.setupGPUMonitoring();
        this.setupEventListeners();
        this.loadThemeSettings();
        this.setupServiceWorker();
    }
    
    // Socket.IO Setup
    setupSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.showToast('Connected to server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showToast('Disconnected from server', 'warning');
        });
        
        // GPU monitoring
        this.socket.on('gpu_stats', (data) => {
            this.updateGPUStats(data);
        });
        
        // Chat updates
        this.socket.on('chat_update', (data) => {
            this.handleChatUpdate(data);
        });
        
        // Model loading progress
        this.socket.on('model_loading', (data) => {
            this.updateModelLoadingProgress(data);
        });
        
        // Model events
        this.socket.on('model_loaded', (data) => {
            this.handleModelLoaded(data);
        });
        
        this.socket.on('model_unloaded', (data) => {
            this.handleModelUnloaded(data);
        });
        
        this.socket.on('model_switched', (data) => {
            this.handleModelSwitched(data);
        });
        
        // RAG indexing progress
        this.socket.on('rag_progress', (data) => {
            this.updateRAGProgress(data);
        });
    }
    
    // GPU Monitoring
    setupGPUMonitoring() {
        // Initialize GPU gauge if element exists
        const gaugeElement = document.getElementById('gpu-gauge');
        if (gaugeElement && typeof GaugeChart !== 'undefined') {
            this.gpuGauge = GaugeChart.gaugeChart(gaugeElement, 300, {
                arcColorsDomain: [0, 33, 66, 100],
                arcColors: ['#5BE12C', '#F5CD19', '#F57C00', '#D32F2F'],
                arcDelimiters: [33, 66, 90],
                arcPadding: 2,
                arcWidth: 0.3,
                gaugeType: 0,
                centralLabel: 'GPU',
                labelsFont: 'Consolas',
                hasNeedle: true,
                needleColor: '#464A4F',
                needleBaseColor: '#464A4F',
                needleSharpCircleSize: 2,
                animationDuration: 1000
            });
        }
        
        // Request initial GPU stats
        if (this.socket) {
            this.socket.emit('request_gpu_stats');
        }
    }
    
    updateGPUStats(data) {
        // Update gauge
        if (this.gpuGauge && data.usage !== undefined) {
            this.gpuGauge.updateNeedle(data.usage);
        }
        
        // Update text displays
        const usageElement = document.getElementById('gpu-usage');
        const vramElement = document.getElementById('vram-usage');
        const tempElement = document.getElementById('gpu-temp');
        
        if (usageElement) {
            usageElement.textContent = `GPU: ${data.usage || 0}%`;
        }
        
        if (vramElement) {
            vramElement.textContent = `${data.vram_used || 0} / ${data.vram_total || 0} MB`;
        }
        
        if (tempElement) {
            tempElement.textContent = `${data.temperature || 0}°C`;
        }
    }
    
    // Event Listeners
    setupEventListeners() {
        // Temperature range slider
        const tempRange = document.getElementById('temperature-range');
        const tempValue = document.getElementById('temp-value');

        if (tempRange && tempValue) {
            tempRange.value = this.settings.temperature;
            tempValue.textContent = this.settings.temperature;
            tempRange.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
                this.settings.temperature = parseFloat(e.target.value);
                this.saveSettings();
            });
        }

        // Top P slider
        const topPRange = document.getElementById('top-p-range');
        const topPValue = document.getElementById('top-p-value');
        if (topPRange && topPValue) {
            topPRange.value = this.settings.topP;
            topPValue.textContent = this.settings.topP;
            topPRange.addEventListener('input', (e) => {
                topPValue.textContent = e.target.value;
                this.settings.topP = parseFloat(e.target.value);
                this.saveSettings();
            });
        }

        // Top K input
        const topKInput = document.getElementById('top-k');
        if (topKInput) {
            topKInput.value = this.settings.topK;
            topKInput.addEventListener('change', (e) => {
                this.settings.topK = parseInt(e.target.value, 10);
                this.saveSettings();
            });
        }

        // Max tokens input
        const maxTokensInput = document.getElementById('max-tokens');
        if (maxTokensInput) {
            maxTokensInput.value = this.settings.maxTokens;
            maxTokensInput.addEventListener('change', (e) => {
                this.settings.maxTokens = parseInt(e.target.value, 10);
                this.saveSettings();
            });
        }

        // Repeat penalty slider
        const repeatPenaltyRange = document.getElementById('repeat-penalty');
        const repeatPenaltyValue = document.getElementById('repeat-penalty-value');
        if (repeatPenaltyRange && repeatPenaltyValue) {
            repeatPenaltyRange.value = this.settings.repeatPenalty;
            repeatPenaltyValue.textContent = this.settings.repeatPenalty;
            repeatPenaltyRange.addEventListener('input', (e) => {
                repeatPenaltyValue.textContent = e.target.value;
                this.settings.repeatPenalty = parseFloat(e.target.value);
                this.saveSettings();
            });
        }

        // Presence penalty slider
        const presencePenaltyRange = document.getElementById('presence-penalty');
        const presencePenaltyValue = document.getElementById('presence-penalty-value');
        if (presencePenaltyRange && presencePenaltyValue) {
            presencePenaltyRange.value = this.settings.presencePenalty;
            presencePenaltyValue.textContent = this.settings.presencePenalty;
            presencePenaltyRange.addEventListener('input', (e) => {
                presencePenaltyValue.textContent = e.target.value;
                this.settings.presencePenalty = parseFloat(e.target.value);
                this.saveSettings();
            });
        }

        // Frequency penalty slider
        const frequencyPenaltyRange = document.getElementById('frequency-penalty');
        const frequencyPenaltyValue = document.getElementById('frequency-penalty-value');
        if (frequencyPenaltyRange && frequencyPenaltyValue) {
            frequencyPenaltyRange.value = this.settings.frequencyPenalty;
            frequencyPenaltyValue.textContent = this.settings.frequencyPenalty;
            frequencyPenaltyRange.addEventListener('input', (e) => {
                frequencyPenaltyValue.textContent = e.target.value;
                this.settings.frequencyPenalty = parseFloat(e.target.value);
                this.saveSettings();
            });
        }
        
        // Settings toggles
        const ragToggle = document.getElementById('rag-enabled');
        const voiceToggle = document.getElementById('voice-enabled');
        
        if (ragToggle) {
            ragToggle.addEventListener('change', (e) => {
                this.settings.ragEnabled = e.target.checked;
                this.saveSettings();
                this.toggleRAGFeatures(e.target.checked);
            });
        }
        
        if (voiceToggle) {
            voiceToggle.addEventListener('change', (e) => {
                this.settings.voiceEnabled = e.target.checked;
                this.saveSettings();
                this.toggleVoiceFeatures(e.target.checked);
            });
        }
        
        // File drag and drop
        this.setupFileDropZone();
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Page visibility
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
    }
    
    // File Drop Zone
    setupFileDropZone() {
        const dropZone = document.body;
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('file-drop-zone', 'drag-over');
        });
        
        dropZone.addEventListener('dragleave', (e) => {
            if (!dropZone.contains(e.relatedTarget)) {
                dropZone.classList.remove('drag-over');
            }
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0 && this.settings.ragEnabled) {
                this.handleFileUpload(files);
            }
        });
    }
    
    // Voice Features
    async setupVoiceRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudioRecording();
            };
            
            return true;
        } catch (error) {
            console.error('Voice recording setup failed:', error);
            this.showToast('Microphone access denied', 'error');
            return false;
        }
    }
    
    async startVoiceRecording() {
        if (!this.mediaRecorder) {
            const success = await this.setupVoiceRecording();
            if (!success) return;
        }
        
        this.audioChunks = [];
        this.mediaRecorder.start();
        this.showToast('Recording started...', 'info');
    }
    
    stopVoiceRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.showToast('Processing audio...', 'info');
        }
    }
    
    async processAudioRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        
        try {
            const response = await fetch('/api/voice/transcribe', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.insertTranscribedText(result.text);
            } else {
                throw new Error('Transcription failed');
            }
        } catch (error) {
            console.error('Voice processing error:', error);
            this.showToast('Voice processing failed', 'error');
        }
    }
    
    insertTranscribedText(text) {
        // Find the message input and insert transcribed text
        const messageInput = document.querySelector('textarea[placeholder*="Type your message"]');
        if (messageInput) {
            messageInput.value += (messageInput.value ? ' ' : '') + text;
            messageInput.dispatchEvent(new Event('input'));
            messageInput.focus();
        }
    }
    
    // Keyboard Shortcuts
    handleKeyboardShortcuts(e) {
        // Ctrl + K: Focus search
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[placeholder*="Search"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Ctrl + N: New chat
        if (e.ctrlKey && e.key === 'n') {
            e.preventDefault();
            const newChatBtn = document.querySelector('button[onclick="newChat()"]');
            if (newChatBtn) {
                newChatBtn.click();
            }
        }
        
        // Ctrl + /: Toggle sidebar
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            const drawerToggle = document.getElementById('drawer-toggle');
            if (drawerToggle) {
                drawerToggle.checked = !drawerToggle.checked;
            }
        }
        
        // Escape: Close modals/dropdowns
        if (e.key === 'Escape') {
            document.querySelectorAll('.dropdown').forEach(dropdown => {
                dropdown.removeAttribute('open');
            });
        }
    }
    
    // Settings Management
    loadSettings() {
        const cfg = window.defaultConfig || {};
        const defaultSettings = {
            theme: 'dark',
            temperature: cfg.temperature || 0.7,
            topP: cfg.top_p || 0.95,
            topK: cfg.top_k || 40,
            maxTokens: cfg.max_tokens || 2048,
            repeatPenalty: cfg.repeat_penalty || 1.1,
            presencePenalty: cfg.presence_penalty || 0.0,
            frequencyPenalty: cfg.frequency_penalty || 0.0,
            ragEnabled: false,
            voiceEnabled: false,
            autoSave: true,
            notifications: true
        };
        
        try {
            const saved = localStorage.getItem('llm_cockpit_settings');
            return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
        } catch (error) {
            console.error('Failed to load settings:', error);
            return defaultSettings;
        }
    }
    
    saveSettings() {
        try {
            localStorage.setItem('llm_cockpit_settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }
    
    // Theme Management
    loadThemeSettings() {
        const themeSelector = document.querySelector('[data-theme]');
        if (themeSelector && this.settings.theme) {
            themeSelector.setAttribute('data-theme', this.settings.theme);
        }
    }
    
    // File Upload Handler
    async handleFileUpload(files) {
        for (const file of files) {
            if (file.size > 20 * 1024 * 1024) { // 20MB limit
                this.showToast(`File "${file.name}" is too large (max 20MB)`, 'error');
                continue;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/rag/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    this.showToast(`File "${file.name}" uploaded successfully`, 'success');
                    
                    // Trigger UI update if chat interface exists
                    if (window.Alpine && window.Alpine.store) {
                        const store = window.Alpine.store('chat');
                        if (store) {
                            store.ragFiles.push({
                                id: result.file_id,
                                name: file.name
                            });
                        }
                    }
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (error) {
                console.error('File upload error:', error);
                this.showToast(`Failed to upload "${file.name}"`, 'error');
            }
        }
    }
    
    // Utility Functions
    showToast(message, type = 'info', duration = 3000) {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} fixed top-4 right-4 w-auto max-w-sm z-50 shadow-lg`;
        toast.innerHTML = `
            <div class="flex items-center">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="btn btn-ghost btn-xs ml-2">✕</button>
            </div>
        `;
        
        // Add to DOM
        document.body.appendChild(toast);
        
        // Auto-remove
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, duration);
    }
    
    toggleRAGFeatures(enabled) {
        // Show/hide RAG-related UI elements
        document.querySelectorAll('[x-show="ragEnabled"]').forEach(el => {
            el.style.display = enabled ? '' : 'none';
        });
    }
    
    toggleVoiceFeatures(enabled) {
        // Show/hide voice-related UI elements
        document.querySelectorAll('[x-show="voiceEnabled"]').forEach(el => {
            el.style.display = enabled ? '' : 'none';
        });
    }
    
    handleResize() {
        // Adjust layout for mobile
        if (window.innerWidth < 768) {
            document.body.classList.add('mobile');
        } else {
            document.body.classList.remove('mobile');
        }
    }
    
    pauseUpdates() {
        // Pause real-time updates when tab is hidden
        if (this.socket) {
            this.socket.emit('pause_updates');
        }
    }
    
    resumeUpdates() {
        // Resume real-time updates when tab is visible
        if (this.socket) {
            this.socket.emit('resume_updates');
        }
    }
    
    // Chat Update Handlers
    handleChatUpdate(data) {
        // Handle real-time chat updates from other clients
        if (data.type === 'new_message') {
            this.showToast('New message received', 'info');
        }
    }
    
    // Model Event Handlers
    handleModelLoaded(data) {
        this.showToast(`Model ${data.model_id} loaded successfully!`, 'success');
        
        // Update UI elements to reflect loaded state
        this.updateModelUI(data.model_id, { loaded: true });
        
        // Enable chat interface if this is the active model
        this.enableChatInterface();
    }
    
    handleModelUnloaded(data) {
        this.showToast(`Model ${data.model_id} unloaded`, 'info');
        
        // Update UI elements to reflect unloaded state
        this.updateModelUI(data.model_id, { loaded: false });
        
        // Disable chat interface if this was the active model
        const currentModel = localStorage.getItem('currentModel');
        if (currentModel === data.model_id) {
            this.disableChatInterface();
        }
    }
    
    handleModelSwitched(data) {
        this.showToast(`Switched to model ${data.model_id}`, 'success');
        
        // Update active model in UI
        this.updateModelUI(data.model_id, { active: true });
        
        // Enable chat interface
        this.enableChatInterface();
    }
    
    updateModelUI(modelId, state) {
        // Update model selection dropdown
        const select = document.getElementById('model-select');
        if (select) {
            // Update current selection if it matches
            if (select.value === modelId) {
                // Trigger the change event to refresh UI
                if (typeof setCurrentModel === 'function') {
                    setCurrentModel(modelId);
                }
            }
        }
        
        // Update any model status indicators
        const statusElements = document.querySelectorAll(`[data-model-id="${modelId}"]`);
        statusElements.forEach(el => {
            if (state.loaded !== undefined) {
                el.dataset.loaded = state.loaded;
                el.classList.toggle('model-loaded', state.loaded);
                el.classList.toggle('model-unloaded', !state.loaded);
            }
            if (state.active !== undefined) {
                el.dataset.active = state.active;
                el.classList.toggle('model-active', state.active);
            }
        });
    }
    
    enableChatInterface() {
        // Enable chat input and send button using Alpine.js store or direct DOM access
        // Since we're using Alpine.js, we need to work with the actual elements
        
        // Try to access Alpine data first
        if (window.Alpine) {
            // Update the currentModel in the chat interface
            const chatContainer = document.querySelector('[x-data*="chatInterface"]');
            if (chatContainer && chatContainer._x_dataStack) {
                const chatData = chatContainer._x_dataStack[0];
                if (chatData && typeof chatData.currentModel !== 'undefined') {
                    chatData.currentModel = localStorage.getItem('currentModel') || '';
                }
            }
        }
        
        // Also update UI elements directly
        const messageInput = document.querySelector('textarea[x-ref="messageInput"]');
        const sendButtons = document.querySelectorAll('button[\\@click*="sendMessage"]');
        
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = 'Type your message... (Ctrl+Enter to send)';
        }
        
        sendButtons.forEach(button => {
            button.disabled = false;
        });
        
        // Show any hidden chat elements
        const chatElements = document.querySelectorAll('.chat-disabled');
        chatElements.forEach(el => {
            el.classList.remove('chat-disabled');
        });
        
        // Update welcome message if visible
        const welcomeContainer = document.querySelector('[x-show="!currentModel"]');
        if (welcomeContainer) {
            welcomeContainer.style.display = 'none';
        }
    }
    
    disableChatInterface() {
        // Disable chat interface when no model is loaded
        
        // Try to access Alpine data first
        if (window.Alpine) {
            const chatContainer = document.querySelector('[x-data*="chatInterface"]');
            if (chatContainer && chatContainer._x_dataStack) {
                const chatData = chatContainer._x_dataStack[0];
                if (chatData && typeof chatData.currentModel !== 'undefined') {
                    chatData.currentModel = '';
                }
            }
        }
        
        // Also update UI elements directly
        const messageInput = document.querySelector('textarea[x-ref="messageInput"]');
        const sendButtons = document.querySelectorAll('button[\\@click*="sendMessage"]');
        
        if (messageInput) {
            messageInput.disabled = true;
            messageInput.placeholder = 'Load a model to start chatting...';
        }
        
        sendButtons.forEach(button => {
            button.disabled = true;
        });
        
        // Hide chat elements that require a model
        const chatElements = document.querySelectorAll('.requires-model');
        chatElements.forEach(el => {
            el.classList.add('chat-disabled');
        });
        
        // Show welcome message
        const welcomeContainer = document.querySelector('[x-show="!currentModel"]');
        if (welcomeContainer) {
            welcomeContainer.style.display = 'block';
        }
    }

    updateModelLoadingProgress(data) {
        // Show model loading progress
        const progressBar = document.getElementById('model-loading-progress');
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
            progressBar.textContent = `Loading ${data.model}... ${data.progress}%`;
        }
    }
    
    updateRAGProgress(data) {
        // Show RAG indexing progress
        this.showToast(`RAG indexing: ${data.progress}% complete`, 'info', 1000);
    }
    
    // Service Worker for offline functionality
    setupServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/static/sw.js')
                .then((registration) => {
                    console.log('ServiceWorker registered:', registration);
                })
                .catch((error) => {
                    console.log('ServiceWorker registration failed:', error);
                });
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.llmCockpit = new LLMCockpit();
});

// Export for use in other scripts
window.LLMCockpit = LLMCockpit; 