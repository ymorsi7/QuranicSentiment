// State management
let state = {
    selectedEmotion: null,
    intensity: 5,
    context: '',
};

// DOM Elements
const emotionButtons = document.querySelectorAll('.emotion-btn');
const intensityContainer = document.querySelector('#intensity-container');
const contextContainer = document.querySelector('#context-container');
const intensitySlider = document.querySelector('#intensity');
const contextInput = document.querySelector('#context');
const recommendationsSection = document.querySelector('#recommendations');

// Event Listeners
emotionButtons.forEach(button => {
    button.addEventListener('click', () => handleEmotionSelect(button));
});

intensitySlider.addEventListener('input', debounce(handleIntensityChange, 300));
contextInput?.addEventListener('input', debounce(handleContextChange, 500));

// Event Handlers
function handleEmotionSelect(button) {
    // Update UI
    emotionButtons.forEach(btn => btn.classList.remove('selected'));
    button.classList.add('selected');
    
    // Show additional inputs
    intensityContainer.style.display = 'block';
    contextContainer.style.display = 'block';
    
    // Update state
    state.selectedEmotion = button.dataset.emotion;
    
    // Get recommendations
    getRecommendations();
}

function handleIntensityChange(event) {
    state.intensity = parseInt(event.target.value);
    getRecommendations();
}

function handleContextChange(event) {
    state.context = event.target.value;
    getRecommendations();
}

// API Interaction
async function getRecommendations() {
    if (!state.selectedEmotion) return;
    
    showLoading();
    
    try {
        const response = await fetch('http://localhost:5000/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                emotion: state.selectedEmotion,
                intensity: state.intensity,
                context: state.context
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch recommendations');
        }
        
        const recommendations = await response.json();
        displayRecommendations(recommendations);
    } catch (error) {
        showError(error.message);
    }
}

// UI Updates
function displayRecommendations(recommendations) {
    const html = recommendations.map(verse => `
        <article class="verse-card">
            <header class="verse-header">
                <span class="verse-title">${verse.surah_name} - Verse ${verse.ayah_no}</span>
                <span class="verse-match">Match: ${(verse.relevance_score * 100).toFixed(1)}%</span>
            </header>
            
            <div class="verse-text">${verse.text}</div>
            
            <footer class="verse-footer">
                <span>Sentiment Level: ${verse.sentiment}</span>
                <button onclick="saveVerse(${JSON.stringify(verse).replace(/"/g, '&quot;')})" 
                        class="save-btn">
                    Save Verse
                </button>
            </footer>
        </article>
    `).join('');
    
    recommendationsSection.innerHTML = html;
}

function showLoading() {
    recommendationsSection.innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
            <p>Finding relevant verses...</p>
        </div>
    `;
}

function showError(message) {
    recommendationsSection.innerHTML = `
        <div class="error">
            <p>Error: ${message}</p>
            <p>Please try again later.</p>
        </div>
    `;
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Local Storage Functionality
function saveVerse(verse) {
    const savedVerses = JSON.parse(localStorage.getItem('savedVerses') || '[]');
    savedVerses.push({
        ...verse,
        savedAt: new Date().toISOString(),
        emotion: state.selectedEmotion,
        intensity: state.intensity
    });
    localStorage.setItem('savedVerses', JSON.stringify(savedVerses));
    alert('Verse saved successfully!');
}

// Initialize
function init() {
    // Check for saved state
    const savedState = localStorage.getItem('lastState');
    if (savedState) {
        state = JSON.parse(savedState);
        // Restore UI state
        if (state.selectedEmotion) {
            const button = document.querySelector(`[data-emotion="${state.selectedEmotion}"]`);
            if (button) handleEmotionSelect(button);
        }
    }
}

init();