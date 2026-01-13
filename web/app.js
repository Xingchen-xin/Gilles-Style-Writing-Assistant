// GSWA Web UI

const API_BASE = '/v1';

// State
let currentSession = {
    id: null,
    inputText: '',
    section: null,
    variants: []
};

// Elements
const inputText = document.getElementById('input-text');
const sectionSelect = document.getElementById('section');
const nVariantsSelect = document.getElementById('n-variants');
const generateBtn = document.getElementById('generate-btn');
const resultsSection = document.getElementById('results-section');
const variantsContainer = document.getElementById('variants-container');
const metaInfo = document.getElementById('meta-info');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const healthStatus = document.getElementById('health-status');

// Strategy descriptions
const STRATEGY_DESCRIPTIONS = {
    'A': 'Conclusion-first',
    'B': 'Background-first',
    'C': 'Methods-first',
    'D': 'Cautious-first'
};

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Check health on load
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();

        let statusText = '';
        let statusClass = '';

        if (data.status === 'healthy') {
            statusText = 'System Ready';
            statusClass = 'healthy';
        } else if (data.status === 'degraded') {
            statusText = 'System Degraded';
            statusClass = 'degraded';
        } else {
            statusText = 'System Error';
            statusClass = 'error';
        }

        if (data.model_loaded) {
            statusText += ` (${data.model_loaded})`;
        }

        if (data.corpus_paragraphs > 0) {
            statusText += ` | ${data.corpus_paragraphs} corpus paragraphs`;
        }

        healthStatus.textContent = statusText;
        healthStatus.className = `health-badge ${statusClass}`;

    } catch (e) {
        healthStatus.textContent = 'Connection Error';
        healthStatus.className = 'health-badge error';
    }
}

// Generate variants
async function generateVariants() {
    const text = inputText.value.trim();

    if (!text) {
        showError('Please enter some text to rewrite.');
        return;
    }

    if (text.length < 10) {
        showError('Text must be at least 10 characters long.');
        return;
    }

    // Show loading
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorDiv.classList.add('hidden');
    generateBtn.disabled = true;

    try {
        const payload = {
            text: text,
            n_variants: parseInt(nVariantsSelect.value),
        };

        if (sectionSelect.value) {
            payload.section = sectionSelect.value;
        }

        const resp = await fetch(`${API_BASE}/rewrite/variants`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || 'API error');
        }

        const data = await resp.json();

        // Store session for feedback
        currentSession = {
            id: generateSessionId(),
            inputText: text,
            section: sectionSelect.value || null,
            variants: data.variants
        };

        displayResults(data);

    } catch (e) {
        showError(`Error: ${e.message}`);
    } finally {
        loading.classList.add('hidden');
        generateBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    variantsContainer.innerHTML = '';

    data.variants.forEach((variant, index) => {
        const card = document.createElement('div');
        card.className = 'variant-card';
        card.dataset.index = index;

        const strategyDesc = STRATEGY_DESCRIPTIONS[variant.strategy] || variant.strategy;

        // Build badges HTML
        let badgesHtml = `<span class="strategy-badge">Strategy ${variant.strategy}: ${strategyDesc}</span>`;
        if (variant.fallback) {
            badgesHtml += `<span class="fallback-badge">Fallback</span>`;
        }

        // Build scores HTML
        let scoresHtml = `
            <span class="score-item ${variant.scores.ngram_max_match >= 12 ? 'warning' : ''}">
                N-gram match: ${variant.scores.ngram_max_match}
            </span>
            <span class="score-item ${variant.scores.embed_top1 >= 0.88 ? 'warning' : ''}">
                Embed sim: ${variant.scores.embed_top1.toFixed(3)}
            </span>
        `;

        if (variant.fallback_reason) {
            scoresHtml += `<span class="score-item warning">Fallback reason: ${escapeHtml(variant.fallback_reason)}</span>`;
        }

        card.innerHTML = `
            <div class="variant-header">
                <div class="variant-badges">
                    ${badgesHtml}
                </div>
                <div class="variant-actions">
                    <button class="copy-btn" onclick="copyText(${index})">Copy</button>
                </div>
            </div>
            <div class="variant-text" id="variant-text-${index}">${escapeHtml(variant.text)}</div>
            <div class="scores">
                ${scoresHtml}
            </div>
            <div class="feedback-section">
                <span class="feedback-label">Rate this variant:</span>
                <div class="feedback-buttons">
                    <button class="feedback-btn best" onclick="rateBest(${index})" title="Best - Use this one">Best</button>
                    <button class="feedback-btn good" onclick="rateGood(${index})" title="Good - Acceptable">Good</button>
                    <button class="feedback-btn bad" onclick="rateBad(${index})" title="Bad - Not acceptable">Bad</button>
                    <button class="feedback-btn edit" onclick="openEditor(${index})" title="Edit - Needs changes">Edit</button>
                </div>
                <div class="feedback-status" id="feedback-status-${index}"></div>
            </div>
        `;

        variantsContainer.appendChild(card);
    });

    // Add submit feedback button
    const submitSection = document.createElement('div');
    submitSection.className = 'submit-feedback-section';
    submitSection.innerHTML = `
        <div class="feedback-notes">
            <label for="user-notes">Additional notes (optional):</label>
            <textarea id="user-notes" placeholder="Any comments about the variants..." rows="2"></textarea>
        </div>
        <button id="submit-feedback-btn" class="submit-feedback-btn" onclick="submitFeedback()">
            Submit Feedback for Training
        </button>
        <div id="submit-status" class="submit-status"></div>
    `;
    variantsContainer.appendChild(submitSection);

    metaInfo.textContent = `Model: ${data.model_version} | Processing time: ${data.processing_time_ms}ms | Session: ${currentSession.id}`;
    resultsSection.classList.remove('hidden');
}

// Feedback ratings storage
let feedbackRatings = {};

function rateBest(index) {
    setRating(index, 'best');
}

function rateGood(index) {
    setRating(index, 'good');
}

function rateBad(index) {
    setRating(index, 'bad');
}

function setRating(index, rating) {
    feedbackRatings[index] = { type: rating };

    // Update UI
    const card = document.querySelector(`.variant-card[data-index="${index}"]`);
    const buttons = card.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => btn.classList.remove('selected'));
    card.querySelector(`.feedback-btn.${rating}`).classList.add('selected');

    const status = document.getElementById(`feedback-status-${index}`);
    status.textContent = `Rated: ${rating.toUpperCase()}`;
    status.className = `feedback-status ${rating}`;
}

// Store original texts for cancel functionality
let originalTexts = {};

function openEditor(index) {
    const textEl = document.getElementById(`variant-text-${index}`);
    const originalText = textEl.textContent;

    // Store original text for cancel
    originalTexts[index] = originalText;

    // Replace with textarea
    const editorHtml = `
        <div class="editor-container" id="editor-${index}">
            <textarea class="edit-textarea" id="edit-textarea-${index}">${escapeHtml(originalText)}</textarea>
            <div class="editor-actions">
                <button class="save-edit-btn" onclick="saveEdit(${index})">Save Edit</button>
                <button class="cancel-edit-btn" onclick="cancelEdit(${index})">Cancel</button>
            </div>
        </div>
    `;

    textEl.innerHTML = editorHtml;
}

function saveEdit(index) {
    const textarea = document.getElementById(`edit-textarea-${index}`);
    const editedText = textarea.value.trim();

    if (!editedText) {
        alert('Edited text cannot be empty');
        return;
    }

    feedbackRatings[index] = { type: 'edited', editedText: editedText };

    // Update UI
    const textEl = document.getElementById(`variant-text-${index}`);
    textEl.textContent = editedText;

    // Update original text for future cancels
    originalTexts[index] = editedText;

    const card = textEl.closest('.variant-card');
    const buttons = card.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => btn.classList.remove('selected'));
    card.querySelector('.feedback-btn.edit').classList.add('selected');

    const status = document.getElementById(`feedback-status-${index}`);
    status.textContent = 'Rated: EDITED';
    status.className = 'feedback-status edited';
}

function cancelEdit(index) {
    const textEl = document.getElementById(`variant-text-${index}`);
    textEl.textContent = originalTexts[index] || '';
}

async function submitFeedback() {
    const submitBtn = document.getElementById('submit-feedback-btn');
    const submitStatus = document.getElementById('submit-status');
    const userNotes = document.getElementById('user-notes').value.trim();

    // Check if any ratings exist
    if (Object.keys(feedbackRatings).length === 0) {
        submitStatus.textContent = 'Please rate at least one variant before submitting.';
        submitStatus.className = 'submit-status error';
        return;
    }

    submitBtn.disabled = true;
    submitStatus.textContent = 'Submitting feedback...';
    submitStatus.className = 'submit-status';

    try {
        // Build variants array
        const variants = Object.entries(feedbackRatings).map(([index, rating]) => ({
            variant_index: parseInt(index),
            feedback_type: rating.type,
            edited_text: rating.editedText || null
        }));

        const payload = {
            session_id: currentSession.id,
            input_text: currentSession.inputText,
            section: currentSession.section,
            variants: variants,
            user_notes: userNotes || null
        };

        const resp = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || 'Feedback submission failed');
        }

        const data = await resp.json();

        submitStatus.textContent = `Feedback submitted successfully! (ID: ${data.feedback_id})`;
        submitStatus.className = 'submit-status success';

        // Reset ratings for next session
        feedbackRatings = {};

    } catch (e) {
        submitStatus.textContent = `Error: ${e.message}`;
        submitStatus.className = 'submit-status error';
    } finally {
        submitBtn.disabled = false;
    }
}

// Copy text to clipboard
function copyText(index) {
    const textEl = document.getElementById(`variant-text-${index}`);
    const text = textEl.textContent;

    navigator.clipboard.writeText(text).then(() => {
        // Brief feedback
        const card = textEl.closest('.variant-card');
        const btn = card.querySelector('.copy-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = originalText;
            btn.classList.remove('copied');
        }, 1500);
    }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    });
}

// Show error
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event listeners
generateBtn.addEventListener('click', generateVariants);

inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        generateVariants();
    }
});

// Clear error when user starts typing
inputText.addEventListener('input', () => {
    errorDiv.classList.add('hidden');
});

// Initialize
checkHealth();
setInterval(checkHealth, 30000);  // Check every 30s
