// GSWA Web UI - Enhanced Version

const API_BASE = '/v1';

// State
let currentSession = {
    id: null,
    inputText: '',
    section: null,
    variants: []
};

// Elements - Rewrite tab
const inputText = document.getElementById('input-text');
const modelSelect = document.getElementById('model-select');
const sectionSelect = document.getElementById('section');
const nVariantsSelect = document.getElementById('n-variants');
const generateBtn = document.getElementById('generate-btn');
const resultsSection = document.getElementById('results-section');
const variantsContainer = document.getElementById('variants-container');
const metaInfo = document.getElementById('meta-info');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const healthStatus = document.getElementById('health-status');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const progressSubtext = document.getElementById('progress-subtext');

// Elements - Style Analysis tab
const evalText = document.getElementById('eval-text');
const analyzeBtn = document.getElementById('analyze-btn');
const analysisResults = document.getElementById('analysis-results');
const analysisLoading = document.getElementById('analysis-loading');

// Elements - Stats tab
const refreshStatsBtn = document.getElementById('refresh-stats-btn');
const exportDpoBtn = document.getElementById('export-dpo-btn');

// Strategy descriptions
const STRATEGY_DESCRIPTIONS = {
    'A': 'Conclusion-first',
    'B': 'Background-first',
    'C': 'Methods-first',
    'D': 'Cautious-first'
};

// Tab Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const tabId = btn.dataset.tab;
        document.getElementById(`tab-${tabId}`).classList.add('active');

        // Load data for stats tab
        if (tabId === 'stats') {
            loadFeedbackStats();
        }
    });
});

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function resetProgress() {
    progressBar.style.width = '0%';
    progressText.textContent = 'Starting...';
    progressSubtext.textContent = '';
}

function updateProgress(event) {
    const overall = Math.round((event.overall_progress || 0) * 100);
    progressBar.style.width = `${overall}%`;
    progressText.textContent = `Generating variants... ${overall}%`;

    const variantIndex = (event.variant_index ?? 0) + 1;
    const variantTotal = event.variant_total ?? 1;
    const tokensGenerated = event.tokens_generated ?? 0;
    const tokensTarget = event.tokens_target ?? 0;
    const fallbackFlag = event.is_fallback ? ' | Fallback' : '';
    progressSubtext.textContent = `Variant ${variantIndex}/${variantTotal} | ${tokensGenerated}/${tokensTarget} tokens${fallbackFlag}`;
}

async function streamSse(resp, onEvent) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split('\n\n');
        buffer = chunks.pop() || '';

        for (const chunk of chunks) {
            const line = chunk.split('\n').find((l) => l.startsWith('data: '));
            if (!line) {
                continue;
            }
            const dataStr = line.slice(6).trim();
            if (!dataStr) {
                continue;
            }
            try {
                const event = JSON.parse(dataStr);
                onEvent(event);
            } catch {
                // Ignore malformed chunks
            }
        }
    }
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

        if (data.available_models > 0) {
            statusText += ` | ${data.available_models} adapters`;
        }

        healthStatus.textContent = statusText;
        healthStatus.className = `health-badge ${statusClass}`;

    } catch (e) {
        healthStatus.textContent = 'Connection Error';
        healthStatus.className = 'health-badge error';
    }
}

// Store loaded models for info display
let loadedModels = [];

// Load available models
async function loadModels() {
    try {
        const resp = await fetch(`${API_BASE}/models`);
        if (!resp.ok) return;
        const data = await resp.json();
        loadedModels = data.models || [];

        // Clear existing options except default
        modelSelect.innerHTML = '<option value="">Default (base model)</option>';

        if (data.models && data.models.length > 0) {
            // Group models by base model
            const groups = {};
            data.models.forEach(model => {
                const base = model.model_short || 'unknown';
                if (!groups[base]) groups[base] = [];
                groups[base].push(model);
            });

            // Add models grouped by base, with recommended first
            Object.keys(groups).forEach(baseKey => {
                const groupModels = groups[baseKey];

                // Create optgroup for each base model
                const optgroup = document.createElement('optgroup');
                const friendlyBase = groupModels[0].display_name.split('(')[0].trim();
                optgroup.label = `${friendlyBase} (${groupModels.length} adapters)`;

                groupModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;

                    // Build display text with key info
                    let displayText = model.display_name;
                    if (model.is_recommended) {
                        displayText = `★ ${displayText} [Latest]`;
                    }
                    option.textContent = displayText;

                    // Rich tooltip with full details
                    option.title = model.description;

                    // Highlight recommended
                    if (model.is_recommended) {
                        option.style.fontWeight = 'bold';
                    }

                    optgroup.appendChild(option);
                });

                modelSelect.appendChild(optgroup);
            });

            // Auto-select recommended model if available
            const recommended = data.models.find(m => m.is_recommended);
            if (recommended) {
                modelSelect.value = recommended.id;
            }
        }
    } catch (e) {
        // Models endpoint not available - keep default only
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

    resetProgress();
    let gotResult = false;

    try {
        const payload = {
            text: text,
            n_variants: parseInt(nVariantsSelect.value),
        };

        if (modelSelect.value) {
            payload.model = modelSelect.value;
        }

        if (sectionSelect.value) {
            payload.section = sectionSelect.value;
        }

        const resp = await fetch(`${API_BASE}/rewrite/variants/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(errText || 'API error');
        }

        await streamSse(resp, (event) => {
            if (event.type === 'variant_start') {
                const variantIndex = (event.variant_index ?? 0) + 1;
                const variantTotal = event.variant_total ?? 1;
                progressText.textContent = `Generating variant ${variantIndex}/${variantTotal}...`;
                return;
            }
            if (event.type === 'progress') {
                updateProgress(event);
                return;
            }
            if (event.type === 'status') {
                progressText.textContent = event.message || 'Regenerating...';
                return;
            }
            if (event.type === 'result') {
                gotResult = true;
                const data = event.data;
                currentSession = {
                    id: generateSessionId(),
                    inputText: text,
                    section: sectionSelect.value || null,
                    variants: data.variants
                };
                displayResults(data);
                loading.classList.add('hidden');
                generateBtn.disabled = false;
                return;
            }
            if (event.type === 'error') {
                throw new Error(event.message || 'Streaming error');
            }
        });
        if (!gotResult) {
            throw new Error('Stream ended without result');
        }
    } catch (e) {
        showError(`Error: ${e.message}`);
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

        // Calculate style score for this variant
        const styleMetrics = calculateStyleMetrics(variant.text);
        const styleScore = computeStyleScore(styleMetrics);

        // Build badges HTML
        let badgesHtml = `<span class="strategy-badge">Strategy ${variant.strategy}: ${strategyDesc}</span>`;
        badgesHtml += `<span class="style-score-badge ${getScoreClass(styleScore)}">Style: ${styleScore}/100</span>`;
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

// Get CSS class based on score
function getScoreClass(score) {
    if (score >= 70) return 'score-high';
    if (score >= 40) return 'score-medium';
    return 'score-low';
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

// ==================== STYLE ANALYSIS ====================

// Style metrics calculation (client-side, matches compare_models.py)
function calculateStyleMetrics(text) {
    if (!text || text.length < 50) {
        return { error: 'Text too short' };
    }

    // Tokenize into sentences and words
    const sentences = text.split(/[.!?]+/).filter(s => s.trim());
    const words = text.toLowerCase().match(/\b[a-zA-Z]+\b/g) || [];

    if (!sentences.length || !words.length) {
        return { error: 'Could not parse text' };
    }

    const metrics = {};

    // 1. Sentence length statistics
    const sentLengths = sentences.map(s => (s.match(/\b\w+\b/g) || []).length);
    metrics.avg_sentence_length = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
    metrics.max_sentence_length = Math.max(...sentLengths);

    // 2. Vocabulary richness (Type-Token Ratio)
    metrics.vocabulary_richness = new Set(words).size / words.length;

    // 3. Discourse markers (Gilles uses these frequently)
    const discourseMarkers = [
        'indeed', 'notably', 'interestingly', 'importantly', 'remarkably',
        'strikingly', 'intriguingly', 'surprisingly', 'unexpectedly',
        'furthermore', 'moreover', 'however', 'nevertheless', 'conversely',
        'accordingly', 'consequently', 'thus', 'hence', 'therefore',
        'specifically', 'particularly', 'especially', 'significantly'
    ];
    const foundMarkers = words.filter(w => discourseMarkers.includes(w));
    metrics.discourse_marker_density = (foundMarkers.length / words.length) * 100;
    metrics.discourse_markers_found = [...new Set(foundMarkers)];

    // 4. Subordination markers
    const subordinators = [
        'which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'while',
        'although', 'whereas', 'whereby', 'wherein', 'because', 'since',
        'unless', 'until', 'after', 'before', 'if', 'whether'
    ];
    const subCount = words.filter(w => subordinators.includes(w)).length;
    metrics.subordination_density = (subCount / words.length) * 100;

    // 5. Academic hedging
    const hedgingWords = [
        'may', 'might', 'could', 'would', 'possibly', 'potentially',
        'likely', 'unlikely', 'perhaps', 'presumably', 'apparently',
        'suggests', 'indicates', 'appears', 'seems'
    ];
    const hedgeCount = words.filter(w => hedgingWords.includes(w)).length;
    metrics.hedging_density = (hedgeCount / words.length) * 100;

    // 6. Scientific precision markers
    const precisionWords = [
        'specifically', 'exclusively', 'predominantly', 'primarily',
        'essentially', 'virtually', 'substantially', 'considerably',
        'markedly', 'drastically', 'dramatically', 'profoundly'
    ];
    const precisionCount = words.filter(w => precisionWords.includes(w)).length;
    metrics.precision_marker_density = (precisionCount / words.length) * 100;

    // 7. Transition phrases
    const textLower = text.toLowerCase();
    const transitionPhrases = [
        'taken together', 'in contrast', 'in addition', 'as a result',
        'on the other hand', 'in this context', 'to this end',
        'of particular interest', 'it is noteworthy', 'it should be noted'
    ];
    metrics.transition_phrases = transitionPhrases.filter(p => textLower.includes(p)).length;

    // 8. Word and sentence count
    metrics.word_count = words.length;
    metrics.sentence_count = sentences.length;

    return metrics;
}

// Compute style score (matches compare_models.py)
function computeStyleScore(metrics) {
    if (metrics.error) {
        return 0;
    }

    let score = 0;

    // 1. Sentence length (target: 20-30 words average)
    const avgLen = metrics.avg_sentence_length || 0;
    if (avgLen >= 20 && avgLen <= 30) score += 20;
    else if (avgLen >= 15 && avgLen <= 35) score += 10;
    else if (avgLen > 10) score += 5;

    // 2. Discourse markers (target: > 0.5% density)
    const dmDensity = metrics.discourse_marker_density || 0;
    if (dmDensity >= 1.0) score += 20;
    else if (dmDensity >= 0.5) score += 15;
    else if (dmDensity >= 0.2) score += 10;
    else if (dmDensity > 0) score += 5;

    // 3. Subordination (target: > 3% density)
    const subDensity = metrics.subordination_density || 0;
    if (subDensity >= 4.0) score += 20;
    else if (subDensity >= 3.0) score += 15;
    else if (subDensity >= 2.0) score += 10;
    else if (subDensity > 0) score += 5;

    // 4. Hedging (target: 0.5-2% - balanced)
    const hedge = metrics.hedging_density || 0;
    if (hedge >= 0.5 && hedge <= 2.0) score += 15;
    else if (hedge >= 0.2 && hedge <= 3.0) score += 10;
    else if (hedge > 0) score += 5;

    // 5. Precision markers
    const precision = metrics.precision_marker_density || 0;
    if (precision >= 0.5) score += 15;
    else if (precision >= 0.2) score += 10;
    else if (precision > 0) score += 5;

    // 6. Transition phrases
    const trans = metrics.transition_phrases || 0;
    if (trans >= 2) score += 10;
    else if (trans >= 1) score += 5;

    return Math.min(score, 100);
}

// Analyze style button handler (client-side quick analysis)
function analyzeStyle() {
    const text = evalText.value.trim();

    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }

    if (text.length < 50) {
        alert('Text must be at least 50 characters for meaningful analysis.');
        return;
    }

    analysisLoading.classList.remove('hidden');
    analysisResults.classList.add('hidden');

    // Small delay for UI feedback
    setTimeout(() => {
        const metrics = calculateStyleMetrics(text);
        const score = computeStyleScore(metrics);

        displayStyleAnalysis(metrics, score);

        analysisLoading.classList.add('hidden');
        analysisResults.classList.remove('hidden');
    }, 300);
}

// Model-based style analysis (server-side, requires LLM)
async function analyzeStyleWithModel() {
    const text = evalText.value.trim();

    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }

    if (text.length < 50) {
        alert('Text must be at least 50 characters for meaningful analysis.');
        return;
    }

    // Show loading
    analysisLoading.classList.remove('hidden');
    analysisResults.classList.add('hidden');
    const modelAnalyzeBtn = document.getElementById('model-analyze-btn');
    if (modelAnalyzeBtn) modelAnalyzeBtn.disabled = true;

    try {
        const resp = await fetch(`${API_BASE}/style/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text }),
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || 'Style analysis failed');
        }

        const data = await resp.json();
        displayModelStyleAnalysis(data);

    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        analysisLoading.classList.add('hidden');
        analysisResults.classList.remove('hidden');
        if (modelAnalyzeBtn) modelAnalyzeBtn.disabled = false;
    }
}

// Display model-based style analysis results
function displayModelStyleAnalysis(data) {
    // Update score circle
    const scoreCircle = document.getElementById('style-score-circle');
    const scoreValue = document.getElementById('style-score-value');
    scoreValue.textContent = data.overall_score;
    scoreCircle.className = `score-circle ${getScoreClass(data.overall_score)}`;

    // Build dimensions display
    const dimensionsHtml = data.dimensions.map(dim => `
        <div class="model-dimension">
            <div class="dimension-header">
                <span class="dimension-name">${dim.name}</span>
                <span class="dimension-score ${getScoreClass(dim.score * 10)}">${dim.score}/10</span>
            </div>
            <div class="dimension-feedback">${dim.feedback}</div>
        </div>
    `).join('');

    // Build suggestions display
    const suggestionsHtml = data.suggestions.length > 0 ? `
        <div class="suggestions-section">
            <h4>Suggestions for Improvement:</h4>
            <ul>
                ${data.suggestions.map(s => `<li>${s}</li>`).join('')}
            </ul>
        </div>
    ` : '';

    // Update the metrics grid with model analysis
    const metricsGrid = document.querySelector('.metrics-grid');
    metricsGrid.innerHTML = `
        <div class="model-analysis-summary">
            <p class="analysis-summary">${data.summary}</p>
            <p class="model-used">Analyzed by: ${data.model_used}</p>
        </div>
        <div class="model-dimensions">
            ${dimensionsHtml}
        </div>
        ${suggestionsHtml}
    `;

    // Hide the marker tags section for model analysis
    const markerTags = document.getElementById('marker-tags');
    if (markerTags) markerTags.parentElement.style.display = 'none';
}

function displayStyleAnalysis(metrics, score) {
    // Update score circle
    const scoreCircle = document.getElementById('style-score-circle');
    const scoreValue = document.getElementById('style-score-value');
    scoreValue.textContent = score;
    scoreCircle.className = `score-circle ${getScoreClass(score)}`;

    // Update individual metrics
    document.getElementById('metric-sent-len').textContent =
        `${metrics.avg_sentence_length?.toFixed(1) || '--'} words`;

    document.getElementById('metric-discourse').textContent =
        `${metrics.discourse_marker_density?.toFixed(2) || '0.00'}%`;

    document.getElementById('metric-subordination').textContent =
        `${metrics.subordination_density?.toFixed(2) || '0.00'}%`;

    document.getElementById('metric-hedging').textContent =
        `${metrics.hedging_density?.toFixed(2) || '0.00'}%`;

    document.getElementById('metric-precision').textContent =
        `${metrics.precision_marker_density?.toFixed(2) || '0.00'}%`;

    document.getElementById('metric-transitions').textContent =
        `${metrics.transition_phrases || 0} found`;

    // Update marker tags
    const markerTags = document.getElementById('marker-tags');
    const markers = metrics.discourse_markers_found || [];
    if (markers.length > 0) {
        markerTags.innerHTML = markers.map(m =>
            `<span class="marker-tag">${m}</span>`
        ).join('');
    } else {
        markerTags.innerHTML = '<span class="no-markers">No discourse markers detected</span>';
    }
}

// ==================== FEEDBACK STATS ====================

async function loadFeedbackStats() {
    try {
        const resp = await fetch(`${API_BASE}/feedback/stats`);
        if (!resp.ok) {
            throw new Error('Failed to load stats');
        }

        const data = await resp.json();

        document.getElementById('stat-sessions').textContent = data.total_sessions;
        document.getElementById('stat-variants').textContent = data.total_variants_rated;
        document.getElementById('stat-best').textContent = data.best_count;
        document.getElementById('stat-good').textContent = data.good_count;
        document.getElementById('stat-bad').textContent = data.bad_count;
        document.getElementById('stat-edited').textContent = data.edited_count;

    } catch (e) {
        console.error('Error loading stats:', e);
    }
}

async function exportDpoData() {
    const exportStatus = document.getElementById('export-status');
    exportDpoBtn.disabled = true;
    exportStatus.textContent = 'Exporting...';
    exportStatus.className = 'export-status';

    try {
        const resp = await fetch(`${API_BASE}/feedback/export-dpo`, {
            method: 'POST'
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || 'Export failed');
        }

        const data = await resp.json();
        exportStatus.textContent = `Exported ${data.pairs_count} training pairs to ${data.output_path}`;
        exportStatus.className = 'export-status success';

    } catch (e) {
        exportStatus.textContent = `Error: ${e.message}`;
        exportStatus.className = 'export-status error';
    } finally {
        exportDpoBtn.disabled = false;
    }
}

// ==================== EVENT LISTENERS ====================

generateBtn.addEventListener('click', generateVariants);
analyzeBtn.addEventListener('click', analyzeStyle);
refreshStatsBtn.addEventListener('click', loadFeedbackStats);
exportDpoBtn.addEventListener('click', exportDpoData);

// Model-based analyze button (added dynamically, so use optional chaining)
const modelAnalyzeBtn = document.getElementById('model-analyze-btn');
if (modelAnalyzeBtn) {
    modelAnalyzeBtn.addEventListener('click', analyzeStyleWithModel);
}

inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        generateVariants();
    }
});

evalText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeStyle();
    }
});

// Clear error when user starts typing
inputText.addEventListener('input', () => {
    errorDiv.classList.add('hidden');
});

// Show model info when selection changes
function updateModelInfo() {
    const modelInfo = document.getElementById('model-info');
    const selectedId = modelSelect.value;

    if (!selectedId || !modelInfo) {
        if (modelInfo) modelInfo.classList.add('hidden');
        return;
    }

    const model = loadedModels.find(m => m.id === selectedId);
    if (!model) {
        modelInfo.classList.add('hidden');
        return;
    }

    // Build info display
    let infoHtml = `<div class="model-info-content">`;
    infoHtml += `<span class="model-param"><b>LoRA rank:</b> ${model.lora_r}</span>`;
    infoHtml += `<span class="model-param"><b>Epochs:</b> ${model.epochs}</span>`;
    infoHtml += `<span class="model-param"><b>Quantization:</b> ${model.quantization}</span>`;
    if (model.started_at) {
        infoHtml += `<span class="model-param"><b>Trained:</b> ${model.started_at.substring(0, 10)}</span>`;
    }
    if (model.is_recommended) {
        infoHtml += `<span class="model-badge recommended">Latest</span>`;
    }
    infoHtml += `</div>`;

    modelInfo.innerHTML = infoHtml;
    modelInfo.classList.remove('hidden');
}

modelSelect.addEventListener('change', updateModelInfo);

// ==================== INITIALIZATION ====================

checkHealth();
loadModels().then(updateModelInfo);
setInterval(checkHealth, 30000);  // Check every 30s
