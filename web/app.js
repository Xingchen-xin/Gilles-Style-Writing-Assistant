// GSWA Web UI

const API_BASE = '/v1';

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
                <button class="copy-btn" onclick="copyText(${index})">Copy</button>
            </div>
            <div class="variant-text" id="variant-text-${index}">${escapeHtml(variant.text)}</div>
            <div class="scores">
                ${scoresHtml}
            </div>
        `;

        variantsContainer.appendChild(card);
    });

    metaInfo.textContent = `Model: ${data.model_version} | Processing time: ${data.processing_time_ms}ms`;
    resultsSection.classList.remove('hidden');
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
