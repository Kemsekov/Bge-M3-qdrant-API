// Configuration - Use relative path for API when served from same origin
let API_BASE = localStorage.getItem('api_base');
if (!API_BASE) {
    // Default to /api for nginx proxy, or localhost for direct access
    API_BASE = '/api';
    localStorage.setItem('api_base', API_BASE);
}

// State
let currentPage = 1;

// API Base Management
function initApiBase() {
    const apiInput = document.getElementById('api-base-url');
    if (apiInput) {
        apiInput.value = API_BASE;
        apiInput.addEventListener('change', () => {
            API_BASE = apiInput.value.trim();
            localStorage.setItem('api_base', API_BASE);
            showNotification('API URL saved: ' + API_BASE, 'success');
        });
    }
}

function resetApiUrl() {
    API_BASE = '/api';
    localStorage.setItem('api_base', API_BASE);
    document.getElementById('api-base-url').value = API_BASE;
    showNotification('API URL reset to /api', 'success');
    testConnection();
}

async function testConnection() {
    const statusEl = document.getElementById('connection-status');
    statusEl.textContent = 'Testing...';
    statusEl.className = '';
    
    try {
        const response = await fetch(`${API_BASE}/health`, { 
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        const data = await response.json();
        
        if (response.ok && data.status === 'healthy') {
            statusEl.textContent = '✓ Connected';
            statusEl.className = 'connected';
            showNotification('Connection successful!', 'success');
        } else {
            throw new Error('Unhealthy response');
        }
    } catch (error) {
        statusEl.textContent = '✗ Disconnected';
        statusEl.className = 'disconnected';
        showNotification('Connection failed: ' + error.message, 'error');
    }
}

function showNotification(message, type = 'info') {
    const colors = {
        success: '#4caf50',
        error: '#f44336',
        info: '#2196f3'
    };
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        border-radius: 8px;
        background: ${colors[type] || colors.info};
        color: white;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(notification);
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Tab Management
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-panel`).classList.remove('hidden');
        });
    });
}

// Add Document
async function handleAddDocument(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('add-result');
    resultDiv.innerHTML = '<div class="loading">Adding document</div>';

    const payload = {
        content: document.getElementById('doc-content').value,
        id: document.getElementById('doc-id').value || null,
        metadata: {}
    };

    const docType = document.getElementById('doc-type').value.trim();
    if (docType) {
        payload.metadata.doc_type = docType;
    }

    const metadataJson = document.getElementById('doc-metadata').value.trim();
    if (metadataJson) {
        try {
            const extraMetadata = JSON.parse(metadataJson);
            payload.metadata = { ...payload.metadata, ...extraMetadata };
        } catch (err) {
            resultDiv.innerHTML = '<div class="error">Invalid JSON in metadata field</div>';
            return;
        }
    }

    try {
        const response = await fetch(`${API_BASE}/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        if (response.ok && data.success) {
            resultDiv.innerHTML = `
                <div class="success">
                    ✅ Document added successfully!<br>
                    <strong>ID:</strong> ${data.document_id}
                </div>
            `;
            document.getElementById('add-form').reset();
        } else {
            resultDiv.innerHTML = `<div class="error">❌ Error: ${data.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

// Load Documents (Paged)
async function loadDocuments() {
    const resultDiv = document.getElementById('view-result');
    const paginationDiv = document.getElementById('view-pagination');
    const docType = document.getElementById('view-doc-type').value;
    const pageSize = parseInt(document.getElementById('page-size').value);

    if (!docType) {
        resultDiv.innerHTML = '<div class="error">Please select a document type</div>';
        return;
    }

    resultDiv.innerHTML = '<div class="loading">Loading documents</div>';
    paginationDiv.innerHTML = '';

    try {
        const url = `${API_BASE}/get_docs_paged?doc_type=${encodeURIComponent(docType)}&page_size=${pageSize}&page_number=${currentPage}`;
        const response = await fetch(url, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();

        if (response.ok) {
            if (data.documents && data.documents.length > 0) {
                resultDiv.innerHTML = `
                    <div class="results">
                        <p><strong>Total:</strong> ${data.total_count} documents | 
                        <strong>Page:</strong> ${data.page_number} of ${data.total_pages}</p>
                        ${data.documents.map(doc => `
                            <div class="result-item">
                                <h4>
                                    <span class="doc-type-badge">${doc.metadata?.doc_type || 'N/A'}</span>
                                    ${doc.id}
                                </h4>
                                <p>${escapeHtml(doc.content)}</p>
                                <div class="meta">
                                    ${doc.metadata && Object.keys(doc.metadata).filter(k => k !== 'doc_type').length > 0 
                                        ? '<strong>Metadata:</strong> ' + JSON.stringify(doc.metadata) 
                                        : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;

                renderPagination(data);
            } else {
                resultDiv.innerHTML = '<div class="error">No documents found for this type</div>';
                paginationDiv.innerHTML = '';
            }
        } else {
            resultDiv.innerHTML = `<div class="error">❌ Error: ${data.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

function renderPagination(data) {
    const paginationDiv = document.getElementById('view-pagination');
    let paginationHtml = '';
    
    if (data.page_number > 1) {
        paginationHtml += `<button class="btn btn-secondary" onclick="changePage(${data.page_number - 1})">← Previous</button>`;
    }
    
    paginationHtml += `<span>Page ${data.page_number} of ${data.total_pages}</span>`;
    
    if (data.page_number < data.total_pages) {
        paginationHtml += `<button class="btn btn-secondary" onclick="changePage(${data.page_number + 1})">Next →</button>`;
    }
    
    paginationDiv.innerHTML = paginationHtml;
}

function changePage(page) {
    currentPage = page;
    loadDocuments();
}

// Search
async function handleSearch(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('search-result');
    resultDiv.innerHTML = '<div class="loading">Searching</div>';

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: document.getElementById('search-query').value,
                top_k: parseInt(document.getElementById('search-top-k').value)
            })
        });
        const data = await response.json();

        if (response.ok && data.results) {
            if (data.results.length > 0) {
                resultDiv.innerHTML = `
                    <div class="results">
                        ${data.results.map((doc, i) => `
                            <div class="result-item">
                                <h4>#${i + 1} - ${doc.id} <span class="score">Score: ${doc.score.toFixed(4)}</span></h4>
                                <p>${escapeHtml(doc.content)}</p>
                                <div class="meta">
                                    ${doc.metadata ? '<strong>Metadata:</strong> ' + JSON.stringify(doc.metadata) : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                resultDiv.innerHTML = '<div class="error">No results found</div>';
            }
        } else {
            resultDiv.innerHTML = `<div class="error">❌ Error: ${data.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

// Ask LLM
async function handleAsk(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('ask-result');
    resultDiv.innerHTML = '<div class="loading">Asking LLM</div>';

    try {
        const response = await fetch(`${API_BASE}/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: document.getElementById('ask-question').value,
                top_k: parseInt(document.getElementById('ask-top-k').value),
                rephrases: parseInt(document.getElementById('ask-rephrases').value)
            })
        });
        const data = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `
                <div class="results">
                    <div class="result-item" style="border-left-color: #764ba2;">
                        <h4>🤖 Answer</h4>
                        <p>${escapeHtml(data.answer)}</p>
                    </div>
                    ${data.rephrases && data.rephrases.length > 0 ? `
                        <div class="rephrases">
                            <h5>Rephrased Questions:</h5>
                            <ul>${data.rephrases.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>
                        </div>
                    ` : ''}
                    ${data.contexts && data.contexts.length > 0 ? `
                        <div class="contexts">
                            <h5>Contexts Used (${data.contexts.length}):</h5>
                            ${data.contexts.map((ctx, i) => `
                                <div class="result-item" style="margin-top: 10px;">
                                    <h4>Context #${i + 1} <span class="score">Score: ${ctx.score?.toFixed(4) || 'N/A'}</span></h4>
                                    <p>${escapeHtml(ctx.content)}</p>
                                    <div class="meta">
                                        <strong>Source:</strong> ${ctx.source_query || 'Original query'}
                                        ${ctx.metadata ? '<br><strong>Metadata:</strong> ' + JSON.stringify(ctx.metadata) : ''}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        } else {
            resultDiv.innerHTML = `<div class="error">❌ Error: ${data.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initApiBase();
    
    document.getElementById('add-form').addEventListener('submit', handleAddDocument);
    document.getElementById('search-form').addEventListener('submit', handleSearch);
    document.getElementById('ask-form').addEventListener('submit', handleAsk);
});
