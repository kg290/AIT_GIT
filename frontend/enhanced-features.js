/**
 * Enhanced Dashboard Features
 * Timeline, Knowledge Graph, Conversational Query, and more
 */

const API_V2_BASE = 'http://localhost:8000/api/v2';

// ==================== Timeline Visualization ====================

async function loadPatientTimeline(patientId) {
    try {
        const response = await fetch(`${API_V2_BASE}/temporal/patient/${patientId}/gantt`);
        const data = await response.json();
        renderGanttChart(data);
    } catch (error) {
        console.error('Error loading timeline:', error);
    }
}

function renderGanttChart(data) {
    const container = document.getElementById('timelineChart');
    if (!container || !data.medications) return;

    const medications = data.medications;
    const startDate = new Date(data.timeline_start);
    const endDate = new Date(data.timeline_end);
    const totalDays = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));

    let html = `
        <div class="gantt-container">
            <div class="gantt-header">
                <div class="gantt-label-col">Medication</div>
                <div class="gantt-timeline-col">
                    <div class="gantt-dates">
    `;

    // Add month markers
    for (let d = new Date(startDate); d <= endDate; d.setMonth(d.getMonth() + 1)) {
        html += `<span class="gantt-month">${d.toLocaleDateString('en', { month: 'short', year: 'numeric' })}</span>`;
    }

    html += `</div></div></div><div class="gantt-body">`;

    medications.forEach(med => {
        const medStart = new Date(med.start);
        const medEnd = med.end ? new Date(med.end) : endDate;
        const leftPercent = ((medStart - startDate) / (endDate - startDate)) * 100;
        const widthPercent = ((medEnd - medStart) / (endDate - startDate)) * 100;

        html += `
            <div class="gantt-row">
                <div class="gantt-label-col">${med.name}</div>
                <div class="gantt-timeline-col">
                    <div class="gantt-bar ${med.active ? 'active' : 'inactive'}" 
                         style="left: ${leftPercent}%; width: ${widthPercent}%;"
                         title="${med.name}: ${med.start} - ${med.end || 'ongoing'}">
                    </div>
                </div>
            </div>
        `;
    });

    html += `</div></div>`;
    container.innerHTML = html;
}

// ==================== Knowledge Graph Visualization ====================

async function loadPatientGraph(patientId) {
    try {
        const response = await fetch(`${API_V2_BASE}/knowledge-graph/patient/${patientId}`);
        const data = await response.json();
        renderKnowledgeGraph(data);
    } catch (error) {
        console.error('Error loading graph:', error);
    }
}

function renderKnowledgeGraph(data) {
    const container = document.getElementById('knowledgeGraphContainer');
    if (!container) return;

    // Simple SVG-based graph rendering
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    const width = container.clientWidth || 600;
    const height = 400;

    // Position nodes in a circle
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 3;

    const nodePositions = {};
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        nodePositions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
        };
    });

    let svg = `<svg width="${width}" height="${height}" class="knowledge-graph-svg">`;

    // Draw edges
    edges.forEach(edge => {
        const from = nodePositions[edge.from_node_id];
        const to = nodePositions[edge.to_node_id];
        if (from && to) {
            svg += `
                <line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" 
                      class="graph-edge" stroke="#d1d5db" stroke-width="2"/>
            `;
        }
    });

    // Draw nodes
    nodes.forEach(node => {
        const pos = nodePositions[node.id];
        const color = getNodeColor(node.node_type);
        svg += `
            <g class="graph-node" transform="translate(${pos.x}, ${pos.y})">
                <circle r="20" fill="${color}" stroke="white" stroke-width="2"/>
                <text y="35" text-anchor="middle" class="graph-label">${node.name}</text>
            </g>
        `;
    });

    svg += '</svg>';
    container.innerHTML = svg;
}

function getNodeColor(type) {
    const colors = {
        'patient': '#3b82f6',
        'medication': '#10b981',
        'condition': '#f59e0b',
        'symptom': '#ef4444',
        'doctor': '#8b5cf6'
    };
    return colors[type] || '#6b7280';
}

// ==================== Conversational Query ====================

async function askQuestion(question, patientId = null) {
    const responseContainer = document.getElementById('queryResponse');
    if (!responseContainer) return;

    responseContainer.innerHTML = '<div class="loading">Processing your question...</div>';

    try {
        const response = await fetch(`${API_V2_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question,
                patient_id: patientId
            })
        });

        const data = await response.json();
        displayQueryResponse(data);
    } catch (error) {
        responseContainer.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

function displayQueryResponse(data) {
    const container = document.getElementById('queryResponse');
    if (!container) return;

    let html = `
        <div class="query-answer">
            <div class="answer-text">${formatMarkdown(data.answer)}</div>
            <div class="answer-confidence">
                <span class="confidence-label">Confidence:</span>
                <span class="confidence-value ${data.confidence > 0.7 ? 'high' : data.confidence > 0.4 ? 'medium' : 'low'}">
                    ${Math.round(data.confidence * 100)}%
                </span>
            </div>
        </div>
    `;

    if (data.evidence && data.evidence.length > 0) {
        html += `
            <div class="query-evidence">
                <h5>Evidence</h5>
                <ul>
                    ${data.evidence.map(e => `<li>${JSON.stringify(e)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (data.related_queries && data.related_queries.length > 0) {
        html += `
            <div class="related-queries">
                <h5>Related Questions</h5>
                <div class="query-suggestions">
                    ${data.related_queries.map(q => 
                        `<button class="suggestion-btn" onclick="askQuestion('${q}')">${q}</button>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function formatMarkdown(text) {
    // Simple markdown formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

// ==================== Human Review Queue ====================

async function loadReviewQueue() {
    try {
        const response = await fetch(`${API_V2_BASE}/review/queue?confidence_threshold=0.7&limit=50`);
        const data = await response.json();
        displayReviewQueue(data.items);
    } catch (error) {
        console.error('Error loading review queue:', error);
    }
}

function displayReviewQueue(items) {
    const container = document.getElementById('reviewQueueList');
    if (!container) return;

    if (!items || items.length === 0) {
        container.innerHTML = '<div class="empty-state">No items pending review</div>';
        return;
    }

    container.innerHTML = items.map(item => `
        <div class="review-item" data-id="${item.id}">
            <div class="review-item-header">
                <span class="review-type badge">${item.entity_type}</span>
                <span class="review-confidence ${item.confidence > 0.5 ? 'medium' : 'low'}">
                    ${Math.round(item.confidence * 100)}%
                </span>
            </div>
            <div class="review-item-value">${item.entity_value}</div>
            <div class="review-item-source">Source: "${item.source_text?.substring(0, 100)}..."</div>
            <div class="review-actions">
                <button class="btn btn-sm btn-success" onclick="confirmExtraction(${item.id})">
                    <i class="fas fa-check"></i> Confirm
                </button>
                <button class="btn btn-sm btn-warning" onclick="correctExtraction(${item.id})">
                    <i class="fas fa-edit"></i> Correct
                </button>
            </div>
        </div>
    `).join('');
}

async function confirmExtraction(evidenceId) {
    try {
        await fetch(`${API_V2_BASE}/review/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                evidence_id: evidenceId,
                is_correct: true,
                reviewer: 'dashboard_user'
            })
        });
        loadReviewQueue(); // Refresh
    } catch (error) {
        alert('Error confirming extraction: ' + error.message);
    }
}

function correctExtraction(evidenceId) {
    const newValue = prompt('Enter the correct value:');
    if (newValue) {
        submitCorrection(evidenceId, newValue);
    }
}

async function submitCorrection(evidenceId, correctedValue) {
    try {
        await fetch(`${API_V2_BASE}/review/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                evidence_id: evidenceId,
                is_correct: false,
                corrected_value: correctedValue,
                reviewer: 'dashboard_user'
            })
        });
        loadReviewQueue(); // Refresh
    } catch (error) {
        alert('Error submitting correction: ' + error.message);
    }
}

// ==================== Audit Trail ====================

async function loadAuditTrail(options = {}) {
    const params = new URLSearchParams();
    if (options.entity_type) params.set('entity_type', options.entity_type);
    if (options.user_name) params.set('user_name', options.user_name);
    if (options.limit) params.set('limit', options.limit);

    try {
        const response = await fetch(`${API_V2_BASE}/audit/trail?${params}`);
        const data = await response.json();
        displayAuditTrail(data.logs);
    } catch (error) {
        console.error('Error loading audit trail:', error);
    }
}

function displayAuditTrail(logs) {
    const container = document.getElementById('auditTrailList');
    if (!container) return;

    if (!logs || logs.length === 0) {
        container.innerHTML = '<div class="empty-state">No audit records found</div>';
        return;
    }

    container.innerHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Action</th>
                    <th>Entity</th>
                    <th>User</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                ${logs.map(log => `
                    <tr>
                        <td>${new Date(log.timestamp).toLocaleString()}</td>
                        <td><span class="badge badge-info">${log.action}</span></td>
                        <td>${log.entity_type} ${log.entity_id ? '#' + log.entity_id : ''}</td>
                        <td>${log.user || 'system'}</td>
                        <td>${log.action_detail || '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// ==================== Compliance Report ====================

async function loadComplianceReport() {
    try {
        const response = await fetch(`${API_V2_BASE}/compliance/report`);
        const data = await response.json();
        displayComplianceReport(data);
    } catch (error) {
        console.error('Error loading compliance report:', error);
    }
}

function displayComplianceReport(report) {
    const container = document.getElementById('complianceReport');
    if (!container) return;

    container.innerHTML = `
        <div class="compliance-summary">
            <div class="compliance-stat">
                <div class="stat-value">${report.total_events}</div>
                <div class="stat-label">Total Events</div>
            </div>
            <div class="compliance-stat">
                <div class="stat-value">${report.corrections_count}</div>
                <div class="stat-label">Corrections</div>
            </div>
            <div class="compliance-stat">
                <div class="stat-value">${Object.keys(report.events_by_user || {}).length}</div>
                <div class="stat-label">Active Users</div>
            </div>
        </div>
        <div class="compliance-breakdown">
            <h5>Events by Action</h5>
            ${Object.entries(report.events_by_action || {}).map(([action, count]) => `
                <div class="breakdown-item">
                    <span class="breakdown-label">${action}</span>
                    <span class="breakdown-value">${count}</span>
                </div>
            `).join('')}
        </div>
    `;
}

// ==================== Drug Interaction Check ====================

async function checkDrugInteractions(medications) {
    try {
        const response = await fetch(`${API_BASE}/api/documents/check-interactions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ medications })
        });
        return await response.json();
    } catch (error) {
        console.error('Error checking interactions:', error);
        return null;
    }
}

function displayInteractionResults(interactions) {
    const container = document.getElementById('interactionResults');
    if (!container) return;

    if (!interactions || interactions.length === 0) {
        container.innerHTML = `
            <div class="alert-box info">
                <span class="alert-icon">✅</span>
                <div class="alert-content">
                    <h5>No Interactions Found</h5>
                    <p>The medications appear to be safe to use together.</p>
                </div>
            </div>
        `;
        return;
    }

    container.innerHTML = interactions.map(int => {
        const severity = int.severity || 'moderate';
        const alertClass = severity === 'major' || severity === 'contraindicated' ? 'danger' : 'warning';
        return `
            <div class="alert-box ${alertClass}">
                <span class="alert-icon">${alertClass === 'danger' ? '🚨' : '⚠️'}</span>
                <div class="alert-content">
                    <h5>${int.drug1} + ${int.drug2}</h5>
                    <p><strong>Severity:</strong> ${severity.toUpperCase()}</p>
                    <p>${int.description}</p>
                    ${int.mechanism ? `<p><strong>Mechanism:</strong> ${int.mechanism}</p>` : ''}
                    ${int.management ? `<p><strong>Management:</strong> ${int.management}</p>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// ==================== Uncertainty Indicators ====================

function displayUncertaintyFlags(flags) {
    const container = document.getElementById('uncertaintyFlags');
    if (!container) return;

    if (!flags || flags.length === 0) {
        container.innerHTML = '<div class="success-message">No uncertainty flags</div>';
        return;
    }

    container.innerHTML = flags.map(flag => {
        const severityClass = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'info',
            'low': 'gray'
        }[flag.severity] || 'info';

        return `
            <div class="uncertainty-flag ${severityClass}">
                <div class="flag-header">
                    <span class="flag-field">${flag.field}</span>
                    <span class="badge badge-${severityClass}">${flag.severity}</span>
                </div>
                <p class="flag-message">${flag.message}</p>
                <div class="flag-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${flag.confidence * 100}%"></div>
                    </div>
                    <span>${Math.round(flag.confidence * 100)}%</span>
                </div>
                ${flag.alternatives && flag.alternatives.length > 0 ? `
                    <div class="flag-alternatives">
                        <span>Alternatives:</span>
                        ${flag.alternatives.map(a => `<span class="tag">${a}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// ==================== Styles ====================

const enhancedStyles = `
<style>
/* Gantt Chart Styles */
.gantt-container {
    overflow-x: auto;
    font-size: 0.875rem;
}

.gantt-header, .gantt-row {
    display: flex;
    border-bottom: 1px solid var(--gray-200);
}

.gantt-label-col {
    width: 150px;
    padding: 0.75rem;
    font-weight: 500;
    flex-shrink: 0;
}

.gantt-timeline-col {
    flex: 1;
    position: relative;
    min-width: 500px;
}

.gantt-dates {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem;
    background: var(--gray-50);
}

.gantt-bar {
    position: absolute;
    height: 24px;
    border-radius: 4px;
    top: 50%;
    transform: translateY(-50%);
}

.gantt-bar.active {
    background: linear-gradient(135deg, #10b981, #059669);
}

.gantt-bar.inactive {
    background: var(--gray-300);
}

/* Knowledge Graph Styles */
.knowledge-graph-svg {
    background: var(--gray-50);
    border-radius: 8px;
}

.graph-node circle {
    cursor: pointer;
    transition: transform 0.2s;
}

.graph-node:hover circle {
    transform: scale(1.1);
}

.graph-label {
    font-size: 11px;
    fill: var(--gray-700);
}

.graph-edge {
    stroke-opacity: 0.5;
}

/* Query Response Styles */
.query-answer {
    background: var(--gray-50);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.answer-text {
    font-size: 0.9375rem;
    line-height: 1.6;
    margin-bottom: 0.75rem;
}

.answer-confidence {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8125rem;
}

.confidence-value.high { color: var(--success); }
.confidence-value.medium { color: var(--warning); }
.confidence-value.low { color: var(--danger); }

.query-suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.suggestion-btn {
    padding: 0.5rem 0.75rem;
    background: white;
    border: 1px solid var(--gray-300);
    border-radius: 6px;
    font-size: 0.8125rem;
    cursor: pointer;
    transition: all 0.15s;
}

.suggestion-btn:hover {
    border-color: var(--primary);
    color: var(--primary);
}

/* Review Queue Styles */
.review-item {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.review-item-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}

.review-item-value {
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

.review-item-source {
    font-size: 0.75rem;
    color: var(--gray-500);
    margin-bottom: 0.75rem;
}

.review-actions {
    display: flex;
    gap: 0.5rem;
}

/* Compliance Report Styles */
.compliance-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.compliance-stat {
    background: var(--gray-50);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}

.compliance-stat .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
}

.breakdown-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--gray-100);
}

/* Uncertainty Flag Styles */
.uncertainty-flag {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.75rem;
}

.uncertainty-flag.danger { background: #fef2f2; border: 1px solid #fca5a5; }
.uncertainty-flag.warning { background: #fffbeb; border: 1px solid #fcd34d; }
.uncertainty-flag.info { background: #f0f9ff; border: 1px solid #7dd3fc; }

.flag-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.flag-field {
    font-weight: 600;
}

.flag-message {
    font-size: 0.875rem;
    color: var(--gray-600);
    margin-bottom: 0.5rem;
}

.flag-confidence {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
}

.flag-alternatives {
    margin-top: 0.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    align-items: center;
    font-size: 0.8125rem;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', enhancedStyles);

// Initialize enhanced features when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // Load review queue if on that page
    if (document.getElementById('reviewQueueList')) {
        loadReviewQueue();
    }
    
    // Load audit trail if on that page
    if (document.getElementById('auditTrailList')) {
        loadAuditTrail({ limit: 50 });
    }
});

// Export functions for global use
window.loadPatientTimeline = loadPatientTimeline;
window.loadPatientGraph = loadPatientGraph;
window.askQuestion = askQuestion;
window.loadReviewQueue = loadReviewQueue;
window.loadAuditTrail = loadAuditTrail;
window.loadComplianceReport = loadComplianceReport;
window.checkDrugInteractions = checkDrugInteractions;
