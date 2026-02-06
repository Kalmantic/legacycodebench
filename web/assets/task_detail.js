/**
 * LegacyCodeBench - Task Drill-Down Logic
 */

// Utility: Get query params
function getQueryParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// Format score (scores are 0-1 scale in JSON, multiply by 100 for display)
function formatScore(score) {
    if (score === null || score === undefined) return '-';
    return Math.round(score * 100) + '%';
}

// Render badges
function renderMethodBadge(method) {
    let color = '#6b7280'; // default gray
    let bg = '#f3f4f6';

    if (method === 'executed') {
        color = '#059669'; // success
        bg = '#ecfdf5';
    } else if (method === 'static') {
        color = '#d97706'; // warning/mid
        bg = '#fffbeb';
    } else if (method === 'error') {
        color = '#dc2626'; // red
        bg = '#fef2f2';
    }

    return `<span style="color: ${color}; background: ${bg}; padding: 2px 8px; border-radius: 4px; font-weight: 500; font-size: 12px; text-transform: uppercase;">${method}</span>`;
}

// Render row
function renderTaskRow(task) {
    // Format critical failures if any
    const cfBadges = task.critical_failures && task.critical_failures.length > 0
        ? task.critical_failures.map(cf =>
            `<span style="color: #dc2626; background: #fef2f2; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-right: 2px;">${cf}</span>`
        ).join('')
        : '';

    // Language badge
    const langBadge = task.language === 'UniBasic'
        ? '<span style="color: #7c3aed; background: #f5f3ff; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-left: 4px;">UB</span>'
        : '';

    return `
    <tr>
      <td style="font-family: var(--font-mono); font-weight: 500;">${task.task_id}${langBadge}</td>
      <td>${task.tier}</td>
      <td class="num" style="font-weight: 600;">${formatScore(task.bf_score)}</td>
      <td>${renderMethodBadge(task.verification_mode)}</td>
      <td style="color: var(--text-secondary); font-size: 13px;">${task.mode_reason || '-'}</td>
      <td style="color: var(--text-secondary); font-size: 13px;">${task.details || cfBadges || '-'}</td>
    </tr>
  `;
}

// Main loader
async function loadTaskDetails() {
    const modelName = getQueryParam('model');
    if (!modelName) {
        document.getElementById('model-title').textContent = 'Error: No model specified';
        return;
    }

    try {
        const paths = [
            'leaderboard_v2.0.json',
            'leaderboard_v1.0.json',
            'leaderboard.json',
            'results/leaderboard.json',
            '../results/leaderboard.json'
        ];
        let data = null;

        for (const path of paths) {
            try {
                const res = await fetch(path);
                if (res.ok) {
                    data = await res.json();
                    console.log('Loaded leaderboard from:', path);
                    break;
                }
            } catch (e) { }
        }

        if (!data) throw new Error('Could not load data');

        // Find model (decode URL parameter) - Case insensitive match
        const decodedModel = decodeURIComponent(modelName).toLowerCase();

        let entry = data.leaderboard.find(e => e.model.toLowerCase() === decodedModel);

        // Fallback: Check "submitter" if model match fails (handles edge cases where name displayed was submitter)
        if (!entry) {
            entry = data.leaderboard.find(e => e.submitter.toLowerCase() === decodedModel);
        }

        if (!entry) throw new Error(`Model "${modelName}" not found in leaderboard`);

        // Format model name for display
        const displayName = entry.submitter && entry.submitter !== entry.model && entry.submitter !== 'Unknown'
            ? `${entry.submitter} / ${entry.model}`
            : entry.model;

        // Check for Language Context
        const langContext = getQueryParam('lang'); // 'COBOL', 'UniBasic', or null

        // Update Header
        let titleSuffix = '';
        if (langContext) titleSuffix = ` (${langContext})`;
        document.getElementById('model-title').textContent = `${displayName}${titleSuffix} - Task Details`;

        // Filter tasks if context provided
        let displayTasks = entry.tasks || [];
        if (langContext && displayTasks.length > 0) {
            displayTasks = displayTasks.filter(t => t.language === langContext);
        }

        // Recalculate stats for the specific view if filtered
        let lcbDisplay = entry.avg_lcb_score;
        let bfDisplay = entry.avg_bf;
        let executed = entry.executed || 0;
        let staticCount = entry.static || 0;
        let totalCount = entry.tasks_total || 0;

        if (langContext && displayTasks.length > 0) {
            totalCount = displayTasks.length;
            executed = displayTasks.filter(t => t.verification_mode === 'executed').length;
            staticCount = displayTasks.filter(t => t.verification_mode === 'static').length;

            // Recalculate averages for this subset
            lcbDisplay = displayTasks.reduce((acc, t) => acc + (t.lcb_score || 0), 0) / totalCount;
            bfDisplay = displayTasks.reduce((acc, t) => acc + (t.bf_score || 0), 0) / totalCount;
        }

        // Render Stats
        const statsHtml = `
      <div class="stat">
        <div class="stat-value">${formatScore(lcbDisplay)}</div>
        <div class="stat-label">LCB Score</div>
      </div>
      <div class="stat">
        <div class="stat-value">${formatScore(bfDisplay)}</div>
        <div class="stat-label">BF Score</div>
      </div>
      <div class="stat">
        <div class="stat-value">${executed}</div>
        <div class="stat-label">Executed</div>
      </div>
      <div class="stat">
        <div class="stat-value">${staticCount}</div>
        <div class="stat-label">Static</div>
      </div>
      <div class="stat">
        <div class="stat-value">${totalCount}</div>
        <div class="stat-label">Total Tasks</div>
      </div>
    `;
        document.getElementById('model-stats').innerHTML = statsHtml;

        // Render Table
        const tbody = document.getElementById('tasks-body');
        if (displayTasks.length > 0) {
            tbody.innerHTML = displayTasks.map(renderTaskRow).join('');
        } else {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem; color: var(--text-light);">
                No detailed task data available for this model${langContext ? ' in ' + langContext : ''}.<br>
                <span style="font-size: 12px;">Run evaluation with V3 evaluator to generate per-task details.</span>
            </td></tr>`;
        }

    } catch (err) {
        console.error(err);
        document.getElementById('tasks-body').innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem; color: red;">Failed to load details: ${err.message}</td></tr>`;
    }
}

document.addEventListener('DOMContentLoaded', loadTaskDetails);
