/**
 * LegacyCodeBench Leaderboard - Dynamic Data Loader
 * Loads and displays evaluation metrics from leaderboard.json
 */

// Model organization mapping
const MODEL_ORGS = {
  'claude-sonnet-4': 'Anthropic',
  'claude-3-opus': 'Anthropic',
  'claude-3-sonnet': 'Anthropic',
  'gpt-4o': 'OpenAI',
  'gpt-4': 'OpenAI',
  'gpt-4-turbo': 'OpenAI',
  'gpt-3.5-turbo': 'OpenAI',
  'docmolt-gpt4o': 'Hexaview',
  'docmolt-gpt4o-mini': 'Hexaview',
  'docmolt-claude': 'Hexaview',
  'Legacy Insights': 'Hexaview',
  'gemini-2.0-flash': 'Google',
  'gemini-2.5-flash': 'Google',
  'gemini-pro': 'Google',
  'llama-3.1-70b': 'Meta',
  'llama-3.1-405b': 'Meta',
  'aws-transform': 'AWS',
  'aws-transform-mainframe': 'AWS',
  'ibm-granite-13b': 'IBM',
};

// Format model name for display
function formatModelName(model) {
  const nameMap = {
    'claude-sonnet-4': 'Claude Sonnet 4',
    'claude-3-opus': 'Claude 3 Opus',
    'gpt-4o': 'GPT-4o',
    'gpt-4': 'GPT-4',
    'docmolt-gpt4o': 'DocMolt GPT-4o',
    'docmolt-gpt4o-mini': 'DocMolt Mini',
    'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'llama-3.1-70b': 'Llama 3.1 70B',
    'aws-transform': 'AWS Transform',
    'aws-transform-mainframe': 'AWS Transform (Mainframe)',
    'ibm-granite-13b': 'IBM Granite 13B',
    'legacy-insights': 'Legacy Insights',
    'Legacy Insights': 'Legacy Insights',
    'Uniview': 'Uniview',
    'uniview': 'Uniview',
  };
  return nameMap[model] || model;
}

// Get organization for a model
function getOrg(entry) {
  if (entry.submitter && entry.submitter !== 'Unknown') {
    return entry.submitter;
  }
  return MODEL_ORGS[entry.model] || 'Unknown';
}

// Format score as percentage
function formatScore(score) {
  return Math.round(score * 100);
}

// Get score color class
function getScoreClass(score) {
  const pct = score * 100;
  if (pct >= 70) return 'score-good';
  if (pct >= 40) return 'score-mid';
  return 'score-low';
}

// Render rank badge
function renderRank(rank) {
  const rankClass = rank <= 3 ? ` rank-${rank}` : '';
  return `<span class="rank${rankClass}">${rank}</span>`;
}

// Render a single leaderboard row
function renderRow(entry, rank, version) {
  const lcbScore = formatScore(entry.avg_lcb_score);
  const sc = formatScore(entry.avg_sc);
  const dq = formatScore(entry.avg_sq || entry.avg_dq || 0); // DQ (was SQ)
  const bf = formatScore(entry.avg_bf);

  // V2.4/V3 Columns - Show T1/T4 Scores instead of Exec/Static
  let t1Val = entry.score_t1;
  if (t1Val === undefined && entry.by_tier && entry.by_tier.T1) {
    t1Val = entry.by_tier.T1.avg_score;
  }
  const t1 = t1Val !== undefined ? formatScore(t1Val) + '%' : '-';

  let t4Val = entry.score_t4;
  if (t4Val === undefined && entry.by_tier && entry.by_tier.T4) {
    t4Val = entry.by_tier.T4.avg_score;
  }
  const t4 = t4Val !== undefined ? formatScore(t4Val) + '%' : '-';

  // Render languages as badges
  const languages = entry.languages && entry.languages.length > 0
    ? entry.languages.map(l =>
      `<span style="font-size: 10px; background: var(--bg-tertiary); padding: 2px 4px; border-radius: 3px; margin-right: 2px;">${l}</span>`
    ).join('')
    : '<span style="font-size: 10px; background: var(--bg-tertiary); padding: 2px 4px; border-radius: 3px;">COBOL</span>';

  // Determine color classes
  const lcbClass = getScoreClass(entry.avg_lcb_score);

  // Conditional Drill-Down Link
  const modelNameDisplay = formatModelName(entry.model);
  const nameCellContent = version === 'v1.0'
    ? `<span style="border-bottom: 1px dotted var(--text-secondary); cursor: not-allowed;" title="Drill-down not available for v1.0">${modelNameDisplay}</span>`
    : `<a href="task_detail.html?model=${encodeURIComponent(entry.model)}" style="color: inherit; text-decoration: none; border-bottom: 1px dotted var(--text-secondary);">${modelNameDisplay}</a>`;

  return `
    <tr>
      <td>${renderRank(rank)}</td>
      <td>
        <div class="model-name">
          ${nameCellContent}
        </div>
        <div class="model-org">${getOrg(entry)}</div>
      </td>
      <td class="center"><span class="score-primary ${lcbClass}">${lcbScore}%</span></td>
      <td class="num">${sc}%</td>
      <td class="num">${dq}%</td>
      <td class="num">${bf}%</td>
      <td class="num">${t1}</td>
      <td class="num">${t4}</td>
      <td class="center">${languages}</td>
    </tr>
  `;
}

// Render the full leaderboard table
function renderLeaderboard(data) {
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) {
    console.error('Leaderboard tbody not found');
    return;
  }

  // Data is already sorted by rank in leaderboard.json
  const leaderboard = data.leaderboard;

  // Filter logic
  const filter = document.getElementById('language-filter');
  const selectedLang = filter ? filter.value : 'All';

  // Show/Hide warning - Logic updated: No longer unsound if we calculate dynamically!
  const warning = document.getElementById('lang-warning');
  if (warning) {
    // With dynamic calculation, the comparison IS sound for that language subset.
    // So we can hide the "statistically unsound" warning or change it to an informational note.
    if (selectedLang !== 'All') {
      warning.style.display = 'block';
      warning.className = 'callout'; // reset style
      // Use existing styles if possible or inline
      warning.style.backgroundColor = '#eff6ff'; // blueish
      warning.style.borderLeftColor = '#2563eb';
      warning.style.color = '#1e3a8a';
      warning.innerHTML = `<strong>Note:</strong> Showing specific scores for <strong>${selectedLang}</strong> tasks only.`;
    } else {
      warning.style.display = 'none';
    }
  }

  // Process rows with dynamic calculation
  let processedRows = leaderboard.map((originalEntry) => {
    let entry = originalEntry;

    // If filtering, calculate specific scores
    if (selectedLang !== 'All' && originalEntry.tasks) {
      const filteredTasks = originalEntry.tasks.filter(t => t.language === selectedLang);

      if (filteredTasks.length === 0) return null; // Model skipped

      // Calculate averages for this subset
      const total = filteredTasks.length;
      const lcbSum = filteredTasks.reduce((sum, t) => sum + (t.lcb_score || 0), 0);
      const scSum = filteredTasks.reduce((sum, t) => sum + (t.sc_score || 0), 0);
      const dqSum = filteredTasks.reduce((sum, t) => sum + (t.dq_score || 0), 0);
      const bfSum = filteredTasks.reduce((sum, t) => sum + (t.bf_score || 0), 0);

      const executedCount = filteredTasks.filter(t => t.verification_mode === 'executed').length;
      const staticCount = filteredTasks.filter(t => t.verification_mode === 'static').length;

      // Recalculate Tier Scores for Subset
      const t1Tasks = filteredTasks.filter(t => t.tier === 'Easy' || t.task_id.includes('-T1-'));
      const t4Tasks = filteredTasks.filter(t => t.tier === 'Hard' || t.task_id.includes('-T4-'));

      const t1Score = t1Tasks.length > 0
        ? t1Tasks.reduce((sum, t) => sum + (t.lcb_score || 0), 0) / t1Tasks.length
        : 0;

      const t4Score = t4Tasks.length > 0
        ? t4Tasks.reduce((sum, t) => sum + (t.lcb_score || 0), 0) / t4Tasks.length
        : 0;

      // Create virtual entry
      entry = {
        ...originalEntry,
        avg_lcb_score: lcbSum / total,
        avg_sc: scSum / total,
        avg_dq: dqSum / total,
        avg_bf: bfSum / total,
        executed: executedCount,
        static: staticCount,
        score_t1: t1Score, // valid for this subset
        score_t4: t4Score  // valid for this subset
      };
    } else if (selectedLang !== 'All' && !originalEntry.tasks) {
      if (!originalEntry.languages || !originalEntry.languages.includes(selectedLang)) return null;
    }

    return entry;
  }).filter(e => e !== null);

  // Sort by LCB Score Descending
  processedRows.sort((a, b) => b.avg_lcb_score - a.avg_lcb_score);

  const currentVersion = data.version; // Ensure data.version exists in JSON, typically "v1.0" or "v2.0"

  let renderedCount = 0;
  const rows = processedRows.map((entry) => {
    renderedCount++;
    return renderRow(entry, renderedCount, currentVersion);
  }).join('');

  if (renderedCount === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 2rem;">No models found for ${selectedLang}</td></tr>`;
  } else {
    tbody.innerHTML = rows;
  }

  // Update Stats
  const totalModelsEl = document.getElementById('total-models');
  if (totalModelsEl) {
    totalModelsEl.textContent = renderedCount;
  }

  const totalTasksEl = document.getElementById('total-tasks');
  if (totalTasksEl) {
    // Hardcoded counts per tier definition
    totalTasksEl.textContent = selectedLang === 'UniBasic' ? '50' : '200';
  }
}

// Show error state
function showError(message) {
  const tbody = document.getElementById('leaderboard-body');
  if (tbody) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align: center; padding: 2rem; color: var(--text-light);">
          ${message}
        </td>
      </tr>
    `;
  }
}

// Show loading state
function showLoading() {
  const tbody = document.getElementById('leaderboard-body');
  if (tbody) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align: center; padding: 2rem; color: var(--text-light);">
          Loading results...
        </td>
      </tr>
    `;
  }
}


// Store data globally
let globalLeaderboardData = null;

// Fetch and render leaderboard data
async function loadLeaderboard(version = 'v2.0') {
  showLoading();

  try {
    // Construct path based on version
    // version is like "v1.0" or "v2.0"
    const timestamp = new Date().getTime();
    const filename = `leaderboard_${version}.json?t=${timestamp}`;

    // Try paths
    const paths = [
      `../${filename}`,
      filename,
      `/${filename}`
    ];

    let data = null;

    for (const path of paths) {
      try {
        console.log(`Attempting to load ${path}...`);
        const response = await fetch(path);
        if (response.ok) {
          data = await response.json();
          console.log('Loaded leaderboard from:', path);
          break;
        }
      } catch (e) {
        // Try next path
      }
    }

    if (data && data.leaderboard && data.leaderboard.length > 0) {
      globalLeaderboardData = data;
      renderLeaderboard(data);

      // Update info text
      const info = document.getElementById('leaderboard-updated');
      if (info) {
        // Format date if available
        let dateStr = '';
        if (data.generated_at) {
          const date = new Date(data.generated_at);
          dateStr = ` (Last updated: ${date.toLocaleDateString()})`;
        }
        info.textContent = `Results: ${version}${dateStr}`;
      }
    } else {
      showError(`No results found for ${version}.`);
    }
  } catch (error) {
    console.error('Failed to load leaderboard:', error);
    showError('Failed to load leaderboard data');
  }
}

// Initialize when DOM is ready
// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Check for URL param ?version=v1.0
  const urlParams = new URLSearchParams(window.location.search);
  const initialVersion = urlParams.get('version') || 'v2.0';

  // Set dropdown initial value if exists
  const vFilter = document.getElementById('version-filter');
  if (vFilter) {
    vFilter.value = initialVersion;
    vFilter.addEventListener('change', (e) => {
      loadLeaderboard(e.target.value);
    });
  }

  loadLeaderboard(initialVersion);

  // Button Toggle Logic
  const langButtons = document.querySelectorAll('.lang-btn');
  const langFilter = document.getElementById('language-filter');

  if (langButtons.length > 0 && langFilter) {
    langButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        // Update Active UI
        langButtons.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');

        // Update Hidden Input
        const lang = e.target.getAttribute('data-lang');
        langFilter.value = lang;

        // Trigger Render
        if (globalLeaderboardData) {
          renderLeaderboard(globalLeaderboardData);
        }
      });
    });
  }
});

