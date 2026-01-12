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
  };
  return nameMap[model] || model;
}

// Get organization for a model
function getOrg(model) {
  return MODEL_ORGS[model] || 'Unknown';
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
function renderRow(entry, rank) {
  const lcbScore = formatScore(entry.avg_lcb_score);
  const sc = formatScore(entry.avg_sc);
  const dq = formatScore(entry.avg_sq || entry.avg_dq || 0); // DQ (was SQ)
  const bf = formatScore(entry.avg_bf);

  // Get tier scores
  const t1Score = entry.by_tier && entry.by_tier.T1 && entry.by_tier.T1.avg_score > 0
    ? formatScore(entry.by_tier.T1.avg_score)
    : '-';
  const t4Score = entry.by_tier && entry.by_tier.T4 && entry.by_tier.T4.avg_score > 0
    ? formatScore(entry.by_tier.T4.avg_score)
    : '-';

  // Determine color classes
  const lcbClass = getScoreClass(entry.avg_lcb_score);
  const t1Class = entry.by_tier && entry.by_tier.T1 && entry.by_tier.T1.avg_score > 0 ? getScoreClass(entry.by_tier.T1.avg_score) : '';
  const t4Class = entry.by_tier && entry.by_tier.T4 && entry.by_tier.T4.avg_score > 0 ? getScoreClass(entry.by_tier.T4.avg_score) : '';

  return `
    <tr>
      <td>${renderRank(rank)}</td>
      <td>
        <div class="model-name">${formatModelName(entry.model)}</div>
        <div class="model-org">${getOrg(entry.model)}</div>
      </td>
      <td class="center"><span class="score-primary ${lcbClass}">${lcbScore}%</span></td>
      <td class="num">${sc}%</td>
      <td class="num">${dq}%</td>
      <td class="num">${bf}%</td>
      <td class="num"><span class="${t1Class}">${t1Score}${t1Score !== '-' ? '%' : ''}</span></td>
      <td class="num"><span class="${t4Class}">${t4Score}${t4Score !== '-' ? '%' : ''}</span></td>
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

  // Render rows
  const rows = leaderboard.map((entry, index) => renderRow(entry, entry.rank || index + 1)).join('');
  tbody.innerHTML = rows;

  // Update stats
  const totalModels = document.getElementById('total-models');
  if (totalModels) {
    totalModels.textContent = data.total_models || leaderboard.length;
  }

  // Update metadata
  const metaUpdated = document.getElementById('leaderboard-updated');
  if (metaUpdated && data.generated_at) {
    const date = new Date(data.generated_at);
    metaUpdated.textContent = `Last updated: ${date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}`;
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

// Fetch and render leaderboard data
async function loadLeaderboard() {
  showLoading();

  try {
    // Try multiple paths for the leaderboard file
    const paths = [
      '../results/leaderboard.json',
      'results/leaderboard.json',
      '/results/leaderboard.json',
    ];

    let data = null;

    for (const path of paths) {
      try {
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
      renderLeaderboard(data);
    } else {
      showError('No leaderboard data available. Run evaluations to generate results.');
    }
  } catch (error) {
    console.error('Failed to load leaderboard:', error);
    showError('Failed to load leaderboard data');
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', loadLeaderboard);
