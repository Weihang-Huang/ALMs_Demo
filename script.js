/* ─────────────────────────────────────────────────────────────────────────
 * ALMs Demo – frontend logic (vanilla JS, no dependencies).
 * ───────────────────────────────────────────────────────────────────────── */

'use strict';

const API = {
  defaultAuthors: 'default_authors',
  demoTexts:      'demo_texts',
  attribute:      'attribute',
};

const state = {
  authors:   [],   // [{name, model, primer}, …]
  demoTexts: {},   // {austen: "...", …}
  lastData:  null, // last /attribute response, kept for re-rendering the viz
};

// ── Bootstrap ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await Promise.all([loadDefaultAuthors(), loadDemoTexts()]);
  } catch (err) {
    showError(`Failed to load defaults: ${err.message}`);
  }
  renderAuthorSlots();
  bindDemoButtons();
  bindAddAuthor();
  bindSubmit();
  bindAnnotationAuthorSelect();
});

// ── Data loading ──────────────────────────────────────────────────────────
async function loadDefaultAuthors() {
  const r = await fetch(API.defaultAuthors);
  if (!r.ok) throw new Error(`GET default_authors → ${r.status}`);
  state.authors = await r.json();
}

async function loadDemoTexts() {
  const r = await fetch(API.demoTexts);
  if (!r.ok) throw new Error(`GET demo_texts → ${r.status}`);
  state.demoTexts = await r.json();
}

// ── Author slots ──────────────────────────────────────────────────────────
function renderAuthorSlots() {
  const container = document.getElementById('author-slots');
  container.innerHTML = '';
  state.authors.forEach((author, idx) =>
    container.appendChild(createAuthorSlot(author, idx))
  );
}

function createAuthorSlot(author, idx) {
  const slot = document.createElement('div');
  slot.className = 'author-slot';
  slot.dataset.idx = String(idx);

  slot.innerHTML = `
    <div class="slot-header">
      <span class="slot-number">${idx + 1}</span>
      <input class="slot-name"  type="text"
             value="${escHtml(author.name  || '')}" placeholder="Author name">
      <input class="slot-model" type="text"
             value="${escHtml(author.model || '')}" placeholder="model-name">
      <button class="btn-icon" type="button" title="Remove this author">✕</button>
    </div>
    <details class="primer-details">
      <summary>Edit primer (style-conditioning excerpt)</summary>
      <textarea class="slot-primer"
                placeholder="Public-domain excerpt that conditions the base model on this author's style">${escHtml(author.primer || '')}</textarea>
    </details>
  `;

  const nameInput   = slot.querySelector('.slot-name');
  const modelInput  = slot.querySelector('.slot-model');
  const primerInput = slot.querySelector('.slot-primer');
  const removeBtn   = slot.querySelector('.btn-icon');

  nameInput  .addEventListener('input', e => state.authors[idx].name   = e.target.value);
  modelInput .addEventListener('input', e => state.authors[idx].model  = e.target.value);
  primerInput.addEventListener('input', e => state.authors[idx].primer = e.target.value);
  removeBtn  .addEventListener('click', () => {
    state.authors.splice(idx, 1);
    renderAuthorSlots();
  });

  return slot;
}

function bindAddAuthor() {
  document.getElementById('add-author').addEventListener('click', () => {
    state.authors.push({
      name:   `Author ${state.authors.length + 1}`,
      model:  'gpt-4o-mini',
      primer: '',
    });
    renderAuthorSlots();
  });
}

// ── Demo text buttons ─────────────────────────────────────────────────────
function bindDemoButtons() {
  document.querySelectorAll('.btn-demo').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.demo;
      const text = state.demoTexts[key];
      if (text) document.getElementById('questioned-text').value = text;
    });
  });
}

// ── Submit ────────────────────────────────────────────────────────────────
function bindSubmit() {
  document.getElementById('attribute-btn').addEventListener('click', doAttribute);
}

async function doAttribute() {
  const text     = document.getElementById('questioned-text').value.trim();
  const apiKey   = document.getElementById('api-key').value.trim();
  const baseUrl  = document.getElementById('base-url').value.trim();
  const tokenViz = document.getElementById('token-viz').checked;

  if (!text) return showError('Please enter or select a questioned text.');

  const authors = collectAuthors();
  if (authors.length < 2) {
    return showError('Please configure at least 2 author slots (model + name).');
  }

  setLoading(true);
  clearResults();

  try {
    const resp = await fetch(API.attribute, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        text, api_key: apiKey, base_url: baseUrl, authors,
        token_viz: tokenViz,
      }),
    });
    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      showError(data.error || `Server error (HTTP ${resp.status})`);
      if (data.details) renderErrorDetails(data.details);
      return;
    }
    state.lastData = data;
    renderResults(data);
    renderCNLLAnnotation(data);
    renderCNLLHeatmap(data);
  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function collectAuthors() {
  return state.authors
    .filter(a => (a.name || '').trim() && (a.model || '').trim())
    .map(a => ({
      name:   a.name.trim(),
      model:  a.model.trim(),
      primer: (a.primer || '').trim(),
    }));
}

// ── Results rendering ─────────────────────────────────────────────────────
function renderResults(data) {
  const { results = [], winner, errors = [] } = data;
  const section = document.getElementById('results-section');
  const content = document.getElementById('results-content');
  section.hidden = false;

  let html = '';
  if (winner) {
    html += `
      <div class="winner-banner">
        <span class="winner-label">Predicted author:</span>
        <span class="winner-name">${escHtml(winner)}</span>
        <span class="winner-note">— lowest perplexity among ${results.length} candidates</span>
      </div>
    `;
  }
  if (results.length) {
    html += `
      <table class="results-table">
        <thead><tr>
          <th>Author</th><th>Model</th><th>PPL</th>
          <th>Tokens</th><th>Method</th>
        </tr></thead><tbody>
    `;
    for (const r of results) {
      const isWin   = r.author === winner;
      const isExact = r.method && !r.method.includes('≈');
      html += `
        <tr class="${isWin ? 'winner-row' : ''}">
          <td>${escHtml(r.author)}${isWin ? '<span class="badge">✓</span>' : ''}</td>
          <td><code>${escHtml(r.model)}</code></td>
          <td class="ppl-value">${Number(r.ppl).toFixed(4)}</td>
          <td>${r.n_tokens ?? '—'}</td>
          <td>
            <span class="method-tag ${isExact ? 'exact' : 'approx'}">${escHtml(r.method || '?')}</span>
            ${r.note ? `<span class="method-note">${escHtml(r.note)}</span>` : ''}
            ${r.fallback_reason ? `<span class="method-note">Fallback used: ${escHtml(r.fallback_reason)}</span>` : ''}
          </td>
        </tr>
      `;
    }
    html += `</tbody></table>
      <div class="formula-box">
        PPL(M, Q) = exp{ −(1/T) Σ log P<sub>M</sub>(x<sub>i</sub> | x<sub>&lt;i</sub>) }
        &nbsp;·&nbsp; predicted = argmin PPL
      </div>
    `;
  }
  if (errors.length) {
    html += '<div class="error-list"><strong>Per-author errors</strong><ul>';
    for (const e of errors) {
      html += `<li><em>${escHtml(e.author)}:</em> ${escHtml(e.error)}</li>`;
    }
    html += '</ul></div>';
  }
  content.innerHTML = html;
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderErrorDetails(details) {
  const content = document.getElementById('results-content');
  if (!Array.isArray(details) || !details.length) return;
  let html = '<div class="error-list"><strong>Details</strong><ul>';
  for (const d of details) {
    html += `<li><em>${escHtml(d.author || '?')}:</em> ${escHtml(d.error || '')}</li>`;
  }
  html += '</ul></div>';
  document.getElementById('results-content').insertAdjacentHTML('beforeend', html);
}

// ── CNLL – color-coded text annotation (Fig 2 / 4) ────────────────────────
function bindAnnotationAuthorSelect() {
  document.getElementById('annotation-author').addEventListener('change', () => {
    if (state.lastData) renderCNLLAnnotation(state.lastData);
  });
}

function renderCNLLAnnotation(data) {
  const section = document.getElementById('annotation-section');
  const textDiv = document.getElementById('annotation-text');
  const sel     = document.getElementById('annotation-author');
  const { results = [], tokens = [], winner } = data;

  if (!tokens.length || !results.length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;

  // Populate the author-selector with winner first, preserving selection.
  const prev = sel.value;
  const ordered = [...results].sort((a, b) => a.ppl - b.ppl).map(r => r.author);
  sel.innerHTML = ordered.map(
    a => `<option value="${escHtml(a)}">${escHtml(a)}${a === winner ? '  (predicted)' : ''}</option>`
  ).join('');
  sel.value = ordered.includes(prev) ? prev : winner;
  const target = sel.value;

  // Determine saturation scale from the max |CNLL| *for this author*.
  let vmax = 0;
  for (const t of tokens) {
    const v = t.cnll && t.cnll[target];
    if (v != null && Math.abs(v) > vmax) vmax = Math.abs(v);
  }
  if (vmax === 0) vmax = 1;

  // Render each word as a span.  Whitespace between words is preserved.
  const pieces = [];
  for (const t of tokens) {
    const cnll = t.cnll ? t.cnll[target] : null;
    const bg = cnllToColor(cnll, vmax);
    const nllTips = t.nlls
      ? Object.entries(t.nlls)
          .map(([a, v]) => `${a}: ${v == null ? '—' : v.toFixed(3)}`)
          .join(' | ')
      : '';
    const cnllTip = cnll == null ? 'no CNLL' : `CNLL=${cnll.toFixed(3)}`;
    pieces.push(
      `<span class="tok" style="background:${bg}" ` +
      `title="${escHtml(t.word)} — ${escHtml(cnllTip)} — ${escHtml(nllTips)}">` +
      `${escHtml(t.word)}</span>`
    );
  }
  textDiv.innerHTML = pieces.join(' ');
}

// Map a CNLL value to a CSS background.  Saturation ∝ |cnll|/vmax.
// Blue = evidence for author (cnll < 0); red = evidence against (cnll > 0).
function cnllToColor(cnll, vmax) {
  if (cnll == null || !isFinite(cnll)) return 'transparent';
  const a = Math.min(1, Math.abs(cnll) / vmax);
  return cnll < 0
    ? `rgba(37, 99, 235, ${a.toFixed(3)})`    // blue
    : `rgba(220, 38, 38, ${a.toFixed(3)})`;   // red
}

// ── CNLL – heatmap (Fig 3 / 7) ────────────────────────────────────────────
function renderCNLLHeatmap(data) {
  const section = document.getElementById('heatmap-section');
  const wrap    = document.getElementById('heatmap-container');
  const meanBox = document.getElementById('heatmap-mean');
  const { results = [], tokens = [], winner } = data;

  if (!tokens.length || !results.length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;

  // Authors in winner-first order.
  const ordered = [...results].sort((a, b) => a.ppl - b.ppl).map(r => r.author);

  // Global saturation scale across the whole matrix.
  let vmax = 0;
  for (const t of tokens) {
    if (!t.cnll) continue;
    for (const a of ordered) {
      const v = t.cnll[a];
      if (v != null && Math.abs(v) > vmax) vmax = Math.abs(v);
    }
  }
  if (vmax === 0) vmax = 1;

  // Build the heatmap table.  Rows = tokens; columns = authors.
  let html = '<table class="heatmap"><thead><tr><th class="tok-col">Token</th>';
  for (const a of ordered) {
    const isWin = a === winner;
    html += `<th class="${isWin ? 'winner-col' : ''}">${escHtml(a)}${isWin ? ' ✓' : ''}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (const t of tokens) {
    html += `<tr><td class="tok-col">${escHtml(t.word)}</td>`;
    for (const a of ordered) {
      const v = t.cnll ? t.cnll[a] : null;
      const bg = cnllToColor(v, vmax);
      const isWin = a === winner;
      const tip = v == null ? '—' : `CNLL=${v.toFixed(3)}`;
      html += `<td class="${isWin ? 'winner-col' : ''}" ` +
              `style="background:${bg}" title="${escHtml(a)} · ${escHtml(t.word)} · ${escHtml(tip)}"></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // Bottom bar chart: mean CNLL per author.
  const means = {};
  for (const a of ordered) {
    const vals = tokens
      .map(t => (t.cnll ? t.cnll[a] : null))
      .filter(v => v != null);
    means[a] = vals.length ? vals.reduce((s, x) => s + x, 0) / vals.length : 0;
  }
  const mMax = Math.max(0.0001, ...Object.values(means).map(Math.abs));

  let mHtml = '<div class="mean-caption">Mean CNLL per candidate author '
            + '(most negative = strongest overall evidence for author)</div>';
  mHtml += '<div class="mean-bars">';
  for (const a of ordered) {
    const v = means[a];
    const pct = Math.min(100, (Math.abs(v) / mMax) * 100);
    const cls = v < 0 ? 'bar-for' : 'bar-against';
    const isWin = a === winner;
    mHtml += `
      <div class="mean-row">
        <div class="mean-label${isWin ? ' winner-label-row' : ''}">${escHtml(a)}${isWin ? ' ✓' : ''}</div>
        <div class="mean-track"><div class="mean-bar ${cls}" style="width:${pct}%"></div></div>
        <div class="mean-value">${v.toFixed(3)}</div>
      </div>
    `;
  }
  mHtml += '</div>';
  meanBox.innerHTML = mHtml;
}

// ── UI helpers ────────────────────────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById('attribute-btn');
  btn.disabled    = on;
  btn.textContent = on ? 'Computing perplexity…' : 'Attribute authorship';
}

function showError(msg) {
  const section = document.getElementById('results-section');
  const content = document.getElementById('results-content');
  section.hidden = false;
  content.innerHTML = `<div class="error-banner">${escHtml(msg)}</div>`;
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearResults() {
  document.getElementById('results-section').hidden    = true;
  document.getElementById('results-content').innerHTML = '';
  document.getElementById('annotation-section').hidden = true;
  document.getElementById('annotation-text').innerHTML = '';
  document.getElementById('heatmap-section').hidden    = true;
  document.getElementById('heatmap-container').innerHTML = '';
  document.getElementById('heatmap-mean').innerHTML    = '';
}

function escHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
