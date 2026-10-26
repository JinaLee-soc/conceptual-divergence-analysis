'use strict';

const labels = ['Corpus', 'Shared clusters', 'Meaning and evidence', 'Review'];
const state = { cases: [], current: null, step: 0, timer: null };
const $ = (id) => document.getElementById(id);
const escapeText = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));

function pause() {
  window.clearInterval(state.timer);
  state.timer = null;
  $('play').textContent = state.step === labels.length - 1 ? 'Replay walkthrough' : 'Play walkthrough';
  $('play').setAttribute('aria-pressed', 'false');
}

function showStep(step) {
  state.step = step;
  const c = state.current;
  const header = `<p class="stage-label">STAGE ${step + 1} OF ${labels.length} · SAVED RESULTS</p>`;
  let content = '';
  if (step === 0) {
    content = `<h2 id="stage-title">Start with two uses of “${escapeText(c.term)}”</h2>
      <p class="lead">The same word can do different work in a draft and in the literature.</p>
      <div class="split"><div class="fact"><span class="count">${c.researcher}</span><strong>Fictional researcher passages</strong><p class="muted">Sentences extracted from synthetic drafts that explicitly use the term.</p></div>
      <div class="fact"><span class="count">${c.literature}</span><strong>Toy literature passages</strong><p class="muted">Term passages from the public abstract corpus used in this saved example.</p></div></div>
      <p>These are passage counts, not numbers of papers. Both sets enter one shared clustering step.</p>`;
  } else if (step === 1) {
    const max = Math.max(...c.clusters.map((row) => row.researcher + row.literature));
    const chart = c.clusters.map((row) => `<div class="cluster"><div class="cluster-label"><strong>Cluster ${row.id}</strong><span>${row.flag === '—' ? 'No concentration or absence flag' : 'Absent from researcher usage'}</span></div>
      <div class="bar" aria-hidden="true"><span class="bar-draft" style="width:${row.researcher / max * 100}%"></span><span class="bar-lit" style="width:${row.literature / max * 100}%"></span></div>
      <p class="cluster-counts">${row.researcher} researcher · ${row.literature} literature · ${((row.researcher / (row.researcher + row.literature)) * 100).toFixed(1)}% researcher share</p></div>`).join('');
    content = `<h2 id="stage-title">Locate the passages in shared clusters</h2>
      <p>Joint spherical k-means selected <strong>${c.k} clusters</strong> in this saved run. Bar length shows the number of passages on a common scale.</p>
      <div class="legend"><span><i class="swatch"></i>Fictional researcher</span><span><i class="swatch literature"></i>Literature</span></div>${chart}
      <p class="note">${c.term === 'novelty' ? 'Both clusters contain both kinds of passages. No structural flag does not mean that the term has the same meaning on both sides.' : 'Cluster 1 contains 18 literature passages and no researcher passages. This marks an absence in the draft, not an error by itself.'}</p>`;
  } else if (step === 2) {
    content = `<h2 id="stage-title">Read what the shared space means</h2><p class="muted">Interpretation from the saved demonstration response</p><p class="lead">${escapeText(c.summary)}</p>
      <h3>Example passages cited in the report</h3>${c.evidence.map((q) => `<div class="evidence-row"><span class="muted">${escapeText(q.corpus)}</span><blockquote>“${escapeText(q.text)}”</blockquote><p class="citation">${escapeText(q.source)}</p></div>`).join('')}
      <details><summary>Read the divergence interpretation</summary>${c.interpretationHtml}</details>`;
  } else {
    content = `<h2 id="stage-title">Review the suggested changes</h2><p>The saved response suggests these edits to the fictional drafts. They are illustrative recommendations for a researcher to assess.</p>${c.recommendationsHtml}
      <p class="note">The number of clusters recurred in only ${c.stability}% of shuffled orderings. Treat this as one plausible partition; quotation matching alone does not validate the interpretation.</p>`;
  }
  $('stage').innerHTML = header + content;
  $('steps').querySelectorAll('button').forEach((button, i) => {
    if (i === step) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  $('previous').disabled = step === 0;
  $('next').disabled = step === labels.length - 1;
  $('play-status').textContent = `${labels[step]} · ${step + 1} / ${labels.length}`;
  if (step === labels.length - 1 || state.timer === null) pause();
}

function selectCase(term) {
  pause();
  state.current = state.cases.find((c) => c.term === term);
  const c = state.current;
  $('case-meta').textContent = `${c.researcher + c.literature} passages · ${c.k} shared clusters · 18 Oct 2026`;
  $('checks').innerHTML = `<div><p><span class="check-value">${c.stability}%</span>cluster-count stability</p><p class="muted">The selected k matched in ${c.stability}% of ${c.trials} shuffled orderings; k ranged ${escapeText(c.kRange)}. Pooled silhouette: ${c.silhouette.toFixed(3)}.</p></div><div><p><span class="check-value">${c.grounded}/${c.quotations}</span>quoted passages matched</p><p class="muted">The saved check matched quotations against prompt passages. This does not establish that a quotation supports an interpretation.</p></div>`;
  $('report').innerHTML = c.reportHtml;
  $('download-md').href = c.markdown;
  $('source').textContent = `Source: ${c.source}`;
  $('hash').textContent = `Source SHA-256: ${c.sha256}`;
  showStep(0);
  pause();
}

async function init() {
  try {
    const response = await fetch('data.json');
    if (!response.ok) throw new Error('Saved examples unavailable');
    const data = await response.json();
    if (data.schemaVersion !== 1 || data.cases.length !== 2) throw new Error('Unexpected demo format');
    state.cases = data.cases;
    labels.forEach((label, i) => {
      const button = document.createElement('button');
      button.textContent = `${i + 1}. ${label}`;
      button.addEventListener('click', () => { pause(); showStep(i); });
      $('steps').append(button);
    });
    $('case-select').addEventListener('change', (event) => selectCase(event.target.value));
    $('previous').addEventListener('click', () => { pause(); showStep(Math.max(0, state.step - 1)); });
    $('next').addEventListener('click', () => { pause(); showStep(Math.min(labels.length - 1, state.step + 1)); });
    $('play').addEventListener('click', () => {
      if (state.timer !== null) { pause(); return; }
      if (state.step === labels.length - 1) showStep(0);
      $('play').textContent = 'Pause walkthrough';
      $('play').setAttribute('aria-pressed', 'true');
      state.timer = window.setInterval(() => showStep(state.step + 1), 10000);
    });
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') pause(); });
    document.querySelector('.masthead a').addEventListener('click', () => { pause(); $('full-report').open = true; });
    document.querySelectorAll('details').forEach((details) => details.addEventListener('toggle', () => { if (details.open) pause(); }));
    selectCase('novelty');
    $('load-status').hidden = true;
    $('demo').hidden = false;
  } catch (error) {
    $('load-status').textContent = 'The saved examples could not load. Reload the page, or download novelty.md or consensus.md from the downloads folder.';
  }
}
init();
