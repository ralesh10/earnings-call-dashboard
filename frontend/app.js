const state = {
  data: null,
  activeModel: null,
  currentCall: null,
  filters: { dir: 'ALL', conf: 'ALL', status: 'VALIDATED' },
  page: 1,
  pageSize: 20,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatPercent(value, signed = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Unavailable';
  const number = Number(value) * 100;
  return `${signed && number > 0 ? '+' : ''}${number.toFixed(1)}%`;
}

function formatPoints(value, signed = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Unavailable';
  const number = Number(value) * 100;
  return `${signed && number > 0 ? '+' : ''}${number.toFixed(1)} pts`;
}

function formatMetric(value) {
  return value === null || value === undefined ? '—' : Number(value).toFixed(3);
}

function activeModelMeta() {
  return state.data.models.find((model) => model.key === state.activeModel) || state.data.models[0];
}

function reliabilityKey() {
  return state.activeModel === 'sentence_hist'
    ? 'sentence_plus_historical_xgboost_depth1_trees100'
    : 'original_logistic';
}

function activeReliability() {
  return state.data.reliability.models.find((model) => model.key === reliabilityKey());
}

function modelFor(call) {
  return call?.models?.[state.activeModel] || null;
}

function statusCategory(status) {
  if (status === 'Out-of-sample holdout' || status === 'Walk-forward validated') return 'VALIDATED';
  if (status === 'Retrospective inference') return 'EXPLORATORY';
  return 'UNAVAILABLE';
}

function statusCategoryLabel(status) {
  if (statusCategory(status) === 'VALIDATED') return 'Validated';
  if (statusCategory(status) === 'EXPLORATORY') return 'Exploratory';
  return 'Unavailable';
}

function statusBadge(status) {
  const category = statusCategory(status);
  const className = category === 'VALIDATED' ? 'up' : category === 'EXPLORATORY' ? 'neutral' : 'muted';
  return `<span class="badge ${className}" title="${escapeHtml(status || 'Unavailable')}">${statusCategoryLabel(status)}</span>`;
}

function signalBadge(model) {
  if (!model || model.prob === null || model.prob === undefined) return '<span class="badge muted">Unavailable</span>';
  const className = model.tone === 'positive' ? 'up' : model.tone === 'negative' ? 'down' : 'neutral';
  const label = model.signal === 'Positive' ? '▲ Positive' : model.signal === 'Negative' ? '▼ Negative' : '• No clear signal';
  const context = `Model probability ${formatPercent(model.prob)}; typical positive-return rate ${formatPercent(model.baseRate)}; difference ${formatPoints(model.differenceFromBaseRate, true)}. ${model.confidenceDescription || model.explanation}`;
  return `<span class="badge ${className}" title="${escapeHtml(context)}">${label} · ${formatPercent(model.prob)}</span>`;
}

function setActiveTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach((button) => {
    const isActive = button.getAttribute('onclick')?.includes(`'${tabId}'`);
    button.classList.toggle('active', Boolean(isActive));
  });
  document.querySelectorAll('.view').forEach((view) => view.classList.remove('active'));
  $(`view-${tabId}`)?.classList.add('active');
}

function switchTab(tabId) {
  setActiveTab(tabId);
  if (tabId === 'screener') renderScreener();
  if (tabId === 'detail') {
    initDetailControls();
    renderDetailView();
  }
  if (tabId === 'reliability') renderReliability();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function populateModelSelector() {
  const select = $('globalModelSelect');
  select.innerHTML = state.data.models.map((model) => (
    `<option value="${escapeHtml(model.key)}">${escapeHtml(model.label)}</option>`
  )).join('');
  select.value = state.activeModel;
}

function updateModelStatus() {
  const meta = activeReliability();
  const metric = meta?.walkForwardAuc ?? meta?.holdoutAuc;
  const label = meta?.walkForwardAuc !== null && meta?.walkForwardAuc !== undefined
    ? 'Walk-forward AUC'
    : 'Latest holdout AUC';
  $('modelStatusBadge').innerText = `${label}: ${formatMetric(metric)}`;
  $('screenerActiveModelLabel').innerText = `Model: ${activeModelMeta().displayName}`;
}

function onModelChange() {
  state.activeModel = $('globalModelSelect').value;
  state.filters.status = state.activeModel === 'finbert' ? 'EXPLORATORY' : 'VALIDATED';
  state.page = 1;
  updateModelStatus();
  updateFilterUI();
  renderOverview();
  renderScreener();
  initDetailControls();
  renderDetailView();
  renderReliability();
  buildTickerTape();
}

function setFilter(type, value, button) {
  state.filters[type] = value;
  state.page = 1;
  document.querySelectorAll(`[data-filter-type="${type}"]`).forEach((pill) => pill.classList.toggle('active', pill.dataset.filterValue === value));
  renderScreener();
}

function buildTickerTape() {
  const scored = state.data.calls.filter((call) => modelFor(call)?.prob !== null && modelFor(call)?.prob !== undefined).slice(0, 16);
  const html = scored.map((call) => {
    const model = modelFor(call);
    const isUp = model.tone === 'positive';
    const isDown = model.tone === 'negative';
    const dirClass = isUp ? 'up' : isDown ? 'down' : '';
    const mark = isUp ? '▲' : isDown ? '▼' : '•';
    return `<div class="tape-item"><b>${escapeHtml(call.sym)}</b> <span class="dir ${dirClass}">${mark} ${formatPercent(model.prob)}</span> (${escapeHtml(call.year)} Q${escapeHtml(call.q)})</div>`;
  }).join('');
  $('tickerTape').innerHTML = html ? html + html : '<div class="tape-item">No scored calls available in the selected artifact.</div>';
}

function updateSpotlight() {
  const meta = activeModelMeta();
  const baseRate = Number(meta?.baseRate ?? 0.5);
  const candidates = state.data.calls
    .map((call) => ({ call, model: modelFor(call) }))
    .filter(({ model }) => model && model.prob !== null && statusCategory(model.status) === 'VALIDATED')
    .sort((a, b) => Math.abs(b.model.prob - baseRate) - Math.abs(a.model.prob - baseRate));
  const selected = candidates[0] || state.data.calls.map((call) => ({ call, model: modelFor(call) })).find(({ model }) => model?.prob !== null);
  if (!selected) {
    $('spotlightCompany').innerText = 'No scored calls available';
    $('spotlightMeta').innerText = 'Choose an artifact with usable predictions.';
    $('spotlightSignal').innerText = 'Unavailable';
    $('spotlightRationale').innerText = 'The selected artifact does not contain a usable prediction for the available calls.';
    $('spotlightFeatures').innerHTML = '';
    return;
  }
  const { call, model } = selected;
  $('spotlightCompany').innerText = `${call.sym} — ${call.co}`;
  $('spotlightMeta').innerText = `${call.year} Q${call.q} · ${call.date} · ${call.timing} · ${statusCategoryLabel(model.status)} · ${model.status}`;
  $('spotlightSignal').className = `badge ${model.tone === 'positive' ? 'up' : model.tone === 'negative' ? 'down' : 'neutral'}`;
  $('spotlightSignal').innerText = `${model.signal} · ${formatPercent(model.prob)}`;
  $('spotlightRationale').innerText = model.explanation;
  $('spotlightFeatures').innerHTML = (model.featureBars || []).slice(0, 3).map((feature) => `
    <div>
      <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
        <span>${escapeHtml(feature.label)} <span class="infoicon">i<span class="tip">${escapeHtml(feature.description)}</span></span></span>
        <span class="text-mono text-${escapeHtml(feature.color)}">${escapeHtml(feature.display)}</span>
      </div>
      <div class="f-bar-track"><div class="f-bar-fill" style="width:${feature.width}%; background:var(--${escapeHtml(feature.color)});"></div></div>
    </div>`).join('');
  $('spotlightOpen').onclick = () => {
    loadDetail(call.sym, `${call.year}-Q${call.q}`);
    switchTab('detail');
  };
}

function renderOverview() {
  const meta = activeModelMeta();
  const reliability = activeReliability();
  $('statCalls').innerText = Number(meta?.events || 0).toLocaleString();
  $('statCompanies').innerText = Number(meta?.companies || 0).toLocaleString();
  $('statAuc').innerText = formatMetric(reliability?.walkForwardAuc ?? reliability?.holdoutAuc);
  $('statMcc').innerText = reliability?.mcc === null || reliability?.mcc === undefined ? '—' : formatMetric(reliability.mcc);
  updateSpotlight();
  const counts = { VALIDATED: 0, EXPLORATORY: 0, UNAVAILABLE: 0 };
  state.data.calls.forEach((call) => { counts[statusCategory(modelFor(call)?.status)] += 1; });
  $('statusValidated').innerText = `${counts.VALIDATED.toLocaleString()} calls carry stored holdout or walk-forward provenance for the active model.`;
  $('statusExploratory').innerText = `${counts.EXPLORATORY.toLocaleString()} calls have retrospective scores that should be treated as exploratory, not deployable evidence.`;
  $('statusUnavailable').innerText = `${counts.UNAVAILABLE.toLocaleString()} calls have no usable prediction in the active model.`;
}

function filteredCalls() {
  const search = $('screenerSearch').value.trim().toLowerCase();
  return state.data.calls.filter((call) => {
    const model = modelFor(call);
    const matchesSearch = !search || call.sym.toLowerCase().includes(search) || call.co.toLowerCase().includes(search);
    const category = statusCategory(model?.status);
    const matchesStatus = state.filters.status === 'ALL' || category === state.filters.status;
    const matchesDirection = state.filters.dir === 'ALL'
      || (state.filters.dir === 'UP' && model?.signal === 'Positive')
      || (state.filters.dir === 'DOWN' && model?.signal === 'Negative');
    const matchesConfidence = state.filters.conf === 'ALL' || model?.confidence === state.filters.conf;
    return matchesSearch && matchesStatus && matchesDirection && matchesConfidence;
  });
}

function renderScreener() {
  if (!state.data) return;
  const calls = filteredCalls();
  const pageCount = Math.max(1, Math.ceil(calls.length / state.pageSize));
  state.page = Math.min(state.page, pageCount);
  const visible = calls.slice((state.page - 1) * state.pageSize, state.page * state.pageSize);
  $('screenerBody').innerHTML = visible.map((call) => {
    const model = modelFor(call);
    const unavailable = !model || model.prob === null;
    const actualClass = call.ret === null || call.ret === undefined ? '' : Number(call.ret) >= 0 ? 'text-up' : 'text-down';
    const actual = call.ret === null || call.ret === undefined ? 'Unavailable' : formatPercent(call.ret, true);
    return `<tr class="${unavailable ? 'unavailable-row' : ''}">
      <td><strong style="color:var(--text); font-family:var(--font-mono);">${escapeHtml(call.sym)}</strong><span style="color:var(--muted);font-size:11.5px;margin-left:6px;">${escapeHtml(call.co)}</span></td>
      <td class="text-mono">${escapeHtml(call.year)} Q${escapeHtml(call.q)}<span style="color:var(--muted-dim);font-size:11px;display:block;">${escapeHtml(call.date)} · ${escapeHtml(call.timing)}</span></td>
      <td>${statusBadge(model?.status || 'Unavailable')}</td>
      <td>${signalBadge(model)}</td>
      <td class="text-mono" style="font-size:11.5px;color:var(--muted);">${escapeHtml(model?.confidence || 'Unavailable')}</td>
      <td class="text-mono font-weight-bold ${actualClass}">${actual}</td>
      <td><button class="btn" style="padding:4px 10px;font-size:11px;" data-call-id="${escapeHtml(call.id)}">View ➔</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="padding:24px;color:var(--muted);">No calls match the current filters.</td></tr>';
  $('screenerBody').querySelectorAll('[data-call-id]').forEach((button) => {
    button.addEventListener('click', () => {
      const call = state.data.calls.find((item) => item.id === button.dataset.callId);
      if (call) { state.currentCall = call; switchTab('detail'); }
    });
  });
  $('screenerCount').innerText = `Showing ${visible.length} of ${calls.length} matching calls`;
  $('screenerPagination').innerHTML = `
    <button class="btn" ${state.page <= 1 ? 'disabled' : ''} id="screenerPrev">← Previous</button>
    <span>Page ${state.page} of ${pageCount}</span>
    <button class="btn" ${state.page >= pageCount ? 'disabled' : ''} id="screenerNext">Next →</button>`;
  $('screenerPrev').onclick = () => { if (state.page > 1) { state.page -= 1; renderScreener(); } };
  $('screenerNext').onclick = () => { if (state.page < pageCount) { state.page += 1; renderScreener(); } };
}

function initDetailControls() {
  if (!state.data) return;
  const companies = [...new Set(state.data.calls.map((call) => call.sym))].sort();
  const companySelect = $('detailCompanySelect');
  const currentSymbol = state.currentCall?.sym || companies[0];
  companySelect.innerHTML = companies.map((symbol) => {
    const call = state.data.calls.find((item) => item.sym === symbol);
    return `<option value="${escapeHtml(symbol)}">${escapeHtml(symbol)} — ${escapeHtml(call.co)}</option>`;
  }).join('');
  companySelect.value = companies.includes(currentSymbol) ? currentSymbol : companies[0];
  companySelect.onchange = onDetailCompanyChange;
  onDetailCompanyChange();
}

function onDetailCompanyChange() {
  const symbol = $('detailCompanySelect').value;
  const calls = state.data.calls.filter((call) => call.sym === symbol).sort((a, b) => String(b.datetime).localeCompare(String(a.datetime)));
  const select = $('detailCallSelect');
  select.innerHTML = calls.map((call) => `<option value="${escapeHtml(call.id)}">${escapeHtml(call.year)} Q${escapeHtml(call.q)} (${escapeHtml(call.date)})</option>`).join('');
  const current = calls.find((call) => call.id === state.currentCall?.id) || calls[0];
  select.value = current?.id || '';
  state.currentCall = current || null;
  select.onchange = onDetailCallChange;
  renderDetailView();
}

function onDetailCallChange() {
  state.currentCall = state.data.calls.find((call) => call.id === $('detailCallSelect').value) || state.currentCall;
  renderDetailView();
}

function loadDetail(symbol, yearQuarter) {
  const [year, quarter] = yearQuarter.split('-Q');
  const call = state.data.calls.find((item) => item.sym === symbol && String(item.year) === String(year) && String(item.q) === String(quarter));
  if (call) state.currentCall = call;
  initDetailControls();
}

function renderFeatureBars(model) {
  if (!model || model.prob === null) return '<div style="padding:24px;background:var(--surface-raised);border:1px dashed var(--border);border-radius:6px;color:var(--muted);">Feature evidence is unavailable because this call has no usable prediction in the active model.</div>';
  if (!model.featureBars?.length) return '<div style="padding:24px;background:var(--surface-raised);border:1px dashed var(--border);border-radius:6px;color:var(--muted);">Human-readable feature evidence is not stored for this artifact.</div>';
  return model.featureBars.map((feature) => `
    <div class="f-bar-wrap">
      <span style="width:180px;font-size:12.5px;">${escapeHtml(feature.label)} <span class="infoicon">i<span class="tip">${escapeHtml(feature.description)}</span></span></span>
      <div class="f-bar-track"><div class="f-bar-fill" style="width:${feature.width}%;background:var(--${escapeHtml(feature.color)});"></div></div>
      <span class="f-bar-val text-${escapeHtml(feature.color)}">${escapeHtml(feature.display)}</span>
    </div>`).join('');
}

function renderDetailView() {
  if (!state.data || !state.currentCall) return;
  const call = state.currentCall;
  const model = modelFor(call);
  $('detailTitle').innerText = `${call.co} (${call.sym}) — ${call.year} Q${call.q}`;
  $('detailSubtitle').innerText = `${call.date} · ${call.timing} · ${statusCategoryLabel(model?.status)} · ${model?.status || 'Unavailable'}`;
  $('metaPresLen').innerText = call.presLen || 'Unavailable';
  $('metaQaLen').innerText = call.qaLen || 'Unavailable';
  $('metaStatus').innerText = `${statusCategoryLabel(model?.status)} · ${model?.status || 'Unavailable'}`;
  $('metaStatus').className = statusCategory(model?.status) === 'VALIDATED' ? 'text-up font-weight-bold' : statusCategory(model?.status) === 'EXPLORATORY' ? 'text-gold font-weight-bold' : 'font-weight-bold';

  const actual = $('detailActualReturn');
  actual.innerText = call.ret === null || call.ret === undefined ? 'Unavailable' : formatPercent(call.ret, true);
  actual.style.color = call.ret === null || call.ret === undefined ? 'var(--muted)' : Number(call.ret) >= 0 ? 'var(--up)' : 'var(--down)';

  const banner = $('detailFallbackBanner');
  if (!model || model.prob === null) {
    banner.style.display = 'flex';
    banner.innerHTML = `<span style="color:var(--neutral);">i</span><div><strong style="color:var(--text);">No prediction available in the active model</strong><br>${escapeHtml(model?.statusDescription || 'This artifact cannot score the selected call.')}</div>`;
  } else if (model.status === 'Retrospective inference') {
    banner.style.display = 'flex';
    banner.innerHTML = `<span style="color:var(--gold);">!</span><div><strong style="color:var(--text);">Exploratory archive record</strong><br>${escapeHtml(model.statusDescription)}</div>`;
  } else {
    banner.style.display = 'none';
  }

  const gaugeFill = $('gaugeFill');
  const probText = $('gaugeProbText');
  const dirText = $('gaugeDirText');
  const confBadge = $('gaugeConfBadge');
  if (!model || model.prob === null) {
    gaugeFill.style.transform = 'rotate(-45deg)';
    gaugeFill.style.borderColor = 'var(--muted-dim)';
    probText.innerText = 'N/A';
    probText.style.color = 'var(--muted)';
    dirText.innerText = 'No clear signal';
    $('gaugeContext').innerText = 'Base rate: unavailable · Difference: unavailable';
    confBadge.className = 'badge muted';
    confBadge.innerText = 'UNAVAILABLE';
    confBadge.title = 'Confidence is unavailable because no probability was produced.';
    $('gaugeConfidenceNote').innerText = 'Confidence compares the model probability with the active model’s typical positive-return rate.';
  } else {
    const isPositive = model.tone === 'positive';
    const color = isPositive ? 'var(--up)' : model.tone === 'negative' ? 'var(--down)' : 'var(--neutral)';
    gaugeFill.style.transform = `rotate(${-45 + model.prob * 180}deg)`;
    gaugeFill.style.borderColor = color;
    probText.innerText = formatPercent(model.prob);
    probText.style.color = color;
    dirText.innerText = model.signal;
    $('gaugeContext').innerText = `Base rate: ${formatPercent(model.baseRate)} · Difference: ${formatPoints(model.differenceFromBaseRate, true)}`;
    confBadge.className = `badge ${model.tone === 'positive' ? 'up' : model.tone === 'negative' ? 'down' : 'neutral'}`;
    confBadge.innerText = `${model.confidence} CONFIDENCE`;
    confBadge.title = model.confidenceDescription || 'Confidence compares distance from the active model’s base rate.';
    $('gaugeConfidenceNote').innerText = model.confidenceDescription || 'Confidence compares the model probability with the active model’s typical positive-return rate.';
  }
  $('featureContainer').innerHTML = renderFeatureBars(model);
  drawEventWindowChart(call);
}

function drawEventWindowChart(call) {
  const canvas = $('eventChartCanvas');
  const wrap = canvas?.parentElement;
  if (!canvas || !wrap) return;
  const stock = Array.isArray(call.priceSeries) ? call.priceSeries.map(Number).filter(Number.isFinite) : [];
  const benchmark = Array.isArray(call.benchmarkSeries) ? call.benchmarkSeries.map(Number).filter(Number.isFinite) : [];
  if (stock.length < 2) {
    canvas.style.display = 'none';
    wrap.innerHTML = '<div style="height:200px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--font-mono);font-size:12px;text-align:center;">Price and benchmark series are not included in the selected artifact.<br>No generated market path is shown.</div>';
    return;
  }
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const normalized = (values) => values.map((value) => (value / values[0]) * 100);
  const series = [normalized(stock)];
  if (benchmark.length === stock.length) series.push(normalized(benchmark));
  const all = series.flat();
  const min = Math.min(...all);
  const max = Math.max(...all);
  const x = (index, length) => (index / Math.max(1, length - 1)) * (w - 20) + 10;
  const y = (value) => h - 12 - ((value - min) / Math.max(0.0001, max - min)) * (h - 24);
  ctx.strokeStyle = '#141a26';
  for (let i = 1; i < 4; i += 1) { ctx.beginPath(); ctx.moveTo(0, (h / 4) * i); ctx.lineTo(w, (h / 4) * i); ctx.stroke(); }
  series.forEach((values, seriesIndex) => {
    ctx.strokeStyle = seriesIndex === 0 ? '#00e599' : '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, index) => index ? ctx.lineTo(x(index, values.length), y(value)) : ctx.moveTo(x(index, values.length), y(value)));
    ctx.stroke();
  });
  const eventIndex = Math.min(1, stock.length - 1);
  ctx.strokeStyle = '#f5b041';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(x(eventIndex, stock.length), 0); ctx.lineTo(x(eventIndex, stock.length), h); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#f5b041'; ctx.font = '11px IBM Plex Mono'; ctx.fillText('Earnings Call', x(eventIndex, stock.length) + 6, 16);
}

function renderReliability() {
  if (!state.data) return;
  const models = state.data.reliability.models;
  const rich = models.find((model) => model.key === 'sentence_plus_historical_xgboost_depth1_trees100');
  const original = models.find((model) => model.key === 'original_logistic');
  $('relPrimaryTitle').innerHTML = `${escapeHtml(rich?.title || 'Rich XGBoost')} <span class="badge gold">${state.activeModel === 'sentence_hist' ? 'Selected Model' : 'Richer candidate'}</span>`;
  $('relFinbertTitle').innerHTML = `${escapeHtml(original?.title || 'Original Logistic')} <span class="badge muted">${state.activeModel === 'finbert' ? 'Selected Model' : 'Reference'}</span>`;
  $('relPrimarySummary').innerText = `Walk-forward AUC ${formatMetric(rich?.walkForwardAuc)} · latest holdout ${formatMetric(rich?.holdoutAuc)} · Brier ${formatMetric(rich?.walkForwardBrier)}.`;
  $('relFinbertSummary').innerText = `Walk-forward AUC ${formatMetric(original?.walkForwardAuc)} · latest holdout ${formatMetric(original?.holdoutAuc)} · Brier ${formatMetric(original?.walkForwardBrier)}.`;
  $('relCardPrimary').classList.toggle('primary', state.activeModel === 'sentence_hist');
  $('relCardFinbert').classList.toggle('primary', state.activeModel === 'finbert');
  $('reliabilityBody').innerHTML = models.map((model) => `<tr class="${(state.activeModel === 'sentence_hist' && model.key === 'sentence_plus_historical_xgboost_depth1_trees100') || (state.activeModel === 'finbert' && model.key === 'original_logistic') ? 'highlight' : ''}">
    <td>${escapeHtml(model.title)}</td><td>${escapeHtml(model.badge)}</td><td>${formatMetric(model.walkForwardAuc)}</td><td>${formatMetric(model.holdoutAuc)}</td><td>${formatMetric(model.walkForwardBrier)}</td><td>${model.events === null || model.events === undefined ? '—' : Number(model.events).toLocaleString()}</td>
  </tr>`).join('');
}

function bindSearch() {
  $('screenerSearch').addEventListener('input', () => { state.page = 1; renderScreener(); });
}

function bindFilters() {
  document.querySelectorAll('[data-filter-type]').forEach((button) => {
    button.addEventListener('click', () => setFilter(button.dataset.filterType, button.dataset.filterValue, button));
  });
}

async function boot() {
  try {
    const response = await fetch('data/app-data.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
    state.data = await response.json();
    state.activeModel = state.data.defaultModel || state.data.models[0]?.key;
    populateModelSelector();
    bindSearch();
    bindFilters();
    updateModelStatus();
    buildTickerTape();
    updateFilterUI();
    renderOverview();
    renderScreener();
    initDetailControls();
    renderReliability();
  } catch (error) {
    document.querySelector('main').innerHTML = `<div class="card-raised"><div class="eyebrow">Data unavailable</div><h1>Could not load the validated artifact export.</h1><p style="color:var(--muted);">${escapeHtml(error.message)}. Run <code>python scripts/export_frontend_data.py</code> and reload the frontend.</p></div>`;
  }
}

function updateFilterUI() {
  document.querySelectorAll('[data-filter-type]').forEach((button) => {
    button.classList.toggle('active', state.filters[button.dataset.filterType] === button.dataset.filterValue);
  });
}

window.switchTab = switchTab;
window.setFilter = (type, value, button) => { setFilter(type, value, button); };
window.onModelChange = onModelChange;
window.loadDetail = loadDetail;

document.addEventListener('DOMContentLoaded', boot);
