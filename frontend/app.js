const state = {
  data: null,
  activeModel: null,
  currentCall: null,
  filters: { dir: 'ALL', conf: 'ALL', status: 'VALIDATED' },
  filtersByModel: {},
  sort: { key: 'datetime', direction: 'desc' },
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

function defaultFiltersFor(modelKey) {
  return {
    dir: 'ALL',
    conf: 'ALL',
    status: modelKey === 'finbert' ? 'EXPLORATORY' : 'VALIDATED',
  };
}

function useModelFilters(modelKey) {
  if (!state.filtersByModel[modelKey]) state.filtersByModel[modelKey] = defaultFiltersFor(modelKey);
  state.filters = state.filtersByModel[modelKey];
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

function directionLabel(model) {
  if (!model || model.prob === null || model.prob === undefined) return 'Unavailable';
  return model.directionLabel || (model.signal === 'Positive' ? 'Positive prediction' : 'Negative prediction');
}

function confidenceBadge(model) {
  if (!model || !model.confidence || model.confidence === 'UNAVAILABLE') return '<span class="badge muted">Unavailable</span>';
  const level = String(model.confidence).toLowerCase();
  return `<span class="badge confidence-${escapeHtml(level)}" title="${escapeHtml(model.signalStrengthDescription || model.confidenceDescription || 'Signal strength is measured against the active model base rate.')}">${escapeHtml(level)} signal strength</span>`;
}

function signalBadge(model) {
  if (!model || model.prob === null || model.prob === undefined) return '<span class="badge muted">Unavailable</span>';
  const className = model.tone === 'positive' ? 'up' : model.tone === 'negative' ? 'down' : 'neutral';
  const label = model.tone === 'positive' ? '▲ Positive prediction' : '▼ Negative prediction';
  const baseRelation = model.baseRateRelation || (Number(model.prob) >= Number(model.baseRate) ? 'Above base rate' : 'Below base rate');
  const threshold = model.predictionThreshold === null || model.predictionThreshold === undefined ? 'Unavailable' : formatPercent(model.predictionThreshold);
  const context = `Model probability ${formatPercent(model.prob)}; binary decision threshold ${threshold}; ${baseRelation}; typical positive-outcome rate ${formatPercent(model.baseRate)}; difference ${formatPoints(model.differenceFromBaseRate, true)}. ${model.signalStrengthDescription || model.confidenceDescription || model.explanation}`;
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
  useModelFilters(state.activeModel);
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
  state.filtersByModel[state.activeModel] = state.filters;
  state.page = 1;
  updateFilterUI();
  renderScreener();
}

function sortValue(call, key) {
  const model = modelFor(call);
  if (key === 'company') return call.sym || call.co || null;
  if (key === 'datetime') return call.datetime || null;
  if (key === 'status') return statusCategory(model?.status);
  if (key === 'direction') return model?.signal || null;
  if (key === 'strength') return model?.confidence || null;
  if (key === 'probability') return model?.prob;
  if (key === 'actualReturn') return call.ret;
  return null;
}

function sortRank(key, value) {
  if (value === null || value === undefined || value === '') return null;
  const ranks = {
    status: { VALIDATED: 0, EXPLORATORY: 1, UNAVAILABLE: 2 },
    direction: { Positive: 0, Negative: 1, Unavailable: 2 },
    strength: { LOW: 0, MEDIUM: 1, HIGH: 2, UNAVAILABLE: 3 },
  };
  return ranks[key]?.[value] ?? null;
}

function sortedCalls(calls) {
  const { key, direction } = state.sort;
  const multiplier = direction === 'asc' ? 1 : -1;
  return [...calls].sort((left, right) => {
    const leftValue = sortValue(left, key);
    const rightValue = sortValue(right, key);
    const leftMissing = leftValue === null || leftValue === undefined || leftValue === '';
    const rightMissing = rightValue === null || rightValue === undefined || rightValue === '';
    if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
    if (leftMissing && rightMissing) return String(right.datetime || '').localeCompare(String(left.datetime || ''));
    let comparison = 0;
    const leftRank = sortRank(key, leftValue);
    const rightRank = sortRank(key, rightValue);
    if (leftRank !== null || rightRank !== null) comparison = (leftRank ?? 99) - (rightRank ?? 99);
    else if (key === 'datetime') comparison = String(leftValue).localeCompare(String(rightValue));
    else if (typeof leftValue === 'number' || typeof rightValue === 'number') comparison = Number(leftValue) - Number(rightValue);
    else comparison = String(leftValue).localeCompare(String(rightValue));
    if (comparison === 0) comparison = String(right.datetime || '').localeCompare(String(left.datetime || ''));
    if (comparison === 0) comparison = String(left.sym || '').localeCompare(String(right.sym || ''));
    return comparison * multiplier;
  });
}

function setSort(key) {
  if (state.sort.key === key) state.sort.direction = state.sort.direction === 'asc' ? 'desc' : 'asc';
  else {
    state.sort.key = key;
    state.sort.direction = key === 'datetime' ? 'desc' : 'asc';
  }
  state.page = 1;
  updateSortUI();
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
  $('spotlightSignal').innerText = `${directionLabel(model)} · ${formatPercent(model.prob)}`;
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
    const matchesConfidence = state.filters.conf === 'ALL' || (model?.signalStrength || model?.confidence) === state.filters.conf;
    return matchesSearch && matchesStatus && matchesDirection && matchesConfidence;
  });
}

function renderScreener() {
  if (!state.data) return;
  const calls = sortedCalls(filteredCalls());
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
      <td>${confidenceBadge(model)}</td>
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
  updateSortUI();
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
  const bars = model.featureBars.map((feature) => `
    <div class="f-bar-wrap">
      <span style="width:180px;font-size:12.5px;">${escapeHtml(feature.label)} <span class="infoicon" tabindex="0">i<span class="tip">${escapeHtml(feature.description)} ${escapeHtml(feature.unitDescription || '')}</span></span></span>
      <div class="f-bar-track"><div class="f-bar-fill" style="width:${feature.width}%;background:var(--${escapeHtml(feature.color)});"></div></div>
      <span class="f-bar-val text-${escapeHtml(feature.color)}">${escapeHtml(feature.display)}</span>
    </div>`).join('');
  const groups = (model.featureGroups || []).map((group) => `
    <details class="feature-group-details">
      <summary>${escapeHtml(group.title)} <span>${escapeHtml(group.summary || '')}</span></summary>
      <p>${escapeHtml(group.description || '')}</p>
      ${(group.details || []).map((detail) => `<div>${escapeHtml(detail)}</div>`).join('')}
    </details>`).join('');
  return `${bars}${groups}`;
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
    dirText.innerText = 'Unavailable';
    $('gaugeContext').innerText = 'Threshold: unavailable · Typical rate: unavailable · Difference: unavailable';
    confBadge.className = 'badge muted';
    confBadge.innerText = 'UNAVAILABLE';
    confBadge.title = 'Signal strength is unavailable because no probability was produced.';
    $('gaugeConfidenceNote').innerText = 'Signal strength compares the model probability with the active model’s typical positive-outcome rate.';
  } else {
    const isPositive = model.tone === 'positive';
    const color = isPositive ? 'var(--up)' : model.tone === 'negative' ? 'var(--down)' : 'var(--neutral)';
    gaugeFill.style.transform = `rotate(${-45 + model.prob * 180}deg)`;
    gaugeFill.style.borderColor = color;
    probText.innerText = formatPercent(model.prob);
    probText.style.color = color;
    dirText.innerText = directionLabel(model);
    const threshold = model.predictionThreshold === null || model.predictionThreshold === undefined ? 'Unavailable' : formatPercent(model.predictionThreshold);
    $('gaugeContext').innerText = `Threshold: ${threshold} · Typical rate: ${formatPercent(model.baseRate)} · Difference: ${formatPoints(model.differenceFromBaseRate, true)}`;
    $('gaugeContext').title = model.baseRateDefinition || 'Typical rate is the dataset-level positive-outcome rate for the active artifact, not a rolling company sentiment average.';
    confBadge.className = `badge confidence-${String(model.confidence || 'unavailable').toLowerCase()}`;
    confBadge.innerText = `${model.confidence || 'UNAVAILABLE'} SIGNAL STRENGTH`;
    confBadge.title = model.signalStrengthDescription || model.confidenceDescription || 'Signal strength compares distance from the active model’s base rate.';
    $('gaugeConfidenceNote').innerText = model.signalStrengthDescription || model.confidenceDescription || 'Signal strength compares the model probability with the active model’s typical positive-outcome rate.';
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
  const badge = $('eventChartBadge');
  const legend = $('eventChartLegend');
  const summary = $('eventChartSummary');
  const note = $('eventChartNote');
  if (stock.length < 2) {
    canvas.style.display = 'none';
    if (badge) { badge.className = 'badge muted'; badge.innerText = 'PRICE DATA UNAVAILABLE'; }
    if (legend) { legend.style.display = 'none'; legend.innerHTML = ''; }
    if (summary) summary.innerText = '';
    if (note) note.innerHTML = '<strong>Evidence unavailable:</strong> Historical price and benchmark series are not included for this call. No market path has been generated or inferred.';
    let empty = wrap.querySelector('.event-chart-empty');
    if (!empty) {
      empty = document.createElement('div');
      empty.className = 'event-chart-empty';
      empty.style.cssText = 'height:240px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--font-mono);font-size:12px;text-align:center;';
      wrap.appendChild(empty);
    }
    empty.innerHTML = 'Price and benchmark series are not included in the selected artifact.<br>No generated market path is shown.';
    return;
  }
  canvas.style.display = 'block';
  wrap.querySelector('.event-chart-empty')?.remove();
  if (badge) { badge.className = 'badge neutral'; badge.innerText = 'HISTORICAL DATA'; }
  const hasBenchmark = benchmark.length === stock.length;
  if (legend) {
    legend.style.display = 'flex';
    legend.innerHTML = `<span class="chart-legend-item"><i class="chart-swatch stock"></i>Stock</span>${hasBenchmark ? `<span class="chart-legend-item"><i class="chart-swatch benchmark"></i>${escapeHtml(call.benchmarkSymbol || 'Benchmark')}</span>` : ''}<span class="chart-legend-item chart-summary">Indexed close · T−5 = 100</span>`;
  }
  const dates = Array.isArray(call.priceDates) ? call.priceDates : [];
  if (summary) {
    const firstClose = `$${stock[0].toFixed(2)}`;
    const lastClose = `$${stock[stock.length - 1].toFixed(2)}`;
    const benchmarkSummary = hasBenchmark ? ` · ${escapeHtml(call.benchmarkSymbol || 'Benchmark')} close: $${benchmark[0].toFixed(2)} → $${benchmark[benchmark.length - 1].toFixed(2)}` : '';
    const dateRange = dates.length >= 2 ? ` · ${escapeHtml(dates[0])} → ${escapeHtml(dates[dates.length - 1])}` : '';
    summary.innerHTML = `Stock close: ${firstClose} → ${lastClose}${benchmarkSummary}${dateRange}`;
  }
  if (note) note.innerHTML = '<strong>How to read it:</strong> Each series is indexed to 100 at T−5 so the stock and benchmark can share one scale. The gold marker is the call; the dashed marker is the end of the five-session evaluation window.';

  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(480, Math.floor(rect.width || 900));
  const h = 240;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const normalized = (values) => values.map((value) => (value / values[0]) * 100);
  const series = [normalized(stock)];
  if (hasBenchmark) series.push(normalized(benchmark));
  const all = series.flat();
  const rawMin = Math.min(...all);
  const rawMax = Math.max(...all);
  const rawRange = Math.max(1, rawMax - rawMin);
  const min = rawMin - rawRange * 0.12;
  const max = rawMax + rawRange * 0.12;
  const left = 48;
  const right = 18;
  const top = 18;
  const bottom = 45;
  const plotHeight = h - top - bottom;
  const plotWidth = w - left - right;
  const x = (index, length) => left + (index / Math.max(1, length - 1)) * plotWidth;
  const y = (value) => top + ((max - value) / Math.max(0.0001, max - min)) * plotHeight;
  ctx.font = '10px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i += 1) {
    const value = max - ((max - min) / 4) * i;
    const lineY = y(value);
    ctx.strokeStyle = '#141a26';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, lineY); ctx.lineTo(w - right, lineY); ctx.stroke();
    ctx.fillStyle = '#748097';
    ctx.fillText(value.toFixed(1), left - 7, lineY + 3);
  }
  ctx.save();
  ctx.translate(11, top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#748097';
  ctx.fillText('Indexed close', 0, 0);
  ctx.restore();
  const eventIndex = Number.isFinite(Number(call.eventIndex)) ? Number(call.eventIndex) : Math.min(1, stock.length - 1);
  const evaluationStart = Number.isFinite(Number(call.evaluationStartIndex)) ? Number(call.evaluationStartIndex) : eventIndex;
  const evaluationEnd = Number.isFinite(Number(call.evaluationEndIndex)) ? Number(call.evaluationEndIndex) : stock.length - 1;
  ctx.fillStyle = 'rgba(56, 189, 248, 0.05)';
  ctx.fillRect(x(evaluationStart, stock.length), top, Math.max(0, x(evaluationEnd, stock.length) - x(evaluationStart, stock.length)), plotHeight);
  if (min <= 100 && max >= 100) {
    ctx.strokeStyle = '#33415c';
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(left, y(100)); ctx.lineTo(w - right, y(100)); ctx.stroke(); ctx.setLineDash([]);
  }
  series.forEach((values, seriesIndex) => {
    ctx.strokeStyle = seriesIndex === 0 ? '#00e599' : '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    values.forEach((value, index) => index ? ctx.lineTo(x(index, values.length), y(value)) : ctx.moveTo(x(index, values.length), y(value)));
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    values.forEach((value, index) => { ctx.beginPath(); ctx.arc(x(index, values.length), y(value), 2.5, 0, Math.PI * 2); ctx.fill(); });
  });
  ctx.strokeStyle = '#f5b041';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(x(eventIndex, stock.length), top); ctx.lineTo(x(eventIndex, stock.length), top + plotHeight); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#f5b041'; ctx.font = '600 10px IBM Plex Mono'; ctx.textAlign = 'left'; ctx.fillText('CALL', Math.min(w - 45, x(eventIndex, stock.length) + 5), top - 5);
  ctx.strokeStyle = '#a7b4c9';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(x(evaluationEnd, stock.length), top); ctx.lineTo(x(evaluationEnd, stock.length), top + plotHeight); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#a7b4c9';
  ctx.textAlign = 'right'; ctx.fillText('T+5', Math.min(w - 5, x(evaluationEnd, stock.length) + 22), top - 5);
  const labels = Array.isArray(call.chartLabels) ? call.chartLabels : stock.map((_, index) => `T${index - 5 >= 0 ? '+' : ''}${index - 5}`);
  const tickSet = new Set([0, 2, 4, Math.max(0, Math.min(stock.length - 1, Math.round(eventIndex))), Math.max(0, Math.min(stock.length - 1, Math.round(evaluationEnd))), stock.length - 1]);
  ctx.textAlign = 'center';
  ctx.font = '10px IBM Plex Mono';
  [...tickSet].sort((a, b) => a - b).forEach((index) => {
    const xx = x(index, stock.length);
    ctx.fillStyle = '#a7b4c9';
    ctx.fillText(labels[index] || `T${index - 5}`, xx, h - 23);
    ctx.fillStyle = '#59667c';
    ctx.fillText(dates[index] || '', xx, h - 9);
  });
  canvas.setAttribute('aria-label', `Historical indexed stock price chart from ${dates[0] || 'T-5'} to ${dates[dates.length - 1] || 'T+5'} with an earnings call marker and five-session evaluation endpoint.`);
}

function renderReliability() {
  if (!state.data) return;
  const models = state.data.reliability.models;
  const activeKey = reliabilityKey();
  const formatDetail = (value) => value === null || value === undefined ? '—' : Number(value).toFixed(3);
  const bestKey = (split, metric, lowerIsBetter = false) => {
    const available = models.filter((model) => model[split]?.[metric] !== null && model[split]?.[metric] !== undefined);
    if (!available.length) return null;
    return available.reduce((best, model) => {
      if (!best) return model;
      const current = Number(model[split][metric]);
      const previous = Number(best[split][metric]);
      return lowerIsBetter ? (current < previous ? model : best) : (current > previous ? model : best);
    }, null)?.key;
  };
  const winnerClass = (model, split, metric, lowerIsBetter = false) => bestKey(split, metric, lowerIsBetter) === model.key ? 'metric-winner' : '';
  const cards = $('reliabilityCards');
  if (cards) {
    cards.innerHTML = models.slice(0, 3).map((model) => `
      <article class="model-card ${model.key === activeKey ? 'primary' : ''}">
        <div class="eyebrow">${escapeHtml(model.badge || 'Stored comparison')}</div>
        <h4>${escapeHtml(model.title)} ${model.key === activeKey ? '<span class="badge gold">Selected Model</span>' : ''}</h4>
        <p style="color:var(--muted);font-size:12.5px;margin-bottom:12px;">${escapeHtml(model.description || '')}</p>
        <div class="reliability-card-grid">
          <span>Walk-forward AUC <strong>${formatDetail(model.walkForwardAuc)}</strong></span>
          <span>Latest holdout <strong>${formatDetail(model.holdoutAuc)}</strong></span>
          <span>Brier score <strong>${formatDetail(model.walkForwardBrier)}</strong></span>
          <span>Events / companies <strong>${model.events == null ? '—' : Number(model.events).toLocaleString()} / ${model.companyCount == null ? '—' : Number(model.companyCount).toLocaleString()}</strong></span>
        </div>
      </article>`).join('');
  }
  $('reliabilityBody').innerHTML = models.map((model) => `<tr class="${(state.activeModel === 'sentence_hist' && model.key === 'sentence_plus_historical_xgboost_depth1_trees100') || (state.activeModel === 'finbert' && model.key === 'original_logistic') ? 'highlight' : ''}">
    <td>${escapeHtml(model.title)}</td><td>${escapeHtml(model.badge)}</td><td class="${winnerClass(model, 'walkForward', 'auc')}">${formatMetric(model.walkForwardAuc)}</td><td class="${winnerClass(model, 'holdout', 'auc')}">${formatMetric(model.holdoutAuc)}</td><td class="${winnerClass(model, 'walkForward', 'brier', true)}">${formatMetric(model.walkForwardBrier)}</td><td>${model.events === null || model.events === undefined ? '—' : Number(model.events).toLocaleString()}</td>
  </tr>`).join('');
  const detail = $('reliabilityDetailBody');
  if (detail) {
    detail.innerHTML = models.map((model) => {
      const walk = model.walkForward || {};
      const holdout = model.holdout || {};
      const cell = (value, className = '') => `<td class="${className}">${value === null || value === undefined ? '—' : Number(value).toFixed(3)}</td>`;
      const row = (split, values) => `<tr><td>${escapeHtml(model.title)}</td><td>${split}</td><td>${values.events == null ? '—' : Number(values.events).toLocaleString()}</td>${cell(values.accuracy, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'accuracy'))}${cell(values.balancedAccuracy, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'balancedAccuracy'))}${cell(values.precision, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'precision'))}${cell(values.recall, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'recall'))}${cell(values.f1, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'f1'))}${cell(values.mcc, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'mcc'))}${cell(values.brier, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'brier', true))}${cell(values.logLoss, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'logLoss', true))}${cell(values.averagePrecision, winnerClass(model, split === 'Walk-forward' ? 'walkForward' : 'holdout', 'averagePrecision'))}<td>${values.trueNegative == null ? '—' : `${values.trueNegative}/${values.falsePositive}/${values.falseNegative}/${values.truePositive}`}</td></tr>`;
      return `${row('Walk-forward', walk)}${row('Holdout', holdout)}`;
    }).join('');
  }
  const methodology = $('reliabilityMethodology');
  if (methodology) {
    const manifest = state.data.reliability.manifest || {};
    const years = Array.isArray(manifest.walk_forward_years) ? manifest.walk_forward_years.join(', ') : 'Unavailable';
    methodology.innerText = `Walk-forward years: ${years} · Holdout cutoff: ${manifest.holdout_cutoff_year ?? 'Unavailable'}`;
  }
  const chart = $('reliabilityChart');
  if (chart) {
    chart.innerHTML = models.map((model) => {
      const walk = Number(model.walkForwardAuc);
      const holdout = Number(model.holdoutAuc);
      const walkPos = Number.isFinite(walk) ? Math.max(0, Math.min(100, walk * 100)) : null;
      const holdoutPos = Number.isFinite(holdout) ? Math.max(0, Math.min(100, holdout * 100)) : null;
      const walkBest = bestKey('walkForward', 'auc') === model.key;
      const holdoutBest = bestKey('holdout', 'auc') === model.key;
      return `<div class="reliability-row"><div class="reliability-row-label">${escapeHtml(model.title)}</div><div class="reliability-track" aria-label="${escapeHtml(model.title)} AUC comparison"><span class="reliability-baseline" style="left:50%"></span>${walkPos === null ? '' : `<span class="reliability-dot walk ${walkBest ? 'best' : ''}" style="left:${walkPos}%" title="Walk-forward AUC ${formatMetric(walk)}${walkBest ? ' · Best' : ''}"></span>`}${holdoutPos === null ? '' : `<span class="reliability-dot holdout ${holdoutBest ? 'best' : ''}" style="left:${holdoutPos}%" title="Holdout AUC ${formatMetric(holdout)}${holdoutBest ? ' · Best' : ''}"></span>`}</div><div class="reliability-row-values">WF ${formatMetric(model.walkForwardAuc)} · HO ${formatMetric(model.holdoutAuc)}</div></div>`;
    }).join('');
  }
}

function bindSearch() {
  if ($('screenerSearch').dataset.bound === 'true') return;
  $('screenerSearch').dataset.bound = 'true';
  $('screenerSearch').addEventListener('input', () => { state.page = 1; renderScreener(); });
}

function bindFilters() {
  document.querySelectorAll('[data-filter-type]').forEach((button) => {
    button.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      setFilter(button.dataset.filterType, button.dataset.filterValue, button);
    };
  });
}

function bindSortControls() {
  document.querySelectorAll('[data-sort-key]').forEach((button) => {
    if (button.dataset.bound === 'true') return;
    button.dataset.bound = 'true';
    button.addEventListener('click', () => setSort(button.dataset.sortKey));
    button.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSort(button.dataset.sortKey); }
    });
  });
}

async function boot() {
  try {
    const response = await fetch('data/app-data.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
    state.data = await response.json();
    state.activeModel = state.data.defaultModel || state.data.models[0]?.key;
    useModelFilters(state.activeModel);
    const buildInfo = $('dataBuildInfo');
    if (buildInfo) buildInfo.innerText = `Artifact export v${state.data.version || '—'} · ${state.data.generatedAt || 'timestamp unavailable'}`;
    populateModelSelector();
    bindSearch();
    bindFilters();
    bindSortControls();
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
    const isActive = state.filters[button.dataset.filterType] === button.dataset.filterValue;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function updateSortUI() {
  document.querySelectorAll('[data-sort-key]').forEach((button) => {
    const active = button.dataset.sortKey === state.sort.key;
    const sortValue = active ? (state.sort.direction === 'asc' ? 'ascending' : 'descending') : 'none';
    button.setAttribute('aria-sort', sortValue);
    button.parentElement?.setAttribute('aria-sort', sortValue);
    const label = button.dataset.label || button.innerText.replace(/[↕↑↓]/g, '').trim();
    button.innerHTML = `${escapeHtml(label)}${active ? (state.sort.direction === 'asc' ? ' ↑' : ' ↓') : ' ↕'}`;
  });
}

window.switchTab = switchTab;
window.setFilter = (type, value, button) => { setFilter(type, value, button); };
window.onModelChange = onModelChange;
window.loadDetail = loadDetail;

// Bind before the data request starts so the controls never appear clickable
// without responding to input.
bindFilters();
bindSortControls();
document.addEventListener('DOMContentLoaded', boot);
