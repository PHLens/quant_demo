(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const UPDATE_OPERATION_STORAGE_KEY = 'r0-data-update-operation-v1';
  const state = { snapshot: null, loadedTabs: new Set(), recoveryOperation: null, updateOperation: null, updateActive: false, updatePlanReady: false, updatePollGeneration: 0 };

  function setText(node, value, fallback = 'unknown') {
    if (node) node.textContent = value == null || value === '' ? fallback : String(value);
  }

  function format(value) {
    if (value == null || value === '') return 'not stored/unknown';
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  async function requestJson(url, options = {}) {
    const headers = { Accept: 'application/json', ...(options.headers || {}) };
    if (options.body != null) headers['Content-Type'] = 'application/json';
    const response = await fetch(url, { ...options, headers });
    let body = {};
    try { body = await response.json(); } catch (_) { body = {}; }
    if (!response.ok) {
      const error = new Error(body.message || body.error || `HTTP ${response.status}`);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  }

  async function fetchAllPages(url) {
    const first = await requestJson(url);
    if (!Array.isArray(first.items) || !first.next_cursor) return first;
    const items = [...first.items]; let cursor = first.next_cursor;
    while (cursor) {
      const nextUrl = new URL(url, window.location.origin);
      nextUrl.searchParams.set('cursor', cursor);
      nextUrl.searchParams.set('limit', '100');
      const page = await requestJson(`${nextUrl.pathname}${nextUrl.search}`);
      items.push(...(page.items || [])); cursor = page.next_cursor;
      if (items.length > Number(first.total)) throw new Error('Paged collection exceeded its declared total.');
    }
    if (items.length !== Number(first.total)) throw new Error('Paged collection did not match its declared total.');
    return { ...first, items, next_cursor: null };
  }

  function confirmAction(message, acceptLabel = 'Confirm') {
    const dialog = $('#confirm-dialog');
    if (!dialog || typeof dialog.showModal !== 'function') return Promise.resolve(false);
    const previous = document.activeElement; const accept = $('#confirm-dialog-accept');
    setText($('#confirm-dialog-message'), message); setText(accept, acceptLabel);
    dialog.returnValue = '';
    return new Promise((resolve) => {
      dialog.addEventListener('close', () => {
        previous?.focus?.(); resolve(dialog.returnValue === 'confirm');
      }, { once: true });
      dialog.showModal(); accept.focus();
    });
  }

  function showMessage(node, message, danger = false) {
    if (!node) return;
    setText(node, message);
    node.className = `notice${danger ? ' danger' : ''}`;
    node.hidden = false;
  }

  function initNavigation() {
    const button = $('[data-menu-button]');
    const sidebar = $('[data-sidebar]');
    if (!button || !sidebar) return;
    button.addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      button.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
        button.setAttribute('aria-expanded', 'false');
        button.focus();
      }
    });
  }

  function renderTable(root, rows, preferred = [], actions = null) {
    root.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) {
      const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'No stored rows.'; root.appendChild(empty); return;
    }
    const columns = [...preferred, ...Object.keys(rows[0]).filter((key) => !preferred.includes(key))].filter((key, index, all) => all.indexOf(key) === index);
    const wrap = document.createElement('div'); wrap.className = 'table-wrap';
    const table = document.createElement('table'); const head = document.createElement('thead'); const hr = document.createElement('tr');
    columns.forEach((column) => { const th = document.createElement('th'); th.scope = 'col'; th.textContent = column.replaceAll('_', ' '); hr.appendChild(th); });
    if (actions) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = 'Actions'; hr.appendChild(th); }
    head.appendChild(hr); table.appendChild(head); const body = document.createElement('tbody');
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      columns.forEach((column) => { const td = document.createElement('td'); td.dataset.label = column.replaceAll('_', ' '); td.textContent = format(row[column]); tr.appendChild(td); });
      if (actions) { const td = document.createElement('td'); td.dataset.label = 'Actions'; actions(row, td); tr.appendChild(td); }
      body.appendChild(tr);
    });
    table.appendChild(body); wrap.appendChild(table); root.appendChild(wrap);
  }

  function drawCurve(root, points) {
    root.replaceChildren();
    if (!points || points.length < 2) { const empty = document.createElement('div'); empty.className = 'chart-empty'; empty.textContent = 'No stored curve.'; root.appendChild(empty); return; }
    const sampled = points.length <= 480 ? points : Array.from({ length: 480 }, (_, index) => points[Math.round(index * (points.length - 1) / 479)]);
    const values = sampled.map((item) => Number(item.value));
    if (values.some((value) => !Number.isFinite(value))) { drawCurve(root, []); return; }
    const width = 960; const height = 320; const pad = 44; const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1;
    const ns = 'http://www.w3.org/2000/svg'; const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.classList.add('chart-svg'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `Published curve ${sampled[0].date} through ${sampled.at(-1).date}`);
    const coords = sampled.map((item, index) => `${pad + index / (sampled.length - 1) * (width - 2 * pad)},${pad + (1 - (Number(item.value) - min) / span) * (height - 2 * pad)}`).join(' ');
    const line = document.createElementNS(ns, 'polyline'); line.setAttribute('points', coords); line.setAttribute('fill', 'none'); line.setAttribute('stroke', '#1d6d62'); line.setAttribute('stroke-width', '3'); svg.appendChild(line); root.appendChild(svg);
  }

  function initTabs(onActivate) {
    const tabs = $$('[role="tab"]');
    tabs.forEach((tab, index) => {
      const activate = () => {
        if (tab.hidden) return;
        tabs.forEach((item) => { const selected = item === tab; item.setAttribute('aria-selected', String(selected)); item.tabIndex = selected ? 0 : -1; $(`#${item.getAttribute('aria-controls')}`).hidden = !selected; });
        onActivate?.(tab.dataset.resource);
      };
      tab.addEventListener('click', activate);
      tab.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault(); const visible = tabs.filter((item) => !item.hidden); const current = visible.indexOf(tab); let next = current;
        if (event.key === 'ArrowRight') next = (current + 1) % visible.length;
        if (event.key === 'ArrowLeft') next = (current - 1 + visible.length) % visible.length;
        if (event.key === 'Home') next = 0; if (event.key === 'End') next = visible.length - 1;
        visible[next].focus(); visible[next].click();
      });
    });
  }

  function subtractMonths(day, months) {
    if (!day) return '';
    const value = new Date(`${day}T00:00:00Z`); const desired = value.getUTCDate(); value.setUTCDate(1); value.setUTCMonth(value.getUTCMonth() - months); value.setUTCDate(Math.min(desired, new Date(Date.UTC(value.getUTCFullYear(), value.getUTCMonth() + 1, 0)).getUTCDate())); return value.toISOString().slice(0, 10);
  }

  async function initSnapshot() {
    const configNode = $('#snapshot-config'); if (!configNode) return;
    const config = JSON.parse(configNode.textContent); const sourceSelect = $('#source-select'); const strategySelect = $('#strategy-select'); const variantSelect = $('#variant-select'); const message = $('#snapshot-message');
    let sources = []; let strategies = []; let variants = [];

    async function loadSources() {
      const payload = await requestJson('/api/r0/sources'); sources = payload.items || []; sourceSelect.replaceChildren();
      sources.filter((item) => !item.related_only || item.source_id === config.source_id).forEach((item) => { const option = document.createElement('option'); option.value = item.source_id; option.textContent = item.display_name; sourceSelect.appendChild(option); });
      sourceSelect.value = sources.some((item) => item.source_id === config.source_id) ? config.source_id : 'selection';
      if (config.initial_range === '6m') { const source = sources.find((item) => item.source_id === sourceSelect.value); if (source?.data_max_date) $('#start-date').value = subtractMonths(source.data_max_date, 6); }
      await loadStrategies();
    }
    async function loadStrategies() {
      const payload = await requestJson(`/api/r0/sources/${encodeURIComponent(sourceSelect.value)}/strategies`); strategies = payload.items || []; strategySelect.replaceChildren();
      strategies.forEach((item) => { const option = document.createElement('option'); option.value = item.strategy_id; option.textContent = item.display_name; strategySelect.appendChild(option); });
      strategySelect.value = strategies.some((item) => item.strategy_id === config.strategy_id) ? config.strategy_id : strategies[0]?.strategy_id || '';
      $('#benchmark-select').value = ['selection', 'a_share_timing'].includes(sourceSelect.value) ? 'csi1000' : 'etf';
      await loadVariants();
    }
    async function loadVariants() {
      const payload = await requestJson(`/api/r0/sources/${encodeURIComponent(sourceSelect.value)}/strategies/${encodeURIComponent(strategySelect.value)}/variants`); variants = payload.items || []; variantSelect.replaceChildren();
      variants.forEach((item) => { const option = document.createElement('option'); option.value = item.variant_id; option.textContent = `${item.default ? 'Default · ' : ''}${JSON.stringify(item.canonical_params)} · ${item.cache_state}`; option.dataset.state = JSON.stringify(item); variantSelect.appendChild(option); });
      const preferred = variants.find((item) => item.default) || variants[0]; if (preferred) variantSelect.value = preferred.variant_id;
      renderVariantState(preferred); await openSnapshot();
    }
    function renderVariantState(variant) {
      if (!variant) return; setText($('#snapshot-state'), variant.cache_state); setText($('#snapshot-freshness'), variant.freshness_state);
      const notice = $('#recovery-notice'); const bootstrap = variant.blocker_code === 'bootstrap_required'; $('#bootstrap-guidance').hidden = !bootstrap;
      if (variant.readable) { notice.hidden = true; return; }
      notice.hidden = false; notice.setAttribute('aria-busy', variant.cache_state === 'recovering' ? 'true' : 'false');
      setText($('#recovery-context'), `${variant.source_id} / ${variant.strategy_id} / ${variant.variant_id}; blocker=${variant.blocker_code || 'none'}`);
      setText($('#recovery-plan'), bootstrap ? `首次 bootstrap 顺序：${(variant.recovery_scopes || []).join(' + ') || '查看 Data Status'} → Preview → Confirm and update → 跟踪完成 → 返回 Snapshot。` : (variant.recovery_steps ? `${variant.recovery_steps.map((step) => step.label).join(' → ')}; ETA ${variant.eta_seconds || 'unknown'}s` : 'No fixed online recovery plan.'));
      $('#recover-cache').hidden = !variant.recovery_supported || !variant.recoverable_now; $('#recover-cache').disabled = variant.cache_state === 'recovering';
      if (variant.operation_id) pollOperation(variant.operation_id, true);
    }
    function warnings(payload) {
      const root = $('#snapshot-warnings'); root.replaceChildren();
      if (payload.current_signal?.data_stale_warning) { const node = document.createElement('div'); node.className = 'notice'; node.textContent = `Stale data: ${payload.current_signal.data_stale_warning}. Update through Data Status.`; root.appendChild(node); }
      if (payload.current_signal?.degraded_reason) { const node = document.createElement('div'); node.className = 'notice'; node.textContent = `Degraded: ${payload.current_signal.degraded_reason}.`; root.appendChild(node); }
    }
    function syncTabs(capabilities) {
      const required = { series: 'series', signals: 'signals', holdings: 'holdings', trades: 'trades', factors: 'factors', configuration: 'configuration' };
      const visible = [];
      $$('[role="tab"]').forEach((tab) => {
        const supported = capabilities.includes(required[tab.dataset.resource]);
        tab.hidden = !supported; tab.tabIndex = -1; tab.setAttribute('aria-selected', 'false');
        const panel = $(`#${tab.getAttribute('aria-controls')}`); panel.hidden = true;
        if (supported) visible.push(tab);
      });
      const first = sourceSelect.value === 'selection_factor' ? visible.find((tab) => tab.dataset.resource === 'factors') : visible[0];
      if (first) { first.hidden = false; first.tabIndex = 0; first.setAttribute('aria-selected', 'true'); $(`#${first.getAttribute('aria-controls')}`).hidden = false; }
    }
    async function openSnapshot() {
      if (!variantSelect.value) return; state.loadedTabs.clear(); state.snapshot = null; const params = new URLSearchParams({ variant_id: variantSelect.value, benchmark: $('#benchmark-select').value, resolution: $('#resolution-select').value });
      if ($('#start-date').value) params.set('start', $('#start-date').value); if ($('#end-date').value) params.set('end', $('#end-date').value);
      try {
        const payload = await requestJson(`/api/r0/sources/${encodeURIComponent(sourceSelect.value)}/strategies/${encodeURIComponent(strategySelect.value)}/snapshots?${params}`); state.snapshot = payload; message.hidden = true; $('#recovery-notice').hidden = true;
        setText($('#snapshot-state'), 'readable'); setText($('#snapshot-freshness'), payload.current_signal?.data_stale_warning ? 'stale' : 'current'); setText($('#snapshot-date'), payload.data_as_of);
        const values = [payload.metrics?.cumulative_return, payload.metrics?.total_return_pct, payload.metrics?.max_drawdown, payload.counts?.total_periods]; $$('#snapshot-metrics strong').forEach((node, index) => setText(node, format(values[index]))); warnings(payload);
        const strategy = strategies.find((item) => item.strategy_id === payload.strategy_id); const capabilities = strategy?.resource_capabilities || [];
        syncTabs(capabilities); localStorage.setItem('r0-last-snapshot', JSON.stringify({ source_id: payload.source_id, strategy_id: payload.strategy_id, snapshot_id: payload.snapshot_id }));
        await loadResource($('[role="tab"][aria-selected="true"]:not([hidden])')?.dataset.resource || 'configuration');
      } catch (error) {
        const bootstrap = error.body?.blocker_code === 'bootstrap_required';
        showMessage(message, bootstrap ? '首次使用需要先初始化数据：前往 Data Status，按页面提示 Preview 并确认 Update；这不是等待代码 Review。' : `${error.body?.error || 'snapshot_error'}: ${error.message}`, true); const variant = variants.find((item) => item.variant_id === variantSelect.value); renderVariantState(variant);
      }
    }
    async function loadResource(resource) {
      if (!state.snapshot || state.loadedTabs.has(resource)) return; const s = state.snapshot; const base = `/api/r0/sources/${encodeURIComponent(s.source_id)}/strategies/${encodeURIComponent(s.strategy_id)}/snapshots/${encodeURIComponent(s.snapshot_id)}`; const view = encodeURIComponent(s.view_id);
      try {
        if (resource === 'series') { const payload = await fetchAllPages(`${base}/series?view_id=${view}&kind=equity&window=full&resolution=${encodeURIComponent(s.canonical_view.resolution)}&limit=100`); drawCurve($('#snapshot-chart'), payload.items); renderTable($('#curve-table'), payload.items, ['date', 'value', 'return']); }
        if (resource === 'signals' && strategies.find((item) => item.strategy_id === s.strategy_id)?.resource_capabilities.includes('signals')) { const payload = await fetchAllPages(`${base}/signals?view_id=${view}&event=all&limit=100`); renderTable($('#signals-table'), payload.items, ['date', 'action', 'position', 'reason_summary', 'target_exposure']); }
        if (resource === 'trades' && strategies.find((item) => item.strategy_id === s.strategy_id)?.resource_capabilities.includes('trades')) { const payload = await fetchAllPages(`${base}/trades?view_id=${view}&projection=detail&limit=100`); renderTable($('#trades-table'), payload.items, ['date', 'action', 'etf_code', 'trade_price', 'cost_price', 'latest_price', 'quantity', 'trade_amount', 'fee_amount', 'commission', 'stamp', 'transfer', 'slippage_cost', 'blocked_by_limit', 'limit_delays', 'holding_value', 'realized_pnl', 'realized_pnl_pct', 'nav']); }
        if (resource === 'holdings' && s.source_id === 'selection') { const periods = await fetchAllPages(`${base}/holding-periods?view_id=${view}&window=full&limit=100`); const rows = []; for (const period of periods.items || []) { const payload = await fetchAllPages(`${period.stocks_url}&limit=100`); rows.push(...payload.items.map((item) => ({ period_id: period.period_id, ...item }))); } renderTable($('#holdings-table'), rows, ['period_id', 'code', 'name', 'market_label', 'industry_l2', 'factor_score', 'rank', 'selection_reason_summary', 'weight', 'return', 'pnl']); }
        if (resource === 'factors') { const root = $('#factors-table'); if (s.source_id === 'selection_factor') { const kind = s.strategy_id === 'sector_heat' ? 'sector_heat' : 'single_factor'; const payload = await fetchAllPages(`${base}/factors?view_id=${view}&kind=${kind}&limit=100`); renderTable(root, payload.items); } else { const metadata = await requestJson(`${base}/factors?view_id=${view}&kind=metadata`); const overview = await fetchAllPages(`${base}/factors?view_id=${view}&kind=overview&limit=100`); const pre = document.createElement('pre'); pre.className = 'plan-box'; pre.textContent = JSON.stringify(metadata.metadata, null, 2); const tableRoot = document.createElement('div'); renderTable(tableRoot, overview.items); const links = document.createElement('div'); links.className = 'card-actions'; (s.related_snapshot_links || []).forEach((link) => { const a = document.createElement('a'); a.className = 'button'; a.href = `/snapshots?source_id=${encodeURIComponent(link.source_id)}&strategy_id=${encodeURIComponent(link.strategy_id)}&initial_range=full`; a.textContent = `${link.strategy_id}: ${link.cache_state}`; links.appendChild(a); }); root.replaceChildren(pre, tableRoot, links); } }
        if (resource === 'configuration') { const payload = await requestJson(`${base}/configuration`); const pre = document.createElement('pre'); pre.className = 'plan-box'; pre.textContent = JSON.stringify(payload, null, 2); $('#configuration-list').replaceChildren(pre); }
        state.loadedTabs.add(resource);
      } catch (error) { const root = $(`#${resource}-table`) || $('#snapshot-message'); showMessage(root, error.message, true); }
    }
    async function pollOperation(operationId, recovery = false) {
      state.recoveryOperation = operationId; const notice = $('#recovery-notice'); notice.hidden = false; notice.setAttribute('aria-busy', 'true');
      try { const operation = await requestJson(`/api/r0/actions/${encodeURIComponent(operationId)}`); setText($('#recovery-plan'), operation.steps.map((step) => `${step.step_id}:${step.status}`).join(' · ')); if (['pending', 'running'].includes(operation.status)) { window.setTimeout(() => pollOperation(operationId, recovery), 1000); return; } notice.setAttribute('aria-busy', 'false'); if (operation.status === 'done') { await loadVariants(); } else { $('#retry-recovery').hidden = false; showMessage(message, `${operation.status}: ${operation.error_code || 'operation failed'}`, true); } } catch (error) { notice.setAttribute('aria-busy', 'false'); showMessage(message, `Operation status unknown: ${error.message}`, true); }
    }
    $('#recover-cache').addEventListener('click', async () => { const variant = variants.find((item) => item.variant_id === variantSelect.value); if (!variant || $('#recover-cache').disabled) return; $('#recover-cache').disabled = true; $('#recovery-notice').setAttribute('aria-busy', 'true'); try { const operation = await requestJson('/api/r0/actions/cache-recover', { method: 'POST', body: JSON.stringify({ source_id: variant.source_id, strategy_id: variant.strategy_id, variant_id: variant.variant_id }) }); await pollOperation(operation.operation_id, true); } catch (error) { $('#recover-cache').disabled = false; showMessage(message, error.message, true); } });
    $('#retry-recovery').addEventListener('click', async () => { if (!state.recoveryOperation) return; const operation = await requestJson(`/api/r0/actions/${state.recoveryOperation}/retry`, { method: 'POST' }); $('#retry-recovery').hidden = true; pollOperation(operation.operation_id, true); });
    sourceSelect.addEventListener('change', loadStrategies); strategySelect.addEventListener('change', loadVariants); variantSelect.addEventListener('change', () => renderVariantState(variants.find((item) => item.variant_id === variantSelect.value))); $('#open-snapshot').addEventListener('click', openSnapshot); initTabs(loadResource);
    try { await loadSources(); } catch (error) { showMessage(message, error.message, true); }
  }

  async function initLegacy() {
    const root = $('#artifact-grid'); if (!root) return;
    try { const payload = await requestJson('/api/r0/legacy-artifacts'); root.replaceChildren(); payload.items.forEach((artifact) => { const card = document.createElement('article'); card.className = 'card artifact-card'; const title = document.createElement('h3'); title.textContent = artifact.label; const meta = document.createElement('p'); meta.textContent = `${artifact.kind} · ${artifact.file_name} · ${artifact.viewer_mode} · ${artifact.download_state}`; card.append(title, meta); if (artifact.download_url) { const link = document.createElement('a'); link.className = 'button'; link.href = artifact.download_url; link.textContent = 'Download validated bytes'; card.appendChild(link); } root.appendChild(card); }); } catch (error) { showMessage(root, error.message, true); }
  }

  async function pollGeneric(operationId, root) {
    try {
      const operation = await requestJson(`/api/r0/actions/${encodeURIComponent(operationId)}`);
      setText(root, `${operation.status}: ${operation.steps.map((step) => `${step.step_id}=${step.status}`).join(', ')}`);
      if (['pending', 'running'].includes(operation.status)) { window.setTimeout(() => pollGeneric(operationId, root), 1000); return; }
    } catch (error) { setText(root, `unknown: ${error.message}`); }
  }

  function compactUpdateOperation(operation) {
    return {
      operation_id: operation.operation_id,
      kind: operation.kind || 'data-update',
      status: operation.status || 'pending',
      error_code: operation.error_code || null,
      created_at: operation.created_at || null,
      finished_at: operation.finished_at || null,
      steps: Array.isArray(operation.steps) ? operation.steps.map((step) => ({
        step_id: step.step_id, scope: step.scope || null, status: step.status || 'pending',
        progress: Number.isFinite(Number(step.progress)) ? Number(step.progress) : 0,
        message: step.message || null, error_code: step.error_code || null,
        blocked_by: Array.isArray(step.blocked_by) ? step.blocked_by : [],
      })) : [],
    };
  }

  function saveUpdateOperation(operation) {
    try { localStorage.setItem(UPDATE_OPERATION_STORAGE_KEY, JSON.stringify(compactUpdateOperation(operation))); } catch (_) {}
  }

  function loadUpdateOperation() {
    try {
      const value = JSON.parse(localStorage.getItem(UPDATE_OPERATION_STORAGE_KEY));
      return value && typeof value.operation_id === 'string' && value.operation_id ? value : null;
    } catch (_) { return null; }
  }

  function updateProgressPercent(operation) {
    const steps = Array.isArray(operation.steps) ? operation.steps : [];
    if (operation.status === 'done') return 100;
    if (!steps.length) return operation.status === 'pending' ? 0 : 1;
    return Math.max(0, Math.min(100, Math.round(steps.reduce((sum, step) => sum + Math.max(0, Math.min(100, Number(step.progress) || 0)), 0) / steps.length)));
  }

  function showUpdateProgressDialog() {
    const dialog = $('#update-progress-dialog');
    if (dialog && !dialog.open && typeof dialog.showModal === 'function') dialog.showModal();
  }

  function renderUpdateOperation(operation) {
    const compact = compactUpdateOperation(operation); const percent = updateProgressPercent(compact); const active = ['pending', 'running'].includes(compact.status);
    state.updateOperation = compact.operation_id; state.updateActive = active;
    const start = $('#start-update'); if (start) start.disabled = active || !state.updatePlanReady;
    setText($('#update-operation-id'), compact.operation_id); setText($('#update-operation-status'), compact.status); setText($('#update-operation-percent'), `${percent}%`);
    const bar = $('#update-progress-bar'); bar.value = percent; bar.textContent = `${percent}%`; bar.setAttribute('aria-valuenow', String(percent));
    setText($('#update-progress'), `${compact.status} · ${percent}% · operation ${compact.operation_id}`);
    $('#open-update-progress').hidden = false; $('#retry-update').hidden = !['partial', 'error'].includes(compact.status);
    const error = $('#update-operation-error'); error.hidden = !compact.error_code; setText(error, compact.error_code, '');
    const list = $('#update-step-list'); list.replaceChildren();
    compact.steps.forEach((step) => {
      const item = document.createElement('li'); item.className = `operation-step status-${step.status}`;
      const head = document.createElement('div'); const title = document.createElement('strong'); const status = document.createElement('span');
      title.textContent = `${step.step_id}${step.scope ? ` · ${step.scope}` : ''}`; status.textContent = `${step.status} · ${step.progress}%`; head.append(title, status);
      const progress = document.createElement('progress'); progress.max = 100; progress.value = step.progress; progress.setAttribute('aria-label', `${step.step_id} progress`);
      item.append(head, progress);
      if (step.message || step.error_code) { const detail = document.createElement('p'); detail.textContent = [step.message, step.error_code].filter(Boolean).join(' · '); item.appendChild(detail); }
      list.appendChild(item);
    });
    if (!compact.steps.length) { const item = document.createElement('li'); item.className = 'operation-step'; item.textContent = 'Operation accepted; waiting for the first status response.'; list.appendChild(item); }
    saveUpdateOperation(compact);
  }

  function renderUpdateUnavailable(cached, error) {
    if (cached) renderUpdateOperation(cached);
    state.updateActive = false; setText($('#update-operation-status'), 'unavailable');
    setText($('#update-progress'), `unavailable · last known ${cached?.status || 'unknown'} · operation ${cached?.operation_id || state.updateOperation || 'unknown'}`);
    const node = $('#update-operation-error'); showMessage(node, `当前服务 boot 无法继续查询此 operation：${error.message}。上方保留的是本浏览器最后已知状态。`, true);
    $('#retry-update').hidden = true; $('#open-update-progress').hidden = false;
  }

  function beginUpdateTracking(operationId, { cached = null, open = false } = {}) {
    state.updateOperation = operationId; const generation = ++state.updatePollGeneration;
    if (cached) renderUpdateOperation(cached); if (open) showUpdateProgressDialog();
    const poll = async () => {
      if (generation !== state.updatePollGeneration) return;
      try {
        const operation = await requestJson(`/api/r0/actions/${encodeURIComponent(operationId)}`);
        if (generation !== state.updatePollGeneration) return;
        renderUpdateOperation(operation);
        if (['pending', 'running'].includes(operation.status)) window.setTimeout(poll, 1000);
      } catch (error) {
        if (generation !== state.updatePollGeneration) return;
        if (error.status === 404) { renderUpdateUnavailable(loadUpdateOperation() || cached, error); return; }
        const node = $('#update-operation-error'); showMessage(node, `进度查询暂时失败：${error.message}。3 秒后重试。`, true); window.setTimeout(poll, 3000);
      }
    };
    poll();
  }

  async function initDataStatus() {
    const root = $('#data-status-list'); if (!root) return; let plan = null;
    try { const payload = await requestJson('/api/r0/data-status'); setText($('#status-checked-at'), payload.checked_at); root.replaceChildren(); payload.scope_status.forEach((scope) => { const row = document.createElement('div'); row.className = 'status-row'; ['scope', 'current_local_date', 'latest_expected_date', 'needs_update', 'reason'].forEach((key) => { const node = document.createElement(key === 'scope' ? 'strong' : 'span'); node.textContent = `${key}: ${format(scope[key])}`; row.appendChild(node); }); root.appendChild(row); }); const restart = payload.restart; setText($('#restart-capability'), restart.available ? 'Required topology verified.' : 'restart_unavailable: systemd socket activation, Type=notify, inherited listener, fixed helper/receipt and server post-send hook are not all available. No receipt, signal or process exit will occur.'); $('#restart-service').disabled = !restart.available; setText($('#restart-service'), restart.available ? 'Request restart' : 'Restart unavailable'); } catch (error) { showMessage(root, error.message, true); }
    $('#preview-update').addEventListener('click', async () => { const scopes = $$('#update-scopes input:checked').map((input) => input.value); if (!scopes.length) { state.updatePlanReady = false; $('#start-update').disabled = true; setText($('#update-plan'), 'Select at least one scope.'); return; } const params = new URLSearchParams({ scopes: scopes.join(','), force: String($('#update-force').checked) }); try { plan = await requestJson(`/api/r0/data-update-plan?${params}`); state.updatePlanReady = true; setText($('#update-plan'), JSON.stringify(plan, null, 2)); $('#start-update').disabled = state.updateActive; if (state.updateActive) setText($('#update-progress'), `operation ${state.updateOperation} 仍在执行；请先查看当前进度。`); } catch (error) { plan = null; state.updatePlanReady = false; $('#start-update').disabled = true; setText($('#update-plan'), error.message); } });
    $('#start-update').addEventListener('click', async () => { if (!plan || state.updateActive || !await confirmAction(`Public-unsafe action. Network pull/write/rebuild starts immediately; it does not wait for code review. Scopes: ${plan.resolved_scopes.join(', ')}. ETA ${plan.steps.reduce((sum, step) => sum + step.eta_seconds, 0)}s. Confirmation prevents mistakes only; it is not access control.`, 'Start update')) return; $('#start-update').disabled = true; $('#retry-update').hidden = true; try { const operation = await requestJson('/api/r0/actions/data-update', { method: 'POST', body: JSON.stringify({ scopes: plan.requested_scopes, force: plan.force }) }); setText($('#update-plan'), JSON.stringify(operation.resolved_plan, null, 2)); const cached = { operation_id: operation.operation_id, kind: 'data-update', status: 'pending', error_code: null, steps: (plan.steps || []).map((step) => ({ step_id: step.step_id, scope: step.scope, status: 'pending', progress: 0, message: 'Accepted; waiting for worker status.', error_code: null, blocked_by: [] })) }; renderUpdateOperation(cached); beginUpdateTracking(operation.operation_id, { cached, open: true }); } catch (error) { setText($('#update-progress'), error.message); $('#start-update').disabled = false; } });
    $('#retry-update').addEventListener('click', async () => { if (!state.updateOperation || !await confirmAction('Retry the failed/partial fixed update steps now? This immediately resumes allowed network/write work and does not wait for code review.', 'Retry failed steps')) return; try { const operation = await requestJson(`/api/r0/actions/${encodeURIComponent(state.updateOperation)}/retry`, { method: 'POST' }); $('#retry-update').hidden = true; const cached = { operation_id: operation.operation_id, kind: 'data-update', status: 'pending', error_code: null, steps: [], created_at: null, finished_at: null }; renderUpdateOperation(cached); beginUpdateTracking(operation.operation_id, { cached, open: true }); } catch (error) { const node = $('#update-operation-error'); showMessage(node, error.message, true); showUpdateProgressDialog(); } });
    $('#open-update-progress').addEventListener('click', showUpdateProgressDialog);
    $('#minimize-update-progress').addEventListener('click', () => $('#update-progress-dialog').close());
    $$('#update-scopes input, #update-force').forEach((input) => input.addEventListener('change', () => { plan = null; state.updatePlanReady = false; $('#start-update').disabled = true; setText($('#update-plan'), 'Selection changed. Preview the fixed plan again before updating.'); }));
    $('#restart-service').addEventListener('click', async () => { if ($('#restart-service').disabled || !await confirmAction('Public-unsafe restart request. A successful 202 is complete only after the boot id changes and health is ready. Confirmation is not access control.', 'Request restart')) return; try { const operation = await requestJson('/api/r0/actions/restart', { method: 'POST' }); setText($('#restart-progress'), `scheduled: ${operation.operation_id}`); pollGeneric(operation.operation_id, $('#restart-progress')); } catch (error) { setText($('#restart-progress'), `${error.body?.error || 'restart_error'}: ${error.message}`); } });
    const stored = loadUpdateOperation(); if (stored) beginUpdateTracking(stored.operation_id, { cached: stored, open: ['pending', 'running'].includes(stored.status) });
  }

  async function initManual() {
    const root = $('#manual-records-table'); if (!root) return; const config = JSON.parse($('#manual-config').textContent); const select = $('#manual-strategy'); const message = $('#manual-message'); let pendingCreate = null;
    async function loadCapabilities() { const payload = await requestJson('/api/r0/manual-records/capabilities'); select.replaceChildren(); payload.items.forEach((item) => { const option = document.createElement('option'); option.value = item.strategy_id; option.textContent = `${item.display_name} (${item.currency})`; select.appendChild(option); }); select.value = payload.items.some((item) => item.strategy_id === config.strategy) ? config.strategy : payload.items[0]?.strategy_id || ''; }
    async function loadRecords() { if (!select.value) return; try { const payload = await fetchAllPages(`/api/r0/manual-records?strategy=${encodeURIComponent(select.value)}&limit=100`); setText($('#manual-record-count'), payload.total, '0'); renderTable(root, payload.items, ['date', 'strategy', 'capital', 'actual_position', 'exec_price', 'shares', 'notes', 'created_at'], (row, cell) => { const button = document.createElement('button'); button.type = 'button'; button.className = 'button'; button.textContent = 'Delete'; button.addEventListener('click', async () => { if (!await confirmAction('Irreversibly delete this payload? This public, unauthenticated action has no identity attribution or recovery.', 'Delete record')) return; try { await requestJson(`/api/r0/manual-records/${encodeURIComponent(row.record_id)}`, { method: 'DELETE' }); await loadRecords(); } catch (error) { showMessage(message, error.message, true); } }); cell.appendChild(button); }); await loadSignalAndReconciliation(); } catch (error) { showMessage(message, error.message, true); } }
    async function loadSignalAndReconciliation() { let context = null; try { context = JSON.parse(localStorage.getItem('r0-last-snapshot')); } catch (_) {} if (!context || context.strategy_id !== select.value) { setText($('#signal-reference'), 'Open an exact snapshot for this strategy first.'); setText($('#reconciliation-view'), 'No exact snapshot selected.'); return; } try { const signal = await requestJson(`/api/r0/manual-records/signal-reference?strategy=${encodeURIComponent(select.value)}&snapshot_id=${encodeURIComponent(context.snapshot_id)}`); const pre = document.createElement('pre'); pre.className = 'plan-box'; pre.textContent = JSON.stringify(signal, null, 2); $('#signal-reference').replaceChildren(pre); const summary = await requestJson(`/api/r0/manual-records/reconciliation?strategy=${encodeURIComponent(select.value)}&snapshot_id=${encodeURIComponent(context.snapshot_id)}`); const series = await fetchAllPages(`${summary.series_url}&limit=100`); renderTable($('#reconciliation-view'), series.items, ['date', 'strategy_nav', 'manual_nav', 'actual_position', 'strategy_target', 'external_flow']); } catch (error) { setText($('#signal-reference'), error.message); setText($('#reconciliation-view'), error.message); } }
    $('#manual-form').addEventListener('submit', async (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); const body = { date: form.get('date'), strategy: select.value, capital: form.get('capital'), signal_target: form.get('signal_target') || null, exec_price: form.get('exec_price') || null, shares: form.get('shares') || null, actual_position: form.get('actual_position') || null, notes: form.get('notes') }; const serialized = JSON.stringify(body); if (!pendingCreate || pendingCreate.body !== serialized) pendingCreate = { body: serialized, key: crypto.randomUUID() }; try { await requestJson('/api/r0/manual-records', { method: 'POST', headers: { 'Idempotency-Key': pendingCreate.key }, body: pendingCreate.body }); pendingCreate = null; event.currentTarget.reset(); $('#record-capital').value = '50000'; await loadRecords(); } catch (error) { showMessage(message, error.message, true); } });
    $('#refresh-records').addEventListener('click', loadRecords); select.addEventListener('change', loadRecords); await loadCapabilities(); await loadRecords();
  }

  initNavigation();
  const page = document.body.dataset.page;
  if (page === 'snapshot') initSnapshot();
  if (page === 'legacy') initLegacy();
  if (page === 'data-status') initDataStatus();
  if (page === 'manual-records') initManual();
})();
