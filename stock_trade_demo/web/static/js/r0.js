(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function text(node, value) {
    if (node) node.textContent = value == null || value === '' ? 'unknown' : String(value);
  }

  function formatBytes(value) {
    if (value == null || Number.isNaN(Number(value))) return 'unknown';
    const bytes = Number(value);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function formatCell(value) {
    if (value == null || value === '') return 'unknown';
    if (typeof value === 'object') {
      const rendered = JSON.stringify(value);
      return rendered.length > 140 ? `${rendered.slice(0, 137)}…` : rendered;
    }
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
    return String(value);
  }

  async function getJson(url) {
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
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

  function initNavigation() {
    const button = $('[data-menu-button]');
    const sidebar = $('[data-sidebar]');
    if (!button || !sidebar) return;
    button.addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      button.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
        button.setAttribute('aria-expanded', 'false');
        button.focus();
      }
    });
  }

  function initTabs() {
    const tabs = $$('[role="tab"]');
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        tabs.forEach((item) => item.setAttribute('aria-selected', item === tab ? 'true' : 'false'));
        $$('[role="tabpanel"]').forEach((panel) => {
          panel.hidden = panel.id !== tab.getAttribute('aria-controls');
        });
      });
    });
  }

  function drawCurve(container, points) {
    container.replaceChildren();
    if (!Array.isArray(points) || points.length < 2) {
      const empty = document.createElement('div');
      empty.className = 'chart-empty';
      empty.textContent = 'No cached curve is available.';
      container.appendChild(empty);
      return;
    }

    const width = 960;
    const height = 340;
    const pad = { top: 24, right: 22, bottom: 42, left: 58 };
    const values = points.map((point) => Number(point.value)).filter(Number.isFinite);
    if (values.length < 2) return drawCurve(container, []);
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) { min -= 0.05; max += 0.05; }

    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('class', 'chart-svg');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `Cached curve from ${points[0].date} to ${points[points.length - 1].date}`);

    const x = (index) => pad.left + (index / (points.length - 1)) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (1 - (value - min) / (max - min)) * (height - pad.top - pad.bottom);

    for (let lineIndex = 0; lineIndex <= 4; lineIndex += 1) {
      const ratio = lineIndex / 4;
      const yPos = pad.top + ratio * (height - pad.top - pad.bottom);
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', String(pad.left));
      line.setAttribute('x2', String(width - pad.right));
      line.setAttribute('y1', String(yPos));
      line.setAttribute('y2', String(yPos));
      line.setAttribute('stroke', '#e3e8ed');
      line.setAttribute('stroke-width', '1');
      svg.appendChild(line);

      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', String(pad.left - 10));
      label.setAttribute('y', String(yPos + 4));
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('fill', '#657386');
      label.setAttribute('font-size', '11');
      label.textContent = (max - ratio * (max - min)).toFixed(2);
      svg.appendChild(label);
    }

    const polygon = document.createElementNS(ns, 'polygon');
    const linePoints = points.map((point, index) => `${x(index)},${y(Number(point.value))}`).join(' ');
    polygon.setAttribute('points', `${pad.left},${height - pad.bottom} ${linePoints} ${width - pad.right},${height - pad.bottom}`);
    polygon.setAttribute('fill', 'rgba(29,109,98,0.10)');
    svg.appendChild(polygon);

    const polyline = document.createElementNS(ns, 'polyline');
    polyline.setAttribute('points', linePoints);
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', '#1d6d62');
    polyline.setAttribute('stroke-width', '2.5');
    polyline.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(polyline);

    [0, points.length - 1].forEach((index) => {
      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', String(x(index)));
      label.setAttribute('y', String(height - 14));
      label.setAttribute('text-anchor', index === 0 ? 'start' : 'end');
      label.setAttribute('fill', '#657386');
      label.setAttribute('font-size', '11');
      label.textContent = points[index].date;
      svg.appendChild(label);
    });
    container.appendChild(svg);
  }

  function renderTable(container, rows, preferredColumns = []) {
    container.replaceChildren();
    if (!Array.isArray(rows) || rows.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'No cached rows are available.';
      container.appendChild(empty);
      return;
    }
    const available = Array.from(new Set(rows.flatMap((row) => Object.keys(row || {}))));
    const columns = [
      ...preferredColumns.filter((column) => available.includes(column)),
      ...available.filter((column) => !preferredColumns.includes(column)),
    ].slice(0, 8);

    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    columns.forEach((column) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = column.replaceAll('_', ' ');
      headRow.appendChild(th);
    });
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      columns.forEach((column) => {
        const td = document.createElement('td');
        td.textContent = formatCell(row[column]);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  function renderConfiguration(container, rows) {
    container.replaceChildren();
    if (!Array.isArray(rows) || rows.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'Configuration metadata is unavailable.';
      container.appendChild(empty);
      return;
    }
    const list = document.createElement('div');
    list.className = 'config-list';
    rows.forEach((row) => {
      const item = document.createElement('div');
      item.className = 'config-row';
      const label = document.createElement('strong');
      label.textContent = row.label || row.key || 'unknown';
      const value = document.createElement('code');
      value.textContent = formatCell(row.value);
      const description = document.createElement('span');
      description.textContent = row.description || 'No description is stored.';
      item.append(label, value, description);
      list.appendChild(item);
    });
    container.appendChild(list);
  }

  async function initSnapshotPage() {
    const configNode = $('#snapshot-config');
    if (!configNode) return;
    const config = JSON.parse(configNode.textContent);
    const select = $('#strategy-select');
    const status = $('#snapshot-status');
    const date = $('#snapshot-date');
    const metricsRoot = $('#snapshot-metrics');
    const chart = $('#snapshot-chart');

    async function loadSnapshot(strategyId) {
      text(status, 'loading');
      status.className = 'badge warning';
      try {
        const payload = await getJson(`/api/r0/sources/${encodeURIComponent(config.id)}/snapshots/${encodeURIComponent(strategyId)}`);
        text(status, payload.available ? (payload.evidence_status || 'unverified') : 'unavailable');
        status.className = payload.available ? 'badge warning' : 'badge danger';
        text(date, payload.data_as_of || 'unknown');
        const metricMap = [
          ['Cumulative value', payload.metrics?.cumulative_value],
          ['Total return', payload.metrics?.total_return],
          ['Max drawdown', payload.metrics?.max_drawdown],
          ['Observations', payload.metrics?.observations],
        ];
        $$('.metric-card', metricsRoot).forEach((card, index) => {
          text($('span', card), metricMap[index][0]);
          text($('strong', card), metricMap[index][1]);
        });
        drawCurve(chart, payload.equity_curve || []);
        renderTable($('#holdings-table'), payload.holdings || [], ['date', 'code', 'name', 'position_label', 'target_exposure', 'holding_units', 'holding_value']);
        renderTable($('#trades-table'), payload.trades || [], ['date', 'action', 'target_exposure', 'trade_price', 'quantity', 'fee_amount', 'nav']);
        renderConfiguration($('#configuration-list'), payload.configuration || []);
        const errorBox = $('#snapshot-error');
        if (payload.available) {
          errorBox.hidden = true;
        } else {
          text(errorBox, payload.message || 'No prebuilt snapshot is available.');
          errorBox.hidden = false;
        }
      } catch (error) {
        text(status, 'unavailable');
        status.className = 'badge danger';
        text(date, 'unknown');
        $$('.metric-card strong', metricsRoot).forEach((node) => text(node, 'unknown'));
        drawCurve(chart, []);
        renderTable($('#holdings-table'), []);
        renderTable($('#trades-table'), []);
        renderConfiguration($('#configuration-list'), []);
        const errorBox = $('#snapshot-error');
        text(errorBox, error.message);
        errorBox.hidden = false;
      }
    }

    try {
      const payload = await getJson(`/api/r0/sources/${encodeURIComponent(config.id)}/strategies`);
      select.replaceChildren();
      (payload.strategies || []).forEach((strategy) => {
        const option = document.createElement('option');
        option.value = strategy.id;
        const storedName = strategy.name || '';
        const label = /^[\x20-\x7e]+$/.test(storedName) ? storedName : strategy.id;
        option.textContent = `${label}${strategy.available ? '' : ' (unavailable)'}`;
        select.appendChild(option);
      });
      const preferred = (payload.strategies || []).find((item) => item.id === config.default_strategy);
      select.value = preferred ? preferred.id : select.options[0]?.value || '';
      select.disabled = select.options.length <= 1;
      if (select.value) await loadSnapshot(select.value);
      else throw new Error('No fixed strategy metadata is available.');
    } catch (error) {
      const errorBox = $('#snapshot-error');
      text(errorBox, error.message);
      errorBox.hidden = false;
      text(status, 'unavailable');
    }
    select.addEventListener('change', () => loadSnapshot(select.value));
  }

  async function initLegacyPage() {
    const root = $('#artifact-grid');
    if (!root) return;
    try {
      const payload = await getJson('/api/r0/artifacts');
      root.replaceChildren();
      (payload.artifacts || []).forEach((artifact) => {
        const card = document.createElement('article');
        card.className = 'card artifact-card';
        const kicker = document.createElement('div');
        kicker.className = 'card-kicker';
        kicker.textContent = artifact.kind;
        const title = document.createElement('h3');
        title.textContent = artifact.label;
        const file = document.createElement('p');
        file.textContent = artifact.file_name;
        const tags = document.createElement('div');
        tags.className = 'artifact-tags';
        [artifact.evidence_status, artifact.provenance, artifact.available ? formatBytes(artifact.size_bytes) : 'unavailable'].forEach((value, index) => {
          const badge = document.createElement('span');
          badge.className = `badge ${index < 2 ? 'warning' : ''}`;
          badge.textContent = value;
          tags.appendChild(badge);
        });
        const actions = document.createElement('div');
        actions.className = 'card-actions';
        if (artifact.download_url) {
          const link = document.createElement('a');
          link.className = 'button';
          link.href = artifact.download_url;
          link.textContent = 'Download checked file';
          actions.appendChild(link);
        } else {
          const disabled = document.createElement('span');
          disabled.className = 'button';
          disabled.setAttribute('aria-disabled', 'true');
          disabled.textContent = 'No public download';
          actions.appendChild(disabled);
        }
        card.append(kicker, title, file, tags, actions);
        root.appendChild(card);
      });
    } catch (error) {
      root.textContent = error.message;
      root.className = 'empty';
    }
  }

  function buildStatusRows(root, sources) {
    root.replaceChildren();
    sources.forEach((source) => {
      const row = document.createElement('div');
      row.className = 'status-row';
      const title = document.createElement('strong');
      title.textContent = source.label;
      const available = document.createElement('span');
      available.textContent = source.available ? 'file present' : 'unavailable';
      const dataDate = document.createElement('span');
      dataDate.textContent = `data as of: ${source.data_as_of}`;
      const vintage = document.createElement('span');
      vintage.textContent = `vintage: ${source.source_vintage}`;
      const evidence = document.createElement('span');
      evidence.textContent = source.evidence_status;
      row.append(title, available, dataDate, vintage, evidence);
      root.appendChild(row);
    });
  }

  async function initDataStatusPage() {
    const root = $('#data-status-list');
    if (!root) return;
    try {
      const payload = await getJson('/api/r0/data-status');
      text($('#status-checked-at'), payload.checked_at);
      buildStatusRows(root, payload.sources || []);
    } catch (error) {
      root.textContent = error.message;
      root.className = 'empty';
    }
  }

  async function initManualRecordsPage() {
    const root = $('#manual-records-table');
    if (!root) return;
    try {
      const payload = await getJson('/api/r0/manual-records');
      if (!payload.available) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = payload.message || 'Manual record storage is unavailable.';
        root.replaceChildren(empty);
        text($('#manual-record-count'), payload.evidence_status || 'unverified');
        return;
      }
      renderTable(root, payload.records || [], ['date', 'strategy', 'signal_target', 'actual_position', 'exec_price', 'shares', 'notes', 'created_at']);
      text($('#manual-record-count'), (payload.records || []).length);
    } catch (error) {
      root.textContent = error.message;
      root.className = 'empty';
    }
  }

  initNavigation();
  initTabs();
  const page = document.body.dataset.page;
  if (page === 'snapshot') initSnapshotPage();
  if (page === 'legacy') initLegacyPage();
  if (page === 'data-status') initDataStatusPage();
  if (page === 'manual-records') initManualRecordsPage();
})();
