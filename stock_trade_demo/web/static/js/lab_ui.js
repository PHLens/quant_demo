(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const ACTIVE_RUN_KEY = 'quantLabActiveRunId';

  async function api(url, options = {}) {
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.details?.unknown_fields
        ? `（未知字段：${payload.details.unknown_fields.join(', ')}）`
        : '';
      throw new Error(`${payload.message || payload.error || `HTTP ${response.status}`}${detail}`);
    }
    return payload;
  }

  function shortHash(value) {
    if (!value) return '—';
    return value.length > 24 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value;
  }

  function windowText(value) {
    if (!value) return '—';
    return `${value.start} → ${value.end}`;
  }

  function metricText(name, value, signed = false) {
    if (value === null || value === undefined) return '—';
    const percentNames = new Set([
      'turnover', 'fee_drag', 'max_drawdown', 'cumulative_return',
      'annual_return', 'benchmark_active_return',
    ]);
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (percentNames.has(name)) {
      const sign = signed && number > 0 ? '+' : '';
      return `${sign}${(number * 100).toFixed(2)}%`;
    }
    if (name === 'total_cost') {
      const sign = signed && number > 0 ? '+' : '';
      return `${sign}¥${number.toFixed(2)}`;
    }
    if (Number.isInteger(number)) {
      const sign = signed && number > 0 ? '+' : '';
      return `${sign}${number}`;
    }
    const sign = signed && number > 0 ? '+' : '';
    return `${sign}${number.toFixed(4)}`;
  }

  async function initLearn() {
    const page = $('[data-learn-page]');
    if (!page) return;
    const grid = $('[data-learn-topics]', page);
    try {
      const payload = await api('/api/learn/topics');
      grid.replaceChildren();
      payload.topics.forEach((topic, index) => {
        const article = document.createElement('article');
        article.className = 'lab-card lab-topic-card';
        const number = document.createElement('span');
        number.className = 'lab-topic-card__number';
        number.textContent = String(index + 1).padStart(2, '0');
        const title = document.createElement('h3');
        title.textContent = topic.title;
        const summary = document.createElement('p');
        summary.textContent = topic.summary;
        const check = document.createElement('small');
        check.textContent = `检查：${topic.check}`;
        article.append(number, title, summary, check);
        grid.append(article);
      });
    } catch (error) {
      grid.textContent = `学习主题读取失败：${error.message}`;
    }
  }

  function fillTemplateIdentity(page, template) {
    const snapshot = template.snapshot;
    const mappings = [
      ['[data-snapshot-id]', snapshot.snapshot_id],
      ['[data-snapshot-hash]', snapshot.content_sha256],
      ['[data-input-window]', windowText(snapshot.input_window)],
      ['[data-validation-window]', windowText(snapshot.validation_window)],
      ['[data-baseline-id]', template.baseline_result_id],
      ['[data-trial-count]', String(template.trial_count)],
    ];
    mappings.forEach(([selector, value]) => {
      const node = $(selector, page);
      if (node) {
        node.textContent = value;
        node.title = value;
      }
    });
    const summary = $('[data-contract-summary]', page);
    if (summary) {
      summary.textContent = [
        `snapshot ${shortHash(snapshot.snapshot_id)}`,
        `validation ${windowText(snapshot.validation_window)}`,
        '二元仓位',
        '信号 close(t) / 成交 ETF open(t+1)',
        'OOS=not_configured',
      ].join(' · ');
    }
  }

  function renderRunStatus(page, run) {
    $('[data-run-empty]', page).hidden = true;
    const detail = $('[data-run-detail]', page);
    detail.hidden = false;
    const status = $('[data-run-status]', page);
    status.textContent = run.status;
    status.parentElement.dataset.state = run.status;
    $('[data-run-id]', page).textContent = run.run_id;
    $('[data-result-id]', page).textContent = run.result_id || '等待 worker 生成';
  }

  function renderRunMetrics(page, metrics) {
    const grid = $('[data-run-metrics]', page);
    grid.replaceChildren();
    [
      ['validation_bars', 'Bars'],
      ['signal_switch_count', 'Signal switches'],
      ['trade_count', 'Trades'],
      ['turnover', 'Turnover'],
      ['max_drawdown', 'Max drawdown'],
      ['total_cost', 'Total cost'],
    ].forEach(([key, label]) => {
      const box = document.createElement('div');
      box.className = 'lab-metric';
      const name = document.createElement('span');
      name.textContent = label;
      const value = document.createElement('strong');
      value.textContent = metricText(key, metrics[key]);
      box.append(name, value);
      grid.append(box);
    });
  }

  async function pollRun(page, runId) {
    const deadline = Date.now() + 120000;
    let lastError = null;
    while (Date.now() < deadline) {
      try {
        const run = await api(`/api/lab/runs/${encodeURIComponent(runId)}`);
        renderRunStatus(page, run);
        if (run.status === 'success' || run.status === 'skipped') {
          const result = await api(`/api/lab/results/${encodeURIComponent(run.result_id)}`);
          renderRunMetrics(page, result.content.evidence.metrics);
          localStorage.setItem('quantLabCandidateResultId', run.result_id);
          localStorage.removeItem(ACTIVE_RUN_KEY);
          const link = $('[data-compare-link]', page);
          link.href = `/compare?candidate=${encodeURIComponent(run.result_id)}`;
          link.hidden = false;
          return run;
        }
        if (run.status === 'failed') {
          localStorage.removeItem(ACTIVE_RUN_KEY);
          throw new Error(run.error?.message || 'worker run failed');
        }
        lastError = null;
      } catch (error) {
        if (!localStorage.getItem(ACTIVE_RUN_KEY)) throw error;
        lastError = error;
        const status = $('[data-run-status]', page);
        status.textContent = 'reconnecting';
        status.parentElement.dataset.state = 'running';
      }
      await new Promise(resolve => window.setTimeout(resolve, 300));
    }
    throw new Error(
      lastError
        ? `运行仍在后台执行，网络恢复后刷新页面将继续查询：${lastError.message}`
        : '运行仍在后台执行，刷新页面将用已保存的 run_id 继续查询',
    );
  }

  async function initLab() {
    const page = $('[data-lab-page]');
    if (!page) return;
    const form = $('[data-experiment-form]', page);
    const errorBox = $('[data-form-error]', page);
    const submit = $('button[type="submit"]', form);
    let template;
    submit.disabled = true;
    submit.textContent = '正在读取模板…';

    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (!template || submit.disabled) return;
      errorBox.hidden = true;
      submit.disabled = true;
      submit.textContent = '正在冻结假设…';
      const values = new FormData(form);
      try {
        const experiment = await api('/api/lab/experiments', {
          method: 'POST',
          body: JSON.stringify({
            template_id: template.template_id,
            title: values.get('title'),
            hypothesis: {
              statement: values.get('statement'),
              primary_observable: 'signal_switch_count',
              expected_direction: values.get('expected_direction'),
              falsification_condition: values.get('falsification_condition'),
              validation_window: template.fixed_policy.validation_window,
            },
          }),
        });
        submit.textContent = '正在创建单变量 Variant…';
        const variant = await api(
          `/api/lab/experiments/${encodeURIComponent(experiment.experiment_id)}/variants`,
          {
            method: 'POST',
            body: JSON.stringify({
              patch: { trend_window: Number(values.get('trend_window')) },
            }),
          },
        );
        submit.textContent = '正在提交异步 run…';
        const run = await api(`/api/lab/variants/${encodeURIComponent(variant.variant_id)}/runs`, {
          method: 'POST',
          body: JSON.stringify({}),
        });
        localStorage.setItem(ACTIVE_RUN_KEY, run.run_id);
        renderRunStatus(page, run);
        submit.textContent = 'Worker 运行中…';
        await pollRun(page, run.run_id);
        const freshTemplate = (await api('/api/lab/templates')).templates[0];
        template = freshTemplate;
        fillTemplateIdentity(page, freshTemplate);
        submit.textContent = '已完成，可再次提交新假设';
      } catch (error) {
        errorBox.hidden = false;
        errorBox.textContent = error.message;
        submit.textContent = localStorage.getItem(ACTIVE_RUN_KEY)
          ? '后台运行中，刷新可继续查询'
          : '重新提交 validation';
      } finally {
        submit.disabled = Boolean(localStorage.getItem(ACTIVE_RUN_KEY));
      }
    });

    try {
      const payload = await api('/api/lab/templates');
      template = payload.templates[0];
      fillTemplateIdentity(page, template);
    } catch (error) {
      errorBox.hidden = false;
      errorBox.textContent = `模板读取失败：${error.message}`;
      $$('button, input, textarea, select', form).forEach(node => { node.disabled = true; });
      return;
    }

    const activeRunId = localStorage.getItem(ACTIVE_RUN_KEY);
    if (activeRunId) {
      submit.textContent = '正在恢复后台 run…';
      try {
        await pollRun(page, activeRunId);
        const freshTemplate = (await api('/api/lab/templates')).templates[0];
        template = freshTemplate;
        fillTemplateIdentity(page, freshTemplate);
        submit.textContent = '后台 run 已恢复，可再次提交新假设';
      } catch (error) {
        errorBox.hidden = false;
        errorBox.textContent = error.message;
        submit.textContent = localStorage.getItem(ACTIVE_RUN_KEY)
          ? '后台运行中，刷新可继续查询'
          : '重新提交 validation';
      }
    } else {
      submit.textContent = '冻结假设并启动 validation';
    }
    submit.disabled = Boolean(localStorage.getItem(ACTIVE_RUN_KEY));
  }

  const metricLabels = {
    validation_bars: 'Validation bars',
    signal_switch_count: 'Signal switches · 主观察量',
    trade_count: 'Completed trades',
    turnover: 'Turnover',
    total_cost: 'Total cost',
    fee_drag: 'Fee drag',
    max_drawdown: 'Max drawdown',
    calmar: 'Calmar',
    cumulative_return: 'Cumulative return',
    annual_return: 'Annual return',
    benchmark_active_return: 'Benchmark active return',
  };

  function renderComparison(page, comparison) {
    const gate = $('[data-gate-card]', page);
    gate.dataset.state = comparison.status;
    $('[data-gate-title]', page).textContent = comparison.comparable
      ? 'Comparable · 口径一致'
      : 'Not comparable · 不显示差值';
    $('[data-gate-copy]', page).textContent = comparison.comparable
      ? '唯一变化字段为 trend_window，可以阅读 validation 绝对值与差值。'
      : `发现 ${comparison.mismatches.length} 个合同不一致字段。`;

    const identity = $('[data-compare-identity]', page);
    identity.hidden = false;
    const snapshot = comparison.data_snapshot || {};
    $('[data-compare-snapshot]', page).textContent = snapshot.snapshot_id || '—';
    $('[data-compare-hash]', page).textContent = snapshot.content_sha256 || '—';
    $('[data-compare-input]', page).textContent = windowText(snapshot.input_window);
    $('[data-compare-window]', page).textContent = windowText(comparison.evaluation?.validation_window);
    $('[data-compare-baseline-id]', page).textContent = comparison.baseline_result_id;

    const section = $('[data-comparison-results]', page);
    section.hidden = false;
    const tbody = $('[data-comparison-body]', page);
    tbody.replaceChildren();
    Object.keys(metricLabels).forEach(name => {
      const row = document.createElement('tr');
      const label = document.createElement('td');
      const baseline = document.createElement('td');
      const candidate = document.createElement('td');
      const delta = document.createElement('td');
      label.textContent = metricLabels[name];
      baseline.textContent = metricText(name, comparison.baseline.metrics[name]);
      candidate.textContent = metricText(name, comparison.candidate.metrics[name]);
      delta.textContent = comparison.comparable
        ? metricText(name, comparison.deltas?.[name], true)
        : '口径不一致，不计算';
      row.append(label, baseline, candidate, delta);
      tbody.append(row);
    });
    $('[data-interpretation-note]', page).textContent = comparison.interpretation.note;
  }

  async function initCompare() {
    const page = $('[data-compare-page]');
    if (!page) return;
    const form = $('[data-compare-form]', page);
    const errorBox = $('[data-compare-error]', page);
    const baselineInput = $('[data-compare-baseline]', page);
    const candidateInput = $('[data-compare-candidate]', page);
    let baselineId;
    try {
      const payload = await api('/api/lab/templates');
      baselineId = payload.templates[0].baseline_result_id;
      baselineInput.value = baselineId;
      const candidate = new URLSearchParams(window.location.search).get('candidate')
        || localStorage.getItem('quantLabCandidateResultId');
      if (candidate) candidateInput.value = candidate;
    } catch (error) {
      errorBox.hidden = false;
      errorBox.textContent = `比较合同读取失败：${error.message}`;
      return;
    }

    form.addEventListener('submit', async event => {
      event.preventDefault();
      errorBox.hidden = true;
      const button = $('button[type="submit"]', form);
      button.disabled = true;
      button.textContent = '正在核对合同…';
      try {
        const comparison = await api('/api/lab/comparisons', {
          method: 'POST',
          body: JSON.stringify({
            baseline_result_id: baselineId,
            candidate_result_id: candidateInput.value.trim(),
          }),
        });
        renderComparison(page, comparison);
      } catch (error) {
        errorBox.hidden = false;
        errorBox.textContent = error.message;
      } finally {
        button.disabled = false;
        button.textContent = '执行可比性检查';
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initLearn();
    initLab();
    initCompare();
  });
})();
