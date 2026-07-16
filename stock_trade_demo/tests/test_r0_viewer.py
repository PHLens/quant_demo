"""v0.1 canonical Viewer, state tuple, pagination and zero-side-effect contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import pandas as pd
import pytest

from web import state
from web.app import create_app
from web.v01.catalog import get_strategy, variants_for
from web.v01.snapshot_store import PUBLISHED_SIGNAL_KEYS, publish_entry


@pytest.fixture()
def app(tmp_path):
    app = create_app()
    app.config.update(
        TESTING=True,
        R0_GENERATION_ROOT=tmp_path / 'generations',
        R0_MANUAL_LEDGER_PATH=tmp_path / 'manual.csv',
        R0_ACTION_MARKER_PATH=tmp_path / 'active-operation.json',
        R0_CURSOR_KEY='test-cursor-key',
    )
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def _signal(strategy_id='csi1000_timing', *, stale=None, degraded=None, target=1.0):
    payload = {key: None for key in PUBLISHED_SIGNAL_KEYS}
    payload.update({
        'strategy_id': strategy_id, 'name': strategy_id, 'index_name': 'Index',
        'etf_code': '510500', 'etf_name': 'ETF', 'as_of_date': '2026-01-04',
        'settled_as_of_date': '2026-01-03', 'data_stale_warning': stale,
        'degraded_reason': degraded, 'target_exposure': target,
        'prev_exposure': 0.5, 'exposure_delta': target - 0.5,
        'rebalance_action': 'add', 'rebalance_label': 'Add', 'signal_action': 'buy',
        'signal_label': 'Buy', 'current_action': 'buy', 'current_position': 1,
        'current_reason': 'stored', 'reason_summary': 'stored', 'bullish_score': 0.8,
        'ref_close': 10.0, 'ref_open': 10.2, 'nav': 1.1, 'settled_nav': 1.09,
        'status': 'research', 'passes_rule14': None, 'exec_basis': 'next_open',
    })
    return payload


def _timing_frame(*, target_state=None, rows=5):
    dates = pd.date_range('2026-01-01', periods=rows, freq='D')
    frame = pd.DataFrame({
        '交易日期': dates, '累积净值': [1 + index * 0.01 for index in range(rows)],
        'strategy_return': [0.0] + [0.01] * (rows - 1), 'signal_action': ['hold'] * (rows - 1) + ['buy'],
        'position': [0] * (rows - 1) + [1], 'reason_summary': ['stored'] * rows,
        'reason_detail': [[] for _ in range(rows)], 'signal_score': [0.5] * rows,
        'strength_score': [0.6] * rows, 'target_exposure': [0.0] * (rows - 1) + [1.0],
        'prev_exposure': [0.0] * rows, 'exposure_change': [0.0] * (rows - 1) + [1.0],
        'rebalance_action': ['flat'] * (rows - 1) + ['enter'], 'close': [10 + index for index in range(rows)],
        'etf_open': [1 + index * 0.1 for index in range(rows)], 'etf_close': [1.05 + index * 0.1 for index in range(rows)],
        'trade_quantity': [0] * (rows - 1) + [100], 'trade_amount': [0] * (rows - 1) + [1000],
        'trade_fee_amount': [0] * (rows - 1) + [1], 'holding_value': [0] * (rows - 1) + [1000],
        'cash_balance': [50000] * rows, 'index_id': ['csi1000'] * rows, 'index_name': ['CSI 1000'] * rows,
    })
    target_state = target_state or {'cache_state': 'ready', 'readable': True, 'freshness_state': 'current', 'freshness_reason': None, 'degradation': None}
    frame.attrs['r0_target_state'] = target_state
    frame.attrs['r0_generated_at'] = '2026-01-05T00:00:00Z'
    frame.attrs['published_current_signal'] = _signal(
        stale=target_state['freshness_reason']['code'] if target_state['freshness_state'] == 'stale' else None,
        degraded=target_state['degradation']['code'] if target_state['degradation'] else None,
    )
    return frame


def _publish(app, frame=None):
    with app.app_context():
        spec = get_strategy('a_share_timing', 'csi1000_timing')
        variant = variants_for(spec)[0]
        publish_entry(spec, variant, frame if frame is not None else _timing_frame())
        return spec, variant


@pytest.mark.parametrize('path,destination', [
    ('/', '/snapshots?source_id=selection&strategy_id=original_ensemble&tab=summary&initial_range=full'),
    ('/timing', '/snapshots?source_id=a_share_timing&strategy_id=csi1000_timing&initial_range=6m'),
    ('/us_timing', '/snapshots?source_id=us_timing&strategy_id=macro_v32_timing&initial_range=6m'),
    ('/hk_timing', '/snapshots?source_id=hk_timing&strategy_id=hsi_timing&initial_range=full'),
    ('/commodity', '/snapshots?source_id=commodity&strategy_id=gold_timing&initial_range=full'),
    ('/live', '/manual-records?strategy=star50_timing'),
])
def test_legacy_html_aliases_are_one_hop_and_discard_query(client, path, destination):
    response = client.get(path + '?force=1&strategy=ignored', follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'] == destination


@pytest.mark.parametrize('path', ['/snapshots', '/legacy-artifacts', '/data-status', '/operation-guide', '/manual-records'])
def test_canonical_pages_share_navigation_and_public_warning(client, path):
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for label in ('Snapshot Explorer', 'Legacy Artifacts', 'Data Status', 'Operation Guide', 'Manual Records'):
        assert label in html
    assert 'Public unsafe mode' in html
    assert '不是访问控制' in html
    assert not re.search(r'API.?key|paper trading|testnet|scheduler', html, re.I)


def test_bootstrap_blocked_recovery_button_is_visually_hidden(client):
    variant = client.get(
        '/api/r0/sources/selection/strategies/original_ensemble/variants'
    ).get_json()['items'][0]
    assert variant['blocker_code'] == 'bootstrap_required'
    assert variant['recoverable_now'] is False

    html = client.get('/snapshots').get_data(as_text=True)
    javascript = client.get('/static/js/r0.js').get_data(as_text=True)
    stylesheet = client.get('/static/css/r0.css').get_data(as_text=True)
    assert 'id="recover-cache"' in html
    assert 'id="bootstrap-guidance"' in html
    assert 'href="/operation-guide"' in html
    assert "variant.blocker_code === 'bootstrap_required'" in javascript
    assert "$('#recover-cache').hidden = !variant.recovery_supported || !variant.recoverable_now" in javascript
    assert re.search(
        r'\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important\s*;',
        stylesheet,
    )


def test_operation_guide_covers_all_pages_and_read_write_boundaries(client):
    html = client.get('/operation-guide').get_data(as_text=True)
    for heading in ('Snapshot Explorer', 'Data Status', 'Legacy Artifacts', 'Manual Records'):
        assert heading in html
    for marker in ('只读浏览', '联网', '写盘', 'bootstrap_required', 'partial', 'error'):
        assert marker in html
    assert '运行 Update 不需要等待 Reviewer' in html
    assert '账户、broker、订单' in html
    for path in ('/snapshots', '/data-status', '/legacy-artifacts', '/manual-records'):
        assert f'href="{path}"' in html


def test_data_status_has_restorable_progress_retry_and_guide_links(client):
    html = client.get('/data-status').get_data(as_text=True)
    javascript = client.get('/static/js/r0.js').get_data(as_text=True)
    for element_id in (
        'open-update-progress', 'update-progress-dialog', 'update-operation-id',
        'update-operation-status', 'update-progress-bar', 'update-step-list',
        'retry-update', 'minimize-update-progress',
    ):
        assert f'id="{element_id}"' in html
    assert 'Update 不需要等待代码 Review' in html
    assert 'href="/operation-guide"' in html
    for contract in (
        "const UPDATE_OPERATION_STORAGE_KEY = 'r0-data-update-operation-v1'",
        'localStorage.setItem(UPDATE_OPERATION_STORAGE_KEY',
        'localStorage.getItem(UPDATE_OPERATION_STORAGE_KEY)',
        'beginUpdateTracking(stored.operation_id',
        "['pending', 'running'].includes(stored.status)",
        '/api/r0/actions/${encodeURIComponent(operationId)}',
        "['partial', 'error'].includes(compact.status)",
        '/retry`, { method: \'POST\' }',
        'showUpdateProgressDialog()',
        'window.setTimeout(poll, 1000)',
        'state.updatePlanReady = true',
        'Selection changed. Preview the fixed plan again before updating.',
    ):
        assert contract in javascript


def test_only_canonical_api_blueprint_is_registered(app, client):
    rules = list(app.url_map.iter_rules())
    api_rules = [rule for rule in rules if rule.rule.startswith('/api/')]
    assert api_rules and all(rule.rule.startswith('/api/r0/') for rule in api_rules)
    assert client.get('/api/backtest').status_code == 404
    assert client.get('/api/timing/latest_signal').status_code == 404
    assert client.post('/api/update_data').status_code == 405
    assert client.post('/api/restart').status_code == 405
    assert client.post('/api/live/record').status_code == 405


def test_catalog_is_bounded_and_missing_variants_remain_visible(client):
    sources = client.get('/api/r0/sources').get_json()
    assert sources['bounded'] is True and sources['max_items'] == 16
    assert [item['source_id'] for item in sources['items']][:3] == ['selection', 'a_share_timing', 'us_timing']
    assert client.get('/api/r0/data-status').get_json()['sources'] == sources['items']
    strategies = client.get('/api/r0/sources/a_share_timing/strategies?include=latest_snapshot').get_json()
    assert strategies['total'] == 3
    item = strategies['items'][0]
    assert set(item['latest_snapshot']) == {
        'variant_id', 'snapshot_id', 'cache_state', 'readable', 'freshness_state',
        'freshness_reason', 'degradation', 'data_as_of', 'cache_generated_at',
        'current_signal', 'signal_state', 'state_reason', 'snapshot_url',
        'operation_id', 'status_url', 'recovery', 'error',
    }
    assert item['latest_snapshot']['cache_state'] == 'missing'
    variant = client.get(
        '/api/r0/sources/a_share_timing/strategies/csi1000_timing/variants'
    ).get_json()['items'][0]
    missing = client.get(
        '/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots'
        f'?variant_id={variant["variant_id"]}'
    )
    assert missing.status_code == 409
    error = missing.get_json()
    assert error['error'] == 'cache_miss' and error['recoverable_now'] is True
    assert error['recovery_plan_id'] == variant['recovery_plan_id']
    assert error['recovery_scopes'] == variant['recovery_scopes']
    assert error['recovery_steps'] == variant['recovery_steps']


@pytest.mark.parametrize('target_state,expected', [
    ({'cache_state': 'ready', 'readable': True, 'freshness_state': 'current', 'freshness_reason': None, 'degradation': None}, ('ready', None, None)),
    ({'cache_state': 'data_stale', 'readable': True, 'freshness_state': 'stale', 'freshness_reason': {'code': 'data_old', 'detail': None}, 'degradation': None}, ('data_stale', 'data_old', None)),
    ({'cache_state': 'degraded', 'readable': True, 'freshness_state': 'current', 'freshness_reason': None, 'degradation': {'code': 'optional_chan_missing', 'detail': None}}, ('degraded', None, 'optional_chan_missing')),
    ({'cache_state': 'degraded', 'readable': True, 'freshness_state': 'stale', 'freshness_reason': {'code': 'data_old', 'detail': None}, 'degradation': {'code': 'optional_chan_missing', 'detail': None}}, ('degraded', 'data_old', 'optional_chan_missing')),
])
def test_four_readable_target_tuples_and_signal_warnings(app, client, target_state, expected):
    _, variant = _publish(app, _timing_frame(target_state=target_state))
    body = client.get('/api/r0/sources/a_share_timing/strategies?include=latest_snapshot').get_json()
    card = next(item for item in body['items'] if item['strategy_id'] == 'csi1000_timing')['latest_snapshot']
    assert (card['cache_state'], card['current_signal']['data_stale_warning'], card['current_signal']['degraded_reason']) == expected
    assert (card['recovery'] is not None) == (target_state['degradation'] is not None)
    variant_row = client.get(
        '/api/r0/sources/a_share_timing/strategies/csi1000_timing/variants'
    ).get_json()['items'][0]
    assert variant_row['recoverable_now'] == (target_state['degradation'] is not None)
    response = client.get(f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots?variant_id={variant.variant_id}')
    assert response.status_code == 200


def test_summary_resources_are_full_paged_and_view_bound(app, client):
    _, variant = _publish(app, _timing_frame(rows=5))
    summary = client.get(f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots?variant_id={variant.variant_id}').get_json()
    base = f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots/{summary["snapshot_id"]}'
    first = client.get(f'{base}/series?view_id={summary["view_id"]}&kind=equity&window=full&resolution=day&limit=2').get_json()
    assert first['total'] == 5 and len(first['items']) == 2 and first['next_cursor']
    second = client.get(f'{base}/series?view_id={summary["view_id"]}&kind=equity&window=full&resolution=day&limit=2&cursor={first["next_cursor"]}').get_json()
    assert second['items'][0]['date'] > first['items'][-1]['date']
    wrong = client.get(f'{base}/signals?view_id={summary["view_id"]}&cursor={first["next_cursor"]}')
    assert wrong.status_code == 400 and wrong.get_json()['error'] == 'invalid_cursor'
    assert client.get(f'{base}/position?view_id=wrong').get_json()['error'] == 'invalid_view_id'


def test_snapshot_change_rejects_old_path(app, client):
    _, variant = _publish(app, _timing_frame(rows=3))
    opened = client.get(f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots?variant_id={variant.variant_id}').get_json()
    _publish(app, _timing_frame(rows=4))
    response = client.get(f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots/{opened["snapshot_id"]}/configuration')
    assert response.status_code == 409
    assert response.get_json()['error'] == 'snapshot_changed'


def test_viewer_gets_never_call_legacy_loaders_network_writes_or_builders(app, client, monkeypatch):
    _, variant = _publish(app)
    forbidden = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('side effect called'))
    for name in ('init_cache', 'init_timing_cache', 'init_us_timing_cache', '_load_disk_cache',
                 '_run_data_update', '_run_index_data_update', '_run_aux_data_update',
                 '_run_factor_update', 'run_backtest_fresh', 'run_timing_backtest_fresh'):
        monkeypatch.setattr(state, name, forbidden)
    summary = client.get(f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots?variant_id={variant.variant_id}').get_json()
    base = f'/api/r0/sources/a_share_timing/strategies/csi1000_timing/snapshots/{summary["snapshot_id"]}'
    urls = [
        '/api/r0/sources', '/api/r0/sources/a_share_timing/strategies?include=latest_snapshot',
        '/api/r0/sources/a_share_timing/strategies/csi1000_timing/variants',
        f'{base}/series?view_id={summary["view_id"]}&kind=equity&window=full&resolution=day',
        f'{base}/signals?view_id={summary["view_id"]}', f'{base}/position?view_id={summary["view_id"]}',
        f'{base}/interval-windows?view_id={summary["view_id"]}', f'{base}/trades?view_id={summary["view_id"]}',
        f'{base}/fees?view_id={summary["view_id"]}', f'{base}/configuration',
        '/api/r0/legacy-artifacts', '/api/r0/data-status', '/api/r0/data-check?scope=index',
        '/api/r0/manual-records/capabilities',
    ]
    assert all(client.get(url).status_code == 200 for url in urls)


def test_canonical_api_rejects_legacy_flags_and_unknown_strategy(client):
    assert client.get('/api/r0/sources/a_share_timing/strategies/csi1000_timing/variant-lookup?force=1').get_json()['error'] == 'invalid_params'
    assert client.get('/api/r0/sources/a_share_timing/strategies/not-here/variants').get_json()['error'] == 'strategy_not_found'
