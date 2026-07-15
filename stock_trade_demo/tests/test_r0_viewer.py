"""R0 viewer contract: honest labels, read-only routes and cache-only data."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from web import state
from web.app import create_app


@pytest.fixture()
def app():
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    with app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    'path',
    [
        '/', '/snapshot/selection', '/snapshot/cn-timing', '/snapshot/us-timing',
        '/snapshot/hk-timing', '/snapshot/commodity', '/legacy-artifacts',
        '/data-status', '/manual-records', '/timing', '/us_timing',
        '/hk_timing', '/commodity',
    ],
)
def test_r0_pages_expose_only_the_four_navigation_destinations(client, path):
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for label in ('Snapshot Explorer', 'Legacy Artifacts', 'Data Status', 'Manual Records'):
        assert label in html
    forbidden = re.compile(r'\b(Project|Run|Compare|OOS|Approved)\b|Frozen\s+Report', re.IGNORECASE)
    assert not forbidden.search(html)


def test_r0_blocks_legacy_mutation_before_handler(client, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('blocked handler was executed')

    monkeypatch.setattr(state, '_run_data_update', forbidden)
    response = client.post('/api/update_data')
    assert response.status_code == 405
    assert response.get_json()['error'] == 'read_only_viewer'


def test_r0_blocks_manual_record_write_before_handler(client, monkeypatch):
    from services import live_trades

    monkeypatch.setattr(live_trades, 'append_record', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('write called')))
    response = client.post('/api/live/record', json={'date': '2026-01-01', 'strategy': 'csi1000_timing'})
    assert response.status_code == 405
    assert response.get_json()['error'] == 'read_only_viewer'

    monkeypatch.setattr(live_trades, 'delete_record', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('delete called')))
    delete_response = client.delete('/api/live/record/1')
    assert delete_response.status_code == 405
    assert delete_response.get_json()['error'] == 'read_only_viewer'


def test_manual_record_storage_is_explicitly_unavailable(client):
    response = client.get('/api/r0/manual-records')
    assert response.status_code == 200
    assert response.get_json() == {
        'available': False,
        'records': [],
        'provenance': 'unknown',
        'evidence_status': 'unverified',
        'message': 'Manual record storage is not exposed by this public R0 viewer.',
    }


def test_legacy_live_surface_is_not_registered(app, client):
    assert not any(rule.rule.startswith('/api/live/') for rule in app.url_map.iter_rules())
    assert client.get('/live').status_code == 404


def test_r0_blocks_fresh_calculation_endpoint(client, monkeypatch):
    monkeypatch.setattr(state, 'run_timing_backtest_fresh', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('calculation called')))
    response = client.get('/api/timing/explore_compare?force=1')
    assert response.status_code == 405
    assert response.get_json()['error'] == 'read_only_viewer'


def test_all_r0_api_rules_are_get_only(app):
    r0_rules = [rule for rule in app.url_map.iter_rules() if rule.rule.startswith('/api/r0/')]
    assert r0_rules
    for rule in r0_rules:
        assert set(rule.methods) <= {'GET', 'HEAD', 'OPTIONS'}


def test_cache_initializers_never_calculate_on_miss(monkeypatch):
    saved = {
        'backtest': dict(state.BACKTEST_CACHE),
        'timing': dict(state.TIMING_CACHE),
        'commodity': dict(state.COMMODITY_CACHE),
        'hk': dict(state.HK_CACHE),
    }
    state.BACKTEST_CACHE.clear()
    state.TIMING_CACHE.clear()
    state.COMMODITY_CACHE.clear()
    state.HK_CACHE.clear()
    monkeypatch.setattr(state, '_load_disk_cache', lambda: False)
    monkeypatch.setattr(state, 'select_and_backtest', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('selection calculated')))
    monkeypatch.setattr(state, 'run_timing_backtest', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('timing calculated')))
    try:
        state.init_cache()
        state.init_timing_cache()
        state.init_commodity_cache()
        state.init_hk_cache()
        assert not state.BACKTEST_CACHE
        assert not state.TIMING_CACHE
        assert not state.COMMODITY_CACHE
        assert not state.HK_CACHE
    finally:
        state.BACKTEST_CACHE.update(saved['backtest'])
        state.TIMING_CACHE.update(saved['timing'])
        state.COMMODITY_CACHE.update(saved['commodity'])
        state.HK_CACHE.update(saved['hk'])


def test_snapshot_api_keeps_legacy_evidence_unverified(client, monkeypatch):
    frame = pd.DataFrame({
        '交易日期': pd.to_datetime(['2026-01-31', '2026-02-28', '2026-03-31']),
        '累积净值': [1.0, 1.05, 1.02],
        '买入股票代码': [['000001'], ['000002'], ['000003']],
    })
    saved = dict(state.BACKTEST_CACHE)
    state.BACKTEST_CACHE.clear()
    state.BACKTEST_CACHE['original_ensemble'] = (frame, None)
    monkeypatch.setattr(state, '_load_disk_cache', lambda: True)
    try:
        response = client.get('/api/r0/sources/selection/snapshots/original_ensemble')
        assert response.status_code == 200
        body = response.get_json()
        assert body['available'] is True
        assert body['evidence_status'] == 'unverified'
        assert body['provenance'] == 'unknown'
        assert body['data_as_of'] == '2026-03-31'
        assert len(body['equity_curve']) == 3
        assert body['metrics']['max_drawdown'] == '-2.86%'
    finally:
        state.BACKTEST_CACHE.clear()
        state.BACKTEST_CACHE.update(saved)


def test_missing_cache_stays_unavailable(client, monkeypatch):
    saved = dict(state.COMMODITY_CACHE)
    state.COMMODITY_CACHE.clear()
    try:
        response = client.get('/api/r0/sources/commodity/snapshots/gold_timing')
        assert response.status_code == 200
        body = response.get_json()
        assert body['available'] is False
        assert body['error'] == 'snapshot_unavailable'
        assert body['evidence_status'] == 'unverified'
        assert body['data_as_of'] == 'unknown'
        assert body['equity_curve'] == []
    finally:
        state.COMMODITY_CACHE.update(saved)


def test_artifact_and_data_status_payloads_never_expose_host_paths(client):
    artifacts = client.get('/api/r0/artifacts')
    assert artifacts.status_code == 200
    artifact_text = artifacts.get_data(as_text=True)
    assert '/root/' not in artifact_text
    assert '/home/' not in artifact_text
    assert '/Users/' not in artifact_text
    body = artifacts.get_json()
    assert {item['kind'] for item in body['artifacts']} >= {'profile', 'sensitivity', 'holdout', 'cache'}
    assert all(item['evidence_status'] == 'unverified' for item in body['artifacts'])
    assert all(item['provenance'] == 'unknown' for item in body['artifacts'])

    status = client.get('/api/r0/data-status')
    assert status.status_code == 200
    status_text = status.get_data(as_text=True)
    assert '/root/' not in status_text
    assert '/home/' not in status_text
    assert '/Users/' not in status_text
    assert all(item['data_as_of'] == 'unknown' for item in status.get_json()['sources'])
