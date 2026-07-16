"""Manual Records ledger, snapshot scope and reconciliation contracts."""
from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from web.app import create_app
from web.v01 import manual_ledger
from web.v01.catalog import get_strategy, variants_for
from web.v01.fingerprints import SCOPE_RESOURCES
from web.v01.snapshot_store import PUBLISHED_SIGNAL_KEYS, publish_entry


@pytest.fixture()
def app(tmp_path):
    app = create_app()
    app.config.update(
        TESTING=True,
        R0_GENERATION_ROOT=tmp_path / 'generations',
        R0_MANUAL_LEDGER_PATH=tmp_path / 'manual.csv',
        R0_ACTION_MARKER_PATH=tmp_path / 'active-operation.json',
        R0_OPERATION_LOG_PATH=tmp_path / 'operations.jsonl',
        R0_CURSOR_KEY='test-cursor-key',
        R0_RESOURCE_FINGERPRINTS={
            resource_id: f'fixture:{resource_id}'
            for resources in SCOPE_RESOURCES.values()
            for resource_id in resources
        },
    )
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def _signal(strategy_id: str, target: float = 0.6):
    result = {key: None for key in PUBLISHED_SIGNAL_KEYS}
    result.update({
        'strategy_id': strategy_id, 'name': strategy_id, 'index_name': 'Index',
        'etf_code': '510050', 'etf_name': 'ETF', 'as_of_date': '2026-01-04',
        'settled_as_of_date': '2026-01-03', 'target_exposure': target,
        'prev_exposure': 0.2, 'exposure_delta': target - 0.2,
        'rebalance_action': 'add', 'rebalance_label': 'Add',
        'signal_action': 'buy', 'signal_label': 'Buy', 'current_action': 'buy',
        'current_position': target, 'current_reason': 'stored',
        'reason_summary': 'stored', 'bullish_score': 0.8, 'ref_close': 10.0,
        'ref_open': 10.0, 'nav': 1.03, 'settled_nav': 1.02,
        'status': 'research', 'exec_basis': 'next_open',
    })
    return result


def _frame(strategy_id: str = 'star50_timing', target: float = 0.6):
    frame = pd.DataFrame({
        '交易日期': pd.date_range('2026-01-01', periods=4, freq='D'),
        '累积净值': [1.0, 1.1, 1.21, 1.331],
        'strategy_return': [0.0, 0.1, 0.1, 0.1],
        'position': [0.0, 0.5, 0.6, 0.6],
        'target_exposure': [0.0, 0.5, 0.6, 0.6],
        'etf_open': [10.0, 11.0, 12.0, 13.0],
        'etf_close': [10.5, 11.5, 12.5, 13.5],
        'signal_action': ['hold', 'buy', 'hold', 'hold'],
        'prev_exposure': [0.0, 0.0, 0.5, 0.6],
        'exposure_change': [0.0, 0.5, 0.1, 0.0],
        'rebalance_action': ['flat', 'enter', 'add', 'hold'],
        'reason_summary': ['stored'] * 4, 'reason_detail': [[] for _ in range(4)],
        'signal_score': [0.5] * 4, 'strength_score': [0.5] * 4,
        'trade_quantity': [0, 100, 0, 0], 'trade_amount': [0, 1000, 0, 0],
        'trade_fee_amount': [0, 1, 0, 0], 'holding_value': [0, 1050, 1150, 1250],
        'cash_balance': [50000, 49000, 49000, 49000],
        'index_id': ['star50'] * 4, 'index_name': ['STAR 50'] * 4,
    })
    frame.attrs.update({
        'r0_target_state': {
            'cache_state': 'ready', 'readable': True,
            'freshness_state': 'current', 'freshness_reason': None,
            'degradation': None,
        },
        'r0_generated_at': '2026-01-05T00:00:00Z',
        'published_current_signal': _signal(strategy_id, target),
    })
    return frame


def _publish(app, source='a_share_timing', strategy='star50_timing', target=0.6):
    with app.app_context():
        spec = get_strategy(source, strategy)
        variant = next(item for item in variants_for(spec) if item.default)
        publish_entry(spec, variant, _frame(strategy, target))
        return variant


def _create(client, *, key=None, **overrides):
    body = {
        'date': '2026-01-02', 'strategy': 'star50_timing',
        'capital': 50000, 'actual_position': 0.2, 'notes': 'research only',
        **overrides,
    }
    return client.post(
        '/api/r0/manual-records', json=body,
        headers={'Idempotency-Key': key or str(uuid4())},
    )


def test_missing_list_is_empty_and_does_not_create_ledger(app, client):
    ledger = Path(app.config['R0_MANUAL_LEDGER_PATH'])
    response = client.get('/api/r0/manual-records?strategy=star50_timing')
    assert response.status_code == 200
    assert response.get_json()['items'] == []
    assert response.get_json()['total'] == 0
    assert not ledger.exists()


def test_create_replay_conflict_delete_and_deleted_replay(app, client):
    key = str(uuid4())
    created = _create(client, key=key)
    assert created.status_code == 201
    record_id = created.get_json()['record']['record_id']
    assert _create(client, key=key).status_code == 200
    assert _create(client, key=key, notes='different').status_code == 409
    assert client.delete(f'/api/r0/manual-records/{record_id}').status_code == 204
    assert client.delete(f'/api/r0/manual-records/{record_id}').status_code == 404
    deleted_replay = _create(client, key=key)
    assert deleted_replay.status_code == 410
    raw = Path(app.config['R0_MANUAL_LEDGER_PATH']).read_text(encoding='utf-8')
    assert raw.splitlines()[0] == ','.join(manual_ledger.HEADER)
    assert 'idempotency_tombstone' in raw
    assert 'research only' not in raw
    operation_log = Path(app.config['R0_OPERATION_LOG_PATH']).read_text(encoding='utf-8')
    assert 'research only' not in operation_log and 'different' not in operation_log


def test_first_mutation_materializes_and_deletes_legacy_row_without_created_at(app, client):
    ledger = Path(app.config['R0_MANUAL_LEDGER_PATH'])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=manual_ledger.LEGACY_HEADER)
        writer.writeheader()
        writer.writerow({
            'record_id': '7', 'date': '2026-01-02', 'strategy': 'star50_timing',
            'signal_target': '0.6', 'actual_position': '0.2', 'exec_price': '',
            'capital': '50000', 'notes': 'legacy research', 'created_at': '',
            'shares': '',
        })
    listed = client.get('/api/r0/manual-records?strategy=star50_timing').get_json()
    record_id = listed['items'][0]['record_id']
    assert listed['items'][0]['created_at'] is None
    assert client.delete(f'/api/r0/manual-records/{record_id}').status_code == 204

    with ledger.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == manual_ledger.HEADER
    assert rows[0]['row_type'] == 'idempotency_tombstone'
    assert rows[0]['created_at'] == '2026-01-02T00:00:00Z'
    assert rows[0]['notes'] == '' and rows[0]['strategy'] == ''
    assert client.delete(f'/api/r0/manual-records/{record_id}').status_code == 404


@pytest.mark.parametrize('body,error', [
    ({'date': 'bad', 'strategy': 'star50_timing', 'actual_position': 0.2}, 'invalid_date'),
    ({'date': '2026-01-02', 'strategy': 'star50_timing', 'actual_position': float('inf')}, 'invalid_params'),
    ({'date': '2026-01-02', 'strategy': 'star50_timing', 'actual_position': 1.1}, 'invalid_params'),
    ({'date': '2026-01-02', 'strategy': 'star50_timing', 'exec_price': 10, 'shares': 100, 'actual_position': 0.7}, 'position_conflict'),
    ({'date': '2026-01-02', 'strategy': 'star50_timing', 'actual_position': 0.2, 'notes': {'html': '<b>x</b>'}}, 'invalid_params'),
    ({'date': '2026-01-02', 'strategy': 'star50_timing', 'actual_position': 0.2, 'notes': '  =1+1'}, 'unsafe_note_prefix'),
    ({'date': '2026-01-02', 'strategy': 'hsi_timing', 'actual_position': 0.2}, 'unsupported_manual_strategy'),
])
def test_create_validation_is_stable(client, body, error):
    response = client.post('/api/r0/manual-records', json=body, headers={'Idempotency-Key': str(uuid4())})
    assert response.status_code == 400
    assert response.get_json()['error'] == error


def test_records_cursor_is_strategy_and_ledger_version_bound(client):
    for date in ('2026-01-01', '2026-01-02', '2026-01-03'):
        assert _create(client, date=date).status_code == 201
    first = client.get('/api/r0/manual-records?strategy=star50_timing&limit=2').get_json()
    assert first['total'] == 3 and len(first['items']) == 2 and first['next_cursor']
    wrong = client.get(f'/api/r0/manual-records?strategy=sp500_timing&cursor={first["next_cursor"]}')
    assert wrong.status_code == 400 and wrong.get_json()['error'] == 'invalid_cursor'
    assert _create(client, date='2026-01-04').status_code == 201
    stale = client.get(f'/api/r0/manual-records?strategy=star50_timing&cursor={first["next_cursor"]}')
    assert stale.status_code == 409 and stale.get_json()['error'] == 'manual_ledger_changed'


def test_snapshot_scope_is_validated_before_ledger_access(app, client, monkeypatch):
    variant = _publish(app)
    with app.app_context():
        spec = get_strategy('a_share_timing', 'star50_timing')
        snapshot_id = __import__('web.v01.snapshot_store', fromlist=['load_snapshot']).load_snapshot(spec, variant).snapshot_id
    monkeypatch.setattr(manual_ledger, 'load', lambda: (_ for _ in ()).throw(AssertionError('ledger read')))
    wrong = client.get('/api/r0/manual-records/signal-reference?strategy=hsi_timing&snapshot_id=s_wrong')
    assert wrong.status_code == 400
    missing = client.get('/api/r0/manual-records/signal-reference?strategy=star50_timing&snapshot_id=legacy')
    assert missing.status_code == 400
    # A valid exact snapshot is the first point at which ledger access is allowed.
    with pytest.raises(AssertionError):
        client.get(f'/api/r0/manual-records/signal-reference?strategy=star50_timing&snapshot_id={snapshot_id}')


def test_signal_reference_uses_only_same_strategy_active_rows(app, client):
    variant = _publish(app)
    first = _create(client, date='2026-01-02', actual_position=0.2).get_json()['record']
    assert _create(client, strategy='sp500_timing', date='2026-01-04', actual_position=0.95).status_code == 201
    latest = _create(client, date='2026-01-03', actual_position=0.4).get_json()['record']
    with app.app_context():
        spec = get_strategy('a_share_timing', 'star50_timing')
        from web.v01.snapshot_store import load_snapshot
        snapshot_id = load_snapshot(spec, variant).snapshot_id
    body = client.get(f'/api/r0/manual-records/signal-reference?strategy=star50_timing&snapshot_id={snapshot_id}').get_json()
    assert body['manual_state'] == {
        'state': 'present', 'latest_manual_position': 0.4,
        'manual_record_id': latest['record_id'], 'manual_record_date': '2026-01-03',
    }
    assert body['action_context']['live_exposure_delta'] == pytest.approx(0.2)
    assert body['action_context']['non_executable'] is True
    assert '不会下单' in body['action_context']['action_rationale']
    assert body['manual_state']['manual_record_id'] != first['record_id']


def test_signal_reference_reads_published_peer_and_risk_context(app, client):
    variant = _publish(app)
    peer_variant = _publish(app, 'us_timing', 'sp500_timing', target=0.2)
    with app.app_context():
        risk_spec = get_strategy('decision_context', 'risk_signals')
        risk_variant = next(item for item in variants_for(risk_spec) if item.default)
        publish_entry(risk_spec, risk_variant, {
            'as_of': '2026-01-04',
            'generated_at': '2026-01-05T00:00:00Z',
            'by_strategy': {
                'star50_timing': {
                    'bullish_risks_dynamic': ['动态风险样本'],
                    'bearish_opportunities_dynamic': [],
                },
            },
        })
        from web.v01.snapshot_store import load_snapshot
        snapshot_id = load_snapshot(
            get_strategy('a_share_timing', 'star50_timing'), variant).snapshot_id
        peer_snapshot_id = load_snapshot(
            get_strategy('us_timing', 'sp500_timing'), peer_variant).snapshot_id
        risk_snapshot_id = load_snapshot(risk_spec, risk_variant).snapshot_id

    body = client.get(
        '/api/r0/manual-records/signal-reference'
        f'?strategy=star50_timing&snapshot_id={snapshot_id}'
    ).get_json()
    context = body['action_context']
    assert context['risk_context_state'] == 'ready'
    assert context['risk_context_snapshot_id'] == risk_snapshot_id
    assert len(context['risk_context_fingerprint']) == 64
    assert context['risk_signals_as_of'] == '2026-01-04'
    assert context['risk_signals_generated_at'] == '2026-01-05T00:00:00Z'
    assert context['peer_snapshot_ids'] == [peer_snapshot_id]
    assert '动态风险样本' in context['risks']
    assert any('S&P 500' in item for item in context['risks'])

    with app.app_context():
        publish_entry(risk_spec, risk_variant, {
            'as_of': '2026-01-04', 'generated_at': 7,
            'by_strategy': {
                'star50_timing': {
                    'bullish_risks_dynamic': ['不得泄漏的损坏内容'],
                    'bearish_opportunities_dynamic': [],
                },
            },
        })
    corrupt = client.get(
        '/api/r0/manual-records/signal-reference'
        f'?strategy=star50_timing&snapshot_id={snapshot_id}'
    ).get_json()['action_context']
    assert corrupt['risk_context_state'] == 'corrupt'
    assert corrupt['risk_context_snapshot_id'] is None
    assert corrupt['risk_context_fingerprint'] is None
    assert '不得泄漏的损坏内容' not in corrupt['risks']


def test_reconciliation_empty_then_same_day_flow_and_pagination(app, client):
    variant = _publish(app)
    with app.app_context():
        spec = get_strategy('a_share_timing', 'star50_timing')
        from web.v01.snapshot_store import load_snapshot
        snapshot_id = load_snapshot(spec, variant).snapshot_id
    base = f'/api/r0/manual-records/reconciliation?strategy=star50_timing&snapshot_id={snapshot_id}'
    empty = client.get(base).get_json()
    assert empty['state'] == 'empty' and empty['series_total'] == 0 and empty['pending_total'] == 0
    first = _create(client, date='2026-01-02', capital=50000, exec_price=10, shares=1000, actual_position=None).get_json()['record']
    second = _create(client, date='2026-01-02', capital=60000, exec_price=10.25, shares=2000, actual_position=None).get_json()['record']
    _create(client, date='2027-01-01', capital=50000, actual_position=0.3)
    summary = client.get(base).get_json()
    assert summary['state'] == 'pending_cache' and summary['pending_total'] == 1
    series = client.get(summary['series_url'] + '&limit=1').get_json()
    assert series['total'] == summary['series_total'] and series['next_cursor']
    first_row = series['items'][0]
    assert first_row['applied_record_count'] == 2
    assert first_row['external_flow'] != 0
    applied = client.get(first_row['applied_records_url']).get_json()
    assert applied['total'] == 2
    assert [item['record_id'] for item in applied['items']] == [first['record_id'], second['record_id']]
    pending = client.get(summary['pending_records_url']).get_json()
    assert pending['total'] == 1
