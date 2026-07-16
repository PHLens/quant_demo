"""Mutation gate, operation, crash-fence and unavailable restart contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from web import state
from web.app import create_app
from web.v01 import actions, fingerprints
from web.v01.catalog import get_strategy, variants_for
from web.v01.snapshot_store import PUBLISHED_SIGNAL_KEYS, publish_entry


class ImmediateThread:
    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class BrokenThread(ImmediateThread):
    def start(self):
        raise RuntimeError('thread start failed')


class PausedThread(ImmediateThread):
    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__class__.instances.append(self)

    def start(self):
        return None

    def run(self):
        self.target(*self.args)


@pytest.fixture(autouse=True)
def _clean_state():
    PausedThread.instances.clear()
    for cache in (
        state.TIMING_CACHE, state.US_TIMING_CACHE, state.HK_CACHE,
        state.COMMODITY_CACHE, state.BACKTEST_CACHE, state.FACTOR_BACKTEST_CACHE,
    ):
        cache.clear()
    yield
    for cache in (
        state.TIMING_CACHE, state.US_TIMING_CACHE, state.HK_CACHE,
        state.COMMODITY_CACHE, state.BACKTEST_CACHE, state.FACTOR_BACKTEST_CACHE,
    ):
        cache.clear()


@pytest.fixture()
def config(tmp_path):
    return {
        'TESTING': True,
        'R0_GENERATION_ROOT': tmp_path / 'generations',
        'R0_MANUAL_LEDGER_PATH': tmp_path / 'manual.csv',
        'R0_ACTION_MARKER_PATH': tmp_path / 'active-operation.json',
        'R0_OPERATION_LOG_PATH': tmp_path / 'operations.jsonl',
        'R0_CURSOR_KEY': 'test-cursor-key',
        'R0_RESOURCE_FINGERPRINTS': {
            resource_id: f'fixture:{resource_id}'
            for resource_id in fingerprints.production_resource_paths(tmp_path)
        },
    }


@pytest.fixture()
def app(config):
    return create_app(config)


@pytest.fixture()
def client(app):
    return app.test_client()


def _signal(strategy_id='star50_timing'):
    result = {key: None for key in PUBLISHED_SIGNAL_KEYS}
    result.update({
        'strategy_id': strategy_id, 'name': strategy_id, 'index_name': 'STAR 50',
        'etf_code': '588000', 'etf_name': 'ETF', 'as_of_date': '2026-01-03',
        'settled_as_of_date': '2026-01-02', 'target_exposure': 0.6,
        'prev_exposure': 0.2, 'exposure_delta': 0.4, 'rebalance_action': 'add',
        'rebalance_label': 'Add', 'signal_action': 'buy', 'signal_label': 'Buy',
        'current_action': 'buy', 'current_position': 0.6, 'current_reason': 'stored',
        'reason_summary': 'stored', 'bullish_score': 0.8, 'ref_close': 10.0,
        'ref_open': 10.1, 'nav': 1.02, 'settled_nav': 1.01,
        'status': 'research', 'exec_basis': 'next_open',
    })
    return result


def _frame(strategy_id='star50_timing'):
    frame = pd.DataFrame({
        '交易日期': pd.date_range('2026-01-01', periods=3, freq='D'),
        '累积净值': [1.0, 1.01, 1.02], 'strategy_return': [0.0, 0.01, 0.01],
        'position': [0.0, 0.2, 0.6], 'target_exposure': [0.0, 0.2, 0.6],
        'prev_exposure': [0.0, 0.0, 0.2], 'exposure_change': [0.0, 0.2, 0.4],
        'rebalance_action': ['flat', 'enter', 'add'],
        'signal_action': ['flat', 'buy', 'buy'], 'reason_summary': ['stored'] * 3,
        'reason_detail': [[], [], []], 'signal_score': [0.5] * 3,
        'strength_score': [0.5] * 3, 'close': [10.0, 10.1, 10.2],
        'etf_open': [10.0, 10.1, 10.2], 'etf_close': [10.05, 10.15, 10.25],
        'trade_quantity': [0, 100, 100], 'trade_amount': [0, 1000, 1000],
        'trade_fee_amount': [0, 1, 1], 'holding_value': [0, 1000, 2000],
        'cash_balance': [50000, 49000, 48000], 'index_id': ['star50'] * 3,
        'index_name': ['STAR 50'] * 3,
    })
    frame.attrs.update({
        'r0_generated_at': '2026-01-04T00:00:00Z',
        'r0_target_state': {
            'cache_state': 'ready', 'readable': True,
            'freshness_state': 'current', 'freshness_reason': None,
            'degradation': None,
        },
        'published_current_signal': _signal(strategy_id),
    })
    return frame


def _selection_frame():
    frame = pd.DataFrame({
        '交易日期': pd.date_range('2026-01-01', periods=3, freq='D'),
        '累积净值': [1.0, 1.01, 1.02],
        '选股下周期涨跌幅': [0.0, 0.01, 0.01],
        '买入个股收益': [[], [], []],
        '当期本金': [100000.0, 100000.0, 101000.0],
        '当期盈亏': [0.0, 1000.0, 1010.0],
    })
    frame.attrs.update({
        'initial_capital': 100000.0,
        'r0_generated_at': '2026-01-04T00:00:00Z',
        'r0_target_state': {
            'cache_state': 'ready', 'readable': True,
            'freshness_state': 'current', 'freshness_reason': None,
            'degradation': None,
        },
    })
    return frame


def _fingerprint(label):
    return hashlib.sha256(label.encode('utf-8')).hexdigest()


def _release_runtime_as_crashed(app, kind, operation_id):
    app.extensions['r0_action_runtime'].gate.release(
        actions.GateOwner(kind, operation_id, None)
    )


def _materialize_resource_paths(paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith('.json'):
            path.write_text('{"fixture":"v1"}', encoding='utf-8')
        else:
            path.write_text('date,value\n2026-01-01,1\n', encoding='utf-8')


def _target():
    spec = get_strategy('a_share_timing', 'star50_timing')
    variant = next(item for item in variants_for(spec) if item.default)
    return spec, variant, {
        'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id,
    }


def _configure_recovery(app, thread_factory):
    _, _, target = _target()
    app.config.update(
        R0_THREAD_FACTORY=thread_factory,
        R0_ACTION_RUNNERS={'index': lambda: None},
        R0_RECOVERY_BUILDERS={
            ('a_share_timing', 'star50_timing'):
                lambda _target: state.TIMING_CACHE.__setitem__('star50_timing', _frame()),
        },
    )
    return target


def test_recovery_publishes_exact_target_and_terminal_status(app, client):
    target = _configure_recovery(app, ImmediateThread)
    cataloged = client.get(
        '/api/r0/sources/a_share_timing/strategies/star50_timing/variants'
    ).get_json()['items'][0]
    response = client.post('/api/r0/actions/cache-recover', json=target)
    assert response.status_code == 202
    accepted = response.get_json()
    status = client.get(accepted['status_url']).get_json()
    assert status['status'] == 'done'
    assert status['original_request']['source_id'] == 'a_share_timing'
    assert status['original_request']['recovery_plan_id'] == cataloged['recovery_plan_id']
    assert status['result']['snapshot_id'].startswith('s_')
    assert status['steps'][-1]['status'] == 'done'
    assert not Path(app.config['R0_ACTION_MARKER_PATH']).exists()
    opened = client.get(
        '/api/r0/sources/a_share_timing/strategies/star50_timing/snapshots'
        f'?variant_id={target["variant_id"]}'
    )
    assert opened.status_code == 200
    pointer_path = next(Path(app.config['R0_GENERATION_ROOT']).rglob('pointer.json'))
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    generation = json.loads(
        (pointer_path.parent / 'generations' / f'{pointer["generation_id"]}.json').read_text(encoding='utf-8')
    )
    assert generation['manifest']['variant_id'] == target['variant_id']
    assert generation['manifest']['canonical_params']
    assert generation['manifest']['data_fingerprint'] == generation['manifest']['payload_digest']
    assert len(generation['manifest']['code_fingerprint']) == 64
    assert set(generation['manifest']['input_fingerprints']) == {
        'dataset:index-daily', 'dataset:etf-daily',
    }
    assert len(generation['manifest']['input_fingerprint']) == 64
    already = client.post('/api/r0/actions/cache-recover', json=target)
    assert already.status_code == 200
    assert already.get_json() == {
        'result': 'already_ready', 'cache_state': 'ready', 'readable': True,
    }


def test_recovery_fences_collateral_when_shared_input_changes(app, client):
    target = _configure_recovery(app, ImmediateThread)
    fingerprints = {'dataset:index': 'index-v1'}
    app.config['R0_RESOURCE_FINGERPRINTS'] = fingerprints
    app.config['R0_ACTION_RUNNERS']['index'] = lambda: fingerprints.__setitem__('dataset:index', 'index-v2')
    response = client.post('/api/r0/actions/cache-recover', json=target)
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    assert status['status'] == 'done'
    roles = {item['role'] for item in status['resolved_plan']['affected_targets']}
    assert roles == {'rebuilt', 'collateral'}
    collateral = client.get(
        '/api/r0/sources/a_share_timing/strategies/chinext_timing/variants'
    ).get_json()['items'][0]
    assert collateral['cache_state'] == 'artifact_stale'
    assert collateral['readable'] is False


def test_same_recovery_is_idempotent_while_gate_blocks_other_mutations(app, client):
    target = _configure_recovery(app, PausedThread)
    first = client.post('/api/r0/actions/cache-recover', json=target)
    assert first.status_code == 202
    again = client.post('/api/r0/actions/cache-recover', json=target)
    assert again.status_code == 202
    assert again.get_json()['operation_id'] == first.get_json()['operation_id']
    conflict = client.post('/api/r0/actions/data-update', json={'scopes': ['index'], 'force': True})
    assert conflict.status_code == 409
    assert conflict.get_json()['error'] == 'operation_in_progress'
    manual = client.post(
        '/api/r0/manual-records',
        json={'date': '2026-01-01', 'strategy': 'star50_timing', 'actual_position': 0.2},
        headers={'Idempotency-Key': str(uuid4())},
    )
    assert manual.status_code == 409
    PausedThread.instances[-1].run()
    assert client.get(first.get_json()['status_url']).get_json()['status'] == 'done'


def test_crash_fence_failure_has_no_overlay_operation_or_worker(app, client, monkeypatch):
    target = _configure_recovery(app, PausedThread)
    value = app.extensions['r0_action_runtime']
    monkeypatch.setattr(value, '_exclusive_json', lambda *_a, **_k: (_ for _ in ()).throw(OSError('fsync')))
    response = client.post('/api/r0/actions/cache-recover', json=target)
    assert response.status_code == 500
    assert response.get_json()['error'] == 'action_crash_fence_failed'
    assert value.operations == {} and value.overlay_targets == {}
    assert value.gate.owner is None and PausedThread.instances == []


def test_exclusive_marker_create_never_replaces_or_deletes_existing_owner(app):
    value = app.extensions['r0_action_runtime']
    marker = Path(app.config['R0_ACTION_MARKER_PATH'])
    marker.write_bytes(b'existing-owner-marker')
    with pytest.raises(FileExistsError):
        value._exclusive_json(marker, {'replacement': True})
    assert marker.read_bytes() == b'existing-owner-marker'


def test_worker_start_failure_removes_marker_overlay_and_operation(app, client):
    target = _configure_recovery(app, BrokenThread)
    response = client.post('/api/r0/actions/cache-recover', json=target)
    value = app.extensions['r0_action_runtime']
    assert response.status_code == 500
    assert response.get_json()['error'] == 'worker_start_failed'
    assert value.operations == {} and value.overlay_targets == {}
    assert value.gate.owner is None
    assert not Path(app.config['R0_ACTION_MARKER_PATH']).exists()


def test_unexpected_worker_exception_reconciles_and_releases_gate(app, client, monkeypatch):
    app.config.update(
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={'index': lambda: None},
    )
    monkeypatch.setattr(actions, '_publish_affected', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('unexpected')))
    response = client.post('/api/r0/actions/data-update', json={'scopes': ['index'], 'force': True})
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    assert status['status'] == 'error'
    assert status['error_code'] == 'worker_failed'
    value = app.extensions['r0_action_runtime']
    assert value.ready is True and value.gate.owner is None
    assert value.overlay_targets == {}
    assert not Path(app.config['R0_ACTION_MARKER_PATH']).exists()


def test_update_independent_root_continues_after_another_root_fails(app, client):
    calls = []
    app.config.update(
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={
            'index': lambda: (_ for _ in ()).throw(RuntimeError('upstream')),
            'aux': lambda: calls.append('aux'),
        },
    )
    response = client.post('/api/r0/actions/data-update', json={'scopes': ['aux', 'index'], 'force': True})
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    step = {item['scope']: item for item in status['steps']}
    assert step['index']['status'] == 'error'
    assert step['aux']['status'] == 'done'
    assert calls == ['aux']
    assert status['status'] in {'partial', 'error'}


def test_factor_plan_adds_stock_only_when_parquet_is_not_valid(app, client, tmp_path):
    parquet = tmp_path / 'stock.parquet'
    app.config['R0_STOCK_PARQUET_PATH'] = parquet
    pd.DataFrame({'value': [1]}).to_parquet(parquet, index=False)
    current = client.get('/api/r0/data-update-plan?scopes=factor&force=false').get_json()
    assert current['resolved_scopes'] == ['factor']
    assert current['prerequisites'] == []
    assert current['steps'][0]['blocked_by'] == []

    parquet.write_bytes(b'not-a-parquet')
    missing = client.get('/api/r0/data-update-plan?scopes=factor&force=false').get_json()
    assert missing['resolved_scopes'] == ['stock', 'factor']
    assert missing['prerequisites'] == [{'scope': 'stock', 'required_by': 'factor'}]
    assert missing['steps'][1]['blocked_by'] == ['stock']


def test_current_update_is_a_no_write_no_target_operation(app, client, monkeypatch):
    monkeypatch.setattr(actions, 'data_check', lambda _scope: {
        'checked_at': '2026-01-01T00:00:00Z',
        'scopes': [{
            'scope': scope, 'current_local_date': '2026-01-01',
            'latest_expected_date': '2026-01-01', 'needs_update': False,
            'reason': 'current', 'unknown': False,
        } for scope in actions.SCOPE_ORDER],
    })
    app.config.update(
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={'index': lambda: (_ for _ in ()).throw(AssertionError('current scope ran'))},
    )
    response = client.post('/api/r0/actions/data-update', json={'scopes': ['index'], 'force': False})
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    assert status['status'] == 'done'
    assert status['steps'][0]['status'] == 'skipped_current'
    assert status['resolved_plan']['write_set'] == []
    assert status['resolved_plan']['affected_targets'] == []
    assert status['result']['targets'] == []


def test_default_stock_update_builds_and_publishes_selection_and_a_timing(app, client, monkeypatch):
    fingerprints = {
        'dataset:index-daily': _fingerprint('index-v1'),
        'dataset:etf-daily': _fingerprint('etf-v1'),
        'dataset:stock-csv': _fingerprint('stock-csv-v1'),
        'dataset:stock-parquet': _fingerprint('stock-parquet-v1'),
    }
    app.config.update(
        R0_RESOURCE_FINGERPRINTS=fingerprints,
        R0_THREAD_FACTORY=ImmediateThread,
    )
    assert not app.config.get('R0_ACTION_RUNNERS')

    def stock_pull():
        fingerprints['dataset:stock-csv'] = _fingerprint('stock-csv-v2')
        fingerprints['dataset:stock-parquet'] = _fingerprint('stock-parquet-v2')
        state._UPDATE_DATA_STATUS.update(stage='done', error=None, running=False)

    monkeypatch.setattr(state, '_run_data_update', stock_pull)
    monkeypatch.setattr(
        state, 'run_backtest_fresh',
        lambda _strategy_id, **_params: (_selection_frame(), {}),
    )
    monkeypatch.setattr(
        state, 'run_timing_backtest_fresh',
        lambda strategy_id, **_params: (_frame(strategy_id), {}, None),
    )

    response = client.post('/api/r0/actions/data-update', json={'scopes': ['stock'], 'force': True})
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    assert status['status'] == 'done'
    published = [item for item in status['result']['targets'] if item['published']]
    assert {item['source_id'] for item in published} == {'selection', 'a_share_timing'}

    for source_id, strategy_id in (
            ('selection', 'original_ensemble'),
            ('a_share_timing', 'star50_timing')):
        variant = next(
            item for item in client.get(
                f'/api/r0/sources/{source_id}/strategies/{strategy_id}/variants'
            ).get_json()['items'] if item['default']
        )
        opened = client.get(
            f'/api/r0/sources/{source_id}/strategies/{strategy_id}/snapshots'
            f'?variant_id={variant["variant_id"]}'
        )
        assert opened.status_code == 200

    generations = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in Path(app.config['R0_GENERATION_ROOT']).glob('*/generations/*.json')
    ]
    selection = next(item for item in generations if (
        item['source_id'], item['strategy_id']) == ('selection', 'original_ensemble'))
    timing = next(item for item in generations if (
        item['source_id'], item['strategy_id']) == ('a_share_timing', 'star50_timing'))
    assert set(selection['manifest']['input_fingerprints']) == {
        'dataset:index-daily', 'dataset:etf-daily',
        'dataset:stock-csv', 'dataset:stock-parquet',
    }
    assert set(timing['manifest']['input_fingerprints']) == {
        'dataset:index-daily', 'dataset:etf-daily',
    }


def test_default_stock_update_builder_failure_is_terminal_and_unreadable(app, client, monkeypatch):
    fingerprints = {
        'dataset:index-daily': _fingerprint('index-v1'),
        'dataset:etf-daily': _fingerprint('etf-v1'),
        'dataset:stock-csv': _fingerprint('stock-csv-v1'),
        'dataset:stock-parquet': _fingerprint('stock-parquet-v1'),
    }
    app.config.update(
        R0_RESOURCE_FINGERPRINTS=fingerprints,
        R0_THREAD_FACTORY=ImmediateThread,
    )
    assert not app.config.get('R0_ACTION_RUNNERS')

    def stock_pull():
        fingerprints['dataset:stock-csv'] = _fingerprint('stock-csv-v2')
        fingerprints['dataset:stock-parquet'] = _fingerprint('stock-parquet-v2')
        state._UPDATE_DATA_STATUS.update(stage='done', error=None, running=False)

    failed = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('fixed builder failed'))
    monkeypatch.setattr(state, '_run_data_update', stock_pull)
    monkeypatch.setattr(state, 'run_backtest_fresh', failed)
    monkeypatch.setattr(state, 'run_timing_backtest_fresh', failed)

    response = client.post('/api/r0/actions/data-update', json={'scopes': ['stock'], 'force': True})
    assert response.status_code == 202
    status = client.get(response.get_json()['status_url']).get_json()
    assert status['status'] == 'error'
    assert status['error_code'] == 'update_failed'
    assert not any(item['published'] for item in status['result']['targets'])

    for source_id, strategy_id in (
            ('selection', 'original_ensemble'),
            ('a_share_timing', 'star50_timing')):
        variant = next(
            item for item in client.get(
                f'/api/r0/sources/{source_id}/strategies/{strategy_id}/variants'
            ).get_json()['items'] if item['default']
        )
        assert variant['readable'] is False
        opened = client.get(
            f'/api/r0/sources/{source_id}/strategies/{strategy_id}/snapshots'
            f'?variant_id={variant["variant_id"]}'
        )
        assert opened.status_code == 409
        assert opened.get_json()['error'] == 'cache_miss'


def test_update_retry_preserves_request_and_reuses_successful_roots(app, client):
    calls = []
    app.config.update(
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={
            'index': lambda: calls.append('index:first'),
            'aux': lambda: (_ for _ in ()).throw(RuntimeError('upstream')),
        },
    )
    first = client.post(
        '/api/r0/actions/data-update',
        json={'scopes': ['aux', 'index'], 'force': True},
    )
    assert first.status_code == 202
    first_status = client.get(first.get_json()['status_url']).get_json()
    assert first_status['status'] == 'error'

    app.config.update(
        R0_THREAD_FACTORY=PausedThread,
        R0_ACTION_RUNNERS={
            'index': lambda: (_ for _ in ()).throw(AssertionError('successful root reran')),
            'aux': lambda: calls.append('aux:retry'),
        },
    )
    retried = client.post(f"{first.get_json()['status_url']}/retry")
    assert retried.status_code == 202
    accepted = retried.get_json()
    assert accepted['retry_of'] == first_status['operation_id']
    assert accepted['resolved_plan']['reused_outputs'] == [
        'dataset:etf-daily', 'dataset:index-daily',
    ]
    assert accepted['resolved_plan']['invalidated_outputs'] == []
    assert accepted['resolved_plan']['write_set'] == [
        'dataset:fred', 'dataset:a-share-macro', 'artifact:risk-signals',
    ]

    retry_status = client.get(accepted['status_url']).get_json()
    assert retry_status['original_request'] == {
        'requested_scopes': ['index', 'aux'], 'force': True,
    }
    steps = {item['scope']: item for item in retry_status['steps']}
    assert steps['index']['status'] == 'skipped_current'
    assert steps['aux']['status'] == 'pending'
    PausedThread.instances[-1].run()
    assert calls == ['index:first', 'aux:retry']


def test_update_retry_invalidates_successful_output_when_fingerprint_changed(app, client):
    fingerprints = {
        'dataset:index-daily': 'index-v1', 'dataset:etf-daily': 'etf-v1',
        'dataset:fred': 'fred-v1', 'dataset:a-share-macro': 'macro-v1',
        'artifact:risk-signals': 'risk-v1',
    }
    calls = []
    app.config.update(
        R0_RESOURCE_FINGERPRINTS=fingerprints,
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={
            'index': lambda: calls.append('index:first'),
            'aux': lambda: (_ for _ in ()).throw(RuntimeError('upstream')),
        },
    )
    first = client.post(
        '/api/r0/actions/data-update',
        json={'scopes': ['aux', 'index'], 'force': True},
    )
    assert first.status_code == 202
    assert client.get(first.get_json()['status_url']).get_json()['status'] == 'error'
    fingerprints['dataset:index-daily'] = 'index-v2'
    app.config.update(
        R0_THREAD_FACTORY=PausedThread,
        R0_ACTION_RUNNERS={
            'index': lambda: calls.append('index:retry'),
            'aux': lambda: calls.append('aux:retry'),
        },
    )
    retried = client.post(f"{first.get_json()['status_url']}/retry")
    assert retried.status_code == 202
    accepted = retried.get_json()
    assert accepted['resolved_plan']['reused_outputs'] == []
    assert accepted['resolved_plan']['invalidated_outputs'] == [
        'dataset:etf-daily', 'dataset:index-daily',
    ]
    steps = {item['scope']: item for item in client.get(accepted['status_url']).get_json()['steps']}
    assert steps['index']['status'] == 'pending'
    PausedThread.instances[-1].run()
    assert calls == ['index:first', 'index:retry', 'aux:retry']


def test_update_fails_if_frozen_input_changes_before_worker(app, client):
    fingerprints = {'dataset:index-daily': 'index-v1', 'dataset:etf-daily': 'etf-v1'}
    calls = []
    app.config.update(
        R0_RESOURCE_FINGERPRINTS=fingerprints,
        R0_THREAD_FACTORY=PausedThread,
        R0_ACTION_RUNNERS={'index': lambda: calls.append('index')},
    )
    accepted = client.post(
        '/api/r0/actions/data-update',
        json={'scopes': ['index'], 'force': True},
    )
    fingerprints['dataset:index-daily'] = 'index-v2'
    PausedThread.instances[-1].run()
    status = client.get(accepted.get_json()['status_url']).get_json()
    assert status['steps'][0]['status'] == 'error'
    assert status['steps'][0]['error_code'] == 'input_changed'
    assert calls == []


def test_recovery_retry_rejects_changed_fixed_plan(app, client, monkeypatch):
    target = _configure_recovery(app, ImmediateThread)
    app.config['R0_RECOVERY_BUILDERS'] = {
        ('a_share_timing', 'star50_timing'):
            lambda _target: (_ for _ in ()).throw(RuntimeError('builder failed')),
    }
    first = client.post('/api/r0/actions/cache-recover', json=target)
    assert first.status_code == 202
    status_url = first.get_json()['status_url']
    assert client.get(status_url).get_json()['status'] == 'error'
    monkeypatch.setattr(actions, 'recovery_plan_id', lambda _spec, _variant: 'recovery-plan-v0.1-changed')
    retried = client.post(f'{status_url}/retry')
    assert retried.status_code == 409
    assert retried.get_json()['error'] == 'recovery_plan_changed'


def test_restart_is_503_before_gate_or_any_receipt_side_effect(app, client):
    value = app.extensions['r0_action_runtime']
    marker = Path(app.config['R0_ACTION_MARKER_PATH'])
    response = client.post('/api/r0/actions/restart', json={})
    assert response.status_code == 503
    assert response.get_json()['error'] == 'restart_unavailable'
    assert response.get_json()['capability']['available'] is False
    assert value.gate.owner is None and not marker.exists()
    status = client.get('/api/r0/data-status').get_json()
    assert status['restart']['available'] is False
    logged = json.loads(Path(app.config['R0_OPERATION_LOG_PATH']).read_text(encoding='utf-8').splitlines()[-1])
    assert logged == {
        'action': 'restart', 'error_code': 'restart_unavailable', 'result': 'error',
        'scope': 'service', 'source_ip': '127.0.0.1', 'time': logged['time'],
    }


def test_corrupt_startup_marker_retained_and_closes_canonical_api(config):
    marker = Path(config['R0_ACTION_MARKER_PATH'])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"schema_version":1,"bad":true}', encoding='utf-8')
    app = create_app(config)
    client = app.test_client()
    health = client.get('/api/r0/health').get_json()
    assert health['ready'] is False and health['error'] == 'interrupted_marker_corrupt'
    closed = client.get('/api/r0/sources')
    assert closed.status_code == 503 and closed.get_json()['error'] == 'interrupted_marker_corrupt'
    assert marker.exists()


def test_second_runtime_cannot_reconcile_or_mutate_while_first_runtime_owns_lock(config):
    first_app = create_app(config)
    target = _configure_recovery(first_app, PausedThread)
    accepted = first_app.test_client().post('/api/r0/actions/cache-recover', json=target).get_json()
    marker = Path(config['R0_ACTION_MARKER_PATH'])
    assert marker.exists()
    marker_before = marker.read_bytes()

    second_app = create_app(config)
    client = second_app.test_client()
    health = client.get('/api/r0/health').get_json()
    assert health['ready'] is False
    assert health['error'] == 'external_operation_in_progress'
    closed = client.post('/api/r0/actions/data-update', json={'scopes': ['stock'], 'force': True})
    assert closed.status_code == 503
    assert closed.get_json()['error'] == 'external_operation_in_progress'
    assert marker.read_bytes() == marker_before

    PausedThread.instances[-1].run()
    assert client.get('/api/r0/sources').status_code == 200
    assert client.get('/api/r0/health').get_json()['ready'] is True
    assert client.get(accepted['status_url']).status_code == 404
    assert not marker.exists()


def test_new_boot_adopts_exact_candidate_after_terminal_cleanup_failure(config, monkeypatch):
    first_app = create_app(config)
    target = _configure_recovery(first_app, ImmediateThread)
    first_runtime = first_app.extensions['r0_action_runtime']
    monkeypatch.setattr(first_runtime, '_remove_marker', lambda: (_ for _ in ()).throw(OSError('unlink')))
    response = first_app.test_client().post('/api/r0/actions/cache-recover', json=target)
    assert response.status_code == 202
    assert first_runtime.ready is False
    marker = Path(config['R0_ACTION_MARKER_PATH'])
    payload = json.loads(marker.read_text(encoding='utf-8'))
    rebuilt = next(item for item in payload['affected_targets'] if item['role'] == 'rebuilt')
    assert rebuilt['commit_state'] == 'durable'
    assert rebuilt['candidate']['input_fingerprints']
    assert len(rebuilt['candidate']['input_fingerprint']) == 64
    _release_runtime_as_crashed(first_app, 'cache-recover', response.get_json()['operation_id'])
    second_app = create_app(config)
    client = second_app.test_client()
    assert client.get('/api/r0/health').get_json()['ready'] is True
    assert client.get(
        '/api/r0/sources/a_share_timing/strategies/star50_timing/snapshots'
        f'?variant_id={target["variant_id"]}'
    ).status_code == 200
    assert not marker.exists()


@pytest.mark.parametrize(
    'replacement',
    [_fingerprint('index-daily-v2'), None],
    ids=['changed', 'unknown'],
)
def test_new_boot_rejects_candidate_when_exact_target_input_changed_or_unknown(
        config, monkeypatch, replacement):
    fingerprints = {
        'dataset:index': _fingerprint('index-aggregate-v1'),
        'dataset:index-daily': _fingerprint('index-daily-v1'),
        'dataset:etf-daily': _fingerprint('etf-daily-v1'),
    }
    config['R0_RESOURCE_FINGERPRINTS'] = fingerprints
    first_app = create_app(config)
    target = _configure_recovery(first_app, ImmediateThread)
    first_runtime = first_app.extensions['r0_action_runtime']
    monkeypatch.setattr(first_runtime, '_remove_marker', lambda: (_ for _ in ()).throw(OSError('unlink')))
    response = first_app.test_client().post('/api/r0/actions/cache-recover', json=target)
    assert response.status_code == 202
    marker = Path(config['R0_ACTION_MARKER_PATH'])
    payload = json.loads(marker.read_text(encoding='utf-8'))
    rebuilt = next(item for item in payload['affected_targets'] if item['role'] == 'rebuilt')
    assert rebuilt['commit_state'] == 'durable'
    assert rebuilt['candidate']['input_fingerprints']['dataset:index-daily'] == _fingerprint('index-daily-v1')
    _release_runtime_as_crashed(first_app, 'cache-recover', response.get_json()['operation_id'])

    fingerprints['dataset:index-daily'] = replacement
    second_app = create_app(config)
    client = second_app.test_client()
    assert client.get('/api/r0/health').get_json()['ready'] is True
    variants = client.get(
        '/api/r0/sources/a_share_timing/strategies/star50_timing/variants'
    ).get_json()['items'][0]
    assert variants['cache_state'] == 'artifact_stale'
    assert variants['readable'] is False
    opened = client.get(
        '/api/r0/sources/a_share_timing/strategies/star50_timing/snapshots'
        f'?variant_id={target["variant_id"]}'
    )
    assert opened.status_code == 409
    assert opened.get_json()['error'] == 'cache_miss'
    assert not marker.exists()


def test_macro_csv_change_without_sidecar_invalidates_production_candidate(
        config, tmp_path, monkeypatch):
    resource_root = tmp_path / 'resource-root'
    resource_paths = fingerprints.production_resource_paths(resource_root)
    _materialize_resource_paths(resource_paths['dataset:aux'])
    config['R0_RESOURCE_ROOT'] = resource_root
    config.pop('R0_RESOURCE_FINGERPRINTS')

    first_app = create_app(config)
    first_app.config.update(
        R0_THREAD_FACTORY=ImmediateThread,
        R0_ACTION_RUNNERS={'aux': lambda: None},
    )
    monkeypatch.setattr(actions, '_built_entry', lambda _spec, _variant: {
        'as_of': '2026-01-01',
        'generated_at': '2026-01-02T00:00:00Z',
        'by_strategy': {},
    })
    first_runtime = first_app.extensions['r0_action_runtime']
    monkeypatch.setattr(first_runtime, '_remove_marker', lambda: (_ for _ in ()).throw(OSError('unlink')))

    with first_app.app_context():
        before = fingerprints.resource_fingerprint('dataset:a-share-macro')
    assert before is not None
    macro_csv = resource_root / 'data/a_share_macro/pe_ttm.csv'
    macro_sidecar = resource_root / 'data/a_share_macro/pe_ttm.csv.meta.json'
    sidecar_before = macro_sidecar.read_bytes()

    response = first_app.test_client().post(
        '/api/r0/actions/data-update',
        json={'scopes': ['aux'], 'force': True},
    )
    assert response.status_code == 202
    marker = Path(config['R0_ACTION_MARKER_PATH'])
    payload = json.loads(marker.read_text(encoding='utf-8'))
    rebuilt = next(item for item in payload['affected_targets'] if item['role'] == 'rebuilt')
    assert (rebuilt['source_id'], rebuilt['strategy_id']) == ('decision_context', 'risk_signals')
    assert rebuilt['commit_state'] == 'durable'
    assert rebuilt['candidate']['input_fingerprints']['dataset:a-share-macro'] == before

    macro_csv.write_text('date,value\n2026-01-01,1\n2026-01-02,2\n', encoding='utf-8')
    assert macro_sidecar.read_bytes() == sidecar_before
    with first_app.app_context():
        after = fingerprints.resource_fingerprint('dataset:a-share-macro')
    assert after is not None and after != before
    _release_runtime_as_crashed(first_app, 'data-update', response.get_json()['operation_id'])

    second_app = create_app(config)
    client = second_app.test_client()
    assert client.get('/api/r0/health').get_json()['ready'] is True
    variant = client.get(
        '/api/r0/sources/decision_context/strategies/risk_signals/variants'
    ).get_json()['items'][0]
    assert variant['cache_state'] == 'artifact_stale'
    assert variant['readable'] is False
    opened = client.get(
        '/api/r0/sources/decision_context/strategies/risk_signals/snapshots'
        f'?variant_id={variant["variant_id"]}'
    )
    assert opened.status_code == 409
    assert opened.get_json()['error'] == 'cache_miss'
    assert not marker.exists()

    original_open = Path.open

    def unreadable(path, *args, **kwargs):
        if path == macro_csv:
            raise PermissionError('fixture unreadable')
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, 'open', unreadable)
        with second_app.app_context():
            assert fingerprints.resource_fingerprint('dataset:a-share-macro') is None

    macro_csv.unlink()
    with second_app.app_context():
        assert fingerprints.resource_fingerprint('dataset:a-share-macro') is None


def test_unexpected_pointer_after_interruption_becomes_artifact_stale(config):
    first_app = create_app(config)
    target = _configure_recovery(first_app, PausedThread)
    response = first_app.test_client().post('/api/r0/actions/cache-recover', json=target)
    with first_app.app_context():
        spec, variant, _ = _target()
        publish_entry(spec, variant, _frame())
    _release_runtime_as_crashed(first_app, 'cache-recover', response.get_json()['operation_id'])
    second_app = create_app(config)
    client = second_app.test_client()
    variants = client.get('/api/r0/sources/a_share_timing/strategies/star50_timing/variants').get_json()
    assert variants['items'][0]['cache_state'] == 'artifact_stale'
    assert variants['items'][0]['readable'] is False
