"""Lab-1 contract and isolation regression suite."""
from __future__ import annotations

import copy
import hashlib
import time

import pytest

from lab.build_baseline import build_baseline_document
from lab.contracts import (
    FIXED_POLICY,
    TEMPLATE_ID,
    VALIDATION_WINDOW,
)
from lab.runner import run_trend_validation, runner_fingerprint
from lab.service import LabService
from lab.snapshot import load_materialized_snapshot
from lab.store import canonical_json
from web import state
from web.app import create_app


@pytest.fixture
def lab_app(tmp_path):
    app = create_app({
        'TESTING': True,
        'LAB_ARTIFACT_ROOT': str(tmp_path / 'lab-artifacts'),
        'LAB_EXECUTOR_KIND': 'thread',
    })
    yield app
    app.extensions['lab_service'].shutdown()


@pytest.fixture
def client(lab_app):
    with lab_app.test_client() as value:
        yield value


def _experiment_payload(**hypothesis_overrides):
    hypothesis = {
        'statement': '趋势窗口变化预计会改变信号切换次数与换手成本。',
        'primary_observable': 'signal_switch_count',
        'expected_direction': 'increase',
        'falsification_condition': '若信号切换次数未增加或成本明显放大，则不支持该假设。',
        'validation_window': VALIDATION_WINDOW,
    }
    hypothesis.update(hypothesis_overrides)
    return {
        'template_id': TEMPLATE_ID,
        'title': '趋势窗口敏感性检查',
        'hypothesis': hypothesis,
    }


def _create_variant(client, trend_window=20):
    response = client.post('/api/lab/experiments', json=_experiment_payload())
    assert response.status_code == 201
    experiment = response.get_json()
    response = client.post(
        f"/api/lab/experiments/{experiment['experiment_id']}/variants",
        json={'patch': {'trend_window': trend_window}},
    )
    assert response.status_code == 201
    return experiment, response.get_json()


def _wait_for_run(client, run_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f'/api/lab/runs/{run_id}')
        assert response.status_code == 200
        run = response.get_json()
        if run['status'] in {'success', 'failed', 'skipped'}:
            return run
        time.sleep(0.01)
    pytest.fail(f'run {run_id} did not finish within {timeout}s')


def _successful_result(client, trend_window=20):
    _, variant = _create_variant(client, trend_window=trend_window)
    response = client.post(f"/api/lab/variants/{variant['variant_id']}/runs", json={})
    assert response.status_code == 202
    run = _wait_for_run(client, response.get_json()['run_id'])
    assert run['status'] == 'success', run.get('error')
    result = client.get(f"/api/lab/results/{run['result_id']}").get_json()
    return variant, run, result


def test_learn_lab_compare_pages_and_read_apis_load(client):
    for path in (
        '/learn',
        '/lab',
        '/compare',
        '/api/learn/topics',
        '/api/lab/templates',
    ):
        response = client.get(path)
        assert response.status_code == 200
    assert b'not_configured' in client.get('/learn').data
    assert b'data-experiment-form' in client.get('/lab').data
    assert b'data-compare-form' in client.get('/compare').data


def test_template_echoes_snapshot_window_baseline_and_locked_contract(client):
    template = client.get('/api/lab/templates').get_json()['templates'][0]
    snapshot = template['snapshot']
    assert snapshot['snapshot_id'] == 'snapshot_csi1000_trend_101_v1_8193bfff89f4ae4b'
    assert snapshot['content_sha256'] == (
        '8193bfff89f4ae4b7afce8f5115f7b4ef661371b6093ad7e076b029eb748ec66'
    )
    assert snapshot['input_window'] == FIXED_POLICY['input_window']
    assert snapshot['validation_window'] == VALIDATION_WINDOW
    assert snapshot['validation_row_count'] == 485
    assert len(snapshot['parents']) == 2
    assert len(template['baseline_result_id']) == 64
    assert template['fixed_policy']['holdout_access'] == 'not_configured'
    assert template['editable_schema']['fields'][0]['enum'] == [20, 50, 100]


def test_packaged_baseline_content_hash_and_runner_fingerprint(client):
    template = client.get('/api/lab/templates').get_json()['templates'][0]
    baseline = client.get(
        f"/api/lab/results/{template['baseline_result_id']}"
    ).get_json()
    assert hashlib.sha256(canonical_json(baseline['content'])).hexdigest() == baseline['result_id']
    assert baseline['content_sha256'] == baseline['result_id']
    assert baseline['baseline_result_id'] == baseline['result_id']
    assert baseline['content']['runner']['fingerprint'] == runner_fingerprint()
    assert baseline['content']['holdout_access'] == 'not_configured'
    assert baseline['content']['data_snapshot'] == template['snapshot']
    assert build_baseline_document()['result_id'] == baseline['result_id']


@pytest.mark.parametrize(
    ('patch', 'error'),
    [
        ({'fast_window': 20}, 'unknown_fields'),
        ({'trend_window': 30}, 'parameter_out_of_range'),
        ({'trend_window': 20.0}, 'invalid_parameter_type'),
        ({'trend_window': True}, 'invalid_parameter_type'),
        ({'trend_window': 50}, 'no_change'),
        ({}, 'missing_edit'),
    ],
)
def test_variant_allowlist_rejects_unknown_inert_invalid_and_baseline(
    client, patch, error,
):
    experiment = client.post(
        '/api/lab/experiments', json=_experiment_payload()
    ).get_json()
    response = client.post(
        f"/api/lab/experiments/{experiment['experiment_id']}/variants",
        json={'patch': patch},
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == error


def test_hypothesis_contract_rejects_changed_primary_or_window(client):
    response = client.post(
        '/api/lab/experiments',
        json=_experiment_payload(primary_observable='return'),
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == 'fixed_primary_observable'

    response = client.post(
        '/api/lab/experiments',
        json=_experiment_payload(
            validation_window={**VALIDATION_WINDOW, 'end': '2026-01-01'},
        ),
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == 'fixed_validation_window'


def test_variant_creation_is_idempotent_for_same_hypothesis(client):
    experiment, variant = _create_variant(client)
    response = client.post(
        f"/api/lab/experiments/{experiment['experiment_id']}/variants",
        json={'patch': {'trend_window': 20}},
    )
    assert response.status_code == 200
    assert response.get_json() == variant


def test_async_run_records_actual_contract_result_identity_and_trace(client):
    variant, run, result = _successful_result(client, trend_window=20)
    content = result['content']
    assert run['status_history'][0]['status'] == 'queued'
    assert any(item['status'] == 'running' for item in run['status_history'])
    assert run['result_id'] == result['result_id'] == result['content_sha256']
    assert hashlib.sha256(canonical_json(content)).hexdigest() == result['result_id']
    assert content['actual_config'] == {'trend_window': 20}
    assert content['variant']['variant_id'] == variant['variant_id']
    assert content['data_snapshot']['content_sha256'] == (
        '8193bfff89f4ae4b7afce8f5115f7b4ef661371b6093ad7e076b029eb748ec66'
    )
    assert content['evaluation']['validation_window'] == VALIDATION_WINDOW
    assert content['fixed_policy'] == FIXED_POLICY
    assert content['baseline_result']['result_id'] == run['baseline_result_id']
    assert content['holdout_access'] == 'not_configured'
    assert content['seed_policy'] == 'none'
    assert content['evidence']['metrics']['validation_bars'] == 485
    assert content['evidence']['trace'][0]['date'] == VALIDATION_WINDOW['start']

    first_trade = content['evidence']['trades'][0]
    assert first_trade['signal_date'] < first_trade['date']
    signal_trace = next(
        row for row in content['evidence']['trace']
        if row['date'] == first_trade['signal_date']
    )
    assert signal_trace['signal'] == first_trade['target_exposure_after']


def test_exact_successful_result_is_reused_as_skipped_attempt(client):
    variant, first_run, _ = _successful_result(client)
    response = client.post(f"/api/lab/variants/{variant['variant_id']}/runs", json={})
    assert response.status_code == 200
    retry = response.get_json()
    assert retry['run_id'] != first_run['run_id']
    assert retry['status'] == 'skipped'
    assert retry['outcome'] == 'reused'
    assert retry['result_id'] == first_run['result_id']


def test_run_command_rejects_all_inert_options(client):
    _, variant = _create_variant(client)
    response = client.post(
        f"/api/lab/variants/{variant['variant_id']}/runs",
        json={'force': True},
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == 'unknown_fields'


def test_get_endpoints_do_not_call_runner(monkeypatch, client):
    def forbidden(*args, **kwargs):
        raise AssertionError('GET attempted fresh computation')

    monkeypatch.setattr('lab.service.run_trend_validation', forbidden)
    template = client.get('/api/lab/templates').get_json()['templates'][0]
    for path in (
        '/learn',
        '/lab',
        '/compare',
        '/api/learn/topics',
        '/api/lab/templates',
        f"/api/lab/results/{template['baseline_result_id']}",
    ):
        assert client.get(path).status_code == 200


def test_pure_runner_is_deterministic_and_only_needs_injected_snapshot():
    snapshot = load_materialized_snapshot()
    first = run_trend_validation(snapshot.worker_payload(), {'trend_window': 100})
    second = run_trend_validation(snapshot.worker_payload(), {'trend_window': 100})
    assert first == second
    assert first['metrics']['validation_bars'] == 485
    assert first['validity']['reproducible'] is True


def test_comparison_returns_delta_only_when_contract_is_strictly_comparable(
    client, lab_app,
):
    _, _, candidate = _successful_result(client)
    template = client.get('/api/lab/templates').get_json()['templates'][0]
    response = client.post('/api/lab/comparisons', json={
        'baseline_result_id': template['baseline_result_id'],
        'candidate_result_id': candidate['result_id'],
    })
    comparison = response.get_json()
    assert response.status_code == 200
    assert comparison['status'] == 'comparable'
    assert comparison['changed_fields'] == ['trend_window']
    assert comparison['deltas'] is not None
    assert comparison['data_snapshot'] == template['snapshot']
    assert comparison['baseline_result_id'] == template['baseline_result_id']
    assert comparison['holdout_access'] == 'not_configured'

    mismatched = copy.deepcopy(candidate)
    mismatched['content']['fixed_policy']['costs']['buy_cost'] = 0.0005
    mismatched_hash = hashlib.sha256(canonical_json(mismatched['content'])).hexdigest()
    mismatched['result_id'] = mismatched_hash
    mismatched['content_sha256'] = mismatched_hash
    lab_app.extensions['lab_service'].store.create_immutable(
        'results', mismatched_hash, mismatched,
    )
    response = client.post('/api/lab/comparisons', json={
        'baseline_result_id': template['baseline_result_id'],
        'candidate_result_id': mismatched_hash,
    })
    comparison = response.get_json()
    assert comparison['status'] == 'not_comparable'
    assert comparison['deltas'] is None
    assert any(
        item['field'] == 'fixed_policy.costs'
        for item in comparison['mismatches']
    )


def test_result_get_rejects_corrupted_content_addressed_artifact(client, lab_app):
    _, _, candidate = _successful_result(client)
    corrupted = copy.deepcopy(candidate)
    corrupted['content']['actual_config']['trend_window'] = 100
    result_path = (
        lab_app.extensions['lab_service'].store.root
        / 'results'
        / f"{candidate['result_id']}.json"
    )
    lab_app.extensions['lab_service'].store._atomic_write(result_path, corrupted)
    response = client.get(f"/api/lab/results/{candidate['result_id']}")
    assert response.status_code == 409
    assert response.get_json()['error'] == 'result_integrity_error'


def test_lab_run_does_not_mutate_legacy_web_caches_or_live_path(client, tmp_path):
    sentinel_backtest = {'legacy': object()}
    sentinel_timing = {'legacy': object()}
    old_backtest = state.BACKTEST_CACHE
    old_timing = state.TIMING_CACHE
    state.BACKTEST_CACHE = sentinel_backtest
    state.TIMING_CACHE = sentinel_timing
    live_path = tmp_path / 'live_trades.csv'
    live_path.write_text('stable\n', encoding='utf-8')
    try:
        _successful_result(client)
        assert state.BACKTEST_CACHE is sentinel_backtest
        assert state.TIMING_CACHE is sentinel_timing
        assert list(state.BACKTEST_CACHE) == ['legacy']
        assert list(state.TIMING_CACHE) == ['legacy']
        assert live_path.read_text(encoding='utf-8') == 'stable\n'
    finally:
        state.BACKTEST_CACHE = old_backtest
        state.TIMING_CACHE = old_timing


def test_default_process_worker_completes_outside_web_process(tmp_path):
    service = LabService(tmp_path / 'process-artifacts', executor_kind='process')
    try:
        experiment = service.create_experiment(_experiment_payload())
        variant, created = service.create_variant(
            experiment['experiment_id'],
            {'patch': {'trend_window': 100}},
        )
        assert created is True
        run, status = service.submit_run(variant['variant_id'], {})
        assert status == 202
        deadline = time.time() + 10.0
        while time.time() < deadline:
            run = service.get_run(run['run_id'])
            if run['status'] in {'success', 'failed'}:
                break
            time.sleep(0.02)
        assert run['status'] == 'success', run.get('error')
        assert service.get_result(run['result_id'])['content']['actual_config'] == {
            'trend_window': 100,
        }
    finally:
        service.shutdown()
