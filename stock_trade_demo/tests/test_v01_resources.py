"""Independent factor targets and strict resource boundary fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from web.app import create_app
from web.v01.catalog import get_strategy, variants_for
from web.v01.fingerprints import SCOPE_RESOURCES
from web.v01.snapshot_store import PUBLISHED_SIGNAL_KEYS, publish_entry


@pytest.fixture()
def app(tmp_path):
    return create_app({
        'TESTING': True,
        'R0_GENERATION_ROOT': tmp_path / 'generations',
        'R0_MANUAL_LEDGER_PATH': tmp_path / 'manual.csv',
        'R0_ACTION_MARKER_PATH': tmp_path / 'active-operation.json',
        'R0_CURSOR_KEY': 'test-cursor-key',
        'R0_RESOURCE_FINGERPRINTS': {
            resource_id: f'fixture:{resource_id}'
            for resources in SCOPE_RESOURCES.values()
            for resource_id in resources
        },
    })


@pytest.fixture()
def client(app):
    return app.test_client()


def _target_state():
    return {
        'cache_state': 'ready', 'readable': True,
        'freshness_state': 'current', 'freshness_reason': None,
        'degradation': None,
    }


def _publish_sector(app, frame=None):
    if frame is None:
        frame = pd.DataFrame([
            ('2026-W01', 'A', 1.0, False, 5),
            ('2026-W01', 'B', 2.0, False, 5),
            ('2026-W02', 'A', 3.0, True, 3),
            ('2026-W02', 'C', 4.0, True, 3),
            ('2026-W03', 'B', 5.0, True, 4),
            ('2026-W03', 'A', 6.0, True, 4),
            ('2026-W04', 'C', 7.0, True, 2),
            ('2026-W04', 'A', 8.0, True, 2),
            ('2026-W04', 'A', 9.0, True, 2),
        ], columns=['week_label', 'industry', 'weekly_ret_pct', 'is_partial', 'n_days_in_week'])
    frame.attrs['r0_target_state'] = _target_state()
    frame.attrs['r0_generated_at'] = '2026-01-31T00:00:00Z'
    with app.app_context():
        spec = get_strategy('selection_factor', 'sector_heat')
        variant = variants_for(spec)[0]
        publish_entry(spec, variant, frame)
        return variant


def _publish_single(app, *, top_k=5, payload=None):
    with app.app_context():
        spec = get_strategy('selection_factor', 'single_factor')
        variant = next(item for item in variants_for(spec) if item.canonical_params['top_k'] == top_k)
        content = payload or {
            'version': 'v0.1', 'saved_at': '2026-01-31T00:00:00Z', 'top_k': top_k,
            'items': [
                {
                    'id': 'size', 'name': 'Size', 'column': 'market_cap',
                    'annual_return': 0.12, 'max_drawdown': -0.08, 'calmar': 1.5,
                    'regime_metrics': {
                        '牛市': {'avg_monthly_return': '1.25%', 'n_periods': 4},
                        '熊市': {'avg_monthly_return': 'N/A', 'n_periods': 0},
                    },
                    'dates': ['2026-01-01', '2026-01-02', '2026-01-03'],
                    'nav': [1.0, 1.1, 1.2],
                },
                {
                    'id': 'pb', 'name': 'PB', 'column': 'pb_inv',
                    'annual_return': 0.08, 'max_drawdown': -0.05, 'calmar': 1.6,
                    'regime_metrics': {
                        '牛市': {'avg_monthly_return': '0.75%', 'n_periods': 3},
                        '熊市': {'avg_monthly_return': '-0.25%', 'n_periods': 2},
                    },
                    'dates': ['2026-01-01', '2026-01-02'], 'nav': [1.0, 1.05],
                },
            ],
        }
        publish_entry(spec, variant, content)
        return variant


def _open(client, source, strategy, variant_id):
    response = client.get(f'/api/r0/sources/{source}/strategies/{strategy}/snapshots?variant_id={variant_id}')
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def test_sector_heat_filtered_axes_sparse_ordinals_and_cursor_binding(app, client):
    variant = _publish_sector(app)
    summary = _open(client, 'selection_factor', 'sector_heat', variant.variant_id)
    base = f'/api/r0/sources/selection_factor/strategies/sector_heat/snapshots/{summary["snapshot_id"]}/factors'
    first = client.get(f'{base}?view_id={summary["view_id"]}&kind=sector_heat&weeks=2&limit=2').get_json()
    assert first['weeks'] == ['2026-W03', '2026-W04']
    assert first['weeks_partial'] == [False, True]
    assert first['weeks_n_days'] == [4, 2]
    assert first['industries'] == ['A', 'B', 'C']
    assert first['total'] == 5
    assert [item['artifact_cell_ordinal'] for item in first['items']] == [4, 5]
    second = client.get(
        f'{base}?view_id={summary["view_id"]}&kind=sector_heat&weeks=2&limit=10'
        f'&cursor={first["next_cursor"]}'
    ).get_json()
    assert [item['artifact_cell_ordinal'] for item in second['items']] == [6, 7, 8]
    wrong = client.get(
        f'{base}?view_id={summary["view_id"]}&kind=sector_heat&weeks=1'
        f'&cursor={first["next_cursor"]}'
    )
    assert wrong.status_code == 400 and wrong.get_json()['error'] == 'invalid_cursor'


def test_sector_heat_inconsistent_week_metadata_fails_before_pointer(app):
    frame = pd.DataFrame([
        ('2026-W01', 'A', 1.0, False, 5),
        ('2026-W01', 'B', 2.0, True, 4),
    ], columns=['week_label', 'industry', 'weekly_ret_pct', 'is_partial', 'n_days_in_week'])
    with pytest.raises(Exception):
        _publish_sector(app, frame)


@pytest.mark.parametrize(('column', 'value'), [
    ('is_partial', 'false'),
    ('n_days_in_week', True),
    ('n_days_in_week', 0),
])
def test_sector_heat_rejects_coercible_week_metadata(app, column, value):
    frame = pd.DataFrame([
        ('2026-W01', 'A', 1.0, False, 5),
        ('2026-W01', 'B', 2.0, False, 5),
    ], columns=['week_label', 'industry', 'weekly_ret_pct', 'is_partial', 'n_days_in_week'])
    frame[column] = value
    with pytest.raises(Exception):
        _publish_sector(app, frame)


def test_single_factor_target_metadata_and_factor_nav_are_independently_paged(app, client):
    variant = _publish_single(app, top_k=5)
    summary = _open(client, 'selection_factor', 'single_factor', variant.variant_id)
    base = f'/api/r0/sources/selection_factor/strategies/single_factor/snapshots/{summary["snapshot_id"]}'
    factors = client.get(f'{base}/factors?view_id={summary["view_id"]}&kind=single_factor&limit=1').get_json()
    assert factors['top_k'] == 5 and factors['total'] == 2
    assert factors['items'][0]['id'] == 'size'
    series = client.get(
        f'{base}/series?view_id={summary["view_id"]}&kind=factor_nav&series_id=size'
        '&window=full&resolution=day&limit=2'
    ).get_json()
    assert series['total'] == 3 and [item['value'] for item in series['items']] == [1.0, 1.1]
    assert series['next_cursor']
    filtered = client.get(
        '/api/r0/sources/selection_factor/strategies/single_factor/snapshots'
        f'?variant_id={variant.variant_id}&start=2026-01-02&end=2026-01-03'
    ).get_json()
    monthly = client.get(
        f'{base}/series?view_id={filtered["view_id"]}&kind=factor_nav&series_id=size'
        '&window=full&resolution=month'
    ).get_json()
    assert monthly['total'] == 1
    assert monthly['items'][0]['date'] == '2026-01-03'
    assert monthly['items'][0]['value'] == 1.2
    empty = client.get(
        '/api/r0/sources/selection_factor/strategies/single_factor/snapshots'
        f'?variant_id={variant.variant_id}&start=2027-01-01'
    )
    assert empty.status_code == 422 and empty.get_json()['error'] == 'empty_view_range'


def test_single_factor_top_k_mismatch_and_regime_schema_fail_publication(app):
    bad_top_k = {
        'version': 'v0.1', 'saved_at': '2026-01-31T00:00:00Z', 'top_k': 3,
        'items': [],
    }
    with pytest.raises(Exception):
        _publish_single(app, top_k=5, payload=bad_top_k)
    bad_regime = {
        'version': 'v0.1', 'saved_at': '2026-01-31T00:00:00Z', 'top_k': 5,
        'items': [{
            'id': 'bad', 'name': 'Bad', 'column': 'bad', 'annual_return': 0.1,
            'max_drawdown': -0.1, 'calmar': 1.0,
            'regime_metrics': {
                '牛市': {'avg_monthly_return': 'N/A', 'n_periods': 1},
                '熊市': {'avg_monthly_return': 'N/A', 'n_periods': 0},
            },
            'dates': [], 'nav': [],
        }],
    }
    with pytest.raises(Exception):
        _publish_single(app, payload=bad_regime)


def _timing_frame(signal):
    frame = pd.DataFrame({
        '交易日期': pd.date_range('2026-01-01', periods=2),
        '累积净值': [1.0, 1.1], 'strategy_return': [0.0, 0.1],
        'position': [0.0, 1.0], 'target_exposure': [0.0, 1.0],
        'prev_exposure': [0.0, 0.0], 'exposure_change': [0.0, 1.0],
        'rebalance_action': ['flat', 'enter'], 'signal_action': ['flat', 'buy'],
        'reason_summary': ['x', 'x'], 'reason_detail': [[], []],
        'signal_score': [0.0, 1.0], 'strength_score': [0.0, 1.0],
        'etf_open': [10.0, 10.0], 'etf_close': [10.0, 11.0],
    })
    frame.attrs.update(
        r0_target_state=_target_state(), r0_generated_at='2026-01-03T00:00:00Z',
        published_current_signal=signal,
    )
    return frame


def test_published_signal_nested_objects_are_closed_and_finite(app):
    signal = {key: None for key in PUBLISHED_SIGNAL_KEYS}
    signal.update({
        'strategy_id': 'star50_timing', 'target_exposure': 1.0,
        'profile_recent_6m': {
            'strategy_total_return_pct': 1.0, 'etf_total_return_pct': 1.0,
            'excess_return_pct': 0.0, 'max_drawdown_pct': -1.0,
            'etf_max_drawdown_pct': float('nan'),
        },
    })
    with app.app_context(), pytest.raises(Exception):
        spec = get_strategy('a_share_timing', 'star50_timing')
        publish_entry(spec, variants_for(spec)[0], _timing_frame(signal))
