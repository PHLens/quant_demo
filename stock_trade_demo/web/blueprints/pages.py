"""R0 Snapshot / Legacy Viewer pages.

The R0 shell intentionally exposes four destinations only. Historical snapshot
URLs remain as aliases so old bookmarks land on a matching read-only source.
"""
from __future__ import annotations

from flask import Blueprint, abort, render_template

bp = Blueprint('pages', __name__)


SNAPSHOT_SOURCES = {
    'selection': {
        'id': 'selection',
        'name': 'A-share selection cache',
        'description': 'Cached stock-selection output. Parameters and evidence fields are display-only.',
        'default_strategy': 'original_ensemble',
        'build_hint': 'python scripts/build_select_cache.py',
    },
    'cn-timing': {
        'id': 'cn-timing',
        'name': 'China timing cache',
        'description': 'Cached China timing curves, positions, fees and signal fields.',
        'default_strategy': 'csi1000_timing',
        'build_hint': 'python scripts/build_timing_cache.py',
    },
    'us-timing': {
        'id': 'us-timing',
        'name': 'US timing cache',
        'description': 'Cached US timing curves, positions, fees and signal fields.',
        'default_strategy': 'macro_v32_timing',
        'build_hint': 'python scripts/build_us_timing_cache.py',
    },
    'hk-timing': {
        'id': 'hk-timing',
        'name': 'Hong Kong timing legacy source',
        'description': 'Legacy Hong Kong source. It is unavailable unless a prebuilt cache is present.',
        'default_strategy': 'hsi_timing',
        'build_hint': 'No R0 web build action is available.',
    },
    'commodity': {
        'id': 'commodity',
        'name': 'Commodity timing legacy source',
        'description': 'Legacy commodity source. It is unavailable unless a prebuilt cache is present.',
        'default_strategy': 'gold_timing',
        'build_hint': 'No R0 web build action is available.',
    },
}


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/snapshot/<source_id>')
def snapshot_page(source_id):
    source = SNAPSHOT_SOURCES.get(source_id)
    if source is None:
        abort(404)
    return render_template('snapshot.html', snapshot_source=source)


@bp.route('/legacy-artifacts')
def legacy_artifacts_page():
    return render_template('legacy.html')


@bp.route('/data-status')
def data_status_page():
    return render_template('data_status.html')


@bp.route('/manual-records')
def manual_records_page():
    return render_template('manual_records.html')


@bp.route('/timing')
def timing_page():
    return snapshot_page('cn-timing')


@bp.route('/us_timing')
def us_timing_page():
    return snapshot_page('us-timing')


@bp.route('/commodity')
def commodity_page():
    return snapshot_page('commodity')


@bp.route('/hk_timing')
def hk_timing_page():
    return snapshot_page('hk-timing')
