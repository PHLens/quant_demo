"""Canonical v0.1 pages plus one-hop, query-discarding HTML aliases."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, request


bp = Blueprint('pages', __name__)


@bp.get('/snapshots')
def snapshots_page():
    return render_template('snapshot.html', snapshot_query={
        'source_id': request.args.get('source_id', 'selection'),
        'strategy_id': request.args.get('strategy_id', 'original_ensemble'),
        'tab': request.args.get('tab', 'summary'),
        'initial_range': request.args.get('initial_range', 'full'),
    })


@bp.get('/legacy-artifacts')
def legacy_artifacts_page():
    return render_template('legacy.html')


@bp.get('/data-status')
def data_status_page():
    return render_template('data_status.html')


@bp.get('/manual-records')
def manual_records_page():
    return render_template('manual_records.html', initial_strategy=request.args.get('strategy', 'star50_timing'))


_ALIASES = {
    '/': '/snapshots?source_id=selection&strategy_id=original_ensemble&tab=summary&initial_range=full',
    '/timing': '/snapshots?source_id=a_share_timing&strategy_id=csi1000_timing&initial_range=6m',
    '/us_timing': '/snapshots?source_id=us_timing&strategy_id=macro_v32_timing&initial_range=6m',
    '/hk_timing': '/snapshots?source_id=hk_timing&strategy_id=hsi_timing&initial_range=full',
    '/commodity': '/snapshots?source_id=commodity&strategy_id=gold_timing&initial_range=full',
    '/live': '/manual-records?strategy=star50_timing',
}


def _alias(destination: str):
    # Never inspect request.args: every legacy page ignored page query and the
    # compatibility contract preserves that behavior exactly.
    return redirect(destination, code=302)


for _index, (_path, _destination) in enumerate(_ALIASES.items()):
    bp.add_url_rule(
        _path,
        endpoint=f'legacy_html_alias_{_index}',
        view_func=lambda destination=_destination: _alias(destination),
        methods=['GET'],
    )
