"""Read-only APIs for the R0 Snapshot / Legacy Viewer.

There is no registration, import, refresh, rebuild, or execution path here.
Every filesystem target is a literal entry below. Legacy evidence is always
reported as unknown/unverified; file presence never upgrades its eligibility.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
from flask import Blueprint, abort, jsonify, send_file

from web import state

bp = Blueprint('r0_viewer_api', __name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
_ALLOWED_DOWNLOAD_SUFFIXES = {'.json': 'application/json', '.csv': 'text/csv'}

_SENSITIVE_KEY_RE = re.compile(
    r'(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|credential|'
    r'private[_-]?key|session|cookie|authorization)(?:$|[_-])',
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r'(?:sk_(?:agent|machine|live|test)_[A-Za-z0-9_-]{8,}|gh[opsu]_[A-Za-z0-9]{20,}|'
    r'AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|'
    r'/(?:Users|home|root)/[^\s,;"\']+|[A-Za-z]:\\(?:Users|Documents and Settings)\\)',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegacyArtifact:
    artifact_id: str
    label: str
    kind: str
    relative_path: str
    downloadable: bool


_ARTIFACTS = (
    LegacyArtifact('profile-csi1000', 'CSI 1000 profile', 'profile', 'strategy/best_profile_csi1000_timing.json', True),
    LegacyArtifact('profile-chinext', 'ChiNext profile', 'profile', 'strategy/best_profile_chinext_timing.json', True),
    LegacyArtifact('profile-star50', 'STAR 50 profile', 'profile', 'strategy/best_profile_star50_timing.json', True),
    LegacyArtifact('profile-sp500', 'S&P 500 profile', 'profile', 'strategy/best_profile_sp500_timing.json', True),
    LegacyArtifact('profile-macro-v32', 'Macro v3.2 profile', 'profile', 'strategy/best_profile_macro_v32_timing.json', True),
    LegacyArtifact('sensitivity-csi1000', 'CSI 1000 sensitivity table', 'sensitivity', 'strategy/sensitivity_csi1000_timing.csv', True),
    LegacyArtifact('sensitivity-chinext', 'ChiNext sensitivity table', 'sensitivity', 'strategy/sensitivity_chinext_timing.csv', True),
    LegacyArtifact('sensitivity-star50', 'STAR 50 sensitivity table', 'sensitivity', 'strategy/sensitivity_star50_timing.csv', True),
    LegacyArtifact('holdout-csi1000', 'CSI 1000 holdout note', 'holdout', 'strategy/holdout_report_csi1000_timing.md', False),
    LegacyArtifact('holdout-chinext', 'ChiNext holdout note', 'holdout', 'strategy/holdout_report_chinext_timing.md', False),
    LegacyArtifact('holdout-star50', 'STAR 50 holdout note', 'holdout', 'strategy/holdout_report_star50_timing.md', False),
    LegacyArtifact('holdout-sp500', 'S&P 500 holdout note', 'holdout', 'strategy/holdout_report_sp500_timing.md', False),
    LegacyArtifact('holdout-macro-v32', 'Macro v3.2 holdout note', 'holdout', 'strategy/holdout_report_macro_v32_timing.md', False),
    LegacyArtifact('curve-macro-v32', 'Macro v3.2 cached curve', 'cached-output', 'strategy/backtest_v32_mix.csv', True),
    LegacyArtifact('curve-sector-heat', 'Sector heat cached curve', 'cached-output', 'strategy/backtest_sector_heat.csv', True),
    LegacyArtifact('cache-web', 'Web cache container', 'cache', 'stock_trade_demo/.cache/web_cache.pkl', False),
    LegacyArtifact('cache-single-factor', 'Single-factor cache container', 'cache', 'stock_trade_demo/.cache/single_factor_results.pkl', False),
)
_ARTIFACTS_BY_ID = {item.artifact_id: item for item in _ARTIFACTS}

_DATA_FILES = (
    ('selection-data', 'Selection data snapshot', 'stock_trade_demo/stock_data.csv'),
    ('index-summary', 'Index data summary', 'data/_idx_summary.csv'),
    ('etf-summary', 'ETF data summary', 'data/_etf_summary.csv'),
    ('macro-summary', 'Macro data summary', 'data/_fred_summary.csv'),
    ('factor-signals', 'Factor signal snapshot', 'strategy/factor_signals_v32.csv'),
)

_SOURCE_IDS = {'selection', 'cn-timing', 'us-timing', 'hk-timing', 'commodity'}


def _safe_resolve(relative_path: str) -> Path:
    root = _REPO_ROOT.resolve()
    candidate = _REPO_ROOT / relative_path
    if candidate.is_symlink():
        raise ValueError('symlink targets are not downloadable')
    current = candidate.parent
    while current != _REPO_ROOT and current != current.parent:
        if current.is_symlink():
            raise ValueError('symlink parents are not downloadable')
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError('path escapes the public resource root')
    return resolved


def _walk_json(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk_json(child)


def _has_sensitive_content(path: Path, raw: bytes) -> bool:
    text = raw.decode('utf-8-sig')
    if _SENSITIVE_VALUE_RE.search(text):
        return True
    if path.suffix.lower() == '.json':
        payload = json.loads(text)
        for key, value in _walk_json(payload):
            if key and _SENSITIVE_KEY_RE.search(key):
                return True
            if isinstance(value, str) and _SENSITIVE_VALUE_RE.search(value):
                return True
        return False
    rows = csv.reader(text.splitlines())
    header = next(rows, [])
    return any(_SENSITIVE_KEY_RE.search(str(column)) for column in header)


def _validated_download(artifact: LegacyArtifact) -> tuple[Path, str]:
    if not artifact.downloadable:
        raise ValueError('artifact is metadata-only')
    path = _safe_resolve(artifact.relative_path)
    mime = _ALLOWED_DOWNLOAD_SUFFIXES.get(path.suffix.lower())
    if mime is None:
        raise ValueError('file type is not allowed')
    size = path.stat().st_size
    if size <= 0 or size > _MAX_DOWNLOAD_BYTES:
        raise ValueError('file size is outside the public limit')
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError('file changed during validation')
    if _has_sensitive_content(path, raw):
        raise ValueError('sensitive content is not downloadable')
    return path, mime


def _artifact_payload(item: LegacyArtifact) -> dict[str, Any]:
    try:
        path = _safe_resolve(item.relative_path)
        exists = path.is_file()
        size = path.stat().st_size if exists else None
    except (FileNotFoundError, OSError, ValueError):
        exists = False
        size = None

    safe_download = False
    if exists and item.downloadable:
        try:
            _validated_download(item)
            safe_download = True
        except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, OSError, ValueError):
            safe_download = False

    return {
        'id': item.artifact_id,
        'label': item.label,
        'kind': item.kind,
        'format': Path(item.relative_path).suffix.lower().lstrip('.') or 'unknown',
        'file_name': Path(item.relative_path).name,
        'available': exists,
        'size_bytes': size,
        'provenance': 'unknown',
        'evidence_status': 'unverified',
        'download_url': f'/api/r0/artifacts/{item.artifact_id}/download' if safe_download else None,
    }


def _read_sidecar(relative_path: str) -> dict[str, Any]:
    sidecar = _REPO_ROOT / f'{relative_path}.meta.json'
    try:
        if sidecar.is_symlink():
            return {}
        resolved = sidecar.resolve(strict=True)
        root = _REPO_ROOT.resolve()
        if root not in resolved.parents:
            return {}
        payload = json.loads(resolved.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _data_status_payload() -> list[dict[str, Any]]:
    rows = []
    for source_id, label, relative_path in _DATA_FILES:
        path = _REPO_ROOT / relative_path
        exists = path.is_file() and not path.is_symlink()
        meta = _read_sidecar(relative_path)
        rows.append({
            'id': source_id,
            'label': label,
            'file_name': Path(relative_path).name,
            'available': exists,
            'size_bytes': path.stat().st_size if exists else None,
            'sidecar_written_at': meta.get('written_at_iso'),
            'data_as_of': 'unknown',
            'source_vintage': 'unknown',
            'provenance': 'unknown',
            'evidence_status': 'unverified',
        })
    return rows


def _load_web_cache_only() -> None:
    if not state.BACKTEST_CACHE or not state.TIMING_CACHE:
        state._load_disk_cache()


def _source_strategy_rows(source_id: str) -> list[dict[str, Any]]:
    if source_id == 'selection':
        strategy_ids = sorted(state.BACKTEST_CACHE.keys())
        if not strategy_ids:
            strategy_ids = [state.get_focused_strategy_id()]
        rows = []
        for strategy_id in strategy_ids:
            strategy = state.build_strategy(strategy_id)
            rows.append({
                'id': strategy_id,
                'name': strategy.get_display_name(),
                'available': strategy_id in state.BACKTEST_CACHE,
            })
        return rows
    if source_id == 'cn-timing':
        mapping = state.TIMING_STRATEGY_MAP
        cache = state.TIMING_CACHE
    elif source_id == 'us-timing':
        mapping = state.US_TIMING_STRATEGY_MAP
        cache = state.US_TIMING_CACHE
    elif source_id == 'hk-timing':
        mapping = state.HK_STRATEGY_MAP
        cache = state.HK_CACHE
    elif source_id == 'commodity':
        mapping = state.COMMODITY_STRATEGY_MAP
        cache = state.COMMODITY_CACHE
    else:
        return []
    return [
        {'id': sid, 'name': cls().get_display_name(), 'available': sid in cache}
        for sid, cls in mapping.items()
    ]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'item'):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _curve_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or len(frame) == 0 or '交易日期' not in frame.columns or '累积净值' not in frame.columns:
        return []
    rows = []
    for date, value in zip(pd.to_datetime(frame['交易日期'], errors='coerce'), frame['累积净值']):
        if pd.isna(date) or pd.isna(value):
            continue
        rows.append({'date': date.strftime('%Y-%m-%d'), 'value': round(float(value), 6)})
    if len(rows) <= 480:
        return rows
    indexes = sorted({round(i * (len(rows) - 1) / 479) for i in range(480)})
    return [rows[i] for i in indexes]


def _basic_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    curve = _curve_from_frame(frame)
    if not curve:
        return {}
    values = pd.Series([point['value'] for point in curve], dtype=float)
    drawdown = values / values.cummax() - 1
    return {
        'cumulative_value': round(float(values.iloc[-1]), 4),
        'total_return': f'{(float(values.iloc[-1]) / float(values.iloc[0]) - 1) * 100:.2f}%' if values.iloc[0] else 'unknown',
        'max_drawdown': f'{float(drawdown.min()) * 100:.2f}%',
        'observations': int(len(frame)),
    }


def _timing_tables(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if frame is None or len(frame) == 0:
        return [], []
    latest = frame.tail(1).iloc[0]
    position = [{
        'date': pd.to_datetime(latest.get('交易日期')).strftime('%Y-%m-%d'),
        'position_label': _json_value(latest.get('position_label')),
        'target_exposure': _json_value(latest.get('target_exposure', latest.get('position'))),
        'holding_units': _json_value(latest.get('holding_units')),
        'holding_value': _json_value(latest.get('holding_value')),
        'cash_balance': _json_value(latest.get('cash_balance')),
        'unrealized_pnl': _json_value(latest.get('unrealized_pnl')),
    }]
    if 'trade_quantity' in frame.columns:
        trade_rows = frame[pd.to_numeric(frame['trade_quantity'], errors='coerce').fillna(0).abs() > 1e-8].tail(100)
    elif 'signal_action' in frame.columns:
        trade_rows = frame[frame['signal_action'].isin(['buy', 'sell'])].tail(100)
    else:
        trade_rows = frame.iloc[0:0]
    trades = []
    for _, row in trade_rows.iterrows():
        trades.append({
            'date': pd.to_datetime(row.get('交易日期')).strftime('%Y-%m-%d'),
            'action': _json_value(row.get('signal_action')),
            'target_exposure': _json_value(row.get('target_exposure', row.get('position'))),
            'trade_price': _json_value(row.get('etf_open')),
            'quantity': _json_value(row.get('trade_quantity')),
            'fee_amount': _json_value(row.get('trade_fee_amount')),
            'nav': _json_value(row.get('累积净值')),
        })
    return position, trades


def _selection_holdings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or len(frame) == 0:
        return []
    for column in ('买入个股收益', '买入股票代码'):
        if column not in frame.columns:
            continue
        for value in reversed(frame[column].tolist()):
            if not isinstance(value, (list, tuple)) or not value:
                continue
            rows = []
            for item in value:
                if isinstance(item, dict):
                    rows.append({str(key): _json_value(val) for key, val in list(item.items())[:10]})
                else:
                    rows.append({'value': _json_value(item)})
            return rows[:100]
    return []


def _cache_for_source(source_id: str) -> dict[str, Any]:
    if source_id in {'selection', 'cn-timing'}:
        _load_web_cache_only()
    elif source_id == 'us-timing':
        state.init_us_timing_cache()
    if source_id == 'selection':
        return state.BACKTEST_CACHE
    if source_id == 'cn-timing':
        return state.TIMING_CACHE
    if source_id == 'us-timing':
        return state.US_TIMING_CACHE
    if source_id == 'hk-timing':
        return state.HK_CACHE
    if source_id == 'commodity':
        return state.COMMODITY_CACHE
    return {}


def _configuration(source_id: str, strategy_id: str) -> list[dict[str, Any]]:
    if source_id == 'selection':
        payload = state.build_strategy(strategy_id).get_factor_metadata()
    elif source_id == 'cn-timing':
        payload = state.build_timing_strategy(strategy_id).get_signal_metadata()
    elif source_id == 'us-timing':
        payload = state.build_us_timing_strategy(strategy_id).get_signal_metadata()
    elif source_id == 'hk-timing':
        payload = state.build_hk_strategy(strategy_id).get_signal_metadata()
    elif source_id == 'commodity':
        payload = state.build_commodity_strategy(strategy_id).get_signal_metadata()
    else:
        return []
    rows = payload.get('parameters') or payload.get('factors') or []
    return [
        {
            'key': row.get('key', 'unknown'),
            'label': row.get('label') or row.get('name') or row.get('key', 'unknown'),
            'value': _json_value(row.get('default')),
            'description': row.get('description') or '',
        }
        for row in rows
        if isinstance(row, dict)
    ]


@bp.get('/api/r0/artifacts')
def api_r0_artifacts():
    return jsonify({'artifacts': [_artifact_payload(item) for item in _ARTIFACTS]})


@bp.get('/api/r0/artifacts/<artifact_id>/download')
def api_r0_artifact_download(artifact_id: str):
    artifact = _ARTIFACTS_BY_ID.get(artifact_id)
    if artifact is None:
        abort(404)
    try:
        path, mime = _validated_download(artifact)
    except FileNotFoundError:
        abort(404)
    except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, OSError, ValueError):
        abort(403)
    return send_file(
        path,
        mimetype=mime,
        as_attachment=True,
        download_name=path.name,
        conditional=False,
        max_age=0,
    )


@bp.get('/api/r0/data-status')
def api_r0_data_status():
    return jsonify({
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'sources': _data_status_payload(),
    })


@bp.get('/api/r0/manual-records')
def api_r0_manual_records():
    """Keep the legacy manual ledger outside the public R0 surface."""
    return jsonify({
        'available': False,
        'records': [],
        'provenance': 'unknown',
        'evidence_status': 'unverified',
        'message': 'Manual record storage is not exposed by this public R0 viewer.',
    })


@bp.get('/api/r0/sources/<source_id>/strategies')
def api_r0_source_strategies(source_id: str):
    if source_id not in _SOURCE_IDS:
        abort(404)
    if source_id in {'selection', 'cn-timing'}:
        _load_web_cache_only()
    elif source_id == 'us-timing':
        state.init_us_timing_cache()
    return jsonify({
        'source_id': source_id,
        'provenance': 'unknown',
        'evidence_status': 'unverified',
        'strategies': _source_strategy_rows(source_id),
    })


@bp.get('/api/r0/sources/<source_id>/snapshots/<strategy_id>')
def api_r0_snapshot(source_id: str, strategy_id: str):
    if source_id not in _SOURCE_IDS:
        abort(404)
    valid_ids = {row['id'] for row in _source_strategy_rows(source_id)}
    if strategy_id not in valid_ids:
        abort(404)
    cache = _cache_for_source(source_id)
    cached = cache.get(strategy_id)
    if cached is None:
        return jsonify({
            'available': False,
            'error': 'snapshot_unavailable',
            'source_id': source_id,
            'strategy_id': strategy_id,
            'provenance': 'unknown',
            'evidence_status': 'unverified',
            'data_as_of': 'unknown',
            'metrics': {},
            'equity_curve': [],
            'holdings': [],
            'trades': [],
            'configuration': _configuration(source_id, strategy_id),
            'message': 'No prebuilt cache is available. R0 does not build or refresh it.',
        })
    frame = cached[0] if source_id == 'selection' and isinstance(cached, tuple) else cached
    if not isinstance(frame, pd.DataFrame):
        return jsonify({
            'available': False,
            'error': 'unsupported_snapshot',
            'source_id': source_id,
            'strategy_id': strategy_id,
            'provenance': 'unknown',
            'evidence_status': 'unverified',
            'data_as_of': 'unknown',
            'metrics': {},
            'equity_curve': [],
            'holdings': [],
            'trades': [],
            'configuration': _configuration(source_id, strategy_id),
            'message': 'The prebuilt cache format cannot be displayed by R0.',
        })
    if source_id == 'selection':
        holdings, trades = _selection_holdings(frame), []
    else:
        holdings, trades = _timing_tables(frame)
    curve = _curve_from_frame(frame)
    return jsonify({
        'available': True,
        'source_id': source_id,
        'strategy_id': strategy_id,
        'provenance': 'unknown',
        'evidence_status': 'unverified',
        'data_as_of': curve[-1]['date'] if curve else 'unknown',
        'metrics': _basic_metrics(frame),
        'equity_curve': curve,
        'holdings': holdings,
        'trades': trades,
        'configuration': _configuration(source_id, strategy_id),
    })
