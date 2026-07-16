"""Production resource fingerprints shared by publication and crash recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from flask import current_app

from web.v01.contracts import canonical_json


SCOPE_RESOURCES = {
    'index': ('dataset:index-daily', 'dataset:etf-daily'),
    'aux': ('dataset:fred', 'dataset:a-share-macro', 'artifact:risk-signals'),
    'stock': ('dataset:stock-csv', 'dataset:stock-parquet'),
    'factor': ('artifact:sector-heat',),
}


def _path_fingerprint(paths: tuple[Path, ...]) -> str | None:
    records = []
    for path in paths:
        try:
            if path.is_dir():
                children = sorted(
                    child for child in path.rglob('*')
                    if child.is_file() and (child.suffix == '.json' or child.name.endswith('.meta.json'))
                )
                records.append({
                    'path': str(path), 'directory': True,
                    'fingerprint': _path_fingerprint(tuple(children)),
                })
                continue
            stat = path.stat()
            raw = path.read_bytes() if stat.st_size <= 1024 * 1024 or path.name.endswith('.meta.json') else b''
            records.append({
                'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'content_digest': hashlib.sha256(raw).hexdigest() if raw else None,
            })
        except OSError:
            records.append({'path': str(path), 'missing': True})
    if not records or all(item.get('missing') for item in records):
        return None
    return hashlib.sha256(canonical_json(records).encode('utf-8')).hexdigest()


def resource_fingerprint(resource_id: str) -> str | None:
    configured = current_app.config.get('R0_RESOURCE_FINGERPRINTS')
    if callable(configured):
        value = configured(resource_id)
        return str(value) if value is not None else None
    if isinstance(configured, dict) and resource_id in configured:
        value = configured[resource_id]
        return str(value) if value is not None else None
    root = Path(__file__).resolve().parents[3]
    resource_paths = {
        'dataset:index': (
            root / 'data/_idx_summary.csv', root / 'data/_idx_summary.csv.meta.json',
            root / 'data/_etf_summary.csv', root / 'data/_etf_summary.csv.meta.json',
        ),
        'dataset:index-daily': (root / 'data/_idx_summary.csv', root / 'data/_idx_summary.csv.meta.json'),
        'dataset:etf-daily': (root / 'data/_etf_summary.csv', root / 'data/_etf_summary.csv.meta.json'),
        'dataset:aux': (
            root / 'data/_fred_summary.csv', root / 'data/_fred_summary.csv.meta.json',
            root / 'data/a_share_macro', root / 'strategy/risk_signals.json',
        ),
        'dataset:fred': (root / 'data/_fred_summary.csv', root / 'data/_fred_summary.csv.meta.json'),
        'dataset:a-share-macro': (root / 'data/a_share_macro',),
        'artifact:risk-signals': (root / 'strategy/risk_signals.json', root / 'strategy/risk_signals.json.meta.json'),
        'dataset:stock': (
            root / 'stock_trade_demo/stock_data.csv', root / 'stock_trade_demo/stock_data.csv.meta.json',
            root / 'stock_trade_demo/stock_data.parquet', root / 'stock_trade_demo/stock_data.parquet.meta.json',
        ),
        'dataset:stock-csv': (root / 'stock_trade_demo/stock_data.csv', root / 'stock_trade_demo/stock_data.csv.meta.json'),
        'dataset:stock-parquet': (root / 'stock_trade_demo/stock_data.parquet', root / 'stock_trade_demo/stock_data.parquet.meta.json'),
        'dataset:factor': (root / 'strategy/backtest_sector_heat.csv', root / 'strategy/backtest_sector_heat.csv.meta.json'),
        'artifact:sector-heat': (root / 'strategy/backtest_sector_heat.csv', root / 'strategy/backtest_sector_heat.csv.meta.json'),
    }
    paths = resource_paths.get(resource_id)
    return _path_fingerprint(paths) if paths is not None else None


def target_input_fingerprints(scopes: Iterable[str]) -> dict[str, str | None]:
    return {resource_id: resource_fingerprint(resource_id) for resource_id in target_input_resources(scopes)}


def target_input_resources(scopes: Iterable[str]) -> tuple[str, ...]:
    resources = []
    for scope in scopes:
        for resource_id in SCOPE_RESOURCES.get(scope, ()):
            if resource_id not in resources:
                resources.append(resource_id)
    return tuple(resources)


def fingerprint_set_digest(values: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(values).encode('utf-8')).hexdigest()


def fingerprints_are_known(values: dict[str, Any]) -> bool:
    return all(isinstance(value, str) and bool(value) for value in values.values())
