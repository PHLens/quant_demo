"""Production resource fingerprints shared by publication and crash recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from flask import current_app

from web.v01.contracts import canonical_json
from web.v01.resource_paths import configured_resource_root


SCOPE_RESOURCES = {
    'index': ('dataset:index-daily', 'dataset:etf-daily'),
    'aux': ('dataset:fred', 'dataset:a-share-macro', 'artifact:risk-signals'),
    'stock': ('dataset:stock-csv', 'dataset:stock-parquet'),
    'factor': ('artifact:sector-heat',),
}


FRED_INPUTS = (
    'fred_FedFundsRate.csv',
    'fred_YieldCurve_10Y2Y.csv',
    'fred_CPI_core.csv',
    'fred_Unemployment.csv',
    'fred_VIX.csv',
    'fred_HighYieldSpread.csv',
    'fred_Treasury10Y.csv',
)
MACRO_INPUTS = ('pe_ttm.csv', 'cn10y.csv', 'sse_daily.csv')


def _with_sidecars(directory: Path, names: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        path
        for name in names
        for path in (directory / name, directory / f'{name}.meta.json')
    )


def production_resource_paths(root: Path) -> dict[str, tuple[Path, ...]]:
    """Exact server-owned files represented by every production resource ID."""
    index_daily = (
        root / 'data/_idx_summary.csv', root / 'data/_idx_summary.csv.meta.json',
    )
    etf_daily = (
        root / 'data/_etf_summary.csv', root / 'data/_etf_summary.csv.meta.json',
    )
    fred = (
        root / 'data/_fred_summary.csv', root / 'data/_fred_summary.csv.meta.json',
        *_with_sidecars(root / 'data', FRED_INPUTS),
    )
    macro = _with_sidecars(root / 'data/a_share_macro', MACRO_INPUTS)
    risk = (
        root / 'strategy/risk_signals.json',
        root / 'strategy/risk_signals.json.meta.json',
    )
    stock_csv = (
        root / 'stock_trade_demo/stock_data.csv',
        root / 'stock_trade_demo/stock_data.csv.meta.json',
    )
    stock_parquet = (
        root / 'stock_trade_demo/stock_data.parquet',
        root / 'stock_trade_demo/stock_data.parquet.meta.json',
    )
    sector_heat = (
        root / 'strategy/backtest_sector_heat.csv',
        root / 'strategy/backtest_sector_heat.csv.meta.json',
    )
    return {
        'dataset:index': index_daily + etf_daily,
        'dataset:index-daily': index_daily,
        'dataset:etf-daily': etf_daily,
        'dataset:aux': fred + macro + risk,
        'dataset:fred': fred,
        'dataset:a-share-macro': macro,
        'artifact:risk-signals': risk,
        'dataset:stock': stock_csv + stock_parquet,
        'dataset:stock-csv': stock_csv,
        'dataset:stock-parquet': stock_parquet,
        'dataset:factor': sector_heat,
        'artifact:sector-heat': sector_heat,
    }


def _path_fingerprint(paths: tuple[Path, ...]) -> str | None:
    records = []
    for path in paths:
        try:
            if path.is_dir():
                children = sorted(child for child in path.rglob('*') if child.is_file())
                if not children or any(child.is_symlink() for child in children):
                    return None
                child_fingerprint = _path_fingerprint(tuple(children))
                if child_fingerprint is None:
                    return None
                records.append({
                    'path': str(path), 'directory': True,
                    'fingerprint': child_fingerprint,
                })
                continue
            if path.is_symlink() or not path.is_file():
                return None
            stat = path.stat()
            if stat.st_size <= 0:
                return None
            with path.open('rb') as stream:
                if stat.st_size <= 1024 * 1024 or path.name.endswith('.meta.json'):
                    sample = stream.read()
                else:
                    head = stream.read(64 * 1024)
                    stream.seek(max(stat.st_size - 64 * 1024, 0))
                    sample = head + stream.read(64 * 1024)
            records.append({
                'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'content_digest': hashlib.sha256(sample).hexdigest(),
            })
        except OSError:
            return None
    if not records:
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
    resource_paths = production_resource_paths(configured_resource_root())
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
