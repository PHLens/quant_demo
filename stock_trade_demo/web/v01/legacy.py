"""The exact 17-slot legacy metadata and safe-download registry."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from web.v01.contracts import ApiError


REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MIME = {'.json': 'application/json', '.csv': 'text/csv'}

SENSITIVE_KEY = re.compile(r'(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|credential|private[_-]?key|session|cookie|authorization)(?:$|[_-])', re.I)
SENSITIVE_VALUE = re.compile(r'(?:sk_(?:agent|machine|live|test)_[A-Za-z0-9_-]{8,}|gh[opsu]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|/(?:Users|home|root)/[^\s,;"\']+|[A-Za-z]:\\(?:Users|Documents and Settings)\\)', re.I)


@dataclass(frozen=True)
class Slot:
    ordinal: int
    artifact_id: str
    label: str
    kind: str
    relative_path: str
    mode: str


_ROWS = (
    ('profile-csi1000', 'CSI 1000 profile', 'profile', 'strategy/best_profile_csi1000_timing.json', 'safe_download'),
    ('profile-chinext', 'ChiNext profile', 'profile', 'strategy/best_profile_chinext_timing.json', 'safe_download'),
    ('profile-star50', 'STAR 50 profile', 'profile', 'strategy/best_profile_star50_timing.json', 'safe_download'),
    ('profile-sp500', 'S&P 500 profile', 'profile', 'strategy/best_profile_sp500_timing.json', 'safe_download'),
    ('profile-macro-v32', 'Macro v3.2 profile', 'profile', 'strategy/best_profile_macro_v32_timing.json', 'safe_download'),
    ('sensitivity-csi1000', 'CSI 1000 sensitivity table', 'sensitivity', 'strategy/sensitivity_csi1000_timing.csv', 'safe_download'),
    ('sensitivity-chinext', 'ChiNext sensitivity table', 'sensitivity', 'strategy/sensitivity_chinext_timing.csv', 'safe_download'),
    ('sensitivity-star50', 'STAR 50 sensitivity table', 'sensitivity', 'strategy/sensitivity_star50_timing.csv', 'safe_download'),
    ('holdout-csi1000', 'CSI 1000 holdout note', 'holdout', 'strategy/holdout_report_csi1000_timing.md', 'metadata_only'),
    ('holdout-chinext', 'ChiNext holdout note', 'holdout', 'strategy/holdout_report_chinext_timing.md', 'metadata_only'),
    ('holdout-star50', 'STAR 50 holdout note', 'holdout', 'strategy/holdout_report_star50_timing.md', 'metadata_only'),
    ('holdout-sp500', 'S&P 500 holdout note', 'holdout', 'strategy/holdout_report_sp500_timing.md', 'metadata_only'),
    ('holdout-macro-v32', 'Macro v3.2 holdout note', 'holdout', 'strategy/holdout_report_macro_v32_timing.md', 'metadata_only'),
    ('curve-macro-v32', 'Macro v3.2 cached curve', 'cached-output', 'strategy/backtest_v32_mix.csv', 'safe_download'),
    ('curve-sector-heat', 'Sector heat cached curve', 'cached-output', 'strategy/backtest_sector_heat.csv', 'safe_download'),
    ('cache-web', 'Web cache container', 'cache', 'stock_trade_demo/.cache/web_cache.pkl', 'metadata_only'),
    ('cache-single-factor', 'Single-factor cache container', 'cache', 'stock_trade_demo/.cache/single_factor_results.pkl', 'metadata_only'),
)
SLOTS = tuple(Slot(index, *row) for index, row in enumerate(_ROWS))
BY_ID = {item.artifact_id: item for item in SLOTS}


def _literal_path(slot: Slot) -> Path:
    path = REPO_ROOT / slot.relative_path
    root = REPO_ROOT.resolve()
    try:
        current = path
        while current != REPO_ROOT:
            if current.is_symlink():
                raise ApiError(403, 'artifact_download_rejected', 'Symlink downloads are forbidden.')
            current = current.parent
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ApiError(404, 'artifact_missing', 'Artifact is missing.') from exc
    if resolved != root and root not in resolved.parents:
        raise ApiError(403, 'artifact_download_rejected', 'Artifact path escaped its allowlist root.')
    return resolved


def _walk(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def validated_download(slot: Slot) -> tuple[Path, str, bytes]:
    if slot.mode != 'safe_download':
        raise ApiError(403, 'artifact_download_rejected', 'This slot is metadata-only.')
    path = _literal_path(slot)
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        raise ApiError(403, 'artifact_download_rejected', 'Artifact type is not allowed.')
    before = path.stat()
    if before.st_size <= 0 or before.st_size > MAX_DOWNLOAD_BYTES:
        raise ApiError(403, 'artifact_download_rejected', 'Artifact size is outside the fixed limit.')
    try:
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
            raise ApiError(403, 'artifact_download_rejected', 'Artifact changed during validation.')
        text = raw.decode('utf-8-sig')
        if SENSITIVE_VALUE.search(text):
            raise ApiError(403, 'artifact_download_rejected', 'Artifact failed the sensitive-content check.')
        if path.suffix.lower() == '.json':
            payload = json.loads(text)
            if any((key and SENSITIVE_KEY.search(key)) or (isinstance(value, str) and SENSITIVE_VALUE.search(value)) for key, value in _walk(payload)):
                raise ApiError(403, 'artifact_download_rejected', 'Artifact failed the sensitive-content check.')
        else:
            header = next(csv.reader(text.splitlines()), [])
            if any(SENSITIVE_KEY.search(str(column)) for column in header):
                raise ApiError(403, 'artifact_download_rejected', 'Artifact failed the sensitive-content check.')
    except ApiError:
        raise
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise ApiError(403, 'artifact_download_rejected', 'Artifact validation failed.') from exc
    return path, mime, raw


def item(slot: Slot) -> dict[str, Any]:
    path = REPO_ROOT / slot.relative_path
    try:
        stat = path.lstat()
        available = path.is_file() and not path.is_symlink()
        size = stat.st_size if available else None
    except OSError:
        available = False
        size = None
    warning = None
    url = None
    if slot.mode == 'metadata_only':
        download_state = 'not_allowed'
    elif not available:
        download_state = 'missing'
    else:
        try:
            validated_download(slot)
            download_state = 'available'
            url = f'/api/r0/downloads/{slot.artifact_id}'
        except ApiError:
            download_state = 'rejected'
            warning = 'artifact_download_rejected'
    return {
        'legacy_artifact_id': slot.artifact_id, 'label': slot.label, 'kind': slot.kind,
        'format': Path(slot.relative_path).suffix.lower().lstrip('.') or 'unknown',
        'file_name': Path(slot.relative_path).name, 'available': available, 'size_bytes': size,
        'provenance': 'unknown', 'evidence_status': 'unverified', 'viewer_mode': slot.mode,
        'download_state': download_state, 'warning_code': warning, 'download_url': url,
        'summary_url': None, 'resource_capabilities': [],
    }


def get_slot(artifact_id: str) -> Slot:
    slot = BY_ID.get(artifact_id)
    if slot is None:
        raise ApiError(404, 'legacy_artifact_not_found', 'Unknown legacy artifact slot.')
    return slot
