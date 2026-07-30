"""Verified, read-only materialized snapshot input for Lab-1."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.contracts import INPUT_WINDOW, VALIDATION_WINDOW

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = _PACKAGE_DIR / 'snapshots' / 'csi1000_trend_101_v1.manifest.json'
_SNAPSHOT_ID_PREFIX = 'snapshot_csi1000_trend_101_v1_'
_SNAPSHOT_FILE = 'csi1000_trend_101_v1.csv'
_CALENDAR = 'CN_A_share_trading_days'
_TIMEZONE = 'Asia/Shanghai'
_MANIFEST_KEYS = {
    'snapshot_id',
    'file',
    'content_sha256',
    'row_count',
    'validation_row_count',
    'input_window',
    'validation_window',
    'as_of',
    'calendar',
    'timezone',
    'parents',
}
_EXPECTED_PARENTS = [
    {
        'artifact_id': 'index_data_csi1000_daily',
        'content_sha256': '3ff072679ed5c88b6cefa009931039a8d83b7a912dc3b06ca59714eaf6e71d47',
        'manifest_sha256': '951f481b2a440b96679b9dd682c18d65948351a291017a7b3bfa4a24e191461a',
        'produced_by': 'index_data.get_index_daily:csi1000',
        'source_rows': 2856,
        'source_written_at': '2026-07-16T13:30:17.655243+00:00',
    },
    {
        'artifact_id': 'timing_etf_csi1000_510980_qfq_daily',
        'content_sha256': '5cb1a27ad16220fc41f5d82d3013919db71dfc69c1c315b6b982e586e7dfe407',
        'manifest_sha256': '2b49a8b345398c0b38ca254b087f0060fab59211013faef4eee6e7fd450eae27',
        'produced_by': 'index_data.get_timing_etf_daily:akshare:510980:qfq',
        'source_rows': 632,
        'source_written_at': '2026-07-14T11:26:27.670281+00:00',
    },
]

_COLUMNS = (
    'date',
    'index_open',
    'index_high',
    'index_low',
    'index_close',
    'index_volume',
    'etf_open',
    'etf_high',
    'etf_low',
    'etf_close',
    'etf_volume',
    'etf_amount',
)


class SnapshotIntegrityError(RuntimeError):
    """The packaged snapshot no longer matches its immutable manifest."""


@dataclass(frozen=True)
class MaterializedSnapshot:
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def public_identity(self) -> dict[str, Any]:
        return {
            'snapshot_id': self.manifest['snapshot_id'],
            'content_sha256': self.manifest['content_sha256'],
            'input_window': dict(self.manifest['input_window']),
            'validation_window': dict(self.manifest['validation_window']),
            'row_count': self.manifest['row_count'],
            'validation_row_count': self.manifest['validation_row_count'],
            'as_of': self.manifest['as_of'],
            'calendar': self.manifest['calendar'],
            'timezone': self.manifest['timezone'],
            'parents': copy.deepcopy(self.manifest['parents']),
        }

    def worker_payload(self) -> dict[str, Any]:
        return {
            'identity': self.public_identity(),
            'rows': [dict(row) for row in self.rows],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_materialized_snapshot(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> MaterializedSnapshot:
    """Load and verify the packaged CSV without consulting any cache/loader."""
    manifest_path = Path(manifest_path)
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise SnapshotIntegrityError(
            f'snapshot manifest fields mismatch: expected={sorted(_MANIFEST_KEYS)} '
            f'actual={sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__}'
        )
    if manifest['file'] != _SNAPSHOT_FILE:
        raise SnapshotIntegrityError(f'snapshot file must be {_SNAPSHOT_FILE}')
    csv_path = manifest_path.parent / manifest['file']
    actual_hash = _sha256(csv_path)
    if actual_hash != manifest.get('content_sha256'):
        raise SnapshotIntegrityError(
            f'snapshot hash mismatch: expected={manifest.get("content_sha256")} '
            f'actual={actual_hash}'
        )
    expected_snapshot_id = f'{_SNAPSHOT_ID_PREFIX}{actual_hash[:16]}'
    if manifest['snapshot_id'] != expected_snapshot_id:
        raise SnapshotIntegrityError(
            f'snapshot_id mismatch: expected={expected_snapshot_id} '
            f'actual={manifest["snapshot_id"]}'
        )
    if manifest['input_window'] != INPUT_WINDOW:
        raise SnapshotIntegrityError('manifest input_window does not match frozen contract')
    if manifest['validation_window'] != VALIDATION_WINDOW:
        raise SnapshotIntegrityError('manifest validation_window does not match frozen contract')
    if manifest['calendar'] != _CALENDAR or manifest['timezone'] != _TIMEZONE:
        raise SnapshotIntegrityError('manifest calendar/timezone does not match frozen contract')
    if manifest['parents'] != _EXPECTED_PARENTS:
        raise SnapshotIntegrityError('manifest parent artifact identities do not match frozen contract')
    for field in ('row_count', 'validation_row_count'):
        if isinstance(manifest[field], bool) or not isinstance(manifest[field], int):
            raise SnapshotIntegrityError(f'manifest {field} must be an integer')

    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise SnapshotIntegrityError(
                f'snapshot columns mismatch: expected={_COLUMNS} actual={reader.fieldnames}'
            )
        previous_date = ''
        for raw in reader:
            date = raw['date']
            if date <= previous_date:
                raise SnapshotIntegrityError('snapshot dates must be unique and ascending')
            previous_date = date
            row: dict[str, Any] = {'date': date}
            for name in _COLUMNS[1:]:
                text = raw[name].strip()
                row[name] = float(text) if text else None
            for name in ('index_open', 'index_high', 'index_low', 'index_close'):
                if row[name] is None or row[name] <= 0:
                    raise SnapshotIntegrityError(f'invalid {name} on {date}')
            if VALIDATION_WINDOW['start'] <= date <= VALIDATION_WINDOW['end']:
                for name in ('etf_open', 'etf_high', 'etf_low', 'etf_close'):
                    if row[name] is None or row[name] <= 0:
                        raise SnapshotIntegrityError(
                            f'missing validation execution price {name} on {date}'
                        )
            rows.append(row)

    if len(rows) != manifest['row_count']:
        raise SnapshotIntegrityError(
            f'snapshot row count mismatch: expected={manifest.get("row_count")} '
            f'actual={len(rows)}'
        )
    if not rows or rows[0]['date'] != INPUT_WINDOW['start'] or rows[-1]['date'] != INPUT_WINDOW['end']:
        raise SnapshotIntegrityError('snapshot input window does not match the frozen contract')
    if manifest['as_of'] != rows[-1]['date']:
        raise SnapshotIntegrityError('manifest as_of does not match the final snapshot row')
    validation_count = sum(
        VALIDATION_WINDOW['start'] <= row['date'] <= VALIDATION_WINDOW['end']
        for row in rows
    )
    if validation_count != manifest['validation_row_count']:
        raise SnapshotIntegrityError(
            'snapshot validation row count does not match manifest'
        )
    return MaterializedSnapshot(manifest=manifest, rows=tuple(rows))
