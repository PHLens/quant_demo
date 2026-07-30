"""Verified, read-only materialized snapshot input for Lab-1."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.contracts import INPUT_WINDOW, VALIDATION_WINDOW

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = _PACKAGE_DIR / 'snapshots' / 'csi1000_trend_101_v1.manifest.json'

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
            'parents': list(self.manifest['parents']),
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
    csv_path = manifest_path.parent / manifest['file']
    actual_hash = _sha256(csv_path)
    if actual_hash != manifest.get('content_sha256'):
        raise SnapshotIntegrityError(
            f'snapshot hash mismatch: expected={manifest.get("content_sha256")} '
            f'actual={actual_hash}'
        )

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
            if row['index_close'] is None or row['index_close'] <= 0:
                raise SnapshotIntegrityError(f'invalid index close on {date}')
            if VALIDATION_WINDOW['start'] <= date <= VALIDATION_WINDOW['end']:
                for name in ('etf_open', 'etf_close'):
                    if row[name] is None or row[name] <= 0:
                        raise SnapshotIntegrityError(
                            f'missing validation execution price {name} on {date}'
                        )
            rows.append(row)

    if len(rows) != int(manifest.get('row_count', -1)):
        raise SnapshotIntegrityError(
            f'snapshot row count mismatch: expected={manifest.get("row_count")} '
            f'actual={len(rows)}'
        )
    if not rows or rows[0]['date'] != INPUT_WINDOW['start'] or rows[-1]['date'] != INPUT_WINDOW['end']:
        raise SnapshotIntegrityError('snapshot input window does not match the frozen contract')
    validation_count = sum(
        VALIDATION_WINDOW['start'] <= row['date'] <= VALIDATION_WINDOW['end']
        for row in rows
    )
    if validation_count != int(manifest.get('validation_row_count', -1)):
        raise SnapshotIntegrityError(
            'snapshot validation row count does not match manifest'
        )
    return MaterializedSnapshot(manifest=manifest, rows=tuple(rows))
