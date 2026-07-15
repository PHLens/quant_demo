"""Atomic, idempotent v0.1 Manual Records ledger."""
from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import UUID, uuid4

from flask import current_app

from services import live_trades
from web.v01.catalog import manual_capability
from web.v01.contracts import ApiError, canonical_json, decode_cursor, encode_cursor, parse_iso_day, parse_limit


HEADER = (
    'schema_version', 'row_type', 'origin', 'record_id', 'idempotency_key',
    'request_hash', 'date', 'strategy', 'capital', 'signal_target', 'exec_price',
    'shares', 'actual_position', 'notes', 'created_at', 'deleted_at',
)
LEGACY_HEADER = (
    'record_id', 'date', 'strategy', 'signal_target', 'actual_position',
    'exec_price', 'capital', 'notes', 'created_at', 'shares',
)
HASH_RE = re.compile(r'^[0-9a-f]{64}$')
LEGACY_ID_RE = re.compile(r'^l_[0-9a-f]{64}$')
UTC_MICRO_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$')
_LOCK = threading.RLock()


@dataclass
class Ledger:
    rows: list[dict[str, str]]
    raw: bytes
    legacy: bool

    @property
    def version(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


def path() -> Path:
    return Path(current_app.config.get('R0_MANUAL_LEDGER_PATH') or live_trades.LIVE_TRADES_FILE)


def _uuid(value: str, *, field: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ApiError(400, 'invalid_params', f'{field} must be a UUID.', field=field) from exc


def _decimal(value: Any, *, field: str, nullable: bool) -> Decimal | None:
    if value is None or value == '':
        if nullable:
            return None
        raise ApiError(400, 'invalid_params', f'{field} is required.', field=field)
    if isinstance(value, bool):
        raise ApiError(400, 'invalid_params', f'{field} must be numeric.', field=field)
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ApiError(400, 'invalid_params', f'{field} must be numeric.', field=field) from exc
    if not result.is_finite():
        raise ApiError(400, 'invalid_params', f'{field} must be finite.', field=field)
    return result


def _decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ''
    normalized = value.normalize()
    text = format(normalized, 'f')
    return '0' if text in {'-0', ''} else text


def require_strategy(raw: str | None):
    if raw is None or not str(raw).strip():
        raise ApiError(400, 'manual_strategy_required', 'A non-empty Manual strategy is required.')
    strategy = str(raw).strip()
    capability = manual_capability(strategy)
    if capability is None:
        raise ApiError(400, 'unsupported_manual_strategy', 'Manual Records supports only the fixed A/US catalog.', strategy=strategy)
    return strategy, capability


def _legacy_id(row: dict[str, str], duplicate_ordinal: int) -> str:
    canonical_row = [row.get(key, '') for key in LEGACY_HEADER if key != 'record_id']
    raw = row.get('strategy', '') + canonical_json(canonical_row) + str(duplicate_ordinal)
    return 'l_' + hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _normalize_legacy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    occurrences: dict[str, int] = {}
    normalized = []
    for row in rows:
        canonical_row = canonical_json([row.get(key, '') for key in LEGACY_HEADER if key != 'record_id'])
        duplicate = occurrences.get(canonical_row, 0)
        occurrences[canonical_row] = duplicate + 1
        item = {key: '' for key in HEADER}
        item.update({
            'schema_version': '1', 'row_type': 'record', 'origin': 'legacy',
            'record_id': _legacy_id(row, duplicate), 'date': row.get('date', ''),
            'strategy': row.get('strategy', ''), 'capital': row.get('capital', ''),
            'signal_target': row.get('signal_target', ''), 'exec_price': row.get('exec_price', ''),
            'shares': row.get('shares', ''), 'actual_position': row.get('actual_position', ''),
            'notes': row.get('notes', ''), 'created_at': row.get('created_at', ''),
        })
        normalized.append(item)
    return normalized


def _parse(raw: bytes) -> Ledger:
    if not raw:
        raise ApiError(500, 'manual_ledger_corrupt', 'An existing Manual ledger is empty.')
    try:
        text = raw.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(text, newline=''))
        header = tuple(reader.fieldnames or ())
        if header not in {HEADER, LEGACY_HEADER}:
            raise ApiError(500, 'manual_ledger_corrupt', 'Manual ledger header is not an approved exact schema.')
        raw_rows = list(reader)
        if any(None in row for row in raw_rows):
            raise ApiError(500, 'manual_ledger_corrupt', 'Manual ledger contains extra columns.')
    except ApiError:
        raise
    except (UnicodeError, csv.Error) as exc:
        raise ApiError(500, 'manual_ledger_corrupt', 'Manual ledger cannot be decoded.') from exc
    legacy = header == LEGACY_HEADER
    rows = _normalize_legacy(raw_rows) if legacy else [{key: row.get(key, '') for key in HEADER} for row in raw_rows]
    _validate_rows(rows)
    return Ledger(rows, raw, legacy)


def _validate_rows(rows: list[dict[str, str]]) -> None:
    ids: set[str] = set()
    keys: set[str] = set()
    for row in rows:
        if row['schema_version'] != '1' or row['row_type'] not in {'record', 'idempotency_tombstone'} or row['origin'] not in {'v0.1', 'legacy'}:
            raise ApiError(500, 'manual_ledger_corrupt', 'Manual ledger row metadata is invalid.')
        record_id = row['record_id']
        if not record_id or record_id in ids:
            raise ApiError(500, 'manual_ledger_corrupt', 'Manual ledger record_id is missing or duplicated.')
        ids.add(record_id)
        if row['origin'] == 'v0.1':
            try:
                if str(UUID(record_id)) != record_id:
                    raise ValueError
            except ValueError as exc:
                raise ApiError(500, 'manual_ledger_corrupt', 'v0.1 record_id is invalid.') from exc
        elif not LEGACY_ID_RE.fullmatch(record_id):
            raise ApiError(500, 'manual_ledger_corrupt', 'legacy record_id is invalid.')
        key = row['idempotency_key']
        request_hash = row['request_hash']
        if key:
            try:
                if str(UUID(key)) != key or key in keys:
                    raise ValueError
            except ValueError as exc:
                raise ApiError(500, 'manual_ledger_corrupt', 'idempotency_key is invalid or duplicated.') from exc
            keys.add(key)
        if request_hash and not HASH_RE.fullmatch(request_hash):
            raise ApiError(500, 'manual_ledger_corrupt', 'request_hash is invalid.')
        if row['origin'] == 'v0.1' and (not key or not request_hash):
            raise ApiError(500, 'manual_ledger_corrupt', 'v0.1 rows require idempotency metadata.')
        if row['row_type'] == 'record':
            if (not row['date'] or not row['strategy'] or not row['capital'] or
                    (row['origin'] == 'v0.1' and not row['created_at']) or row['deleted_at']):
                raise ApiError(500, 'manual_ledger_corrupt', 'Record row is missing required payload.')
            try:
                if date.fromisoformat(row['date']).isoformat() != row['date']:
                    raise ValueError
                if row['created_at']:
                    created_at = row['created_at'].replace('Z', '+00:00')
                    if datetime.fromisoformat(created_at).tzinfo is None:
                        raise ValueError
                capital = Decimal(row['capital'])
                if not capital.is_finite() or capital <= 0:
                    raise ValueError
                for field in ('signal_target', 'exec_price', 'shares', 'actual_position'):
                    if row[field] and not Decimal(row[field]).is_finite():
                        raise ValueError
            except (ValueError, InvalidOperation) as exc:
                raise ApiError(500, 'manual_ledger_corrupt', 'Record row payload is invalid.') from exc
            if row['origin'] == 'v0.1':
                try:
                    if manual_capability(row['strategy']) is None or not UTC_MICRO_RE.fullmatch(row['created_at']):
                        raise ValueError
                    exec_price = Decimal(row['exec_price']) if row['exec_price'] else None
                    shares = Decimal(row['shares']) if row['shares'] else None
                    actual = Decimal(row['actual_position']) if row['actual_position'] else None
                    if (exec_price is None) != (shares is None):
                        raise ValueError
                    if exec_price is not None:
                        derived = exec_price * shares / capital
                        if exec_price <= 0 or shares < 0 or exec_price * shares > capital:
                            raise ValueError
                        if actual is None or abs(actual - derived) > Decimal('0.0001'):
                            raise ValueError
                    elif actual is None:
                        raise ValueError
                    if actual < 0 or actual > 1 or len(row['notes']) > 500:
                        raise ValueError
                    stripped = row['notes'].lstrip(' ')
                    if stripped and stripped[0] in {'=', '+', '-', '@', '\t', '\r'}:
                        raise ValueError
                    canonical = {
                        'actual_position': row['actual_position'],
                        'capital': row['capital'], 'date': row['date'],
                        'exec_price': row['exec_price'], 'notes': row['notes'],
                        'shares': row['shares'], 'signal_target': row['signal_target'],
                        'strategy': row['strategy'],
                    }
                    if hashlib.sha256(canonical_json(canonical).encode('utf-8')).hexdigest() != request_hash:
                        raise ValueError
                except (ValueError, InvalidOperation, ZeroDivisionError) as exc:
                    raise ApiError(500, 'manual_ledger_corrupt', 'v0.1 record invariants are invalid.') from exc
        else:
            payload_fields = ('date', 'strategy', 'capital', 'signal_target', 'exec_price', 'shares', 'actual_position', 'notes')
            if any(row[field] for field in payload_fields) or not row['created_at'] or not row['deleted_at']:
                raise ApiError(500, 'manual_ledger_corrupt', 'Tombstone contains payload or lacks deleted_at.')
            try:
                created_at = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
                deleted_at = datetime.fromisoformat(row['deleted_at'].replace('Z', '+00:00'))
                if created_at.tzinfo is None or deleted_at.tzinfo is None:
                    raise ValueError
                if row['origin'] == 'v0.1' and (
                        not UTC_MICRO_RE.fullmatch(row['created_at'])
                        or not UTC_MICRO_RE.fullmatch(row['deleted_at'])):
                    raise ValueError
                if row['origin'] == 'legacy' and (key or request_hash):
                    raise ValueError
            except ValueError as exc:
                raise ApiError(500, 'manual_ledger_corrupt', 'Tombstone timestamps are invalid.') from exc


@contextmanager
def _exclusive_file_lock():
    """Serialize writers across processes without locking the replaced inode."""
    ledger_path = path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f'.{ledger_path.name}.lock')
    with lock_path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load() -> Ledger:
    ledger_path = path()
    with _LOCK:
        try:
            with ledger_path.open('rb') as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
                except OSError:
                    pass
                raw = stream.read()
        except FileNotFoundError:
            return Ledger([], b'', False)
    return _parse(raw)


def active(ledger: Ledger, strategy: str) -> list[dict[str, str]]:
    result = [row for row in ledger.rows if row['row_type'] == 'record' and row['strategy'] == strategy]
    # Payload parsing occurs only after row_type + strategy filtering.
    for row in result:
        try:
            datetime.fromisoformat(row['date'])
            Decimal(row['capital'])
            for key in ('signal_target', 'exec_price', 'shares', 'actual_position'):
                if row[key]:
                    value = Decimal(row[key])
                    if not value.is_finite():
                        raise ValueError
        except (ValueError, InvalidOperation) as exc:
            raise ApiError(500, 'manual_ledger_corrupt', 'Active Manual row contains invalid payload.') from exc
    return result


def record(row: dict[str, str]) -> dict[str, Any]:
    return {
        'record_id': row['record_id'], 'origin': row['origin'], 'date': row['date'],
        'strategy': row['strategy'], 'capital': float(Decimal(row['capital'])),
        'signal_target': float(Decimal(row['signal_target'])) if row['signal_target'] else None,
        'exec_price': float(Decimal(row['exec_price'])) if row['exec_price'] else None,
        'shares': float(Decimal(row['shares'])) if row['shares'] else None,
        'actual_position': float(Decimal(row['actual_position'])) if row['actual_position'] else None,
        'notes': row['notes'] or None, 'created_at': row['created_at'] or None,
    }


def list_records(strategy: str, *, cursor: str | None, limit_raw: str | None) -> dict[str, Any]:
    limit = parse_limit(limit_raw)
    ledger = load()
    rows = active(ledger, strategy)
    rows.sort(key=lambda row: (row['date'], row['created_at'], row['record_id']), reverse=True)
    decoded = decode_cursor(cursor)
    start = 0
    if decoded is not None:
        binding = decoded.get('binding') or {}
        if binding.get('strategy') != strategy or binding.get('resource') != 'records' or binding.get('direction') != 'DESC':
            raise ApiError(400, 'invalid_cursor', 'Cursor belongs to a different Manual collection.')
        if binding.get('ledger_version') != ledger.version:
            raise ApiError(409, 'manual_ledger_changed', 'The Manual ledger changed between pages.', ledger_version=ledger.version)
        last = decoded.get('last')
        for index, row in enumerate(rows):
            if [row['date'], row['created_at'], row['record_id']] == last:
                start = index + 1
                break
        else:
            raise ApiError(400, 'invalid_cursor', 'Cursor position does not exist.')
    page_rows = rows[start:start + limit]
    next_cursor = None
    if start + len(page_rows) < len(rows) and page_rows:
        next_cursor = encode_cursor({
            'binding': {'strategy': strategy, 'ledger_version': ledger.version, 'resource': 'records', 'direction': 'DESC'},
            'last': [page_rows[-1]['date'], page_rows[-1]['created_at'], page_rows[-1]['record_id']],
        })
    return {'strategy': strategy, 'ledger_version': ledger.version, 'resource': 'records', 'items': [record(row) for row in page_rows], 'total': len(rows), 'next_cursor': next_cursor}


def normalize_body(payload: Any) -> tuple[dict[str, str], str]:
    if not isinstance(payload, dict):
        raise ApiError(400, 'invalid_params', 'JSON body must be an object.')
    allowed = {'date', 'strategy', 'capital', 'signal_target', 'exec_price', 'shares', 'actual_position', 'notes'}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ApiError(400, 'invalid_params', 'Unknown Manual fields.', invalid_params=unknown)
    day = parse_iso_day(payload.get('date'), field='date', nullable=False)
    strategy, _ = require_strategy(payload.get('strategy'))
    capital = _decimal(payload.get('capital', 50000), field='capital', nullable=False)
    if capital <= 0:
        raise ApiError(400, 'invalid_params', 'capital must be positive.', field='capital')
    signal_target = _decimal(payload.get('signal_target'), field='signal_target', nullable=True)
    exec_price = _decimal(payload.get('exec_price'), field='exec_price', nullable=True)
    shares = _decimal(payload.get('shares'), field='shares', nullable=True)
    actual = _decimal(payload.get('actual_position'), field='actual_position', nullable=True)
    if (exec_price is None) != (shares is None):
        raise ApiError(400, 'invalid_params', 'exec_price and shares must be supplied together.')
    if exec_price is not None:
        if exec_price <= 0 or shares < 0 or exec_price * shares > capital:
            raise ApiError(400, 'invalid_params', 'Execution pair is outside the allowed capital range.')
        derived = exec_price * shares / capital
        if actual is not None and abs(actual - derived) > Decimal('0.0001'):
            raise ApiError(400, 'position_conflict', 'actual_position conflicts with exec_price * shares / capital.')
        actual = derived
    elif actual is None:
        raise ApiError(400, 'invalid_params', 'actual_position is required without an execution pair.')
    if actual is None or actual < 0 or actual > 1:
        raise ApiError(400, 'invalid_params', 'actual_position must be within [0,1].')
    raw_notes = payload.get('notes')
    if raw_notes is not None and not isinstance(raw_notes, str):
        raise ApiError(400, 'invalid_params', 'notes must be plain text.', field='notes')
    notes = '' if raw_notes is None else raw_notes
    if len(notes) > 500:
        raise ApiError(400, 'invalid_params', 'notes exceeds 500 characters.', field='notes')
    stripped = notes.lstrip(' ')
    if stripped and stripped[0] in {'=', '+', '-', '@', '\t', '\r'}:
        raise ApiError(400, 'unsafe_note_prefix', 'notes starts with a spreadsheet-formula prefix.')
    canonical = {
        'actual_position': _decimal_text(actual), 'capital': _decimal_text(capital),
        'date': day, 'exec_price': _decimal_text(exec_price), 'notes': notes,
        'shares': _decimal_text(shares), 'signal_target': _decimal_text(signal_target),
        'strategy': strategy,
    }
    request_hash = hashlib.sha256(canonical_json(canonical).encode('utf-8')).hexdigest()
    return canonical, request_hash


def _serialize(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, '') for key in HEADER})
    return stream.getvalue().encode('utf-8')


def _materialized_rows(ledger: Ledger) -> list[dict[str, str]]:
    rows = [dict(row) for row in ledger.rows]
    if not ledger.legacy:
        return rows
    # Legacy rows without created_at sort/reconcile as the start of their day.
    # Freeze that same interpretation when the first v0.1 mutation migrates the
    # file so a later delete can retain a valid, payload-free tombstone.
    for row in rows:
        if row['origin'] == 'legacy' and row['row_type'] == 'record' and not row['created_at']:
            row['created_at'] = f'{row["date"]}T00:00:00Z'
    return rows


def _atomic_write(rows: list[dict[str, str]]) -> None:
    ledger_path = path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    staging = ledger_path.with_name(f'.{ledger_path.name}.{uuid4()}.tmp')
    raw = _serialize(rows)
    try:
        with staging.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, ledger_path)
        directory_fd = os.open(ledger_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def create(payload: Any, idempotency_key: str | None) -> tuple[dict[str, Any], int]:
    key = _uuid(idempotency_key or '', field='Idempotency-Key')
    body, request_hash = normalize_body(payload)
    with _LOCK:
        with _exclusive_file_lock():
            ledger = load()
            rows = _materialized_rows(ledger)
            for row in rows:
                if row['idempotency_key'] != key:
                    continue
                if row['request_hash'] != request_hash:
                    raise ApiError(409, 'idempotency_conflict', 'Idempotency-Key was used with a different canonical body.')
                if row['row_type'] == 'idempotency_tombstone':
                    raise ApiError(410, 'idempotency_record_deleted', 'The idempotent record was deleted.')
                result = record(row)
                result['replayed'] = True
                return result, 200
            now = datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
            row = {field: '' for field in HEADER}
            row.update({
                'schema_version': '1', 'row_type': 'record', 'origin': 'v0.1',
                'record_id': str(uuid4()), 'idempotency_key': key, 'request_hash': request_hash,
                'date': body['date'], 'strategy': body['strategy'], 'capital': body['capital'],
                'signal_target': body['signal_target'], 'exec_price': body['exec_price'],
                'shares': body['shares'], 'actual_position': body['actual_position'],
                'notes': body['notes'], 'created_at': now,
            })
            rows.append(row)
            _validate_rows(rows)
            _atomic_write(rows)
            return record(row), 201


def delete(record_id: str) -> None:
    with _LOCK:
        with _exclusive_file_lock():
            ledger = load()
            rows = _materialized_rows(ledger)
            target = next((row for row in rows if row['record_id'] == record_id and row['row_type'] == 'record'), None)
            if target is None:
                raise ApiError(404, 'record_not_found', 'Manual record does not exist.')
            now = datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
            for field in ('date', 'strategy', 'capital', 'signal_target', 'exec_price', 'shares', 'actual_position', 'notes'):
                target[field] = ''
            target['row_type'] = 'idempotency_tombstone'
            target['deleted_at'] = now
            _validate_rows(rows)
            _atomic_write(rows)
