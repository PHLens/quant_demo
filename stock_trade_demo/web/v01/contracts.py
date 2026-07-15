"""Shared closed-contract primitives for the v0.1 HTTP surface."""
from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import hmac
import json
import math
from typing import Any

from flask import current_app, jsonify


DEFAULT_LIMIT = 200
MAX_LIMIT = 1000


class ApiError(Exception):
    """Stable API error with a closed JSON body."""

    def __init__(self, status: int, code: str, message: str | None = None, **details: Any):
        super().__init__(message or code)
        self.status = status
        self.code = code
        self.message = message or code
        self.details = details

    def response(self):
        payload = {'error': self.code, 'message': self.message}
        payload.update(self.details)
        return jsonify(payload), self.status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'), sort_keys=True)


def digest_id(prefix: str, value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical_json(value).encode('utf-8')
    return f'{prefix}_{hashlib.sha256(raw).hexdigest()}'


def ensure_allowed_args(args: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(args.keys()) - allowed_set)
    if unknown:
        raise ApiError(400, 'invalid_params', 'Unsupported query parameters.', invalid_params=unknown)


def require_no_args(args: Mapping[str, Any]) -> None:
    ensure_allowed_args(args, ())


def parse_limit(raw: str | None) -> int:
    if raw is None:
        return DEFAULT_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, 'invalid_pagination', 'limit must be an integer.') from exc
    if value < 1 or value > MAX_LIMIT:
        raise ApiError(400, 'invalid_pagination', f'limit must be between 1 and {MAX_LIMIT}.')
    return value


def parse_iso_day(raw: str | None, *, field: str, nullable: bool = True) -> str | None:
    if raw is None or raw == '':
        if nullable:
            return None
        raise ApiError(400, 'invalid_date', f'{field} is required.', field=field)
    try:
        return date.fromisoformat(str(raw)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ApiError(400, 'invalid_date', f'{field} must be an ISO date.', field=field) from exc


def finite_number(value: Any, *, nullable: bool = True, field: str = 'value') -> float | int | None:
    if value is None or value == '':
        if nullable:
            return None
        raise ValueError(f'{field} is required')
    if isinstance(value, bool):
        raise ValueError(f'{field} must be numeric')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be finite')
    return number


def nullable_json_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar values without inventing zero values."""
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)):
        return value
    return value


def decimal_string(value: Any, step: Any) -> str:
    try:
        decimal = Decimal(str(value))
        quantum = Decimal(str(step))
    except (InvalidOperation, ValueError) as exc:
        raise ApiError(400, 'invalid_params', 'Invalid decimal parameter.') from exc
    if not decimal.is_finite() or not quantum.is_finite() or quantum <= 0:
        raise ApiError(400, 'invalid_params', 'Invalid decimal parameter.')
    snapped = (decimal / quantum).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * quantum
    places = max(0, -quantum.as_tuple().exponent)
    return format(snapped, f'.{places}f')


def _cursor_key() -> bytes:
    key = current_app.config.get('R0_CURSOR_KEY', 'quant-demo-v0.1-cursor-key')
    return key if isinstance(key, bytes) else str(key).encode('utf-8')


def encode_cursor(payload: Mapping[str, Any]) -> str:
    raw = canonical_json(payload).encode('utf-8')
    body = base64.urlsafe_b64encode(raw).rstrip(b'=')
    signature = hmac.new(_cursor_key(), body, hashlib.sha256).digest()
    return f'{body.decode("ascii")}.{base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")}'


def decode_cursor(token: str | None) -> dict[str, Any] | None:
    if token is None:
        return None
    try:
        body_text, signature_text = token.split('.', 1)
        body = body_text.encode('ascii')
        signature = base64.urlsafe_b64decode(signature_text + '=' * (-len(signature_text) % 4))
        expected = hmac.new(_cursor_key(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError('signature mismatch')
        raw = base64.urlsafe_b64decode(body_text + '=' * (-len(body_text) % 4))
        payload = json.loads(raw.decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('cursor payload is not an object')
        return payload
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, 'invalid_cursor', 'Cursor is invalid.') from exc


def paginate(
    items: Sequence[dict[str, Any]],
    *,
    limit: int,
    cursor: str | None,
    binding: Mapping[str, Any],
    sort_key,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Page an already canonically sorted collection with a bound cursor."""
    decoded = decode_cursor(cursor)
    start = 0
    if decoded is not None:
        if decoded.get('binding') != dict(binding):
            raise ApiError(400, 'invalid_cursor', 'Cursor belongs to a different resource or view.')
        last = decoded.get('last')
        found = False
        for index, item in enumerate(items):
            if list(sort_key(item)) == last:
                start = index + 1
                found = True
                break
        if not found:
            raise ApiError(400, 'invalid_cursor', 'Cursor position no longer exists.')
    page = list(items[start:start + limit])
    next_cursor = None
    if start + len(page) < len(items) and page:
        next_cursor = encode_cursor({'binding': dict(binding), 'last': list(sort_key(page[-1]))})
    return page, len(items), next_cursor


def bounded_envelope(resource: str, items: Sequence[dict[str, Any]], max_items: int, **extra: Any) -> dict[str, Any]:
    if len(items) > max_items:
        raise ApiError(500, 'bounded_collection_overflow', f'{resource} exceeds its hard limit.')
    payload = {'resource': resource, 'bounded': True, 'max_items': max_items, 'items': list(items), 'total': len(items)}
    payload.update(extra)
    return payload


def assert_closed_object(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or list(value.keys()) != list(keys):
        raise ValueError(f'{name} must contain exactly the declared keys in canonical order')
    return value
