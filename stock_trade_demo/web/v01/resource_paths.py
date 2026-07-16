"""Resolve the server-owned data root independently from the code release."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from flask import current_app, has_app_context

from web.v01.contracts import ApiError


CODE_REPO_ROOT = Path(__file__).resolve().parents[3]
STOCK_RESOURCE_NAMES = (
    'stock_data.csv',
    'stock_data.csv.meta.json',
    'stock_data.parquet',
    'stock_data.parquet.meta.json',
)


def configured_resource_root() -> Path:
    """Return the configured persistent root, with a local-code fallback."""
    configured = current_app.config.get('R0_RESOURCE_ROOT') if has_app_context() else None
    return Path(configured).expanduser() if configured else CODE_REPO_ROOT


def stock_resource_dir() -> Path:
    return configured_resource_root() / 'stock_trade_demo'


def resource_root_diagnostic(scopes: Iterable[str] = ()) -> dict[str, object]:
    """Inspect the mutation data-root contract without creating or changing it."""
    configured = current_app.config.get('R0_RESOURCE_ROOT') if has_app_context() else None
    testing = bool(current_app.config.get('TESTING')) if has_app_context() else False
    try:
        root = configured_resource_root()
    except (TypeError, ValueError, OSError, RuntimeError):
        return {
            'ready': False,
            'code': 'resource_root_unavailable',
            'message': 'The configured persistent data root is invalid.',
        }
    if configured is None and testing:
        return {'ready': True, 'code': None, 'message': 'Test-local data-root fallback is ready.'}
    if configured is None and not testing:
        return {
            'ready': False,
            'code': 'resource_root_not_configured',
            'message': 'A persistent R0_RESOURCE_ROOT is required for data mutations.',
        }
    if not root.is_absolute():
        return {
            'ready': False,
            'code': 'resource_root_not_persistent',
            'message': 'The persistent data root must be configured as an absolute path.',
        }
    if root.is_symlink():
        return {
            'ready': False,
            'code': 'resource_root_unavailable',
            'message': 'The configured persistent data root must not be a symlink.',
        }
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return {
            'ready': False,
            'code': 'resource_root_unavailable',
            'message': 'The configured persistent data root does not exist or cannot be resolved.',
        }
    if not resolved.is_dir():
        return {
            'ready': False,
            'code': 'resource_root_unavailable',
            'message': 'The configured persistent data root is not a directory.',
        }
    if not testing and resolved == CODE_REPO_ROOT.resolve():
        return {
            'ready': False,
            'code': 'resource_root_not_persistent',
            'message': 'The data root must be independent from the active code release.',
        }
    if 'stock' in set(scopes):
        project = resolved / 'stock_trade_demo'
        invalid = []
        if not project.is_dir() or not os.access(project, os.R_OK | os.W_OK | os.X_OK):
            invalid.extend(STOCK_RESOURCE_NAMES)
        for name in STOCK_RESOURCE_NAMES:
            path = project / name
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    invalid.append(name)
                    continue
                with path.open('rb') as stream:
                    if not stream.read(1):
                        invalid.append(name)
            except OSError:
                invalid.append(name)
        if invalid:
            return {
                'ready': False,
                'code': 'stock_resource_unavailable',
                'message': 'The persistent stock dataset or its integrity sidecars are missing or unreadable.',
            }
    return {'ready': True, 'code': None, 'message': 'Persistent data root is ready.'}


def require_resource_root(scopes: Iterable[str]) -> Path:
    """Fail before mutation admission when the persistent root is unusable."""
    diagnostic = resource_root_diagnostic(scopes)
    if diagnostic['ready'] is not True:
        raise ApiError(
            503,
            str(diagnostic['code']),
            str(diagnostic['message']),
        )
    return configured_resource_root().resolve(strict=True)
