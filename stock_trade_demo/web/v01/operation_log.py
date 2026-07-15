"""Minimal append-only troubleshooting log for public-unsafe mutations."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any

from flask import current_app

from web.v01.contracts import canonical_json, utc_now


def append_request_result(*, action: str, scope: str | list[str] | None, source_ip: str, status: int, error_code: str | None) -> None:
    path = Path(current_app.config['R0_OPERATION_LOG_PATH'])
    payload: dict[str, Any] = {
        'time': utc_now(), 'action': action, 'scope': scope,
        'source_ip': source_ip, 'result': 'accepted' if status == 202 else ('success' if status < 400 else 'error'),
        'error_code': error_code,
    }
    raw = (canonical_json(payload) + '\n').encode('utf-8')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # This is explicitly a basic troubleshooting log, not an authorization
        # control or tamper-evident audit prerequisite.
        return
