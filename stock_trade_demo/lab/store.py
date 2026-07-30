"""Small JSON artifact store isolated from all Research/live caches."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{7,127}$')


class ArtifactConflictError(RuntimeError):
    pass


class ArtifactNotFoundError(KeyError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


class JsonArtifactStore:
    KINDS = ('experiments', 'variants', 'runs', 'results')

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        for kind in self.KINDS:
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, artifact_id: str) -> Path:
        if kind not in self.KINDS:
            raise ValueError(f'unsupported artifact kind: {kind}')
        if not isinstance(artifact_id, str) or not _ID_RE.fullmatch(artifact_id):
            raise ValueError(f'invalid artifact id: {artifact_id!r}')
        return self.root / kind / f'{artifact_id}.json'

    def _atomic_write(self, path: Path, value: Any) -> None:
        payload = canonical_json(value)
        tmp = path.with_suffix(f'.tmp-{os.getpid()}-{threading.get_ident()}')
        try:
            with tmp.open('wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def create_immutable(self, kind: str, artifact_id: str, value: Any) -> bool:
        path = self._path(kind, artifact_id)
        with self._lock:
            if path.exists():
                existing = self.read(kind, artifact_id)
                if canonical_json(existing) != canonical_json(value):
                    raise ArtifactConflictError(
                        f'immutable artifact identity collision: {kind}/{artifact_id}'
                    )
                return False
            self._atomic_write(path, value)
            return True

    def write_run(self, run_id: str, value: Any) -> None:
        with self._lock:
            self._atomic_write(self._path('runs', run_id), value)

    def read(self, kind: str, artifact_id: str) -> dict[str, Any]:
        path = self._path(kind, artifact_id)
        try:
            with path.open('r', encoding='utf-8') as handle:
                value = json.load(handle)
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f'{kind}/{artifact_id}') from exc
        if not isinstance(value, dict):
            raise ArtifactConflictError(f'artifact is not a JSON object: {path}')
        return value

    def list(self, kind: str) -> list[dict[str, Any]]:
        if kind not in self.KINDS:
            raise ValueError(f'unsupported artifact kind: {kind}')
        values = []
        with self._lock:
            for path in sorted((self.root / kind).glob('*.json')):
                with path.open('r', encoding='utf-8') as handle:
                    value = json.load(handle)
                if isinstance(value, dict):
                    values.append(value)
        return values
