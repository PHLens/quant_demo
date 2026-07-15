"""Fixed action plans, one logical mutation gate, and crash-fence admission."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
from uuid import uuid4

from flask import Flask, current_app

from web import state
from web.v01.catalog import SCOPE_ORDER, find_variant, get_strategy, recovery_plan_id
from web.v01.contracts import ApiError, canonical_json, utc_now
from web.v01.snapshot_store import (
    PUBLISHED_SIGNAL_KEYS, inspect_target, load_snapshot, pointer_value,
    publish_entry, target_fence, write_target_fence,
)


STEP_LIMIT = 64
MARKER_KEYS = {
    'schema_version', 'operation_id', 'kind', 'accepted_boot_id', 'plan_id',
    'plan_hash', 'marker_revision', 'marker_digest', 'write_progress',
    'affected_targets',
}
PUBLIC_UNSAFE_WARNING = (
    'Public unsafe mode: any internet user or bot can view capital, position and notes; '
    'create or irreversibly delete Manual records without identity; repeatedly pull data '
    'or rebuild caches to consume upstream, CPU and disk; and request service restart. '
    'POST, confirmation, 409 conflicts and the process lock prevent mistakes/concurrency only; '
    'they are not access control or a recovery guarantee. Partial non-atomic upstream updates '
    'can leave mixed old/new cache inputs.'
)


@dataclass
class GateOwner:
    kind: str
    operation_id: str | None
    request_id: str | None


class MutationGate:
    def __init__(self):
        self._lock = threading.RLock()
        self._owner: GateOwner | None = None

    def acquire(self, owner: GateOwner) -> None:
        with self._lock:
            if self._owner is not None:
                raise ApiError(
                    409, 'operation_in_progress', 'Another mutation owns the logical gate.',
                    existing_kind=self._owner.kind,
                    existing_operation_id=self._owner.operation_id,
                    existing_request_id=self._owner.request_id,
                )
            self._owner = owner

    def release(self, owner: GateOwner) -> None:
        with self._lock:
            if self._owner == owner:
                self._owner = None

    @property
    def owner(self) -> GateOwner | None:
        with self._lock:
            return self._owner


@dataclass
class Operation:
    operation_id: str
    kind: str
    target: dict[str, Any] | None
    original_request: dict[str, Any]
    plan_id: str
    plan_hash: str
    resolved_plan: dict[str, Any]
    status: str = 'pending'
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error_code: str | None = None
    retry_of: str | None = None
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            'operation_id': self.operation_id, 'status_url': f'/api/r0/actions/{self.operation_id}',
            'kind': self.kind, 'target': deepcopy(self.target), 'original_request': deepcopy(self.original_request),
            'plan_id': self.plan_id, 'plan_hash': self.plan_hash, 'resolved_plan': deepcopy(self.resolved_plan),
            'status': self.status, 'steps': deepcopy(self.steps), 'result': deepcopy(self.result),
            'error_code': self.error_code, 'retry_of': self.retry_of,
            'created_at': self.created_at, 'finished_at': self.finished_at,
        }


class ActionRuntime:
    def __init__(self, app: Flask):
        self.app = app
        self.gate = MutationGate()
        self.operations: dict[str, Operation] = {}
        self.overlay_targets: dict[tuple[str, str, str], tuple[str, str]] = {}
        self.ephemeral_targets: dict[tuple[str, str, str], Any] = {}
        self.boot_id = str(uuid4())
        self.started_at = utc_now()
        self.ready = True
        self.startup_error: str | None = None
        self._lock = threading.RLock()
        self._inspect_marker()

    @property
    def marker_path(self) -> Path:
        configured = self.app.config.get('R0_ACTION_MARKER_PATH')
        return Path(configured or (Path(self.app.instance_path) / 'r0-active-operation.json'))

    def _inspect_marker(self) -> None:
        marker = self.marker_path
        if not marker.exists():
            return
        try:
            payload = self._read_marker()
            self._reconcile_marker(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            self.ready = False
            self.startup_error = 'interrupted_marker_corrupt'

    @staticmethod
    def _marker_digest(payload: dict[str, Any]) -> str:
        return hashlib.sha256(canonical_json({key: value for key, value in payload.items() if key != 'marker_digest'}).encode('utf-8')).hexdigest()

    def _read_marker(self) -> dict[str, Any]:
        payload = json.loads(self.marker_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or set(payload) != MARKER_KEYS:
            raise ValueError('marker schema mismatch')
        if payload.get('schema_version') != 1 or payload.get('kind') not in {'data-update', 'cache-recover'}:
            raise ValueError('marker metadata mismatch')
        if not isinstance(payload.get('marker_revision'), int) or payload['marker_revision'] < 0:
            raise ValueError('marker revision is invalid')
        if payload.get('marker_digest') != self._marker_digest(payload):
            raise ValueError('marker digest mismatch')
        targets = payload.get('affected_targets')
        writes = payload.get('write_progress')
        if not isinstance(targets, list) or not isinstance(writes, list) or len(targets) > 256 or len(writes) > STEP_LIMIT:
            raise ValueError('marker collections are invalid')
        target_ids = []
        for item in targets:
            expected = {
                'source_id', 'strategy_id', 'variant_id', 'role', 'persistence',
                'pre_target_state', 'pre_pointer_digest', 'commit_state', 'candidate',
            }
            if not isinstance(item, dict) or set(item) != expected:
                raise ValueError('marker target schema mismatch')
            get_strategy(item['source_id'], item['strategy_id'])
            find_variant(item['source_id'], item['strategy_id'], item['variant_id'])
            if item['role'] not in {'rebuilt', 'collateral'} or item['persistence'] not in {'persistent', 'boot_ephemeral'}:
                raise ValueError('marker target metadata mismatch')
            if item['commit_state'] not in {'none', 'intent', 'durable'}:
                raise ValueError('marker target commit state mismatch')
            if item['candidate'] is not None:
                if not isinstance(item['candidate'], dict) or set(item['candidate']) != {
                        'generation_id', 'pointer_value', 'pointer_digest', 'manifest_digest', 'snapshot_id'}:
                    raise ValueError('marker candidate schema mismatch')
                candidate = item['candidate']
                pointer = candidate['pointer_value']
                if not isinstance(pointer, dict) or set(pointer) != {
                        'source_id', 'strategy_id', 'variant_id', 'generation_id',
                        'payload_digest', 'generated_at'}:
                    raise ValueError('marker candidate pointer schema mismatch')
                if (pointer['source_id'], pointer['strategy_id'], pointer['variant_id']) != (
                        item['source_id'], item['strategy_id'], item['variant_id']):
                    raise ValueError('marker candidate target mismatch')
                if (candidate['generation_id'] != pointer['generation_id']
                        or candidate['manifest_digest'] != pointer['payload_digest']
                        or candidate['generation_id'] != candidate['manifest_digest']
                        or candidate['pointer_digest'] != hashlib.sha256(
                            canonical_json(pointer).encode('utf-8')).hexdigest()
                        or not isinstance(candidate['snapshot_id'], str)
                        or not candidate['snapshot_id'].startswith('s_')):
                    raise ValueError('marker candidate digest mismatch')
            target_ids.append((item['source_id'], item['strategy_id'], item['variant_id']))
        if target_ids != sorted(target_ids) or len(target_ids) != len(set(target_ids)):
            raise ValueError('marker targets are not unique canonical order')
        for item in writes:
            if not isinstance(item, dict) or set(item) != {'resource_id', 'pre_fingerprint', 'state', 'durable_fingerprint'}:
                raise ValueError('marker write schema mismatch')
            if item['state'] not in {'not_started', 'write_intent', 'write_durable'}:
                raise ValueError('marker write state mismatch')
            if (item['state'] == 'write_durable') != (item['durable_fingerprint'] is not None):
                raise ValueError('marker durable fingerprint mismatch')
        return payload

    def _reconcile_marker(self, payload: dict[str, Any]) -> None:
        """Converge an interrupted action without resuming its worker/status."""
        try:
            with self.app.app_context():
                for item in payload['affected_targets']:
                    if item['persistence'] == 'boot_ephemeral':
                        # Ephemeral targets are absent in every new ActionRuntime.
                        continue
                    spec = get_strategy(item['source_id'], item['strategy_id'])
                    variant = find_variant(item['source_id'], item['strategy_id'], item['variant_id'])
                    _, current_digest = pointer_value(spec, variant)
                    pre = item['pre_target_state']
                    restored = False
                    writes_unchanged = True
                    for write in payload['write_progress']:
                        if not _write_relevant_to_target(write['resource_id'], item):
                            continue
                        if write['state'] == 'not_started':
                            continue
                        observed = _resource_fingerprint(write['resource_id'])
                        if write['pre_fingerprint'] is None or observed != write['pre_fingerprint']:
                            writes_unchanged = False
                            break
                    if current_digest == item['pre_pointer_digest'] and writes_unchanged:
                        if pre.get('readable'):
                            snapshot = load_snapshot(spec, variant)
                            restored = snapshot is not None and snapshot.snapshot_id == pre.get('snapshot_id')
                        else:
                            restored = True
                    candidate = item['candidate']
                    if not restored and candidate is not None and current_digest == candidate['pointer_digest']:
                        snapshot = load_snapshot(spec, variant)
                        restored = snapshot is not None and snapshot.snapshot_id == candidate['snapshot_id']
                    if restored:
                        continue
                    write_target_fence(spec, variant, payload['operation_id'], pre, reason='interrupted_operation')
            self._remove_marker()
            self.ready = True
            self.startup_error = None
        except Exception:
            self.ready = False
            self.startup_error = 'interrupted_reconciliation_failed'

    def _update_marker(self, operation_id: str, update: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            payload = self._read_marker()
            if payload['operation_id'] != operation_id:
                raise OSError('active marker owner changed')
            update(payload)
            payload['marker_revision'] += 1
            payload['marker_digest'] = self._marker_digest(payload)
            self._atomic_json(self.marker_path, payload)
            return payload

    def ensure_ready(self) -> None:
        if not self.ready:
            raise ApiError(503, self.startup_error or 'interrupted_reconciliation_failed', 'Startup crash-fence reconciliation has not completed.')

    def _atomic_json(self, target: Path, payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f'.{target.name}.{uuid4()}.tmp')
        raw = canonical_json(payload).encode('utf-8')
        try:
            with staging.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                staging.unlink()
            except FileNotFoundError:
                pass

    def _persist_marker(self, operation: Operation, affected_targets: list[dict[str, str]], pre_target_state: list[dict[str, Any]]) -> None:
        pre_by_target = {
            (item['source_id'], item['strategy_id'], item['variant_id']): item
            for item in pre_target_state
        }
        marker_targets = []
        for target in sorted(affected_targets, key=lambda item: (item['source_id'], item['strategy_id'], item['variant_id'])):
            pre = pre_by_target[(target['source_id'], target['strategy_id'], target['variant_id'])]
            marker_targets.append({
                'source_id': target['source_id'], 'strategy_id': target['strategy_id'],
                'variant_id': target['variant_id'], 'role': target.get('role', 'rebuilt'),
                'persistence': 'boot_ephemeral' if target['source_id'] in {'hk_timing', 'commodity'} else 'persistent',
                'pre_target_state': pre, 'pre_pointer_digest': pre.get('pointer_digest'),
                'commit_state': 'none', 'candidate': None,
            })
        write_set = operation.resolved_plan.get('write_set') or []
        write_progress = [
            {
                'resource_id': resource_id,
                'pre_fingerprint': _resource_fingerprint(resource_id),
                'state': 'not_started', 'durable_fingerprint': None,
            }
            for resource_id in write_set
        ]
        payload = {
            'schema_version': 1, 'operation_id': operation.operation_id, 'kind': operation.kind,
            'accepted_boot_id': self.boot_id, 'plan_id': operation.plan_id,
            'plan_hash': operation.plan_hash, 'marker_revision': 0,
            'marker_digest': None, 'write_progress': write_progress,
            'affected_targets': marker_targets,
        }
        payload['marker_digest'] = self._marker_digest(payload)
        try:
            self._atomic_json(self.marker_path, payload)
        except OSError as exc:
            raise ApiError(500, 'action_crash_fence_failed', 'Crash-fence could not be made durable before worker admission.') from exc

    def _remove_marker(self) -> None:
        try:
            self.marker_path.unlink()
            directory = os.open(self.marker_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileNotFoundError:
            return

    def operation(self, operation_id: str) -> Operation:
        operation = self.operations.get(operation_id)
        if operation is None:
            raise ApiError(404, 'operation_not_found', 'Operation does not exist in this boot.')
        return operation

    def overlay(self, source_id: str, strategy_id: str, variant_id: str) -> tuple[str, str] | None:
        return self.overlay_targets.get((source_id, strategy_id, variant_id))

    def ephemeral_entry(self, source_id: str, strategy_id: str, variant_id: str):
        return self.ephemeral_targets.get((source_id, strategy_id, variant_id))

    @contextmanager
    def manual_mutation(self, request_id: str):
        self.ensure_ready()
        owner = GateOwner('manual', None, request_id)
        self.gate.acquire(owner)
        try:
            yield
        finally:
            self.gate.release(owner)


def runtime() -> ActionRuntime:
    return current_app.extensions['r0_action_runtime']


def init_runtime(app: Flask) -> ActionRuntime:
    value = ActionRuntime(app)
    app.extensions['r0_action_runtime'] = value
    return value


def _check_scope(scope: str) -> dict[str, Any]:
    # Do not call the legacy "check" helpers: several of them use ensure/get
    # loaders that fetch and write on a cache miss.  v0.1 check reads only
    # literal local summary/sidecar files.
    root = Path(__file__).resolve().parents[3]
    files = {
        'index': (root / 'data/_idx_summary.csv', root / 'data/_etf_summary.csv'),
        'aux': (root / 'data/_fred_summary.csv', root / 'strategy/risk_signals.json'),
        'stock': (root / 'stock_trade_demo/stock_data.csv.meta.json', root / 'stock_trade_demo/stock_data.parquet.meta.json'),
        'factor': (root / 'strategy/backtest_sector_heat.csv',),
    }[scope]
    dates = []
    for file_path in files:
        try:
            if file_path.suffix == '.json' or file_path.name.endswith('.meta.json'):
                payload = json.loads(file_path.read_text(encoding='utf-8'))
                for key in ('data_max_date', 'max_date', 'data_as_of', 'latest_date', 'written_at_iso', 'generated_at'):
                    value = payload.get(key) if isinstance(payload, dict) else None
                    if isinstance(value, str) and len(value) >= 10:
                        dates.append(value[:10])
                        break
            else:
                stat = file_path.stat()
                dates.append(datetime.fromtimestamp(stat.st_mtime, timezone.utc).date().isoformat())
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            continue
    current = max(dates) if dates else None
    expected = datetime.now(timezone.utc).date().isoformat()
    needs = current < expected if current is not None else None
    return {
        'scope': scope, 'current_local_date': current, 'latest_expected_date': expected,
        'needs_update': bool(needs) if needs is not None else None,
        'reason': 'local_date_behind' if needs else 'current' if needs is False else 'local_metadata_missing',
        'unknown': needs is None,
    }


def data_check(scope: str | None) -> dict[str, Any]:
    if scope is not None and scope not in SCOPE_ORDER:
        raise ApiError(400, 'invalid_params', 'scope must be index, aux, stock or factor.')
    rows = [_check_scope(item) for item in SCOPE_ORDER if scope is None or item == scope]
    return {'checked_at': utc_now(), 'scopes': rows}


def _normalized_scopes(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, 'invalid_params', 'scopes must be a non-empty array.')
    if any(not isinstance(item, str) or item not in SCOPE_ORDER for item in raw):
        raise ApiError(400, 'invalid_params', 'scopes contains an unsupported value.')
    return [scope for scope in SCOPE_ORDER if scope in set(raw)]


def _stock_parquet_ready() -> bool:
    configured = current_app.config.get('R0_STOCK_PARQUET_PATH')
    path = Path(configured) if configured else Path(__file__).resolve().parents[3] / 'stock_trade_demo/stock_data.parquet'
    try:
        from pyarrow.parquet import read_metadata
        metadata = read_metadata(path)
        return metadata.num_rows > 0 and metadata.num_columns > 0
    except (ImportError, OSError, ValueError):
        return False


def _path_fingerprint(paths: tuple[Path, ...]) -> str | None:
    """Cheap durable fingerprint over atomic sidecars and file identities.

    The large stock artifacts are intentionally not read end-to-end on an HTTP
    preflight. Their atomic-write sidecars carry the content sample/checkpoint;
    file size and nanosecond mtime make replacement visible as well.
    """
    records = []
    for path in paths:
        try:
            if path.is_dir():
                children = sorted(
                    child for child in path.rglob('*')
                    if child.is_file() and (child.suffix == '.json' or child.name.endswith('.meta.json'))
                )
                child_fingerprint = _path_fingerprint(tuple(children))
                records.append({'path': str(path), 'directory': True, 'fingerprint': child_fingerprint})
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


def _resource_fingerprint(resource_id: str) -> str | None:
    configured = current_app.config.get('R0_RESOURCE_FINGERPRINTS')
    if callable(configured):
        value = configured(resource_id)
        return str(value) if value is not None else None
    if isinstance(configured, dict) and resource_id in configured:
        value = configured[resource_id]
        return str(value) if value is not None else None
    if resource_id.startswith('artifact:') and resource_id.count('/') == 2:
        source_id, strategy_id, variant_id = resource_id[len('artifact:'):].split('/', 2)
        try:
            _, digest = pointer_value(get_strategy(source_id, strategy_id), find_variant(source_id, strategy_id, variant_id))
            return digest
        except (KeyError, TypeError, ValueError):
            return None
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


def _write_relevant_to_target(resource_id: str, target: dict[str, Any]) -> bool:
    if resource_id.startswith('artifact:') and resource_id.count('/') == 2:
        return resource_id == (
            f'artifact:{target["source_id"]}/{target["strategy_id"]}/{target["variant_id"]}'
        )
    scope = None
    if resource_id in {'dataset:index', 'dataset:index-daily', 'dataset:etf-daily'}:
        scope = 'index'
    elif resource_id in {'dataset:aux', 'dataset:fred', 'dataset:a-share-macro', 'artifact:risk-signals'}:
        scope = 'aux'
    elif resource_id in {'dataset:stock', 'dataset:stock-csv', 'dataset:stock-parquet'}:
        scope = 'stock'
    elif resource_id in {'dataset:factor', 'artifact:sector-heat'}:
        scope = 'factor'
    if scope is None:
        return True
    spec = get_strategy(target['source_id'], target['strategy_id'])
    return scope in spec.recovery_scopes


def _target_input_fingerprint(target: dict[str, Any]) -> str | None:
    spec = get_strategy(target['source_id'], target['strategy_id'])
    values = {
        scope: _resource_fingerprint(f'dataset:{scope}')
        for scope in spec.recovery_scopes
    }
    if not values or all(value is None for value in values.values()):
        return None
    return hashlib.sha256(canonical_json(values).encode('utf-8')).hexdigest()


def update_plan(scopes: list[str], force: bool) -> dict[str, Any]:
    requested = [scope for scope in SCOPE_ORDER if scope in set(scopes)]
    resolved = list(requested)
    prerequisites = []
    if 'factor' in requested and 'stock' not in resolved and not _stock_parquet_ready():
        resolved.append('stock')
        resolved.sort(key=SCOPE_ORDER.index)
        prerequisites.append({'scope': 'stock', 'required_by': 'factor'})
    checks = {item['scope']: item for item in data_check(None)['scopes']}
    scope_writes = {
        'index': ['dataset:index-daily', 'dataset:etf-daily'],
        'aux': ['dataset:fred', 'dataset:a-share-macro', 'artifact:risk-signals'],
        'stock': ['dataset:stock-csv', 'dataset:stock-parquet'],
        'factor': ['artifact:sector-heat'],
    }
    steps = []
    for ordinal, scope in enumerate(resolved):
        explicit = scope in requested
        check = checks[scope]
        if explicit and force:
            decision = 'forced'
        elif check['needs_update'] is False:
            decision = 'skipped_current'
        else:
            decision = 'needed'
        steps.append({
            'topological_ordinal': ordinal, 'step_id': scope, 'scope': scope,
            'decision': decision, 'explicit': explicit, 'forced': bool(explicit and force),
            'blocked_by': ['stock'] if scope == 'factor' and 'stock' in resolved else [],
            'origin': 'explicit' if explicit else 'prerequisite',
            'depends_on': ['stock'] if scope == 'factor' and 'stock' in resolved else [],
            'effective_mode': decision, 'write_set': scope_writes[scope],
            'affected_targets': _affected_update_targets([scope]), 'input_fingerprints': {},
            'eta_seconds': {'index': 180, 'aux': 120, 'stock': 300, 'factor': 120}[scope],
        })
        steps[-1]['input_fingerprints'] = {
            resource_id: _resource_fingerprint(resource_id)
            for resource_id in steps[-1]['write_set']
        }
    write_set = []
    for step in steps:
        for resource_id in step['write_set']:
            if resource_id not in write_set:
                write_set.append(resource_id)
    plan = {
        'requested_scopes': requested, 'force': force, 'resolved_scopes': resolved,
        'prerequisites': prerequisites, 'steps': steps, 'write_set': write_set,
    }
    plan_hash = hashlib.sha256(canonical_json(plan).encode('utf-8')).hexdigest()
    plan['plan_id'] = f'data-update-v0.1-{plan_hash[:16]}'
    plan['plan_hash'] = plan_hash
    return plan


def preview_update_plan(scopes_text: str | None, force_text: str | None) -> dict[str, Any]:
    if scopes_text is None or not scopes_text.strip():
        raise ApiError(400, 'invalid_params', 'scopes is required.')
    scopes = _normalized_scopes([value.strip() for value in scopes_text.split(',') if value.strip()])
    if force_text is None:
        force = False
    elif force_text == 'true':
        force = True
    elif force_text == 'false':
        force = False
    else:
        raise ApiError(400, 'invalid_params', 'force must be true or false.')
    return update_plan(scopes, force)


def _step_status(step: dict[str, Any]) -> dict[str, Any]:
    status = 'skipped_current' if step['decision'] == 'skipped_current' else 'pending'
    return {
        **step, 'status': status, 'progress': 100 if status == 'skipped_current' else 0,
        'message': 'Local input is current.' if status == 'skipped_current' else 'Pending',
        'started_at': None, 'finished_at': None, 'error_code': None,
    }


def _affected_update_targets(resolved_scopes: list[str]) -> list[dict[str, str]]:
    from web.v01.catalog import STRATEGY_SPECS, variants_for
    affected = []
    scope_set = set(resolved_scopes)
    for spec in STRATEGY_SPECS:
        if not scope_set.intersection(spec.recovery_scopes):
            continue
        for variant in variants_for(spec):
            affected.append({
                'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
                'variant_id': variant.variant_id, 'role': 'rebuilt',
            })
    return affected


def _pre_states(affected: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for target in affected:
        spec = get_strategy(target['source_id'], target['strategy_id'])
        variant = find_variant(target['source_id'], target['strategy_id'], target['variant_id'])
        target_state, snapshot = inspect_target(spec, variant)
        pointer, pointer_digest = pointer_value(spec, variant)
        fence = target_fence(spec, variant)
        rows.append({
            'source_id': target['source_id'], 'strategy_id': target['strategy_id'],
            'variant_id': target['variant_id'], **target_state.as_fields(),
            'freshness_fields': {'generated_at': target_state.generated_at},
            'blocker_code': target_state.blocker_code,
            'degraded_metadata': target_state.degradation,
            'snapshot_id': snapshot.snapshot_id if snapshot else None,
            'pointer': pointer, 'pointer_digest': pointer_digest,
            'input_fingerprint': _target_input_fingerprint(target),
            'invalidated_by_operation_id': fence.get('invalidated_by_operation_id') if isinstance(fence, dict) else None,
        })
    return rows


def _runner(scope: str) -> Callable[[], Any]:
    configured = current_app.config.get('R0_ACTION_RUNNERS') or {}
    if scope in configured:
        return configured[scope]
    return {
        'index': state._run_index_data_update,
        'aux': state._run_aux_data_update,
        'stock': state._run_data_update,
        'factor': state._run_factor_update,
    }[scope]


def _run_scope(scope: str) -> None:
    configured = current_app.config.get('R0_ACTION_RUNNERS') or {}
    runner = _runner(scope)
    runner()
    if scope in configured:
        return
    status = {
        'index': state._INDEX_UPDATE_STATUS,
        'aux': state._AUX_UPDATE_STATUS,
        'stock': state._UPDATE_DATA_STATUS,
        'factor': state._FACTOR_UPDATE_STATUS,
    }[scope]
    if status.get('stage') != 'done' or status.get('error'):
        raise RuntimeError(f'{scope} runner reported terminal error')


def _built_entry(spec, variant):
    if spec.source_id == 'selection':
        return state.BACKTEST_CACHE.get(spec.strategy_id)
    if spec.source_id == 'a_share_timing':
        return state.TIMING_CACHE.get(spec.strategy_id)
    if spec.source_id == 'us_timing':
        return state.US_TIMING_CACHE.get(spec.strategy_id)
    if spec.source_id == 'hk_timing':
        return state.HK_CACHE.get(spec.strategy_id)
    if spec.source_id == 'commodity':
        return state.COMMODITY_CACHE.get(spec.strategy_id)
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'sector_heat':
        value = state._SECTOR_HEAT_CACHE.get('data')
        return value if value is not None else state._load_sector_heat()
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor':
        value = state.FACTOR_BACKTEST_CACHE.get(f'top_k={int(variant.canonical_params["top_k"])}')
        if value is not None:
            items = deepcopy(value)
            for item in items:
                for key in ('annual_return', 'max_drawdown'):
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.endswith('%'):
                        item[key] = float(raw[:-1]) / 100.0
            return {
                'version': 'v0.1', 'saved_at': utc_now(),
                'top_k': int(variant.canonical_params['top_k']), 'items': items,
            }
    return None


def _default_recovery_builder(target: dict[str, str]) -> None:
    """Run exactly one server-cataloged target after its fixed pull steps."""
    spec = get_strategy(target['source_id'], target['strategy_id'])
    variant = find_variant(target['source_id'], target['strategy_id'], target['variant_id'])
    params = dict(variant.canonical_params)
    if spec.source_id == 'selection':
        result, evaluation = state.run_backtest_fresh(spec.strategy_id, **params)
        state.BACKTEST_CACHE[spec.strategy_id] = (result, evaluation)
        return
    if spec.source_id == 'a_share_timing':
        result, _, _ = state.run_timing_backtest_fresh(spec.strategy_id, **params)
        state.TIMING_CACHE[spec.strategy_id] = result
        return
    if spec.source_id == 'us_timing':
        state.ensure_us_timing_panel_loaded(force_reload=True)
        strategy = state.build_us_timing_strategy(spec.strategy_id, **params)
        _, benchmark = state._get_benchmark_series(strategy.get_index_id())
        signal_frame = strategy.run(state.US_TIMING_PANEL.copy())
        state.US_TIMING_CACHE[spec.strategy_id] = state.run_timing_backtest(signal_frame, strategy, benchmark_returns=benchmark)
        return
    if spec.source_id == 'hk_timing':
        state.ensure_hk_panel_loaded(force_reload=True)
        strategy = state.build_hk_strategy(spec.strategy_id, **params)
        _, benchmark = state._get_benchmark_series(strategy.get_index_id())
        signal_frame = strategy.run(state.HK_PANEL.copy())
        state.HK_CACHE[spec.strategy_id] = state.run_timing_backtest(signal_frame, strategy, benchmark_returns=benchmark)
        return
    if spec.source_id == 'commodity':
        state.ensure_commodity_panel_loaded(force_reload=True)
        strategy = state.build_commodity_strategy(spec.strategy_id, **params)
        _, benchmark = state._get_benchmark_series(strategy.get_index_id())
        signal_frame = strategy.run(state.COMMODITY_PANEL.copy())
        state.COMMODITY_CACHE[spec.strategy_id] = state.run_timing_backtest(signal_frame, strategy, benchmark_returns=benchmark)
        return
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'sector_heat':
        state._SECTOR_HEAT_CACHE.update(mtime=0, data=None)
        if state._load_sector_heat() is None:
            raise RuntimeError('sector heat output is missing')
        return
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor':
        top_k = int(variant.canonical_params['top_k'])
        state.FACTOR_BACKTEST_CACHE[f'top_k={top_k}'] = state._run_single_factor_backtest(top_k=top_k)
        return
    raise RuntimeError('fixed recovery builder is unavailable')


def _freeze_entry(spec, entry):
    frame = entry[0] if spec.source_id == 'selection' and isinstance(entry, tuple) else entry
    if not hasattr(frame, 'attrs'):
        return entry
    frozen = frame.copy(deep=False)
    frozen.attrs = dict(frame.attrs)
    frozen.attrs['r0_generated_at'] = utc_now()
    target_state = frozen.attrs.get('r0_target_state')
    expected_state_keys = {
        'cache_state', 'readable', 'freshness_state', 'freshness_reason',
        'degradation',
    }
    if not isinstance(target_state, dict) or set(target_state) != expected_state_keys or target_state.get('readable') is not True:
        target_state = {
            'cache_state': 'ready', 'readable': True,
            'freshness_state': 'current', 'freshness_reason': None,
            'degradation': None,
        }
    else:
        target_state = deepcopy(target_state)
    frozen.attrs['r0_target_state'] = target_state
    if 'published_current_signal' in spec.capabilities:
        existing = frozen.attrs.get('published_current_signal')
        if isinstance(existing, dict) and set(existing) == set(PUBLISHED_SIGNAL_KEYS) and existing.get('strategy_id') == spec.strategy_id:
            raw = existing
        else:
            if spec.source_id == 'a_share_timing':
                strategy = state.build_timing_strategy(spec.strategy_id)
            elif spec.source_id == 'us_timing':
                strategy = state.build_us_timing_strategy(spec.strategy_id)
            elif spec.source_id == 'hk_timing':
                strategy = state.build_hk_strategy(spec.strategy_id)
            else:
                strategy = state.build_commodity_strategy(spec.strategy_id)
            raw = state._build_latest_signal(spec.strategy_id, strategy, frozen, state._load_best_profile(spec.strategy_id))
        signal = {key: raw.get(key) for key in PUBLISHED_SIGNAL_KEYS}
        signal['strategy_id'] = spec.strategy_id
        signal['data_stale_warning'] = (
            target_state['freshness_reason']['code']
            if target_state['freshness_state'] == 'stale' else None
        )
        signal['degraded_reason'] = (
            target_state['degradation']['code']
            if target_state['degradation'] is not None else None
        )
        frozen.attrs['published_current_signal'] = signal
    return (frozen, entry[1] if len(entry) > 1 else None) if spec.source_id == 'selection' and isinstance(entry, tuple) else frozen


def _mark_step_writes(operation: Operation, step: dict[str, Any], state_name: str) -> None:
    resources = set(step.get('write_set') or [])
    if not resources:
        return

    def update(payload):
        for item in payload['write_progress']:
            if item['resource_id'] not in resources:
                continue
            item['state'] = state_name
            item['durable_fingerprint'] = _resource_fingerprint(item['resource_id']) if state_name == 'write_durable' else None
            if state_name == 'write_durable' and item['durable_fingerprint'] is None:
                raise OSError(f'durable resource fingerprint unavailable: {item["resource_id"]}')

    runtime()._update_marker(operation.operation_id, update)


def _publish_affected(
    operation: Operation,
    *,
    only_target: dict[str, str] | None = None,
    blocked_scopes: set[str] | None = None,
) -> list[dict[str, Any]]:
    targets = [only_target] if only_target is not None else operation.resolved_plan.get('affected_targets')
    if targets is None:
        targets = _affected_update_targets(operation.resolved_plan.get('resolved_scopes', []))
    results = []
    for target in targets:
        spec = get_strategy(target['source_id'], target['strategy_id'])
        variant = find_variant(target['source_id'], target['strategy_id'], target['variant_id'])
        if blocked_scopes and set(spec.recovery_scopes).intersection(blocked_scopes):
            results.append({**target, 'published': False, 'error_code': 'target_dependency_failed'})
            continue
        entry = _built_entry(spec, variant)
        if entry is None:
            results.append({**target, 'published': False, 'error_code': 'target_output_missing'})
            continue
        try:
            frozen = _freeze_entry(spec, entry)
            if spec.source_id in {'hk_timing', 'commodity'}:
                value = runtime()
                value.ephemeral_targets[(spec.source_id, spec.strategy_id, variant.variant_id)] = frozen
                snapshot = load_snapshot(spec, variant)
                if snapshot is None:
                    raise ValueError('ephemeral target did not reopen')
                results.append({
                    **target, 'published': True, 'generation_id': None,
                    'snapshot_id': snapshot.snapshot_id, 'ephemeral': True,
                })
                continue

            def marker_target(payload):
                return next(item for item in payload['affected_targets'] if (
                    item['source_id'], item['strategy_id'], item['variant_id']
                ) == (spec.source_id, spec.strategy_id, variant.variant_id))

            def before_commit(candidate):
                runtime()._update_marker(operation.operation_id, lambda payload: marker_target(payload).update(
                    candidate=candidate, commit_state='intent'))

            def after_commit(candidate):
                runtime()._update_marker(operation.operation_id, lambda payload: marker_target(payload).update(
                    candidate=candidate, commit_state='durable'))

            generation_id = publish_entry(
                spec, variant, frozen,
                before_pointer_commit=before_commit,
                after_pointer_commit=after_commit,
            )
            snapshot = load_snapshot(spec, variant)
            if snapshot is None:
                raise ValueError('committed target did not reopen')
            results.append({**target, 'published': True, 'generation_id': generation_id, 'snapshot_id': snapshot.snapshot_id})
        except Exception:
            results.append({**target, 'published': False, 'error_code': 'target_validation_failed'})
    return results


def _reconcile_unpublished_targets(operation: Operation, publications: list[dict[str, Any]]) -> bool:
    """Preserve exact pre-state only when pointer and relevant inputs stayed unchanged."""
    published = {
        (item['source_id'], item['strategy_id'], item['variant_id'])
        for item in publications if item.get('published')
    }
    try:
        marker = runtime()._read_marker()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return True
    pre_by_target = {
        (item['source_id'], item['strategy_id'], item['variant_id']): item
        for item in operation.resolved_plan.get('pre_target_state', [])
    }
    failed = False
    for target in marker['affected_targets']:
        key = (target['source_id'], target['strategy_id'], target['variant_id'])
        if key in published:
            continue
        spec = get_strategy(target['source_id'], target['strategy_id'])
        variant = find_variant(target['source_id'], target['strategy_id'], target['variant_id'])
        _, pointer_digest = pointer_value(spec, variant)
        unchanged = pointer_digest == target['pre_pointer_digest']
        for write in marker['write_progress']:
            if not _write_relevant_to_target(write['resource_id'], target):
                continue
            if write['state'] == 'not_started':
                continue
            observed = _resource_fingerprint(write['resource_id'])
            if write['pre_fingerprint'] is None or observed != write['pre_fingerprint']:
                unchanged = False
                break
        if unchanged:
            continue
        try:
            write_target_fence(
                spec, variant, operation.operation_id, pre_by_target.get(key, {}),
                reason='operation_input_changed',
            )
        except OSError:
            failed = True
    return failed


def _finish_action(
    value: ActionRuntime,
    operation: Operation,
    owner: GateOwner,
    *,
    terminal_status: str,
    error_code: str | None,
    stabilization_failed: bool,
) -> None:
    if stabilization_failed:
        operation.status = 'running'
        operation.error_code = 'target_reconciliation_failed'
        value.ready = False
        value.startup_error = 'interrupted_reconciliation_failed'
        return
    try:
        value._remove_marker()
    except OSError:
        operation.status = 'running'
        operation.error_code = 'marker_cleanup_failed'
        value.ready = False
        value.startup_error = 'interrupted_reconciliation_failed'
        return
    for key, overlay in list(value.overlay_targets.items()):
        if overlay[0] == operation.operation_id:
            value.overlay_targets.pop(key, None)
    operation.status = terminal_status
    operation.error_code = error_code
    operation.finished_at = utc_now()
    value.gate.release(owner)


def _run_update(operation_id: str) -> None:
    value = runtime()
    operation = value.operation(operation_id)
    owner = GateOwner('data-update', operation_id, None)
    operation.status = 'running'
    any_error = False
    stabilization_failed = False
    for step in operation.steps:
        if step['status'] == 'skipped_current':
            continue
        blocked = [prior['step_id'] for prior in operation.steps if prior['status'] == 'error' and prior['step_id'] in step.get('blocked_by', [])]
        if blocked:
            step.update(status='skipped_dependency', blocked_by=blocked, progress=0, message='Dependency failed.', finished_at=utc_now(), error_code='skipped_dependency')
            any_error = True
            continue
        step.update(status='running', started_at=utc_now(), progress=1, message='Running fixed scope update.')
        try:
            observed_inputs = {
                resource_id: _resource_fingerprint(resource_id)
                for resource_id in step.get('input_fingerprints', {})
            }
            if observed_inputs != step.get('input_fingerprints', {}):
                step.update(
                    status='error', progress=100, message='Frozen action input changed before execution.',
                    finished_at=utc_now(), error_code='input_changed',
                )
                any_error = True
                continue
            _mark_step_writes(operation, step, 'write_intent')
            _run_scope(step['scope'])
            _mark_step_writes(operation, step, 'write_durable')
            step.update(
                status='done', progress=100, message='Fixed scope update completed.',
                finished_at=utc_now(), output_fingerprints={
                    resource_id: _resource_fingerprint(resource_id)
                    for resource_id in step.get('write_set', [])
                },
            )
        except Exception:
            step.update(status='error', progress=100, message='Fixed scope update failed.', finished_at=utc_now(), error_code='scope_update_failed')
            any_error = True
    failed_scopes = {step['scope'] for step in operation.steps if step['status'] in {'error', 'skipped_dependency'}}
    publications = _publish_affected(operation, blocked_scopes=failed_scopes)
    operation.result = {'targets': publications}
    if any(not item['published'] for item in publications):
        any_error = True
    stabilization_failed = _reconcile_unpublished_targets(operation, publications)
    completed = any(step['status'] in {'done', 'skipped_current'} for step in operation.steps) or any(item['published'] for item in publications)
    terminal = 'partial' if any_error and completed else ('error' if any_error else 'done')
    error_code = 'update_partial' if terminal == 'partial' else ('update_failed' if terminal == 'error' else None)
    _finish_action(value, operation, owner, terminal_status=terminal, error_code=error_code, stabilization_failed=stabilization_failed)


def _admit(operation: Operation, affected: list[dict[str, str]], worker: Callable[[str], None]) -> Operation:
    value = runtime()
    value.ensure_ready()
    owner = GateOwner(operation.kind, operation.operation_id, None)
    value.gate.acquire(owner)
    release_on_error = True
    try:
        pre_states = operation.resolved_plan.get('pre_target_state') or _pre_states(affected)
        value._persist_marker(operation, affected, pre_states)
        for target in affected:
            value.overlay_targets[(target['source_id'], target['strategy_id'], target['variant_id'])] = (operation.operation_id, 'updating' if operation.kind == 'data-update' else 'recovering')
        value.operations[operation.operation_id] = operation
        thread_factory = current_app.config.get('R0_THREAD_FACTORY') or threading.Thread
        try:
            thread = thread_factory(target=_worker_entry, args=(current_app._get_current_object(), worker, operation.operation_id), daemon=True)
            thread.start()
        except Exception as exc:
            value.operations.pop(operation.operation_id, None)
            for target in affected:
                value.overlay_targets.pop((target['source_id'], target['strategy_id'], target['variant_id']), None)
            try:
                value._remove_marker()
            except OSError:
                value.ready = False
                value.startup_error = 'interrupted_reconciliation_failed'
                release_on_error = False
                raise ApiError(500, 'worker_start_cleanup_failed', 'Worker start failed and crash-fence cleanup did not stabilize.') from exc
            raise ApiError(500, 'worker_start_failed', 'Worker could not start; no operation was accepted.') from exc
        return operation
    except Exception:
        if release_on_error:
            value.gate.release(owner)
        raise


def _worker_entry(app: Flask, worker: Callable[[str], None], operation_id: str) -> None:
    with app.app_context():
        try:
            worker(operation_id)
        except Exception:
            value = runtime()
            operation = value.operations.get(operation_id)
            if operation is None:
                return
            try:
                marker = value._read_marker()
                value._reconcile_marker(marker)
            except Exception:
                operation.status = 'running'
                operation.error_code = 'worker_reconciliation_failed'
                value.ready = False
                value.startup_error = 'interrupted_reconciliation_failed'
                return
            for key, overlay in list(value.overlay_targets.items()):
                if overlay[0] == operation_id:
                    value.overlay_targets.pop(key, None)
            operation.status = 'error'
            operation.error_code = 'worker_failed'
            operation.finished_at = utc_now()
            value.gate.release(GateOwner(operation.kind, operation_id, None))


def start_update(payload: Any, *, retry_of: Operation | None = None) -> Operation:
    if not isinstance(payload, dict) or set(payload) != {'scopes', 'force'} or not isinstance(payload.get('force'), bool):
        raise ApiError(400, 'invalid_params', 'Body must be exactly {scopes:[...], force:boolean}.')
    scopes = _normalized_scopes(payload['scopes'])
    force = payload['force']
    plan = update_plan(scopes, force)
    execution_scopes = {
        step['scope'] for step in plan['steps']
        if step['decision'] != 'skipped_current'
    }
    reused_outputs = []
    invalidated_outputs = []
    if retry_of is not None:
        retry_steps = {
            step['step_id'] for step in retry_of.steps
            if step['status'] in {'error', 'skipped_dependency'}
        }
        # Include only missing ancestors of failed descendants. A successful
        # ancestor is reused unless current preflight no longer says current.
        for step in plan['steps']:
            if step['step_id'] in retry_steps:
                step['decision'] = 'forced' if step['explicit'] and force else 'needed'
                step['effective_mode'] = step['decision']
                continue
            previous_step = next((item for item in retry_of.steps if item['step_id'] == step['step_id']), None)
            if previous_step and previous_step['status'] in {'done', 'skipped_current'}:
                expected = previous_step.get('output_fingerprints') or previous_step.get('input_fingerprints') or {}
                observed = {
                    resource_id: _resource_fingerprint(resource_id)
                    for resource_id in previous_step.get('write_set') or []
                }
                if expected and all(
                        expected.get(resource_id) is not None
                        and expected.get(resource_id) == observed.get(resource_id)
                        for resource_id in previous_step.get('write_set') or []):
                    step['decision'] = 'skipped_current'
                    step['effective_mode'] = 'skipped_current'
                    reused_outputs.extend(previous_step.get('write_set') or [])
                else:
                    step['decision'] = 'forced' if step['explicit'] and force else 'needed'
                    step['effective_mode'] = step['decision']
                    invalidated_outputs.extend(previous_step.get('write_set') or [])
        execution_scopes = {step['scope'] for step in plan['steps'] if step['decision'] != 'skipped_current'}
    plan['reused_outputs'] = sorted(set(reused_outputs))
    plan['invalidated_outputs'] = sorted(set(invalidated_outputs))
    plan['write_set'] = list(dict.fromkeys(
        resource_id
        for step in plan['steps'] if step['decision'] != 'skipped_current'
        for resource_id in step['write_set']
    ))
    operation_id = str(uuid4())
    affected = _affected_update_targets([scope for scope in SCOPE_ORDER if scope in execution_scopes])
    plan['affected_targets'] = affected
    plan['pre_target_state'] = _pre_states(affected)
    plan_hash = hashlib.sha256(canonical_json({key: value for key, value in plan.items() if key not in {'plan_id', 'plan_hash'}}).encode('utf-8')).hexdigest()
    plan['plan_hash'] = plan_hash
    plan['plan_id'] = f'data-update-v0.1-{plan_hash[:16]}'
    operation = Operation(
        operation_id, 'data-update', None,
        {'requested_scopes': scopes, 'force': force}, plan['plan_id'], plan['plan_hash'], plan,
        steps=[_step_status(step) for step in plan['steps']], retry_of=retry_of.operation_id if retry_of else None,
    )
    return _admit(operation, affected, _run_update)


def _recovery_plan(source_id: str, strategy_id: str, variant_id: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    spec = get_strategy(source_id, strategy_id)
    variant = find_variant(source_id, strategy_id, variant_id)
    if not spec.recovery_supported:
        raise ApiError(400, 'recovery_unsupported', 'The fixed catalog has no recovery recipe for this target.')
    target_state, snapshot = inspect_target(spec, variant)
    if snapshot and target_state.cache_state == 'ready':
        return {'already_ready': True, 'snapshot_id': snapshot.snapshot_id}, []
    if snapshot and target_state.cache_state == 'data_stale':
        raise ApiError(409, 'recovery_blocked', 'Stale data must be updated through Data Status.', blocker_code='data_stale_use_data_update')
    if target_state.blocker_code:
        raise ApiError(409, 'recovery_blocked', 'A fixed recovery prerequisite is not satisfied.', blocker_code=target_state.blocker_code)
    steps = []
    for ordinal, scope in enumerate(spec.recovery_scopes):
        steps.append({
            'topological_ordinal': ordinal, 'step_id': f'pull-{scope}', 'scope': scope,
            'decision': 'needed', 'explicit': False, 'forced': False, 'blocked_by': [],
            'origin': 'prerequisite', 'depends_on': [], 'effective_mode': 'needed',
            'write_set': [f'dataset:{scope}'],
            'affected_targets': _affected_update_targets([scope]),
            'input_fingerprints': {}, 'eta_seconds': 90,
        })
    steps.append({
        'topological_ordinal': len(steps), 'step_id': 'build-target', 'scope': 'build',
        'decision': 'needed', 'explicit': True, 'forced': False,
        'blocked_by': [item['step_id'] for item in steps],
        'origin': 'explicit', 'depends_on': [item['step_id'] for item in steps],
        'effective_mode': 'needed',
        'write_set': [f'artifact:{source_id}/{strategy_id}/{variant_id}'],
        'affected_targets': [{
            'source_id': source_id, 'strategy_id': strategy_id,
            'variant_id': variant_id, 'role': 'rebuilt',
        }], 'input_fingerprints': {},
        'eta_seconds': spec.eta_seconds or 120,
    })
    plan = {
        'plan_version': 'v0.1',
        'recovery_plan_id': recovery_plan_id(spec, variant),
        'target': {'source_id': source_id, 'strategy_id': strategy_id, 'variant_id': variant_id},
        'recovery_scopes': list(spec.recovery_scopes), 'steps': steps,
        'write_set': [resource for step in steps for resource in step['write_set']],
    }
    plan_hash = hashlib.sha256(canonical_json(plan).encode('utf-8')).hexdigest()
    plan.update(plan_id=f'recovery-v0.1-{plan_hash[:16]}', plan_hash=plan_hash)
    exact_key = (source_id, strategy_id, variant_id)
    affected_by_key = {}
    for target in _affected_update_targets(list(spec.recovery_scopes)):
        key = (target['source_id'], target['strategy_id'], target['variant_id'])
        affected_by_key[key] = {**target, 'role': 'collateral'}
    affected_by_key[exact_key] = {
        'source_id': source_id, 'strategy_id': strategy_id,
        'variant_id': variant_id, 'role': 'rebuilt',
    }
    affected = [affected_by_key[key] for key in sorted(affected_by_key)]
    for step in steps:
        step['affected_targets'] = [
            affected_by_key[(target['source_id'], target['strategy_id'], target['variant_id'])]
            for target in step['affected_targets']
        ]
        step['input_fingerprints'] = {
            resource_id: _resource_fingerprint(resource_id)
            for resource_id in step['write_set']
        }
    plan['affected_targets'] = affected
    plan['pre_target_state'] = _pre_states(affected)
    plan_hash = hashlib.sha256(canonical_json({key: value for key, value in plan.items() if key not in {'plan_id', 'plan_hash'}}).encode('utf-8')).hexdigest()
    plan.update(plan_id=f'recovery-v0.1-{plan_hash[:16]}', plan_hash=plan_hash)
    return plan, affected


def _run_recovery(operation_id: str) -> None:
    value = runtime()
    operation = value.operation(operation_id)
    owner = GateOwner('cache-recover', operation_id, None)
    operation.status = 'running'
    terminal = 'done'
    error_code = None
    stabilization_failed = False
    publications: list[dict[str, Any]] = []
    for step in operation.steps:
        blocked = [
            prior['step_id'] for prior in operation.steps
            if prior['status'] == 'error' and prior['step_id'] in step.get('blocked_by', [])
        ]
        if blocked:
            step.update(
                status='skipped_dependency', blocked_by=blocked, progress=0,
                message='Dependency failed.', finished_at=utc_now(),
                error_code='skipped_dependency',
            )
            terminal = 'error'
            error_code = 'recovery_failed'
            continue
        step.update(status='running', started_at=utc_now(), progress=1, message='Running fixed recovery step.')
        try:
            observed_inputs = {
                resource_id: _resource_fingerprint(resource_id)
                for resource_id in step.get('input_fingerprints', {})
            }
            if observed_inputs != step.get('input_fingerprints', {}):
                step.update(
                    status='error', progress=100,
                    message='Frozen recovery input changed before execution.',
                    finished_at=utc_now(), error_code='input_changed',
                )
                terminal = 'error'
                error_code = 'recovery_failed'
                continue
            _mark_step_writes(operation, step, 'write_intent')
            if step['scope'] == 'build':
                builders = current_app.config.get('R0_RECOVERY_BUILDERS') or {}
                key = (operation.target['source_id'], operation.target['strategy_id'])
                builder = builders.get(key)
                (builder or _default_recovery_builder)(operation.target)
                publications = _publish_affected(operation, only_target=operation.target)
                if not publications or not publications[0]['published']:
                    raise RuntimeError('target-specific output missing or invalid')
                operation.result = {
                    'snapshot_id': publications[0]['snapshot_id'],
                    'snapshot_url': f'/api/r0/sources/{operation.target["source_id"]}/strategies/{operation.target["strategy_id"]}/snapshots?variant_id={operation.target["variant_id"]}',
                }
            else:
                _run_scope(step['scope'])
            _mark_step_writes(operation, step, 'write_durable')
            step.update(status='done', progress=100, message='Fixed recovery step completed.', finished_at=utc_now())
        except Exception:
            step.update(status='error', progress=100, message='Fixed recovery step failed.', finished_at=utc_now(), error_code='recovery_step_failed')
            terminal = 'error'
            error_code = 'recovery_failed'
    stabilization_failed = _reconcile_unpublished_targets(operation, publications)
    _finish_action(
        value, operation, owner, terminal_status=terminal,
        error_code=error_code, stabilization_failed=stabilization_failed,
    )


def start_recovery(payload: Any) -> tuple[Operation | None, dict[str, Any] | None]:
    if not isinstance(payload, dict) or set(payload) != {'source_id', 'strategy_id', 'variant_id'} or not all(isinstance(payload.get(key), str) and payload[key] for key in payload):
        raise ApiError(400, 'invalid_params', 'Body must be exactly the non-empty source/strategy/variant triple.')
    value = runtime()
    target_key = (payload['source_id'], payload['strategy_id'], payload['variant_id'])
    overlay = value.overlay_targets.get(target_key)
    if overlay and overlay[1] == 'recovering':
        return value.operation(overlay[0]), None
    plan, affected = _recovery_plan(**payload)
    if plan.get('already_ready'):
        return None, {'result': 'already_ready', 'cache_state': 'ready', 'readable': True}
    operation_id = str(uuid4())
    original = {**payload, 'recovery_plan_id': plan['recovery_plan_id'], 'plan_version': plan['plan_version']}
    operation = Operation(operation_id, 'cache-recover', dict(payload), original, plan['plan_id'], plan['plan_hash'], plan, steps=[_step_status(step) for step in plan['steps']])
    return _admit(operation, affected, _run_recovery), None


def retry(operation_id: str, payload: Any) -> Operation:
    if payload not in (None, {}) and payload != b'':
        raise ApiError(400, 'invalid_params', 'Retry body must be empty.')
    previous = runtime().operation(operation_id)
    if previous.kind == 'cache-recover' and previous.status == 'error':
        spec = get_strategy(previous.target['source_id'], previous.target['strategy_id'])
        variant = find_variant(**previous.target)
        current_recovery_plan_id = recovery_plan_id(spec, variant)
        if (previous.original_request.get('plan_version') != 'v0.1'
                or previous.original_request.get('recovery_plan_id') != current_recovery_plan_id):
            raise ApiError(
                409, 'recovery_plan_changed',
                'The fixed recovery recipe changed; reopen the catalog and submit a new recovery request.',
                previous_recovery_plan_id=previous.original_request.get('recovery_plan_id'),
                current_recovery_plan_id=current_recovery_plan_id,
            )
        operation, ready = start_recovery({key: previous.target[key] for key in ('source_id', 'strategy_id', 'variant_id')})
        if ready is not None:
            raise ApiError(409, 'operation_not_retryable', 'The target is already ready.')
    elif previous.kind == 'data-update' and previous.status in {'error', 'partial'}:
        operation = start_update(
            {'scopes': previous.original_request['requested_scopes'], 'force': previous.original_request['force']},
            retry_of=previous,
        )
    else:
        raise ApiError(409, 'operation_not_retryable', 'Only terminal failed/partial actions can be retried.')
    operation.retry_of = operation_id
    return operation


def restart_preflight() -> dict[str, Any]:
    # The current app has no server-level post-send hook integration and the
    # inspected deployment has no socket/service units. Environment flags can
    # never upgrade this to available without the missing code integration.
    return {
        'available': False,
        'error': 'restart_unavailable',
        'requirements': {
            'systemd_socket_activation': False, 'single_serving_web_process': False,
            'type_notify': False, 'inherited_listener': False, 'server_post_send_hook': False,
            'fixed_receipt_helper': False, 'unit_invocation_identity': False,
        },
    }


def request_restart() -> None:
    # Preflight is deliberately evaluated before gate acquisition or any
    # receipt/write/signal/process operation.
    raise ApiError(503, 'restart_unavailable', 'Required systemd socket-activation/notify/receipt topology is unavailable.', capability=restart_preflight())
