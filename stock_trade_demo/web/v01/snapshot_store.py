"""Immutable snapshot identities over already-loaded, fixed catalog artifacts.

This module is intentionally load-only: it never calls a cache initializer,
builder, strategy, subprocess, network client, or filesystem writer.  The
application startup/recovery path is responsible for loading/publishing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from flask import current_app

from web import state
from web.v01.catalog import StrategySpec, VariantSpec, get_strategy, recovery_plan_id
from web.v01.contracts import ApiError, canonical_json, decode_cursor, digest_id, encode_cursor, parse_iso_day


PUBLISHED_SIGNAL_KEYS = (
    'strategy_id', 'name', 'index_name', 'etf_code', 'etf_name', 'as_of_date',
    'settled_as_of_date', 'data_stale_warning', 'degraded_reason', 'target_exposure',
    'prev_exposure', 'exposure_delta', 'rebalance_action', 'rebalance_label',
    'signal_action', 'signal_label', 'current_action', 'current_position',
    'current_reason', 'reason_summary', 'bullish_score', 'ref_close', 'ref_open',
    'nav', 'settled_nav', 'status', 'passes_rule14', 'exec_basis',
    'profile_recent_6m', 'experiment_meta',
)
PROFILE_RECENT_KEYS = (
    'strategy_total_return_pct', 'etf_total_return_pct', 'excess_return_pct',
    'max_drawdown_pct', 'etf_max_drawdown_pct',
)
EXPERIMENT_KEYS = (
    'training_cutoff', 'holdout_start', 'holdout_end', 'holdout_bars',
    'training_recent_6m', 'training_full_pre_cutoff', 'holdout_metrics',
)
TRAINING_METRIC_KEYS = (
    'final_nav', 'total_return', 'annual_return', 'max_drawdown', 'calmar',
    'rebalance_count', 'avg_exposure', 'etf_total_return',
    'etf_max_drawdown', 'excess_return', 'dd_excess',
)
HOLDOUT_METRIC_KEYS = (
    'final_nav', 'total_return', 'annual_return', 'max_drawdown', 'calmar',
    'rebalance_count', 'avg_exposure',
)


@dataclass(frozen=True)
class TargetState:
    cache_state: str
    readable: bool
    freshness_state: str
    freshness_reason: dict[str, Any] | None
    degradation: dict[str, Any] | None
    generated_at: str | None = None
    blocker_code: str | None = None

    def as_fields(self) -> dict[str, Any]:
        return {
            'cache_state': self.cache_state,
            'readable': self.readable,
            'freshness_state': self.freshness_state,
            'freshness_reason': self.freshness_reason,
            'degradation': self.degradation,
        }


@dataclass(frozen=True)
class LoadedSnapshot:
    spec: StrategySpec
    variant: VariantSpec
    frame: pd.DataFrame | None
    document: dict[str, Any]
    snapshot_id: str
    generated_at: str
    target_state: TargetState


def _reason(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'code', 'detail'}:
        raise ValueError(f'{name} must be null or a closed code/detail object')
    if not isinstance(value['code'], str) or not value['code']:
        raise ValueError(f'{name}.code is required')
    if value['detail'] is not None and not isinstance(value['detail'], str):
        raise ValueError(f'{name}.detail must be null or string')
    return {'code': value['code'], 'detail': value['detail']}


def _derive_readable_state(raw: dict[str, Any] | None, generated_at: str | None) -> TargetState:
    raw = raw or {}
    expected = {'cache_state', 'readable', 'freshness_state', 'freshness_reason', 'degradation'}
    if raw and set(raw) != expected:
        raise ValueError('published target_state must be an exact closed object')
    if raw and raw.get('readable') is not True:
        raise ValueError('a published generation must declare readable=true')
    freshness = raw.get('freshness_state', 'current')
    freshness_reason = _reason(raw.get('freshness_reason'), name='freshness_reason')
    degradation = _reason(raw.get('degradation'), name='degradation')
    if freshness not in {'current', 'stale'}:
        raise ValueError('readable snapshots require current or stale freshness')
    if freshness == 'current' and freshness_reason is not None:
        raise ValueError('current freshness cannot carry a reason')
    if freshness == 'stale' and freshness_reason is None:
        raise ValueError('stale freshness requires a reason')
    cache_state = 'degraded' if degradation is not None else ('data_stale' if freshness == 'stale' else 'ready')
    if raw.get('cache_state') not in (None, cache_state):
        raise ValueError('cache_state does not match tuple precedence')
    return TargetState(cache_state, True, freshness, freshness_reason, degradation, generated_at)


def unreadable_state(cache_state: str, *, blocker_code: str | None = None) -> TargetState:
    if cache_state == 'missing':
        freshness = 'unknown'
        reason = {'code': 'freshness_unknown', 'detail': None}
    elif cache_state in {'recovering', 'updating'}:
        freshness = 'updating'
        reason = {'code': 'freshness_updating', 'detail': None}
    else:
        freshness = 'error'
        reason = {'code': 'freshness_unreadable', 'detail': None}
    return TargetState(cache_state, False, freshness, reason, None, None, blocker_code)


def _cache_entry(spec: StrategySpec, variant: VariantSpec) -> Any:
    fixture_map = current_app.config.get('R0_SNAPSHOT_FIXTURES') or {}
    fixture = fixture_map.get((spec.source_id, spec.strategy_id, variant.variant_id))
    if fixture is not None:
        return fixture
    action_runtime = current_app.extensions.get('r0_action_runtime')
    if action_runtime is not None:
        ephemeral = action_runtime.ephemeral_entry(spec.source_id, spec.strategy_id, variant.variant_id)
        if ephemeral is not None:
            return ephemeral
    return _read_generation(spec, variant)


def _generation_root() -> Path:
    configured = current_app.config.get('R0_GENERATION_ROOT')
    return Path(configured or (Path(current_app.instance_path) / 'r0-generations'))


def _target_dir(spec: StrategySpec, variant: VariantSpec) -> Path:
    target_hash = hashlib.sha256(f'{spec.source_id}\0{spec.strategy_id}\0{variant.variant_id}'.encode('utf-8')).hexdigest()
    return _generation_root() / target_hash


def _code_fingerprint() -> str:
    web_root = Path(__file__).resolve().parents[1]
    paths = sorted((web_root / 'v01').glob('*.py')) + [
        web_root / 'blueprints/v01_api.py', web_root / 'state.py',
    ]
    digest = hashlib.sha256()
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            raw = b''
        digest.update(path.relative_to(web_root).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def pointer_value(spec: StrategySpec, variant: VariantSpec) -> tuple[dict[str, Any] | None, str | None]:
    """Return the exact current pointer and its byte digest without opening legacy data."""
    try:
        raw = (_target_dir(spec, variant) / 'pointer.json').read_bytes()
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('pointer is not an object')
        return value, hashlib.sha256(raw).hexdigest()
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None, 'invalid'


def target_fence(spec: StrategySpec, variant: VariantSpec) -> dict[str, Any] | None:
    try:
        payload = json.loads((_target_dir(spec, variant) / 'readability-fence.json').read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('target_id') != [spec.source_id, spec.strategy_id, variant.variant_id]:
            raise ValueError('invalid target fence')
        return payload
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return {'reason': 'fence_corrupt'}


def write_target_fence(
    spec: StrategySpec,
    variant: VariantSpec,
    operation_id: str,
    diagnostic_pre_target_state: dict[str, Any],
    *,
    reason: str = 'interrupted_or_changed_input',
) -> None:
    payload = {
        'target_id': [spec.source_id, spec.strategy_id, variant.variant_id],
        'invalidated_by_operation_id': operation_id,
        'reason': reason,
        'observed_input_fingerprint': None,
        'observed_pointer_digest': None,
        'diagnostic_pre_target_state': diagnostic_pre_target_state,
    }
    _atomic_write(_target_dir(spec, variant) / 'readability-fence.json', canonical_json(payload).encode('utf-8'))


def _read_generation(spec: StrategySpec, variant: VariantSpec) -> dict[str, Any] | None:
    target = _target_dir(spec, variant)
    pointer_path = target / 'pointer.json'
    try:
        pointer_raw = pointer_path.read_bytes()
        pointer = json.loads(pointer_raw.decode('utf-8'))
        expected_pointer = {'source_id', 'strategy_id', 'variant_id', 'generation_id', 'payload_digest', 'generated_at'}
        if set(pointer) != expected_pointer:
            raise ValueError('pointer schema mismatch')
        if (pointer['source_id'], pointer['strategy_id'], pointer['variant_id']) != (spec.source_id, spec.strategy_id, variant.variant_id):
            raise ValueError('pointer target mismatch')
        generation_path = target / 'generations' / f'{pointer["generation_id"]}.json'
        raw = generation_path.read_bytes()
        raw_digest = hashlib.sha256(raw).hexdigest()
        if raw_digest != pointer['payload_digest'] or raw_digest != pointer['generation_id']:
            raise ValueError('generation digest mismatch')
        payload = json.loads(raw.decode('utf-8'))
        expected_payload = {
            'schema_version', 'source_id', 'strategy_id', 'variant_id',
            'generated_at', 'target_state', 'manifest', 'artifact',
        }
        if set(payload) != expected_payload or payload['schema_version'] != 1:
            raise ValueError('generation schema mismatch')
        if (payload['source_id'], payload['strategy_id'], payload['variant_id']) != (spec.source_id, spec.strategy_id, variant.variant_id):
            raise ValueError('generation target mismatch')
        if payload['generated_at'] != pointer['generated_at']:
            raise ValueError('generation timestamp mismatch')
        artifact = payload['artifact']
        if not isinstance(artifact, dict) or set(artifact) != {'kind', 'content'}:
            raise ValueError('artifact schema mismatch')
        manifest = payload['manifest']
        if not isinstance(manifest, dict) or set(manifest) != {
                'variant_id', 'canonical_params', 'code_fingerprint',
                'data_fingerprint', 'payload_digest', 'generated_at'}:
            raise ValueError('generation manifest schema mismatch')
        artifact_digest = hashlib.sha256(canonical_json(artifact).encode('utf-8')).hexdigest()
        if (manifest['variant_id'] != variant.variant_id
                or manifest['canonical_params'] != _jsonable(variant.canonical_params)
                or manifest['generated_at'] != payload['generated_at']
                or manifest['data_fingerprint'] != artifact_digest
                or manifest['payload_digest'] != artifact_digest
                or manifest['code_fingerprint'] != _code_fingerprint()):
            raise ValueError('generation manifest validation failed')
        entry: dict[str, Any] = {'target_state': payload['target_state'], 'generated_at': payload['generated_at']}
        if artifact['kind'] == 'frame':
            content = artifact['content']
            columns = content.get('columns')
            rows = content.get('rows')
            if not isinstance(columns, list) or not isinstance(rows, list):
                raise ValueError('frame artifact is invalid')
            records = []
            for expected_ordinal, row in enumerate(rows):
                if row.get('ordinal') != expected_ordinal or not isinstance(row.get('values'), dict):
                    raise ValueError('frame row ordinal mismatch')
                if set(row['values']) != set(columns):
                    raise ValueError('frame row columns mismatch')
                records.append([row['values'][column] for column in columns])
            frame = pd.DataFrame(records, columns=columns)
            if '交易日期' in frame.columns:
                frame['交易日期'] = pd.to_datetime(frame['交易日期'], errors='coerce')
            frame.attrs.update(content.get('attrs') or {})
            frame.attrs['r0_target_state'] = payload['target_state']
            frame.attrs['r0_generated_at'] = payload['generated_at']
            entry['frame'] = frame
        elif artifact['kind'] == 'payload':
            entry.update(artifact['content'])
        else:
            raise ValueError('unsupported artifact kind')
        entry['document'] = artifact
        return entry
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {'_generation_error': str(exc)}


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f'.{path.name}.{uuid4()}.tmp')
    try:
        with staging.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def publish_entry(
    spec: StrategySpec,
    variant: VariantSpec,
    entry: Any,
    *,
    target_state: dict[str, Any] | None = None,
    before_pointer_commit: Callable[[dict[str, Any]], None] | None = None,
    after_pointer_commit: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Publish one immutable target generation and atomically switch pointer.

    The caller must own the mutation gate and crash-fence.  Validation happens
    against the same load/projection structures before the pointer replace.
    """
    frame = _frame_from_entry(spec, entry)
    document, generated_at = _document_from_entry(spec, variant, entry, frame)
    state_fields = target_state or (frame.attrs.get('r0_target_state') if frame is not None else None) or {
        'cache_state': 'ready', 'readable': True, 'freshness_state': 'current',
        'freshness_reason': None, 'degradation': None,
    }
    validated_state = _derive_readable_state(state_fields, generated_at)
    candidate_snapshot_id = digest_id('s', {
        'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id, 'document': document,
    })
    candidate = LoadedSnapshot(spec, variant, frame, document, candidate_snapshot_id, generated_at, validated_state)
    _signal_from(candidate)
    # Exercise the production projection code before pointer commit.  The
    # imports are local to avoid a module cycle.
    from web.v01 import projections
    if frame is not None and len(frame):
        low, high = _date_bounds(candidate)
        view = {'start': low, 'end': high, 'benchmark_id': 'csi1000' if spec.source_id in {'selection', 'a_share_timing'} else 'etf', 'resolution': 'day'}
        view_id = 'validation'
        projections.summary(candidate, view, view_id)
        if 'series' in spec.capabilities:
            projections.series(candidate, frame, kind='equity', window='full', resolution='day', benchmark_id=None, series_id=None)
        if 'signals' in spec.capabilities:
            projections.signals(frame, 'all')
            projections._position(frame)
            projections.trades(frame, 'detail')
        if 'fees' in spec.capabilities:
            projections.fees(candidate, frame)
        if 'interval_windows' in spec.capabilities:
            projections.interval_windows(candidate, frame, view_id)
        if 'holding_periods' in spec.capabilities:
            projections.holding_periods(candidate, frame, 'full')
        if spec.source_id == 'selection':
            projections.factor_metadata(candidate)
            projections.factor_overview(candidate)
        if spec.source_id == 'selection_factor' and spec.strategy_id == 'sector_heat':
            projections.sector_heat(candidate, 8)
    elif frame is None:
        projections.summary(
            candidate,
            {'start': None, 'end': None, 'benchmark_id': 'etf', 'resolution': 'day'},
            'validation',
        )
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor':
        _, factor_items = projections.single_factor(candidate)
        for item in factor_items:
            projections.series(
                candidate, None, kind='factor_nav', window='full',
                resolution='day', benchmark_id=None, series_id=item['series_id'],
            )
    projections.configuration(candidate)
    artifact_digest = hashlib.sha256(canonical_json(document).encode('utf-8')).hexdigest()
    manifest = {
        'variant_id': variant.variant_id,
        'canonical_params': _jsonable(variant.canonical_params),
        'code_fingerprint': _code_fingerprint(),
        'data_fingerprint': artifact_digest,
        'payload_digest': artifact_digest,
        'generated_at': generated_at,
    }
    payload = {
        'schema_version': 1, 'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id, 'generated_at': generated_at,
        'target_state': state_fields, 'manifest': manifest, 'artifact': document,
    }
    raw = canonical_json(payload).encode('utf-8')
    generation_id = hashlib.sha256(raw).hexdigest()
    target = _target_dir(spec, variant)
    generation_path = target / 'generations' / f'{generation_id}.json'
    if not generation_path.exists():
        _atomic_write(generation_path, raw)
    pointer = {
        'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id, 'generation_id': generation_id,
        'payload_digest': hashlib.sha256(raw).hexdigest(), 'generated_at': generated_at,
    }
    pointer_raw = canonical_json(pointer).encode('utf-8')
    candidate = {
        'generation_id': generation_id,
        'pointer_value': pointer,
        'pointer_digest': hashlib.sha256(pointer_raw).hexdigest(),
        'manifest_digest': hashlib.sha256(raw).hexdigest(),
        'snapshot_id': candidate_snapshot_id,
    }
    pointer_path = target / 'pointer.json'
    try:
        previous_pointer = pointer_path.read_bytes()
    except FileNotFoundError:
        previous_pointer = None
    if before_pointer_commit is not None:
        before_pointer_commit(candidate)
    pointer_committed = False
    try:
        _atomic_write(pointer_path, pointer_raw)
        pointer_committed = True
        if after_pointer_commit is not None:
            after_pointer_commit(candidate)
    except Exception:
        if pointer_committed:
            if previous_pointer is None:
                try:
                    pointer_path.unlink()
                    directory = os.open(target, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                except FileNotFoundError:
                    pass
            else:
                _atomic_write(pointer_path, previous_pointer)
        raise
    try:
        (target / 'readability-fence.json').unlink()
        directory = os.open(target, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileNotFoundError:
        pass
    return generation_id


def _frame_from_entry(spec: StrategySpec, entry: Any) -> pd.DataFrame | None:
    if isinstance(entry, dict) and isinstance(entry.get('frame'), pd.DataFrame):
        return entry['frame']
    if spec.source_id == 'selection' and isinstance(entry, tuple) and entry:
        return entry[0] if isinstance(entry[0], pd.DataFrame) else None
    return entry if isinstance(entry, pd.DataFrame) else None


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (pd.Timestamp, date)):
        return pd.to_datetime(value).isoformat()
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)):
        return value
    return str(value)


def _frame_content(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        rows.append({'ordinal': ordinal, 'values': {str(column): _jsonable(row[column]) for column in frame.columns}})
    attrs = {
        str(key): _jsonable(value)
        for key, value in frame.attrs.items()
        if key in {
            'initial_capital', 'c_rate', 't_rate', 'sell_cost', 'buy_cost', 'market',
            'settlement_mode', 'limit_pct', 'slippage_bps', 'cash_interest_rate',
            'commission_rate', 'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
            'limit_max_delay_days', 'profit_lock_enabled', 'profit_lock_drawdown',
            'profit_lock_level_1', 'profit_lock_level_2', 'profit_lock_level_3',
            'exposure_mode', 'etf_code', 'etf_name', 'published_current_signal',
            'r0_target_state', 'r0_generated_at', 'configuration', 'factor_metadata',
            'factor_overview', 'profile_summary', 'strategy_meta', 'split',
            'best_profile', 'changelog', 'timing_metadata', 'selection_metadata',
            'effective_configuration', 'provenance_label', 'etf_inception_date',
            'metrics',
        }
    }
    return {'columns': [str(column) for column in frame.columns], 'rows': rows, 'attrs': attrs}


def _document_from_entry(spec: StrategySpec, variant: VariantSpec, entry: Any, frame: pd.DataFrame | None) -> tuple[dict[str, Any], str]:
    if isinstance(entry, dict) and entry.get('_generation_error'):
        raise ValueError(entry['_generation_error'])
    if isinstance(entry, dict) and isinstance(entry.get('document'), dict):
        document = _jsonable(entry['document'])
        generated_at = str(entry.get('generated_at') or document.get('generated_at') or 'unknown')
        return document, generated_at
    if frame is not None:
        generated_at = str(frame.attrs.get('r0_generated_at') or 'boot-loaded')
        return {'kind': 'frame', 'content': _frame_content(frame)}, generated_at
    if isinstance(entry, dict):
        generated_at = str(entry.get('saved_at') or entry.get('generated_at') or 'boot-loaded')
        return {'kind': 'payload', 'content': _jsonable(entry)}, generated_at
    raise ValueError('unsupported snapshot artifact')


def _signal_from(snapshot: LoadedSnapshot) -> dict[str, Any] | None:
    if 'published_current_signal' not in snapshot.spec.capabilities:
        return None
    signal = None
    if snapshot.frame is not None:
        signal = snapshot.frame.attrs.get('published_current_signal')
    if signal is None:
        signal = snapshot.document.get('published_current_signal')
    if not isinstance(signal, dict) or set(signal) != set(PUBLISHED_SIGNAL_KEYS):
        raise ValueError('published current signal is missing or not closed')
    ordered = {key: signal.get(key) for key in PUBLISHED_SIGNAL_KEYS}
    if ordered['strategy_id'] != snapshot.spec.strategy_id:
        raise ValueError('published current signal strategy mismatch')
    numeric_keys = {
        'target_exposure', 'prev_exposure', 'exposure_delta', 'current_position',
        'bullish_score', 'ref_close', 'ref_open', 'nav', 'settled_nav',
    }
    for key in numeric_keys:
        value = ordered[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            raise ValueError(f'published current signal {key} must be finite number or null')
    for key in set(PUBLISHED_SIGNAL_KEYS) - numeric_keys - {'passes_rule14', 'profile_recent_6m', 'experiment_meta'}:
        value = ordered[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f'published current signal {key} must be string or null')
    if ordered['passes_rule14'] is not None and not isinstance(ordered['passes_rule14'], bool):
        raise ValueError('published current signal passes_rule14 must be boolean or null')
    ordered['profile_recent_6m'] = _closed_metric_object(
        ordered['profile_recent_6m'], PROFILE_RECENT_KEYS, 'profile_recent_6m')
    ordered['experiment_meta'] = _experiment_object(ordered['experiment_meta'])
    ordered = {key: _jsonable(ordered[key]) for key in PUBLISHED_SIGNAL_KEYS}
    expected_stale = snapshot.target_state.freshness_reason['code'] if snapshot.target_state.freshness_state == 'stale' else None
    expected_degraded = snapshot.target_state.degradation['code'] if snapshot.target_state.degradation else None
    if ordered['data_stale_warning'] != expected_stale or ordered['degraded_reason'] != expected_degraded:
        raise ValueError('published signal warning tuple mismatch')
    return ordered


def _closed_metric_object(value: Any, keys: tuple[str, ...], name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name} must be null or an exact closed object')
    result = {}
    for key in keys:
        item = value[key]
        if item is not None:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError(f'{name}.{key} must be finite number or null')
            if key == 'rebalance_count' and (not isinstance(item, int) or item < 0):
                raise ValueError(f'{name}.{key} must be nonnegative integer or null')
        result[key] = item
    return result


def _experiment_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(EXPERIMENT_KEYS):
        raise ValueError('experiment_meta must be null or an exact closed object')
    result = {}
    for key in ('training_cutoff', 'holdout_start', 'holdout_end'):
        item = value[key]
        if item is not None:
            try:
                if date.fromisoformat(item).isoformat() != item:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f'experiment_meta.{key} must be ISO day or null') from exc
        result[key] = item
    bars = value['holdout_bars']
    if bars is not None and (isinstance(bars, bool) or not isinstance(bars, int) or bars < 0):
        raise ValueError('experiment_meta.holdout_bars must be nonnegative integer or null')
    result['holdout_bars'] = bars
    result['training_recent_6m'] = _closed_metric_object(value['training_recent_6m'], TRAINING_METRIC_KEYS, 'training_recent_6m')
    result['training_full_pre_cutoff'] = _closed_metric_object(value['training_full_pre_cutoff'], TRAINING_METRIC_KEYS, 'training_full_pre_cutoff')
    result['holdout_metrics'] = _closed_metric_object(value['holdout_metrics'], HOLDOUT_METRIC_KEYS, 'holdout_metrics')
    return result


def load_snapshot(spec: StrategySpec, variant: VariantSpec) -> LoadedSnapshot | None:
    entry = _cache_entry(spec, variant)
    if entry is None:
        return None
    frame = _frame_from_entry(spec, entry)
    document, generated_at = _document_from_entry(spec, variant, entry, frame)
    snapshot_id = digest_id('s', {
        'source_id': spec.source_id,
        'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id,
        'document': document,
    })
    raw_state = None
    if frame is not None:
        raw_state = frame.attrs.get('r0_target_state')
    if raw_state is None and isinstance(entry, dict):
        raw_state = entry.get('target_state')
    target_state = _derive_readable_state(raw_state, generated_at)
    snapshot = LoadedSnapshot(spec, variant, frame, document, snapshot_id, generated_at, target_state)
    _signal_from(snapshot)
    return snapshot


def inspect_target(spec: StrategySpec, variant: VariantSpec) -> tuple[TargetState, LoadedSnapshot | None]:
    action_runtime = current_app.extensions.get('r0_action_runtime')
    if action_runtime is not None:
        overlay = action_runtime.overlay(spec.source_id, spec.strategy_id, variant.variant_id)
        if overlay is not None:
            return unreadable_state(overlay[1]), None
    if target_fence(spec, variant) is not None:
        return unreadable_state('artifact_stale'), None
    try:
        snapshot = load_snapshot(spec, variant)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return unreadable_state('corrupt'), None
    if snapshot is None:
        blocker = None
        if spec.source_id == 'selection' and not state.BACKTEST_CACHE:
            blocker = 'bootstrap_required'
        if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor' and not state.BACKTEST_CACHE:
            blocker = 'bootstrap_required'
        return unreadable_state('missing', blocker_code=blocker), None
    return snapshot.target_state, snapshot


def current_snapshot(source_id: str, strategy_id: str, variant: VariantSpec) -> LoadedSnapshot:
    spec = get_strategy(source_id, strategy_id)
    target_state, snapshot = inspect_target(spec, variant)
    if snapshot is None:
        code = 'cache_recovering' if target_state.cache_state == 'recovering' else (
            'snapshot_updating' if target_state.cache_state == 'updating' else 'cache_miss'
        )
        action_runtime = current_app.extensions.get('r0_action_runtime')
        overlay = action_runtime.overlay(source_id, strategy_id, variant.variant_id) if action_runtime else None
        supported = bool(spec.recovery_supported)
        recoverable = supported and target_state.blocker_code is None and target_state.cache_state in {
            'missing', 'corrupt', 'artifact_stale',
        }
        steps = None
        if supported:
            steps = [
                {'step_id': f'pull-{scope}', 'label': f'Needed {scope} data', 'scope': scope}
                for scope in spec.recovery_scopes
            ] + [{
                'step_id': 'build-target',
                'label': 'Build and validate exact catalog target', 'scope': 'build',
            }]
        raise ApiError(
            409, code, 'The exact catalog variant has no readable canonical snapshot.',
            source_id=source_id, strategy_id=strategy_id, variant_id=variant.variant_id,
            cache_state=target_state.cache_state, readable=False,
            freshness_state=target_state.freshness_state,
            freshness_reason=target_state.freshness_reason, degradation=None,
            recovery_supported=supported, recoverable_now=recoverable,
            blocker_code=target_state.blocker_code,
            recovery_plan_id=recovery_plan_id(spec, variant) if supported else None,
            recovery_scopes=list(spec.recovery_scopes) if supported else None,
            recovery_steps=steps, eta_seconds=spec.eta_seconds if supported else None,
            operation_id=overlay[0] if overlay else None,
            status_url=f'/api/r0/actions/{overlay[0]}' if overlay else None,
        )
    return snapshot


def exact_snapshot(source_id: str, strategy_id: str, variant: VariantSpec, snapshot_id: str) -> LoadedSnapshot:
    snapshot = current_snapshot(source_id, strategy_id, variant)
    if snapshot.snapshot_id != snapshot_id:
        raise ApiError(409, 'snapshot_changed', 'The target now points to a different snapshot.', current_snapshot_id=snapshot.snapshot_id)
    return snapshot


def _date_bounds(snapshot: LoadedSnapshot) -> tuple[str | None, str | None]:
    frame = snapshot.frame
    if frame is None or '交易日期' not in frame.columns or len(frame) == 0:
        content = snapshot.document.get('content', {})
        low = content.get('data_min_date')
        high = content.get('data_max_date')
        candidates = []
        for item in content.get('factors') or content.get('items') or []:
            if isinstance(item, dict) and isinstance(item.get('dates'), list):
                candidates.extend(item['dates'])
        if content.get('as_of') is not None:
            candidates.append(content['as_of'])
        parsed = pd.to_datetime(pd.Series(candidates), errors='coerce').dropna()
        derived_low = parsed.min().date().isoformat() if len(parsed) else None
        derived_high = parsed.max().date().isoformat() if len(parsed) else None
        return low or derived_low, high or derived_high
    dates = pd.to_datetime(frame['交易日期'], errors='coerce').dropna()
    if len(dates) == 0:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def snapshot_date_bounds(snapshot: LoadedSnapshot) -> tuple[str | None, str | None]:
    return _date_bounds(snapshot)


def source_date_bounds(source_id: str) -> tuple[str | None, str | None]:
    minimums: list[str] = []
    maximums: list[str] = []
    from web.v01.catalog import STRATEGY_SPECS, variants_for
    for spec in STRATEGY_SPECS:
        if spec.source_id != source_id:
            continue
        for variant in variants_for(spec):
            _, snapshot = inspect_target(spec, variant)
            if snapshot is None:
                continue
            low, high = _date_bounds(snapshot)
            if low:
                minimums.append(low)
            if high:
                maximums.append(high)
    return (min(minimums) if minimums else None, max(maximums) if maximums else None)


def canonical_view(snapshot: LoadedSnapshot, raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
    start = parse_iso_day(raw.get('start'), field='start')
    end = parse_iso_day(raw.get('end'), field='end')
    if start and end and start > end:
        raise ApiError(400, 'invalid_range', 'start must not be after end.')
    available_start, available_end = _date_bounds(snapshot)
    effective_start = start or available_start
    effective_end = end or available_end
    if (effective_start and available_end and effective_start > available_end) or (
            effective_end and available_start and effective_end < available_start):
        raise ApiError(422, 'empty_view_range', 'The valid date range contains no stored points.')
    benchmark = raw.get('benchmark')
    if benchmark is None:
        benchmark = 'csi1000' if snapshot.spec.source_id in {'selection', 'a_share_timing'} else 'etf'
    allowed_benchmarks = {'csi1000', 'csi500', 'hs300', 'chinext', 'star50'} if snapshot.spec.source_id in {'selection', 'a_share_timing'} else {'etf'}
    if benchmark not in allowed_benchmarks:
        raise ApiError(400, 'invalid_view_filter', 'Unsupported benchmark.')
    resolution = raw.get('resolution') or 'day'
    if resolution not in {'day', 'month', 'quarter', 'year'}:
        raise ApiError(400, 'invalid_view_filter', 'Unsupported resolution.')
    view = {'start': effective_start, 'end': effective_end, 'benchmark_id': benchmark, 'resolution': resolution}
    view_id = 'vw_' + encode_cursor({'snapshot_id': snapshot.snapshot_id, 'canonical_view': view})
    return view, view_id


def validate_view(snapshot: LoadedSnapshot, view_id: str | None) -> dict[str, Any]:
    if not view_id:
        raise ApiError(400, 'invalid_view_id', 'view_id is required.')
    if not view_id.startswith('vw_'):
        raise ApiError(400, 'invalid_view_id', 'view_id is invalid for this snapshot.')
    try:
        stored = decode_cursor(view_id[3:])
    except ApiError as exc:
        raise ApiError(400, 'invalid_view_id', 'view_id is invalid for this snapshot.') from exc
    if not stored or stored.get('snapshot_id') != snapshot.snapshot_id or not isinstance(stored.get('canonical_view'), dict):
        raise ApiError(400, 'invalid_view_id', 'view_id is invalid for this snapshot.')
    return dict(stored['canonical_view'])


def remember_view(snapshot: LoadedSnapshot, view_id: str, view: dict[str, Any]) -> None:
    """Compatibility no-op: view identity is fully stateless and signed."""
    return None


def filter_frame(snapshot: LoadedSnapshot, view: dict[str, Any]) -> pd.DataFrame | None:
    frame = snapshot.frame
    if frame is None:
        return None
    result = frame.copy(deep=False)
    if '交易日期' not in result.columns:
        return result
    dates = pd.to_datetime(result['交易日期'], errors='coerce')
    mask = dates.notna()
    if view.get('start'):
        mask &= dates >= pd.Timestamp(view['start'])
    if view.get('end'):
        mask &= dates <= pd.Timestamp(view['end'])
    result = result.loc[mask].copy(deep=False).reset_index(drop=True)
    result.attrs = dict(frame.attrs)
    if len(result) == 0:
        raise ApiError(422, 'empty_view_range', 'The valid date range contains no stored points.')
    return result


def published_signal(snapshot: LoadedSnapshot) -> dict[str, Any] | None:
    return _signal_from(snapshot)
