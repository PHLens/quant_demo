"""Experiment, async run, immutable result, and strict comparison service."""
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.contracts import (
    ALLOWED_TREND_WINDOWS,
    BASELINE_TREND_WINDOW,
    COMPARABILITY_FIELDS,
    EVIDENCE_DOMAIN_ID,
    FIXED_POLICY,
    RESEARCH_FAMILY_ID,
    RESULT_SCHEMA_VERSION,
    RUNNER_CONTRACT_VERSION,
    TEMPLATE_ID,
    TEMPLATE_VERSION,
    TRIAL_FAMILY_ID,
    VALIDATION_WINDOW,
    public_template,
)
from lab.runner import run_trend_validation, runner_fingerprint
from lab.snapshot import (
    DEFAULT_MANIFEST_PATH,
    SnapshotIntegrityError,
    load_materialized_snapshot,
)
from lab.store import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    JsonArtifactStore,
    canonical_json,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_PATH = _PACKAGE_DIR / 'baselines' / 'csi1000_trend_101_v1.json'


class LabError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 400,
                 details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_id(prefix: str, value: Any, length: int = 24) -> str:
    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    return f'{prefix}_{digest[:length]}'


def _uuid_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex}'


def _expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabError('invalid_request', f'{label} must be a JSON object')
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise LabError(
            'unknown_fields',
            f'{label} contains unsupported fields',
            details={'unknown_fields': unknown, 'allowed_fields': sorted(allowed)},
        )


def _nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class LabService:
    """Personal Lab-1 service with no dependency on mutable Research state."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        executor_kind: str = 'process',
        snapshot_manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    ):
        self.store = JsonArtifactStore(artifact_root)
        self._submission_lock = threading.RLock()
        self.snapshot_manifest_path = Path(snapshot_manifest_path)
        self.baseline_path = Path(baseline_path)
        self.executor_kind = executor_kind
        self._executor = None
        self._executor_lock = threading.Lock()
        self._baseline = self._load_baseline()
        self._reconcile_interrupted_runs()
        atexit.register(self.shutdown)

    def _load_baseline(self) -> dict[str, Any]:
        try:
            with self.baseline_path.open('r', encoding='utf-8') as handle:
                baseline = json.load(handle)
        except FileNotFoundError as exc:
            raise RuntimeError(f'packaged Lab baseline missing: {self.baseline_path}') from exc
        content = baseline.get('content')
        if not isinstance(content, dict):
            raise RuntimeError('packaged Lab baseline has no content object')
        actual_hash = hashlib.sha256(canonical_json(content)).hexdigest()
        if actual_hash != baseline.get('content_sha256') or actual_hash != baseline.get('result_id'):
            raise RuntimeError('packaged Lab baseline content identity mismatch')
        current_fp = runner_fingerprint()
        if _nested_get(content, 'runner.fingerprint') != current_fp:
            raise RuntimeError(
                'packaged Lab baseline runner fingerprint is stale; rebuild baseline offline'
            )
        snapshot = load_materialized_snapshot(self.snapshot_manifest_path)
        if content.get('data_snapshot') != snapshot.public_identity():
            raise RuntimeError('packaged Lab baseline snapshot public identity is stale')
        return baseline

    @property
    def baseline_result_id(self) -> str:
        return self._baseline['result_id']

    def _get_executor(self):
        with self._executor_lock:
            if self._executor is None:
                if self.executor_kind == 'thread':
                    self._executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix='quant-lab1',
                    )
                elif self.executor_kind == 'process':
                    self._executor = ProcessPoolExecutor(max_workers=1)
                else:
                    raise RuntimeError(f'unsupported Lab executor: {self.executor_kind}')
            return self._executor

    def shutdown(self) -> None:
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_result_document(
        self,
        result: dict[str, Any],
        result_id: str,
    ) -> dict[str, Any]:
        content = result.get('content')
        if not isinstance(content, dict):
            raise LabError(
                'result_integrity_error',
                'result content is missing or invalid',
                status=409,
            )
        actual_hash = hashlib.sha256(canonical_json(content)).hexdigest()
        if (
            actual_hash != result.get('result_id')
            or actual_hash != result.get('content_sha256')
            or actual_hash != result_id
        ):
            raise LabError(
                'result_integrity_error',
                'result content hash does not match its immutable identity',
                status=409,
                details={'expected_result_id': result_id, 'actual_content_sha256': actual_hash},
            )
        return result

    def _reconcile_interrupted_runs(self) -> None:
        results_by_run: dict[str, dict[str, Any]] = {}
        for result in self.store.list('results'):
            result_id = result.get('result_id')
            if not isinstance(result_id, str):
                continue
            try:
                verified = self._validate_result_document(result, result_id)
            except LabError:
                continue
            producing_run_id = _nested_get(verified, 'provenance.producing_run_id')
            if isinstance(producing_run_id, str):
                results_by_run[producing_run_id] = verified

        for run in self.store.list('runs'):
            if run.get('status') not in {'queued', 'running'}:
                continue
            now = utc_now()
            recovered = results_by_run.get(run.get('run_id'))
            if (
                recovered is not None
                and _nested_get(recovered, 'content.variant.variant_id') == run.get('variant_id')
            ):
                run['status'] = 'success'
                run['outcome'] = 'recovered_success'
                run['finished_at'] = now
                run['result_id'] = recovered['result_id']
                run['error'] = None
                run.setdefault('status_history', []).append({
                    'status': 'success',
                    'at': now,
                    'reason': 'result_artifact_recovered',
                })
                self.store.write_run(run['run_id'], run)
                continue
            run['status'] = 'failed'
            run['outcome'] = 'interrupted'
            run['finished_at'] = now
            run['error'] = {
                'code': 'worker_interrupted',
                'message': 'Web process restarted before this run completed; submit a retry.',
            }
            run.setdefault('status_history', []).append({
                'status': 'failed',
                'at': now,
                'reason': 'worker_interrupted',
            })
            self.store.write_run(run['run_id'], run)

    def _current_snapshot(self):
        try:
            return load_materialized_snapshot(self.snapshot_manifest_path)
        except (SnapshotIntegrityError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise LabError(
                'snapshot_integrity_error',
                'the frozen Lab snapshot is unavailable or failed integrity verification',
                status=503,
                details={'reason': str(exc)},
            ) from exc

    def _variant_contract_mismatches(
        self,
        variant: dict[str, Any],
        snapshot,
    ) -> list[str]:
        expected = {
            'template_id': TEMPLATE_ID,
            'template_version': TEMPLATE_VERSION,
            'data_snapshot': snapshot.public_identity(),
            'runner': {
                'contract_version': RUNNER_CONTRACT_VERSION,
                'fingerprint': runner_fingerprint(),
            },
            'fixed_policy': FIXED_POLICY,
            'baseline_result_id': self.baseline_result_id,
            'holdout_access': 'not_configured',
        }
        return [
            field
            for field, value in expected.items()
            if variant.get(field) != value
        ]

    def template(self) -> dict[str, Any]:
        snapshot = self._current_snapshot()
        variant_ids = {
            variant['variant_id']
            for variant in self.store.list('variants')
            if variant.get('trial_family_id') == TRIAL_FAMILY_ID
        }
        return public_template(
            snapshot=snapshot.public_identity(),
            runner_fingerprint=runner_fingerprint(),
            baseline_result_id=self.baseline_result_id,
            trial_count=len(variant_ids),
        )

    def learn_topics(self) -> dict[str, Any]:
        return {
            'topics': [
                {
                    'id': 'selection-vs-timing',
                    'title': '先分清：选什么，还是何时持有',
                    'summary': 'Selection 回答买什么；Timing 回答何时、持有多少。',
                    'check': 'CSI1000 Trend 101 是 Timing 模板，不负责挑选成分股。',
                },
                {
                    'id': 'cache-vs-experiment',
                    'title': '缓存读取不是新实验',
                    'summary': 'Research 页面只读取既有 artifact；Lab 的 POST 才创建异步 validation run。',
                    'check': 'GET 只能查询，不会触发 fresh computation。',
                },
                {
                    'id': 'risk-before-return',
                    'title': '先看样本、回撤与成本',
                    'summary': '收益放在有效性、最大回撤、换手与费用之后解释。',
                    'check': '高收益若来自更高换手，先检查 fee drag 与交易样本。',
                },
                {
                    'id': 'validation-not-oos',
                    'title': 'Validation 不是独立 OOS',
                    'summary': '当前 Lab 只展示固定 validation；所有结果明确 OOS=not_configured。',
                    'check': 'recent/validation 都不能被称为未见样本。',
                },
            ],
            'worked_example': {
                'template_id': TEMPLATE_ID,
                'formula': 'close(t) > MA_N(t) → position=1；否则 position=0',
                'clock': FIXED_POLICY['clock'],
                'baseline_trend_window': BASELINE_TREND_WINDOW,
                'baseline_result_id': self.baseline_result_id,
            },
        }

    def create_experiment(self, payload: Any) -> dict[str, Any]:
        body = _expect_object(payload, 'request')
        _reject_unknown(body, {'template_id', 'title', 'hypothesis'}, 'request')
        if body.get('template_id') != TEMPLATE_ID:
            raise LabError('unknown_template', f'only template {TEMPLATE_ID} is enabled')
        title = body.get('title')
        if not isinstance(title, str) or not (3 <= len(title.strip()) <= 120):
            raise LabError('invalid_title', 'title must contain 3-120 characters')
        hypothesis = _expect_object(body.get('hypothesis'), 'hypothesis')
        _reject_unknown(
            hypothesis,
            {
                'statement',
                'primary_observable',
                'expected_direction',
                'falsification_condition',
                'validation_window',
            },
            'hypothesis',
        )
        statement = hypothesis.get('statement')
        falsification = hypothesis.get('falsification_condition')
        if not isinstance(statement, str) or len(statement.strip()) < 10:
            raise LabError('invalid_hypothesis', 'statement must contain at least 10 characters')
        if not isinstance(falsification, str) or len(falsification.strip()) < 10:
            raise LabError(
                'invalid_falsification_condition',
                'falsification_condition must contain at least 10 characters',
            )
        if hypothesis.get('primary_observable') != 'signal_switch_count':
            raise LabError(
                'fixed_primary_observable',
                'primary_observable must be signal_switch_count for this template',
            )
        if hypothesis.get('expected_direction') not in {
            'increase', 'decrease', 'no_material_change',
        }:
            raise LabError(
                'invalid_expected_direction',
                'expected_direction must be increase, decrease, or no_material_change',
            )
        if hypothesis.get('validation_window') != VALIDATION_WINDOW:
            raise LabError(
                'fixed_validation_window',
                'validation_window must exactly match the server-frozen window',
                details={'expected': VALIDATION_WINDOW},
            )

        registered_at = utc_now()
        revision_content = {
            'revision': 1,
            'statement': statement.strip(),
            'primary_observable': 'signal_switch_count',
            'expected_direction': hypothesis['expected_direction'],
            'falsification_condition': falsification.strip(),
            'validation_window': copy.deepcopy(VALIDATION_WINDOW),
            'registered_at': registered_at,
        }
        revision_id = _hash_id('hypothesis', revision_content)
        experiment_id = _uuid_id('experiment')
        experiment = {
            'experiment_id': experiment_id,
            'template_id': TEMPLATE_ID,
            'template_version': TEMPLATE_VERSION,
            'title': title.strip(),
            'research_family_id': RESEARCH_FAMILY_ID,
            'evidence_domain_id': EVIDENCE_DOMAIN_ID,
            'trial_family_id': TRIAL_FAMILY_ID,
            'hypothesis_revision': {
                'hypothesis_revision_id': revision_id,
                **revision_content,
            },
            'baseline_result_id': self.baseline_result_id,
            'holdout_access': 'not_configured',
            'created_at': registered_at,
        }
        self.store.create_immutable('experiments', experiment_id, experiment)
        return experiment

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        try:
            experiment = self.store.read('experiments', experiment_id)
        except (ArtifactNotFoundError, ValueError) as exc:
            raise LabError('experiment_not_found', 'experiment not found', status=404) from exc
        variants = [
            value for value in self.store.list('variants')
            if value.get('experiment_id') == experiment_id
        ]
        variant_ids = {value['variant_id'] for value in variants}
        runs = [
            value for value in self.store.list('runs')
            if value.get('variant_id') in variant_ids
        ]
        return {
            **experiment,
            'variants': variants,
            'runs': sorted(runs, key=lambda value: value.get('created_at', '')),
            'trial_count': len({
                value['variant_id']
                for value in self.store.list('variants')
                if value.get('trial_family_id') == TRIAL_FAMILY_ID
            }),
        }

    def create_variant(self, experiment_id: str, payload: Any) -> tuple[dict[str, Any], bool]:
        experiment = self.get_experiment(experiment_id)
        body = _expect_object(payload, 'request')
        _reject_unknown(body, {'patch'}, 'request')
        patch = _expect_object(body.get('patch'), 'patch')
        _reject_unknown(patch, {'trend_window'}, 'patch')
        if set(patch) != {'trend_window'}:
            raise LabError(
                'missing_edit',
                'patch must contain exactly the editable field trend_window',
            )
        trend_window = patch['trend_window']
        if isinstance(trend_window, bool) or not isinstance(trend_window, int):
            raise LabError('invalid_parameter_type', 'trend_window must be an integer')
        if trend_window not in ALLOWED_TREND_WINDOWS:
            raise LabError(
                'parameter_out_of_range',
                f'trend_window must be one of {list(ALLOWED_TREND_WINDOWS)}',
            )
        if trend_window == BASELINE_TREND_WINDOW:
            raise LabError(
                'no_change',
                'trend_window=50 is the locked baseline; choose 20 or 100 for a Variant',
            )

        snapshot = self._current_snapshot()
        identity = {
            'template_id': TEMPLATE_ID,
            'template_version': TEMPLATE_VERSION,
            'schema_version': '1',
            'patch': {'trend_window': trend_window},
            'hypothesis_revision_id': experiment['hypothesis_revision']['hypothesis_revision_id'],
            'snapshot_id': snapshot.manifest['snapshot_id'],
            'snapshot_content_sha256': snapshot.manifest['content_sha256'],
            'baseline_result_id': self.baseline_result_id,
            'fixed_policy': FIXED_POLICY,
            'runner_contract_version': RUNNER_CONTRACT_VERSION,
            'runner_fingerprint': runner_fingerprint(),
        }
        variant_id = _hash_id('variant', identity)
        try:
            existing = self.store.read('variants', variant_id)
        except ArtifactNotFoundError:
            existing = None
        if existing is not None:
            return existing, False
        variant = {
            'variant_id': variant_id,
            'experiment_id': experiment_id,
            'template_id': TEMPLATE_ID,
            'template_version': TEMPLATE_VERSION,
            'hypothesis_revision_id': identity['hypothesis_revision_id'],
            'trial_family_id': TRIAL_FAMILY_ID,
            'canonical_patch': {'trend_window': trend_window},
            'actual_config': {'trend_window': trend_window},
            'changed_fields': ['trend_window'],
            'data_snapshot': snapshot.public_identity(),
            'runner': {
                'contract_version': RUNNER_CONTRACT_VERSION,
                'fingerprint': identity['runner_fingerprint'],
            },
            'fixed_policy': copy.deepcopy(FIXED_POLICY),
            'baseline_result_id': self.baseline_result_id,
            'holdout_access': 'not_configured',
            'created_at': utc_now(),
        }
        created = self.store.create_immutable('variants', variant_id, variant)
        return self.store.read('variants', variant_id), created

    def _result_for_variant(self, variant_id: str) -> dict[str, Any] | None:
        for result in self.store.list('results'):
            result_id = result.get('result_id')
            if (
                isinstance(result_id, str)
                and _nested_get(result, 'content.variant.variant_id') == variant_id
            ):
                try:
                    return self._validate_result_document(result, result_id)
                except LabError:
                    continue
        return None

    def _active_run_for_variant(self, variant_id: str) -> dict[str, Any] | None:
        active = [
            run
            for run in self.store.list('runs')
            if (
                run.get('variant_id') == variant_id
                and run.get('status') in {'queued', 'running'}
            )
        ]
        if not active:
            return None
        return sorted(active, key=lambda value: value.get('created_at', ''))[0]

    def submit_run(self, variant_id: str, payload: Any) -> tuple[dict[str, Any], int]:
        body = {} if payload is None else _expect_object(payload, 'request')
        _reject_unknown(body, set(), 'request')
        with self._submission_lock:
            try:
                variant = self.store.read('variants', variant_id)
            except (ArtifactNotFoundError, ValueError) as exc:
                raise LabError('variant_not_found', 'variant not found', status=404) from exc

            snapshot = self._current_snapshot()
            mismatches = self._variant_contract_mismatches(variant, snapshot)
            if mismatches:
                raise LabError(
                    'variant_contract_stale',
                    'Variant identity no longer matches the current frozen Lab contract; '
                    'create a new Variant before running.',
                    status=409,
                    details={'mismatched_fields': mismatches, 'action': 'create_new_variant'},
                )

            run_id = _uuid_id('run')
            now = utc_now()
            run = {
                'run_id': run_id,
                'variant_id': variant_id,
                'hypothesis_revision_id': variant['hypothesis_revision_id'],
                'status': 'queued',
                'outcome': None,
                'actual_config': copy.deepcopy(variant['actual_config']),
                'data_snapshot': copy.deepcopy(variant['data_snapshot']),
                'runner': copy.deepcopy(variant['runner']),
                'fixed_policy': copy.deepcopy(variant['fixed_policy']),
                'baseline_result_id': variant['baseline_result_id'],
                'holdout_access': 'not_configured',
                'created_at': now,
                'started_at': None,
                'finished_at': None,
                'result_id': None,
                'error': None,
                'status_history': [{'status': 'queued', 'at': now}],
            }

            reusable = self._result_for_variant(variant_id)
            if reusable is not None:
                run['status'] = 'skipped'
                run['outcome'] = 'reused'
                run['finished_at'] = now
                run['result_id'] = reusable['result_id']
                run['status_history'].append({
                    'status': 'skipped',
                    'at': now,
                    'reason': 'exact_result_reused',
                })
                self.store.write_run(run_id, run)
                return run, 200

            active = self._active_run_for_variant(variant_id)
            if active is not None:
                response = copy.deepcopy(active)
                response['deduplicated'] = True
                return response, 202

            self.store.write_run(run_id, run)
            try:
                future = self._get_executor().submit(
                    run_trend_validation,
                    snapshot.worker_payload(),
                    copy.deepcopy(variant['actual_config']),
                )
            except Exception as exc:
                finished_at = utc_now()
                run['status'] = 'failed'
                run['outcome'] = 'failed'
                run['finished_at'] = finished_at
                run['error'] = {
                    'code': 'executor_submit_failed',
                    'message': 'validation worker did not accept the run',
                    'exception_type': type(exc).__name__,
                }
                run['status_history'].append({
                    'status': 'failed',
                    'at': finished_at,
                    'reason': 'executor_submit_failed',
                })
                self.store.write_run(run_id, run)
                raise LabError(
                    'worker_unavailable',
                    'validation worker is unavailable; retry with a new run request',
                    status=503,
                    details={'run_id': run_id},
                ) from exc

            started_at = utc_now()
            run['status'] = 'running'
            run['started_at'] = started_at
            run['status_history'].append({'status': 'running', 'at': started_at})
            self.store.write_run(run_id, run)
            future.add_done_callback(
                lambda completed, rid=run_id, var=copy.deepcopy(variant):
                    self._finish_run(rid, var, completed)
            )
            return run, 202

    def _finish_run(self, run_id: str, variant: dict[str, Any], future) -> None:
        with self._submission_lock:
            try:
                evidence = future.result()
                experiment = self.store.read('experiments', variant['experiment_id'])
                content = {
                    'schema_version': RESULT_SCHEMA_VERSION,
                    'artifact_label': 'Research Experiment',
                    'template': {
                        'template_id': TEMPLATE_ID,
                        'template_version': TEMPLATE_VERSION,
                    },
                    'variant': {
                        'variant_id': variant['variant_id'],
                        'canonical_patch': copy.deepcopy(variant['canonical_patch']),
                    },
                    'hypothesis_revision': copy.deepcopy(experiment['hypothesis_revision']),
                    'actual_config': copy.deepcopy(variant['actual_config']),
                    'data_snapshot': copy.deepcopy(variant['data_snapshot']),
                    'runner': copy.deepcopy(variant['runner']),
                    'fixed_policy': copy.deepcopy(variant['fixed_policy']),
                    'evaluation': {
                        'purpose': 'validation',
                        'input_window': copy.deepcopy(
                            variant['data_snapshot']['input_window']
                        ),
                        'validation_window': copy.deepcopy(VALIDATION_WINDOW),
                        'evidence_domain_id': EVIDENCE_DOMAIN_ID,
                    },
                    'baseline_result': {
                        'result_id': variant['baseline_result_id'],
                        'role': 'locked_default',
                    },
                    'seed_policy': 'none',
                    'holdout_access': 'not_configured',
                    'holdout_result': 'not_evaluated',
                    'robustness': 'not_required',
                    'evidence': evidence,
                }
                content_hash = hashlib.sha256(canonical_json(content)).hexdigest()
                result = {
                    'result_id': content_hash,
                    'content_sha256': content_hash,
                    'hash_scope': 'content',
                    'baseline_result_id': variant['baseline_result_id'],
                    'content': content,
                    'provenance': {
                        'producing_run_id': run_id,
                        'built_at': utc_now(),
                    },
                }
                outcome = 'success'
                try:
                    self.store.create_immutable('results', content_hash, result)
                except ArtifactConflictError:
                    existing = self.store.read('results', content_hash)
                    self._validate_result_document(existing, content_hash)
                    if existing.get('content') != content:
                        raise
                    outcome = 'reused_after_compute'
                run = self.store.read('runs', run_id)
                finished_at = utc_now()
                run['status'] = 'success'
                run['outcome'] = outcome
                run['finished_at'] = finished_at
                run['result_id'] = content_hash
                run['status_history'].append({
                    'status': 'success',
                    'at': finished_at,
                    'reason': outcome,
                })
                self.store.write_run(run_id, run)
            except Exception as exc:
                try:
                    run = self.store.read('runs', run_id)
                except Exception:
                    return
                finished_at = utc_now()
                run['status'] = 'failed'
                run['outcome'] = 'failed'
                run['finished_at'] = finished_at
                run['error'] = {
                    'code': 'run_failed',
                    'message': str(exc),
                    'exception_type': type(exc).__name__,
                }
                run['status_history'].append({'status': 'failed', 'at': finished_at})
                self.store.write_run(run_id, run)

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            return self.store.read('runs', run_id)
        except (ArtifactNotFoundError, ValueError) as exc:
            raise LabError('run_not_found', 'run not found', status=404) from exc

    def get_result(self, result_id: str) -> dict[str, Any]:
        if result_id == self.baseline_result_id:
            result = copy.deepcopy(self._baseline)
        else:
            try:
                result = self.store.read('results', result_id)
            except (ArtifactNotFoundError, ValueError) as exc:
                raise LabError('result_not_found', 'result not found', status=404) from exc
        return self._validate_result_document(result, result_id)

    def compare(self, payload: Any) -> dict[str, Any]:
        body = _expect_object(payload, 'request')
        _reject_unknown(
            body,
            {'baseline_result_id', 'candidate_result_id'},
            'request',
        )
        baseline_id = body.get('baseline_result_id')
        candidate_id = body.get('candidate_result_id')
        if baseline_id != self.baseline_result_id:
            raise LabError(
                'baseline_not_locked',
                'baseline_result_id must be the template locked baseline',
                details={'expected': self.baseline_result_id},
            )
        if not isinstance(candidate_id, str):
            raise LabError('invalid_candidate', 'candidate_result_id is required')
        baseline = self.get_result(baseline_id)
        candidate = self.get_result(candidate_id)
        baseline_content = baseline['content']
        candidate_content = candidate['content']

        mismatches = []
        for field in COMPARABILITY_FIELDS:
            left = _nested_get(baseline_content, field)
            right = _nested_get(candidate_content, field)
            if left is None or right is None or left != right:
                mismatches.append({
                    'field': field,
                    'baseline': left,
                    'candidate': right,
                })
        baseline_config = baseline_content.get('actual_config', {})
        candidate_config = candidate_content.get('actual_config', {})
        changed_fields = sorted(
            key
            for key in set(baseline_config) | set(candidate_config)
            if baseline_config.get(key) != candidate_config.get(key)
        )
        if changed_fields != ['trend_window']:
            mismatches.append({
                'field': 'actual_config.changed_fields',
                'baseline': baseline_config,
                'candidate': candidate_config,
            })
        comparable = not mismatches

        metric_names = (
            'signal_switch_count',
            'turnover',
            'trade_count',
            'total_cost',
            'fee_drag',
            'max_drawdown',
            'calmar',
            'cumulative_return',
            'annual_return',
            'benchmark_active_return',
        )
        baseline_metrics = _nested_get(baseline_content, 'evidence.metrics') or {}
        candidate_metrics = _nested_get(candidate_content, 'evidence.metrics') or {}
        deltas = None
        if comparable:
            deltas = {}
            for name in metric_names:
                left = baseline_metrics.get(name)
                right = candidate_metrics.get(name)
                deltas[name] = (
                    round(float(right) - float(left), 8)
                    if isinstance(left, (int, float)) and isinstance(right, (int, float))
                    else None
                )

        return {
            'comparison_id': _hash_id('comparison', {
                'baseline_result_id': baseline_id,
                'candidate_result_id': candidate_id,
                'comparability_fields': COMPARABILITY_FIELDS,
            }),
            'status': 'comparable' if comparable else 'not_comparable',
            'comparable': comparable,
            'mismatches': mismatches,
            'changed_fields': changed_fields,
            'baseline': {
                'result_id': baseline_id,
                'actual_config': baseline_config,
                'metrics': baseline_metrics,
            },
            'candidate': {
                'result_id': candidate_id,
                'actual_config': candidate_config,
                'metrics': candidate_metrics,
            },
            'deltas': deltas,
            'data_snapshot': copy.deepcopy(candidate_content.get('data_snapshot')),
            'evaluation': copy.deepcopy(candidate_content.get('evaluation')),
            'baseline_result_id': self.baseline_result_id,
            'holdout_access': 'not_configured',
            'interpretation': {
                'primary_observable': 'signal_switch_count',
                'verdict': 'interpret_manually',
                'note': '差值只描述同口径 validation 敏感性，不是排名、OOS 或自动晋级结论。',
            },
        }


def default_artifact_root() -> Path:
    project_dir = _PACKAGE_DIR.parent
    return Path(
        os.environ.get(
            'QUANT_LAB_ARTIFACT_DIR',
            str(project_dir / 'data' / 'lab_artifacts'),
        )
    )
