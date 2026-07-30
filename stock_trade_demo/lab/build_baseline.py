"""Offline-only builder for the packaged Trend 101 baseline Result.

This module is never imported by the Web blueprint or worker.  It exists so a
reviewed runner/snapshot change can rebuild the locked baseline reproducibly:

    PYTHONPATH=stock_trade_demo python -m lab.build_baseline --check
    PYTHONPATH=stock_trade_demo python -m lab.build_baseline --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.contracts import (
    BASELINE_TREND_WINDOW,
    EVIDENCE_DOMAIN_ID,
    FIXED_POLICY,
    RESULT_SCHEMA_VERSION,
    RUNNER_CONTRACT_VERSION,
    TEMPLATE_ID,
    TEMPLATE_VERSION,
    VALIDATION_WINDOW,
)
from lab.runner import run_trend_validation, runner_fingerprint
from lab.snapshot import load_materialized_snapshot
from lab.store import canonical_json

DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'baselines' / 'csi1000_trend_101_v1.json'


def build_baseline_document(*, built_at: str | None = None) -> dict[str, Any]:
    snapshot = load_materialized_snapshot()
    evidence = run_trend_validation(
        snapshot.worker_payload(),
        {'trend_window': BASELINE_TREND_WINDOW},
    )
    content = {
        'schema_version': RESULT_SCHEMA_VERSION,
        'artifact_label': 'Worked Example',
        'template': {
            'template_id': TEMPLATE_ID,
            'template_version': TEMPLATE_VERSION,
        },
        'variant': {
            'variant_id': 'variant_baseline_csi1000_trend_101_v1',
            'canonical_patch': {'trend_window': BASELINE_TREND_WINDOW},
        },
        'hypothesis_revision': {
            'hypothesis_revision_id': 'hypothesis_worked_example_csi1000_trend_101_v1',
            'revision': 1,
            'statement': '以趋势窗口 50 作为锁定教学基线，供单变量 validation 敏感性比较。',
            'primary_observable': 'signal_switch_count',
            'expected_direction': 'no_material_change',
            'falsification_condition': '该基线仅是参照，不承担候选有效性或 OOS 结论。',
            'validation_window': VALIDATION_WINDOW,
            'registered_at': '2026-07-30T00:00:00+00:00',
        },
        'actual_config': {'trend_window': BASELINE_TREND_WINDOW},
        'data_snapshot': snapshot.public_identity(),
        'runner': {
            'contract_version': RUNNER_CONTRACT_VERSION,
            'fingerprint': runner_fingerprint(),
        },
        'fixed_policy': FIXED_POLICY,
        'evaluation': {
            'purpose': 'validation',
            'input_window': snapshot.public_identity()['input_window'],
            'validation_window': VALIDATION_WINDOW,
            'evidence_domain_id': EVIDENCE_DOMAIN_ID,
        },
        'baseline_result': {'role': 'self_locked_baseline'},
        'seed_policy': 'none',
        'holdout_access': 'not_configured',
        'holdout_result': 'not_evaluated',
        'robustness': 'not_required',
        'evidence': evidence,
    }
    content_hash = hashlib.sha256(canonical_json(content)).hexdigest()
    return {
        'result_id': content_hash,
        'content_sha256': content_hash,
        'hash_scope': 'content',
        'baseline_result_id': content_hash,
        'content': content,
        'provenance': {
            'producing_run_id': 'run_baseline_csi1000_trend_101_v1',
            'built_at': built_at or datetime.now(timezone.utc).isoformat(),
            'build_mode': 'offline_packaged_baseline',
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generated = build_baseline_document()
    if args.check:
        with args.output.open('r', encoding='utf-8') as handle:
            packaged = json.load(handle)
        if packaged.get('result_id') != generated['result_id']:
            print(
                f"stale baseline: packaged={packaged.get('result_id')} "
                f"generated={generated['result_id']}"
            )
            return 1
        print(f"baseline current: {generated['result_id']}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            generated,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + '\n',
        encoding='utf-8',
    )
    print(f"wrote {args.output}: {generated['result_id']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
