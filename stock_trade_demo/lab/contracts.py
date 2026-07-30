"""Frozen contract for the first validation-only teaching template."""
from __future__ import annotations

TEMPLATE_ID = 'csi1000_trend_101'
TEMPLATE_VERSION = '1.0.0'
SCHEMA_VERSION = '1'
RUNNER_CONTRACT_VERSION = 'lab_timing_snapshot_v1'
RESULT_SCHEMA_VERSION = 'lab_result_v1'
RESEARCH_FAMILY_ID = 'timing_csi1000_trend_101'
EVIDENCE_DOMAIN_ID = 'cn_csi1000_daily_v1'
TRIAL_FAMILY_ID = f'{RESEARCH_FAMILY_ID}:{EVIDENCE_DOMAIN_ID}'

BASELINE_TREND_WINDOW = 50
ALLOWED_TREND_WINDOWS = (20, 50, 100)

INPUT_WINDOW = {
    'start': '2023-07-03',
    'end': '2025-12-31',
    'purpose': 'indicator_warmup_plus_validation',
}
VALIDATION_WINDOW = {
    'start': '2024-01-02',
    'end': '2025-12-31',
    'purpose': 'validation',
}

FIXED_POLICY = {
    'market': 'CN',
    'universe': 'CSI1000',
    'signal_source': {
        'instrument': 'CSI1000 index',
        'field': 'close',
    },
    'execution_instrument': {
        'code': '510980',
        'name': '中证1000ETF',
        'price_adjustment': 'qfq',
    },
    'benchmark': {
        'id': 'csi1000_buy_and_hold',
        'name': 'CSI1000 buy-and-hold',
        'price_field': 'index_close',
    },
    'exposure': {
        'mode': 'binary',
        'flat': 0.0,
        'invested': 1.0,
    },
    'initial_capital': 50000.0,
    'costs': {
        # These are the same separately reported fields used by the legacy
        # timing replay contract.  buy/sell cost is applied to notional in
        # addition to the granular fee components below.
        'buy_cost': 0.001,
        'sell_cost': 0.001,
        'slippage_bps': 5.0,
        'cash_interest_rate': 0.015,
        'commission_rate': 0.0001,
        'commission_min': 5.0,
        'stamp_tax_rate': 0.0,
        'transfer_fee_rate': 0.00001,
    },
    'clock': {
        'signal': 'index close(t)',
        'execute': 'next trading day ETF open(t+1)',
        'mark': 'next trading day ETF close(t+1)',
        'timezone': 'Asia/Shanghai',
        'calendar': 'CN_A_share_trading_days',
    },
    'settlement': 'T+1',
    'limit_pct': 0.10,
    'limit_max_delay_days': 5,
    'seed_policy': 'none',
    'input_window': INPUT_WINDOW,
    'validation_window': VALIDATION_WINDOW,
    'holdout_access': 'not_configured',
}

EDITABLE_SCHEMA = {
    'schema_version': SCHEMA_VERSION,
    'max_changed_fields': 1,
    'fields': [
        {
            'key': 'trend_window',
            'type': 'integer',
            'enum': list(ALLOWED_TREND_WINDOWS),
            'default': BASELINE_TREND_WINDOW,
            'unit': 'trading_days',
            'label': '趋势窗口',
            'help': '收盘价高于 N 日趋势线时持有，否则空仓。',
            'server_mapping': 'trend_window',
        },
    ],
}

COMPARABILITY_FIELDS = (
    'template.template_id',
    'template.template_version',
    'data_snapshot.snapshot_id',
    'data_snapshot.content_sha256',
    'runner.contract_version',
    'runner.fingerprint',
    'fixed_policy.benchmark',
    'fixed_policy.costs',
    'fixed_policy.clock',
    'fixed_policy.initial_capital',
    'fixed_policy.settlement',
    'fixed_policy.limit_pct',
    'fixed_policy.limit_max_delay_days',
    'evaluation.validation_window',
    'seed_policy',
)


def public_template(*, snapshot: dict, runner_fingerprint: str,
                    baseline_result_id: str, trial_count: int) -> dict:
    """Return the versioned allowlist and all locked policy fields."""
    return {
        'template_id': TEMPLATE_ID,
        'template_version': TEMPLATE_VERSION,
        'schema_version': SCHEMA_VERSION,
        'runner_contract_version': RUNNER_CONTRACT_VERSION,
        'runner_fingerprint': runner_fingerprint,
        'research_family_id': RESEARCH_FAMILY_ID,
        'evidence_domain_id': EVIDENCE_DOMAIN_ID,
        'trial_family_id': TRIAL_FAMILY_ID,
        'trial_count': int(trial_count),
        'name': 'CSI1000 Trend 101',
        'module': 'timing',
        'artifact_label': 'Worked Example / Research Experiment',
        'formula': 'position(t) = 1 if CSI1000 close(t) > MA_N(t), else 0',
        'question': '趋势窗口从 50 改为 20 或 100，是否按预期改变信号切换次数与换手？',
        'primary_observable': 'signal_switch_count',
        'secondary_observables': [
            'turnover',
            'max_drawdown',
            'calmar',
            'total_cost',
            'benchmark_active_return',
        ],
        'editable_schema': EDITABLE_SCHEMA,
        'fixed_policy': FIXED_POLICY,
        'snapshot': snapshot,
        'baseline_result_id': baseline_result_id,
        'holdout_access': 'not_configured',
        'promotion': 'manual_review_only',
    }
