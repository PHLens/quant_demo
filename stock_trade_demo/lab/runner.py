"""Deterministic Lab-only CSI1000 Trend 101 computation.

The runner receives materialized rows.  It performs no file, network, cache,
registry, Flask, or ``web.state`` access.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from lab.contracts import (
    ALLOWED_TREND_WINDOWS,
    FIXED_POLICY,
    RUNNER_CONTRACT_VERSION,
    VALIDATION_WINDOW,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


def runner_fingerprint() -> str:
    """Fingerprint code plus the frozen execution contract."""
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256()
    digest.update(source)
    digest.update(_canonical_json({
        'contract_version': RUNNER_CONTRACT_VERSION,
        'fixed_policy': FIXED_POLICY,
        'allowed_windows': ALLOWED_TREND_WINDOWS,
    }))
    return digest.hexdigest()


def _round(value: float | None, digits: int = 8) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _max_drawdown(navs: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for nav in navs:
        peak = max(peak, nav)
        if peak > 0:
            worst = min(worst, nav / peak - 1.0)
    return worst


def _execute_buy(cash: float, open_price: float, costs: dict[str, float]) -> tuple[float, float, float, float]:
    execution_price = open_price * (1.0 + costs['slippage_bps'] / 10000.0)
    rate = costs['buy_cost'] + costs['commission_rate'] + costs['transfer_fee_rate']
    estimated_notional = cash / (1.0 + rate)
    commission = max(estimated_notional * costs['commission_rate'], costs['commission_min'])
    notional = max(
        0.0,
        (cash - commission) / (1.0 + costs['buy_cost'] + costs['transfer_fee_rate']),
    )
    aggregate_cost = notional * costs['buy_cost']
    transfer = notional * costs['transfer_fee_rate']
    total_fee = aggregate_cost + commission + transfer
    shares = notional / execution_price if execution_price > 0 else 0.0
    remaining_cash = max(0.0, cash - notional - total_fee)
    slippage_cost = shares * max(0.0, execution_price - open_price)
    return remaining_cash, shares, total_fee + slippage_cost, notional


def _execute_sell(shares: float, open_price: float, costs: dict[str, float]) -> tuple[float, float, float]:
    execution_price = open_price * (1.0 - costs['slippage_bps'] / 10000.0)
    notional = shares * execution_price
    commission = max(notional * costs['commission_rate'], costs['commission_min'])
    aggregate_cost = notional * costs['sell_cost']
    stamp = notional * costs['stamp_tax_rate']
    transfer = notional * costs['transfer_fee_rate']
    slippage_cost = shares * max(0.0, open_price - execution_price)
    total_fee = aggregate_cost + commission + stamp + transfer + slippage_cost
    return max(0.0, notional - aggregate_cost - commission - stamp - transfer), total_fee, notional


def run_trend_validation(snapshot_payload: dict[str, Any], actual_config: dict[str, Any]) -> dict[str, Any]:
    """Run one deterministic validation from explicitly injected snapshot rows."""
    if set(actual_config) != {'trend_window'}:
        raise ValueError('runner accepts exactly one mapped field: trend_window')
    trend_window = actual_config['trend_window']
    if isinstance(trend_window, bool) or trend_window not in ALLOWED_TREND_WINDOWS:
        raise ValueError(f'unsupported trend_window: {trend_window!r}')

    rows = snapshot_payload.get('rows')
    identity = snapshot_payload.get('identity')
    if not isinstance(rows, list) or not rows or not isinstance(identity, dict):
        raise ValueError('materialized snapshot payload is required')

    rolling: deque[float] = deque()
    rolling_sum = 0.0
    prepared: list[dict[str, Any]] = []
    previous_date = ''
    for raw in rows:
        date = raw['date']
        if date <= previous_date:
            raise ValueError('snapshot rows must be unique and ascending')
        previous_date = date
        close = float(raw['index_close'])
        if close <= 0:
            raise ValueError(f'invalid index close on {date}')
        rolling.append(close)
        rolling_sum += close
        if len(rolling) > trend_window:
            rolling_sum -= rolling.popleft()
        trend_line = rolling_sum / trend_window if len(rolling) == trend_window else None
        signal = int(trend_line is not None and close > trend_line)
        prepared.append({
            **raw,
            'trend_line': trend_line,
            'signal': signal,
        })

    validation = [
        row for row in prepared
        if VALIDATION_WINDOW['start'] <= row['date'] <= VALIDATION_WINDOW['end']
    ]
    if not validation:
        raise ValueError('snapshot has no rows inside the fixed validation window')

    policy = FIXED_POLICY
    costs = policy['costs']
    initial_capital = float(policy['initial_capital'])
    cash = initial_capital
    shares = 0.0
    current_target = 0
    previous_signal = 0
    previous_etf_close: float | None = None
    buy_bar_index: int | None = None
    pending_attempts = 0
    total_cost = 0.0
    total_trade_notional = 0.0
    signal_switch_count = 0
    trades: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    navs: list[float] = []

    for bar_index, row in enumerate(validation):
        signal = int(row['signal'])
        if signal != previous_signal:
            signal_switch_count += 1

        # close(t) signal becomes the desired exposure at open(t+1).
        desired_target = previous_signal
        etf_open = float(row['etf_open'])
        etf_close = float(row['etf_close'])
        if etf_open <= 0 or etf_close <= 0:
            raise ValueError(f'invalid ETF execution bar on {row["date"]}')

        if cash > 0 and costs['cash_interest_rate'] > 0:
            cash += cash * costs['cash_interest_rate'] / 252.0

        action = 'hold' if current_target else 'flat'
        notional = 0.0
        fee = 0.0
        blocked = False
        limit_pct = float(policy['limit_pct'])
        if desired_target != current_target:
            if desired_target == 1:
                blocked = (
                    previous_etf_close is not None
                    and etf_open >= previous_etf_close * (1.0 + limit_pct - 1e-10)
                )
                if not blocked:
                    cash, shares, fee, notional = _execute_buy(cash, etf_open, costs)
                    if shares <= 0:
                        raise ValueError(f'buy produced zero shares on {row["date"]}')
                    current_target = 1
                    buy_bar_index = bar_index
                    action = 'buy'
            else:
                blocked = (
                    previous_etf_close is not None
                    and etf_open <= previous_etf_close * (1.0 - limit_pct + 1e-10)
                )
                t_plus_one_ready = buy_bar_index is None or bar_index > buy_bar_index
                if not blocked and t_plus_one_ready:
                    proceeds, fee, notional = _execute_sell(shares, etf_open, costs)
                    cash += proceeds
                    shares = 0.0
                    current_target = 0
                    buy_bar_index = None
                    action = 'sell'
                elif not t_plus_one_ready:
                    blocked = True

            if blocked:
                pending_attempts += 1
                action = 'blocked'
                if pending_attempts > int(policy['limit_max_delay_days']):
                    action = 'dropped'
                    pending_attempts = 0
            else:
                pending_attempts = 0

        if action in {'buy', 'sell'}:
            total_cost += fee
            total_trade_notional += notional
            trades.append({
                'date': row['date'],
                'action': action,
                'signal_date': validation[bar_index - 1]['date'] if bar_index > 0 else None,
                'etf_open': _round(etf_open, 6),
                'notional': _round(notional, 4),
                'fee': _round(fee, 4),
                'target_exposure_after': current_target,
            })

        equity = cash + shares * etf_close
        nav = equity / initial_capital
        navs.append(nav)
        trace.append({
            'date': row['date'],
            'index_close': _round(float(row['index_close']), 4),
            'trend_line': _round(row['trend_line'], 4),
            'signal': signal,
            'executed_target': desired_target,
            'actual_exposure': current_target,
            'etf_open': _round(etf_open, 6),
            'etf_close': _round(etf_close, 6),
            'trade_action': action,
            'trade_notional': _round(notional, 4),
            'fee': _round(fee, 4),
            'nav': _round(nav, 8),
        })
        previous_signal = signal
        previous_etf_close = etf_close

    daily_returns = [
        navs[index] / navs[index - 1] - 1.0
        for index in range(1, len(navs))
        if navs[index - 1] > 0
    ]
    final_nav = navs[-1]
    cumulative_return = final_nav - 1.0
    years = max(len(validation) / 252.0, 1.0 / 252.0)
    annual_return = final_nav ** (1.0 / years) - 1.0
    max_drawdown = _max_drawdown(navs)
    volatility = statistics.pstdev(daily_returns) * math.sqrt(252.0) if len(daily_returns) > 1 else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else None
    benchmark_return = (
        float(validation[-1]['index_close']) / float(validation[0]['index_close']) - 1.0
    )
    sample_status = 'sufficient' if len(validation) >= 252 and len(trades) >= 2 else 'insufficient'

    return {
        'metrics': {
            'validation_bars': len(validation),
            'trade_count': len(trades),
            'signal_switch_count': signal_switch_count,
            'turnover': _round(total_trade_notional / initial_capital),
            'total_cost': _round(total_cost, 4),
            'fee_drag': _round(total_cost / initial_capital),
            'final_nav': _round(final_nav),
            'cumulative_return': _round(cumulative_return),
            'annual_return': _round(annual_return),
            'max_drawdown': _round(max_drawdown),
            'annualized_volatility': _round(volatility),
            'calmar': _round(calmar),
            'benchmark_return': _round(benchmark_return),
            'benchmark_active_return': _round(cumulative_return - benchmark_return),
            'average_exposure': _round(
                sum(item['actual_exposure'] for item in trace) / len(trace)
            ),
        },
        'validity': {
            'status': sample_status,
            'reproducible': True,
            'snapshot_verified': True,
            'validation_bars': len(validation),
            'warnings': [] if sample_status == 'sufficient' else [
                'validation bars or completed trades are below the teaching threshold',
            ],
        },
        'trades': trades,
        'trace': trace,
    }
