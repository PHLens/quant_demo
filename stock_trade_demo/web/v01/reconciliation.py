"""Cache-only Manual signal reference and reconciliation v1."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from decimal import Decimal
import hashlib
import math
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from web.v01 import manual_ledger
from web.v01.catalog import MANUAL_CAPABILITIES, get_strategy, variants_for
from web.v01.contracts import ApiError, canonical_json, decode_cursor, encode_cursor, parse_limit
from web.v01.snapshot_store import inspect_target, published_signal


VERSION = 'manual_reconciliation_v1'
ACTION_LABELS = {
    'hold': '继续持有', 'enter': '建仓', 'add': '加仓', 'trim': '减仓',
    'exit': '退出', 'flat': '空仓',
}
STATIC_RISKS = {
    'bullish': '研究信号偏多时，台账仓位仍可能因成交、费用和资金流与模型目标产生偏差。',
    'bearish': '研究信号偏空时，市场反弹可能使低仓位相对基准落后。',
    'neutral': '研究信号中性时，方向变化仍可能快于手工台账更新。',
}
STATIC_OPPORTUNITIES = {
    'bullish': '已发布信号可用于观察目标仓位与手工研究记录的差异。',
    'bearish': '低仓位可保留后续研究信号转强时的观察空间。',
    'neutral': '中性状态可用于等待更多已发布信号形成一致方向。',
}


def _require_snapshot(strategy: str, snapshot_id: str | None):
    capability = manual_ledger.require_strategy(strategy)[1]
    source_id = capability[0]
    if snapshot_id is None or not str(snapshot_id).strip():
        raise ApiError(400, 'manual_snapshot_required', 'A non-empty canonical snapshot_id is required.')
    if not str(snapshot_id).startswith('s_'):
        raise ApiError(400, 'manual_snapshot_not_canonical', 'Legacy identities are not valid Manual snapshot references.')
    spec = get_strategy(source_id, strategy)
    saw_different = False
    for variant in variants_for(spec):
        target_state, snapshot = inspect_target(spec, variant)
        if snapshot is not None:
            saw_different = True
            if snapshot.snapshot_id == snapshot_id:
                return capability, snapshot
        elif target_state.cache_state in {'recovering', 'updating'}:
            code = 'cache_recovering' if target_state.cache_state == 'recovering' else 'snapshot_updating'
            raise ApiError(409, code, 'The exact Manual target is not currently readable.')
    if saw_different:
        raise ApiError(409, 'snapshot_changed', 'The referenced snapshot is no longer current.')
    raise ApiError(404, 'snapshot_not_found', 'Canonical snapshot does not exist.')


def _latest(rows: list[dict[str, str]]) -> dict[str, str] | None:
    return max(rows, key=lambda row: (row['date'], row['created_at'], row['record_id'])) if rows else None


def _derive_action(live: float, target: float, tolerance: float = 0.005) -> str:
    delta = target - live
    if abs(delta) < tolerance:
        return 'flat' if target <= tolerance else 'hold'
    if delta > 0:
        return 'enter' if live <= tolerance else 'add'
    return 'exit' if target <= tolerance else 'trim'


def _bounded_strings(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64 or any(
            not isinstance(item, str) or not item for item in value):
        raise ValueError(f'{name} must be a bounded non-empty string array')
    return list(value)


def _risk_context(strategy: str, bias: str) -> dict[str, Any]:
    risks = [STATIC_RISKS[bias]]
    opportunities = [STATIC_OPPORTUNITIES[bias]]
    peer_snapshot_ids = []
    for source_id, peer_strategy, display_name, *_ in MANUAL_CAPABILITIES:
        if peer_strategy == strategy:
            continue
        spec = get_strategy(source_id, peer_strategy)
        variant = next(item for item in variants_for(spec) if item.default)
        _, peer_snapshot = inspect_target(spec, variant)
        if peer_snapshot is None:
            continue
        signal = published_signal(peer_snapshot)
        peer_target = signal.get('target_exposure') if signal else None
        if peer_target is None:
            continue
        peer_target = float(peer_target)
        if bias == 'bullish' and peer_target <= 0.3:
            risks.append(f'{display_name} 已发布目标仓位为 {peer_target:.0%}，与当前偏多视角存在分歧。')
            peer_snapshot_ids.append(peer_snapshot.snapshot_id)
        elif bias == 'bearish' and peer_target >= 0.5:
            opportunities.append(f'{display_name} 已发布目标仓位为 {peer_target:.0%}，可作为交叉观察方向。')
            peer_snapshot_ids.append(peer_snapshot.snapshot_id)

    risk_spec = get_strategy('decision_context', 'risk_signals')
    risk_variant = next(item for item in variants_for(risk_spec) if item.default)
    risk_state, risk_snapshot = inspect_target(risk_spec, risk_variant)
    base_risks = list(risks)
    base_opportunities = list(opportunities)
    if risk_snapshot is None:
        return {
            'risks': risks, 'opportunities': opportunities,
            'risk_signals_as_of': None, 'risk_signals_generated_at': None,
            'risk_context_state': (
                'corrupt' if risk_state.cache_state in {'corrupt', 'artifact_stale'} else 'missing'
            ),
            'risk_context_snapshot_id': None, 'risk_context_fingerprint': None,
            'peer_snapshot_ids': sorted(set(peer_snapshot_ids)),
        }

    try:
        content = risk_snapshot.document.get('content')
        if not isinstance(content, dict):
            raise ValueError('risk context content must be an object')
        by_strategy = content.get('by_strategy') or {}
        if not isinstance(by_strategy, dict):
            raise ValueError('risk context by_strategy must be an object')
        dynamic = by_strategy.get(strategy) or {}
        if not isinstance(dynamic, dict):
            raise ValueError('strategy risk context must be an object')
        dynamic_risks = []
        dynamic_opportunities = []
        if bias == 'bullish':
            dynamic_risks = _bounded_strings(
                dynamic.get('bullish_risks_dynamic'), name='bullish_risks_dynamic')
        elif bias == 'bearish':
            dynamic_opportunities = _bounded_strings(
                dynamic.get('bearish_opportunities_dynamic'),
                name='bearish_opportunities_dynamic')
        as_of = content.get('as_of')
        generated_at = content.get('generated_at')
        if as_of is not None and not isinstance(as_of, str):
            raise ValueError('risk context as_of must be string or null')
        if generated_at is not None and not isinstance(generated_at, str):
            raise ValueError('risk context generated_at must be string or null')
        risks = base_risks + dynamic_risks
        opportunities = base_opportunities + dynamic_opportunities
        if len(risks) > 64 or len(opportunities) > 64:
            raise ApiError(500, 'bounded_collection_overflow', 'Manual risk context exceeds its hard limit.')
        fingerprint = hashlib.sha256(canonical_json(content).encode('utf-8')).hexdigest()
        return {
            'risks': risks, 'opportunities': opportunities,
            'risk_signals_as_of': as_of,
            'risk_signals_generated_at': generated_at,
            'risk_context_state': 'ready',
            'risk_context_snapshot_id': risk_snapshot.snapshot_id,
            'risk_context_fingerprint': fingerprint,
            'peer_snapshot_ids': sorted(set(peer_snapshot_ids)),
        }
    except ApiError:
        raise
    except (TypeError, ValueError):
        return {
            'risks': base_risks, 'opportunities': base_opportunities,
            'risk_signals_as_of': None, 'risk_signals_generated_at': None,
            'risk_context_state': 'corrupt', 'risk_context_snapshot_id': None,
            'risk_context_fingerprint': None,
            'peer_snapshot_ids': sorted(set(peer_snapshot_ids)),
        }


def signal_reference(strategy: str, snapshot_id: str | None) -> dict[str, Any]:
    capability, snapshot = _require_snapshot(strategy, snapshot_id)
    # Ledger access intentionally occurs only after capability + snapshot scope.
    ledger = manual_ledger.load()
    rows = manual_ledger.active(ledger, strategy)
    latest = _latest(rows)
    manual_position = float(Decimal(latest['actual_position'])) if latest and latest['actual_position'] else 0.0
    signal = published_signal(snapshot)
    if signal is None:
        raise ApiError(500, 'snapshot_schema_invalid', 'Manual strategy snapshot lacks a frozen published signal.')
    target = float(signal.get('target_exposure') or 0.0)
    delta = target - manual_position
    action = _derive_action(manual_position, target)
    initial_capital = capability[5]
    lot_size = capability[4]
    reference_price = signal.get('ref_open') or signal.get('ref_close')
    shares = None
    if reference_price and float(reference_price) > 0:
        shares = math.floor(initial_capital * target / float(reference_price) / lot_size) * lot_size
    rationale = (
        f'研究台账参考、不可执行、不会下单：初始资金 {initial_capital:g}，整手 {lot_size:g}，'
        f'参考价 {reference_price if reference_price is not None else "unknown"}，目标股数 {shares if shares is not None else "unknown"}。'
    )
    if target >= 0.5 or action in {'enter', 'add'}:
        bias = 'bullish'
    elif target <= 0.3 or action in {'exit', 'trim', 'flat'}:
        bias = 'bearish'
    else:
        bias = 'neutral'
    context = _risk_context(strategy, bias)
    return {
        'strategy': strategy, 'snapshot_id': snapshot.snapshot_id,
        'published_current_signal': signal,
        'manual_state': {
            'state': 'present' if latest else 'empty', 'latest_manual_position': manual_position,
            'manual_record_id': latest['record_id'] if latest else None,
            'manual_record_date': latest['date'] if latest else None,
        },
        'action_context': {
            'live_position': manual_position, 'live_exposure_delta': delta,
            'live_rebalance_action': action, 'live_rebalance_label': ACTION_LABELS[action],
            'action_rationale': rationale, 'view_bias': bias,
            **context, 'non_executable': True,
        },
    }


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _cache_rows(snapshot) -> list[dict[str, Any]]:
    frame = snapshot.frame
    if frame is None or '交易日期' not in frame.columns:
        return []
    rows = []
    dates = pd.to_datetime(frame['交易日期'], errors='coerce')
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        signal_date = None if pd.isna(dates.iloc[ordinal]) else dates.iloc[ordinal].date().isoformat()
        execution = row.get('execution_date')
        if execution in (None, '') and ordinal + 1 < len(frame) and not pd.isna(dates.iloc[ordinal + 1]):
            execution = dates.iloc[ordinal + 1].date().isoformat()
        elif execution not in (None, ''):
            parsed = pd.to_datetime(execution, errors='coerce')
            execution = None if pd.isna(parsed) else parsed.date().isoformat()
        else:
            execution = None
        rows.append({
            'artifact_row_ordinal': ordinal, 'signal_date': signal_date,
            'execution_date': execution, 'etf_open': row.get('etf_open'),
            'etf_close': row.get('etf_close'), 'strategy_nav': row.get('累积净值'),
            'strategy_target': row.get('target_exposure', row.get('position')),
        })
    return sorted(rows, key=lambda item: (item['execution_date'] or '9999-99-99', item['artifact_row_ordinal']))


def _active_records(ledger, strategy: str) -> list[dict[str, Any]]:
    rows = []
    for row in manual_ledger.active(ledger, strategy):
        rows.append({
            'record_id': row['record_id'], 'date': row['date'],
            'created_at': row['created_at'] or f'{row["date"]}T00:00:00Z',
            'capital': float(Decimal(row['capital'])),
            'exec_price': float(Decimal(row['exec_price'])) if row['exec_price'] else None,
            'shares': float(Decimal(row['shares'])) if row['shares'] else None,
            'actual_position': float(Decimal(row['actual_position'])) if row['actual_position'] else None,
        })
    return sorted(rows, key=lambda item: (item['date'], item['created_at'], item['record_id']))


def _build(snapshot, strategy: str, ledger):
    records = _active_records(ledger, strategy)
    cache_rows = _cache_rows(snapshot)
    execution_keys = [(item['execution_date'], item['artifact_row_ordinal']) for item in cache_rows if item['execution_date']]
    mapped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pending = []
    for record in records:
        candidate = next((index for index, row in enumerate(cache_rows) if row['execution_date'] and row['execution_date'] >= record['date']), None)
        if candidate is None:
            pending.append({'pending_ordinal': len(pending), 'record_id': record['record_id']})
        else:
            mapped[candidate].append(record)
    if not mapped:
        return {
            'state': 'empty' if not records else ('pending_cache' if pending else 'ready'),
            'initial_nav': 1.0, 'initial_position': 0.0, 'approximate_mode': False,
            'pending': pending, 'series': [], 'applied': {},
            'final_strategy_nav': None, 'final_manual_nav': None,
        }
    first_index = min(mapped)
    first_row = cache_rows[first_index]
    first_records = mapped[first_index]
    baseline = first_records[0]
    approximate = baseline['exec_price'] is None
    event_price = baseline['exec_price'] or _finite_positive(first_row['etf_open'])
    if event_price is None:
        raise ApiError(422, 'reconciliation_input_missing', 'Baseline event price is missing.', date=first_row['execution_date'], fields=['etf_open'])
    capital = baseline['capital']
    if baseline['exec_price'] is not None:
        shares = baseline['shares']
        cash = capital - event_price * shares
    else:
        shares = capital * baseline['actual_position'] / event_price
        cash = capital * (1 - baseline['actual_position'])
    initial_position = shares * event_price / capital
    manual_nav = 1.0
    last_mark = event_price
    start_strategy_nav = _finite_positive(first_row['strategy_nav'])
    if start_strategy_nav is None:
        raise ApiError(422, 'reconciliation_input_missing', 'Baseline strategy NAV is missing.', date=first_row['execution_date'], fields=['strategy_nav'])

    def equity(price: float) -> float:
        return cash + shares * price

    def mark(price: Any, row_date: str, field: str):
        nonlocal manual_nav, last_mark
        q = _finite_positive(price)
        denominator = equity(last_mark)
        if q is None or denominator <= 0 or not math.isfinite(denominator):
            raise ApiError(422, 'reconciliation_input_missing', 'Required mark input is missing or nonpositive.', date=row_date, fields=[field])
        manual_nav *= equity(q) / denominator
        last_mark = q

    series = []
    applied: dict[str, list[dict[str, Any]]] = {}
    for row_index in range(first_index, len(cache_rows)):
        cache_row = cache_rows[row_index]
        if not cache_row['execution_date']:
            continue
        row_records = mapped.get(row_index, [])
        flow_total = 0.0
        if row_index != first_index:
            mark(cache_row['etf_open'], cache_row['execution_date'], 'etf_open')
        applications = []
        start_at = 1 if row_index == first_index else 0
        for record in row_records[start_at:]:
            q = record['exec_price'] or _finite_positive(cache_row['etf_open'])
            if q is None:
                raise ApiError(422, 'reconciliation_input_missing', 'Manual event price is missing.', date=cache_row['execution_date'], fields=['etf_open'])
            mark(q, cache_row['execution_date'], 'exec_price' if record['exec_price'] is not None else 'etf_open')
            external_flow = record['capital'] - equity(q)
            flow_total += external_flow
            if record['exec_price'] is not None:
                shares = record['shares']
                cash = record['capital'] - q * shares
            else:
                approximate = True
                shares = record['capital'] * record['actual_position'] / q
                cash = record['capital'] * (1 - record['actual_position'])
            applications.append({'application_ordinal': len(applications), 'record_id': record['record_id']})
        if row_index == first_index:
            applications.insert(0, {'application_ordinal': 0, 'record_id': baseline['record_id']})
            for index, item in enumerate(applications):
                item['application_ordinal'] = index
        close = _finite_positive(cache_row['etf_close'])
        if close is None:
            raise ApiError(422, 'reconciliation_input_missing', 'etf_close is missing or nonpositive.', date=cache_row['execution_date'], fields=['etf_close'])
        mark(close, cache_row['execution_date'], 'etf_close')
        current_equity = equity(close)
        reconciliation_id_placeholder = 'pending'
        row_id = None
        series.append({
            'reconciliation_row_ordinal': len(series), '_cache_row': cache_row,
            'date': cache_row['execution_date'], 'signal_date': cache_row['signal_date'],
            'strategy_nav': float(cache_row['strategy_nav']) / start_strategy_nav,
            'manual_nav': manual_nav, 'actual_position': shares * close / current_equity,
            'strategy_target': float(cache_row['strategy_target']) if cache_row['strategy_target'] is not None else None,
            'mark_price': close, 'mark_date': cache_row['execution_date'], 'share_units': shares,
            'cash_balance': cash, 'equity': current_equity, 'external_flow': flow_total,
            'applied_record_count': len(applications), '_applications': applications,
        })
    return {
        'state': 'pending_cache' if pending else 'ready', 'initial_nav': 1.0,
        'initial_position': initial_position, 'approximate_mode': approximate, 'pending': pending,
        'series': series, 'applied': applied,
        'final_strategy_nav': series[-1]['strategy_nav'] if series else None,
        'final_manual_nav': series[-1]['manual_nav'] if series else None,
    }


def reconciliation_id(strategy: str, snapshot_id: str, ledger_version: str) -> str:
    return hashlib.sha256(canonical_json({'version': VERSION, 'strategy': strategy, 'snapshot_id': snapshot_id, 'ledger_version': ledger_version}).encode('utf-8')).hexdigest()


def _canonical_urls(strategy: str, snapshot_id: str, rid: str, resource: str, row_id: str | None = None) -> str:
    params = {'strategy': strategy, 'snapshot_id': snapshot_id, 'reconciliation_id': rid, 'resource': resource}
    if row_id is not None:
        params['row_id'] = row_id
    return '/api/r0/manual-records/reconciliation?' + urlencode(params)


def _finalize_rows(result: dict[str, Any], rid: str, strategy: str, snapshot_id: str):
    applied = {}
    for item in result['series']:
        cache_row = item.pop('_cache_row')
        applications = item.pop('_applications')
        row_id = 'r_' + hashlib.sha256(canonical_json({'reconciliation_id': rid, 'execution_date': cache_row['execution_date'], 'artifact_row_ordinal': cache_row['artifact_row_ordinal']}).encode('utf-8')).hexdigest()
        item['row_id'] = row_id
        item['applied_records_url'] = _canonical_urls(strategy, snapshot_id, rid, 'applied_records', row_id)
        applied[row_id] = applications
    result['applied'] = applied


def _page(items, *, resource: str, strategy: str, snapshot_id: str, rid: str, ledger_version: str, row_id: str | None, cursor: str | None, limit_raw: str | None, sort_key):
    limit = parse_limit(limit_raw)
    decoded = decode_cursor(cursor)
    binding = {'strategy': strategy, 'snapshot_id': snapshot_id, 'reconciliation_id': rid, 'ledger_version': ledger_version, 'resource': resource, 'row_id': row_id}
    start = 0
    if decoded is not None:
        other = decoded.get('binding') or {}
        for key in ('strategy', 'snapshot_id', 'reconciliation_id', 'resource', 'row_id'):
            if other.get(key) != binding.get(key):
                raise ApiError(400, 'invalid_cursor', 'Cursor belongs to another reconciliation collection.')
        if other.get('ledger_version') != ledger_version:
            raise ApiError(409, 'manual_ledger_changed', 'Manual ledger changed between reconciliation pages.', ledger_version=ledger_version)
        last = decoded.get('last')
        for index, item in enumerate(items):
            if list(sort_key(item)) == last:
                start = index + 1
                break
        else:
            raise ApiError(400, 'invalid_cursor', 'Reconciliation cursor position does not exist.')
    page = items[start:start + limit]
    next_cursor = None
    if start + len(page) < len(items) and page:
        next_cursor = encode_cursor({'binding': binding, 'last': list(sort_key(page[-1]))})
    return page, len(items), next_cursor


def _validate_collection_identity(
    args: dict[str, str], *, strategy: str, snapshot_id: str,
    ledger_version: str, current_reconciliation_id: str,
) -> None:
    submitted = args['reconciliation_id']
    decoded = decode_cursor(args.get('cursor'))
    if decoded is not None:
        other = decoded.get('binding') or {}
        expected = {
            'strategy': strategy, 'snapshot_id': snapshot_id,
            'reconciliation_id': submitted,
            'resource': args.get('resource'), 'row_id': args.get('row_id'),
        }
        for key, value in expected.items():
            if other.get(key) != value:
                raise ApiError(400, 'invalid_cursor', 'Cursor belongs to another reconciliation collection.')
        if other.get('ledger_version') != ledger_version:
            raise ApiError(409, 'manual_ledger_changed', 'Manual ledger changed between reconciliation pages.', ledger_version=ledger_version)
    if submitted != current_reconciliation_id:
        raise ApiError(409, 'manual_reconciliation_changed', 'Reconciliation identity changed.', reconciliation_id=current_reconciliation_id)


def response(args: dict[str, str]) -> dict[str, Any]:
    strategy = args.get('strategy')
    snapshot_id = args.get('snapshot_id')
    _, snapshot = _require_snapshot(strategy, snapshot_id)
    resource = args.get('resource') or 'summary'
    if resource not in {'summary', 'series', 'pending_records', 'applied_records'}:
        raise ApiError(400, 'invalid_params', 'Unknown reconciliation resource.')
    if resource == 'summary' and any(key in args for key in ('reconciliation_id', 'cursor', 'limit', 'row_id')):
        raise ApiError(400, 'invalid_params', 'Summary rejects collection parameters.')
    if resource != 'summary' and not args.get('reconciliation_id'):
        raise ApiError(400, 'manual_reconciliation_id_required', 'reconciliation_id is required for collections.')
    ledger = manual_ledger.load()
    rid = reconciliation_id(strategy, snapshot.snapshot_id, ledger.version)
    if resource != 'summary':
        _validate_collection_identity(
            args, strategy=strategy, snapshot_id=snapshot.snapshot_id,
            ledger_version=ledger.version, current_reconciliation_id=rid,
        )
    result = _build(snapshot, strategy, ledger)
    _finalize_rows(result, rid, strategy, snapshot.snapshot_id)
    if resource == 'summary':
        return {
            'strategy': strategy, 'snapshot_id': snapshot.snapshot_id, 'ledger_version': ledger.version,
            'reconciliation_id': rid, 'state': result['state'], 'initial_nav': result['initial_nav'],
            'initial_position': result['initial_position'], 'approximate_mode': result['approximate_mode'],
            'pending_total': len(result['pending']), 'pending_records_url': _canonical_urls(strategy, snapshot.snapshot_id, rid, 'pending_records'),
            'series_total': len(result['series']), 'series_url': _canonical_urls(strategy, snapshot.snapshot_id, rid, 'series'),
            'final_strategy_nav': result['final_strategy_nav'], 'final_manual_nav': result['final_manual_nav'],
        }
    submitted = args['reconciliation_id']
    row_id = args.get('row_id')
    if resource == 'applied_records':
        if not row_id:
            raise ApiError(400, 'reconciliation_row_required', 'row_id is required for applied_records.')
        if row_id not in result['applied']:
            raise ApiError(404, 'reconciliation_row_not_found', 'row_id does not exist.')
        items = result['applied'][row_id]
        sort_key = lambda item: (item['application_ordinal'], item['record_id'])
    elif resource == 'pending_records':
        if row_id:
            raise ApiError(400, 'invalid_params', 'row_id applies only to applied_records.')
        items = result['pending']
        sort_key = lambda item: (item['pending_ordinal'], item['record_id'])
    else:
        if row_id:
            raise ApiError(400, 'invalid_params', 'row_id applies only to applied_records.')
        items = result['series']
        sort_key = lambda item: (item['reconciliation_row_ordinal'], item['row_id'])
    page, total, next_cursor = _page(items, resource=resource, strategy=strategy, snapshot_id=snapshot.snapshot_id, rid=submitted, ledger_version=ledger.version, row_id=row_id, cursor=args.get('cursor'), limit_raw=args.get('limit'), sort_key=sort_key)
    payload = {
        'strategy': strategy, 'snapshot_id': snapshot.snapshot_id, 'ledger_version': ledger.version,
        'reconciliation_id': rid, 'resource': resource, 'items': page, 'total': total,
        'next_cursor': next_cursor,
    }
    if resource == 'applied_records':
        payload['row_id'] = row_id
    return payload
