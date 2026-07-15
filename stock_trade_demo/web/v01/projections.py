"""Pure, source-specific projections from one already-loaded snapshot."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
import re
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from web.v01.catalog import get_strategy, variants_for
from web.v01.contracts import ApiError, nullable_json_scalar
from web.v01.snapshot_store import (
    LoadedSnapshot, filter_frame, inspect_target, published_signal,
    snapshot_date_bounds,
)


def _value(row: Any, key: str, fallback: Any = None) -> Any:
    try:
        value = row.get(key, fallback)
    except AttributeError:
        value = fallback
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (pd.Timestamp, date)):
        return pd.to_datetime(value).date().isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _day(value: Any) -> str | None:
    try:
        parsed = pd.to_datetime(value, errors='coerce')
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _nav_column(frame: pd.DataFrame) -> str | None:
    for key in ('累积净值', '累计净值', 'nav', 'value'):
        if key in frame.columns:
            return key
    return None


def _return_column(snapshot: LoadedSnapshot, frame: pd.DataFrame) -> str | None:
    candidates = ('选股下周期涨跌幅', 'strategy_return', 'return') if snapshot.spec.source_id == 'selection' else ('strategy_return', 'return', '选股下周期涨跌幅')
    return next((key for key in candidates if key in frame.columns), None)


def _metric_payload(snapshot: LoadedSnapshot, frame: pd.DataFrame) -> dict[str, Any]:
    nav_key = _nav_column(frame)
    values = pd.to_numeric(frame[nav_key], errors='coerce').dropna() if nav_key else pd.Series(dtype=float)
    returns_key = _return_column(snapshot, frame)
    returns = pd.to_numeric(frame[returns_key], errors='coerce').fillna(0.0) if returns_key else pd.Series(dtype=float)
    initial = _number(frame.attrs.get('initial_capital')) or (100000.0 if snapshot.spec.source_id == 'selection' else 50000.0)
    final_nav = float(values.iloc[-1]) if len(values) else None
    total_return = final_nav - 1.0 if final_nav is not None else None
    dates = pd.to_datetime(frame.get('交易日期'), errors='coerce').dropna() if '交易日期' in frame.columns else pd.Series(dtype='datetime64[ns]')
    annual = None
    if final_nav is not None and final_nav > 0 and len(dates) >= 2:
        days = max((dates.max() - dates.min()).days, 1)
        annual = final_nav ** (365.0 / days) - 1.0
    max_drawdown = max_start = max_end = None
    if len(values):
        peaks = values.cummax()
        drawdown = values / peaks - 1.0
        end_index = drawdown.idxmin()
        max_drawdown = float(drawdown.loc[end_index])
        end_position = list(values.index).index(end_index)
        prior = values.iloc[:end_position + 1]
        start_index = prior.idxmax()
        if '交易日期' in frame.columns:
            max_start = _day(frame.loc[start_index, '交易日期'])
            max_end = _day(frame.loc[end_index, '交易日期'])
    calmar = annual / abs(max_drawdown) if annual is not None and max_drawdown not in (None, 0) else None
    attrs_metrics = frame.attrs.get('metrics') if isinstance(frame.attrs.get('metrics'), dict) else {}
    return {
        'cumulative_return': final_nav,
        'annual_return': annual,
        'max_drawdown': max_drawdown,
        'max_dd_start': max_start,
        'max_dd_end': max_end,
        'calmar_ratio': calmar,
        'final_capital': initial * final_nav if final_nav is not None else None,
        'total_return_pct': total_return * 100.0 if total_return is not None else None,
        'total_pnl': initial * total_return if total_return is not None else None,
        'beta': _number(attrs_metrics.get('Beta') or attrs_metrics.get('beta')),
        'annual_alpha': _number(attrs_metrics.get('annual_alpha')),
        'information_ratio': _number(attrs_metrics.get('information_ratio') or attrs_metrics.get('信息比率')),
        'r_squared': _number(attrs_metrics.get('r_squared') or attrs_metrics.get('R-squared')),
        'up_capture': _number(attrs_metrics.get('up_capture')),
        'down_capture': _number(attrs_metrics.get('down_capture')),
        'avg_exposure': _number(pd.to_numeric(frame.get('target_exposure'), errors='coerce').mean()) if 'target_exposure' in frame.columns else None,
        'rebalance_count': int((pd.to_numeric(frame.get('trade_quantity'), errors='coerce').fillna(0).abs() > 1e-8).sum()) if 'trade_quantity' in frame.columns else None,
        'fee_ratio': _number(pd.to_numeric(frame.get('trade_cost'), errors='coerce').fillna(0).sum()) if 'trade_cost' in frame.columns else None,
    }


def _dates(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if '交易日期' not in frame.columns or len(frame) == 0:
        return None, None
    values = pd.to_datetime(frame['交易日期'], errors='coerce').dropna()
    return (values.min().date().isoformat(), values.max().date().isoformat()) if len(values) else (None, None)


def _timing_signal_summary(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.iloc[-1]
    return {
        'current_action': _value(row, 'signal_action'),
        'current_position': _value(row, 'position'),
        'current_reason': _value(row, 'reason_summary'),
        'target_exposure': _number(_value(row, 'target_exposure', _value(row, 'position'))),
        'prev_exposure': _number(_value(row, 'prev_exposure')),
        'exposure_change': _number(_value(row, 'exposure_change')),
        'rebalance_action': _value(row, 'rebalance_action', _value(row, 'signal_action')),
    }


def _position(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.iloc[-1]
    attrs = frame.attrs
    invested = _number(_value(row, 'invested_amount'))
    unrealized = _number(_value(row, 'unrealized_pnl'))
    return {
        'etf_code': _value(row, 'etf_code', attrs.get('etf_code')),
        'etf_name': _value(row, 'etf_name', attrs.get('etf_name')),
        'etf_close': _number(_value(row, 'etf_close')),
        'holding_units': _number(_value(row, 'holding_units')),
        'holding_value': _number(_value(row, 'holding_value')),
        'cash_balance': _number(_value(row, 'cash_balance')),
        'entry_price': _number(_value(row, 'entry_price')),
        'entry_date': _day(_value(row, 'entry_date')),
        'unrealized_pnl': unrealized,
        'unrealized_pnl_pct': unrealized / invested * 100 if unrealized is not None and invested else None,
        'position_label': _value(row, 'position_label'),
        'target_exposure': _number(_value(row, 'target_exposure', _value(row, 'position'))),
        'prev_exposure': _number(_value(row, 'prev_exposure')),
        'rebalance_action': _value(row, 'rebalance_action', _value(row, 'signal_action')),
    }


def _trade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if 'trade_quantity' in frame.columns:
        return frame[pd.to_numeric(frame['trade_quantity'], errors='coerce').fillna(0).abs() > 1e-8]
    if 'signal_action' in frame.columns:
        return frame[frame['signal_action'].isin(['buy', 'sell'])]
    return frame.iloc[0:0]


def _trade_summary(frame: pd.DataFrame) -> dict[str, Any]:
    trades = _trade_rows(frame)
    realized = pd.to_numeric(trades.get('realized_pnl'), errors='coerce').dropna() if 'realized_pnl' in trades.columns else pd.Series(dtype=float)
    pos = _position(frame)
    return {
        'trade_count': len(trades),
        'completed_trade_count': int((trades.get('signal_action') == 'sell').sum()) if 'signal_action' in trades.columns else 0,
        'total_realized_pnl': float(realized.sum()) if len(realized) else 0.0,
        'last_realized_pnl': float(realized.iloc[-1]) if len(realized) else None,
        'current_unrealized_pnl': pos['unrealized_pnl'],
        'current_holding_value': pos['holding_value'],
        'avg_exposure': _number(pd.to_numeric(frame.get('target_exposure'), errors='coerce').mean()) if 'target_exposure' in frame.columns else None,
        'rebalance_count': len(trades),
    }


def _etf_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get('etf_close'), errors='coerce') if 'etf_close' in frame.columns else pd.Series(dtype=float)
    valid = values[values > 0].dropna()
    row = frame.iloc[-1]
    nav_key = _nav_column(frame)
    nav = pd.to_numeric(frame[nav_key], errors='coerce').dropna() if nav_key else pd.Series(dtype=float)
    etf_return = (float(valid.iloc[-1]) / float(valid.iloc[0]) - 1) * 100 if len(valid) else None
    strategy_return = (float(nav.iloc[-1]) / float(nav.iloc[0]) - 1) * 100 if len(nav) and nav.iloc[0] else None
    return {
        'etf_code': _value(row, 'etf_code', frame.attrs.get('etf_code')),
        'etf_name': _value(row, 'etf_name', frame.attrs.get('etf_name')),
        'start_price': float(valid.iloc[0]) if len(valid) else None,
        'end_price': float(valid.iloc[-1]) if len(valid) else None,
        'return_pct': etf_return,
        'strategy_return_pct': strategy_return,
        'strategy_excess_pct': strategy_return - etf_return if strategy_return is not None and etf_return is not None else None,
    }


def _related_links(snapshot: LoadedSnapshot) -> list[dict[str, Any]]:
    if snapshot.spec.source_id != 'selection':
        return []
    links = []
    for strategy_id in ('sector_heat', 'single_factor'):
        spec = get_strategy('selection_factor', strategy_id)
        for variant in variants_for(spec):
            if not variant.default:
                continue
            target_state, related = inspect_target(spec, variant)
            links.append({
                'source_id': spec.source_id,
                'strategy_id': spec.strategy_id,
                'variant_id': variant.variant_id,
                'snapshot_id': related.snapshot_id if related else None,
                'cache_state': target_state.cache_state,
                'url': f'/api/r0/sources/{spec.source_id}/strategies/{spec.strategy_id}/snapshots?variant_id={variant.variant_id}',
            })
    return sorted(links, key=lambda item: (item['source_id'], item['strategy_id'], item['variant_id']))


def summary(snapshot: LoadedSnapshot, view: dict[str, Any], view_id: str) -> dict[str, Any]:
    frame = filter_frame(snapshot, view)
    if frame is None:
        content = snapshot.document.get('content', {})
        stored_summary = content.get('summary') if isinstance(content.get('summary'), dict) else {}
        metrics = stored_summary.get('metrics') if isinstance(stored_summary.get('metrics'), dict) else {}
        metric_keys = (
            'cumulative_return', 'annual_return', 'max_drawdown', 'max_dd_start',
            'max_dd_end', 'calmar_ratio', 'final_capital', 'total_return_pct',
            'total_pnl', 'beta', 'annual_alpha', 'information_ratio', 'r_squared',
            'up_capture', 'down_capture', 'avg_exposure', 'rebalance_count', 'fee_ratio',
        )
        available_start, data_as_of = snapshot_date_bounds(snapshot)
        item_count = len(content.get('items') or content.get('factors') or [])
        return {
            'source_id': snapshot.spec.source_id, 'strategy_id': snapshot.spec.strategy_id,
            'variant_id': snapshot.variant.variant_id, 'snapshot_id': snapshot.snapshot_id,
            'view_id': view_id, 'canonical_view': view,
            'data_as_of': data_as_of, 'cache_generated_at': snapshot.generated_at,
            'date_range': {
                'start': available_start, 'end': data_as_of,
                'etf_inception_date': None,
            },
            'metrics': {key: nullable_json_scalar(metrics.get(key)) for key in metric_keys},
            'current_signal': published_signal(snapshot), 'position_snapshot': None,
            'interval_windows_link': None, 'benchmark': view.get('benchmark_id'),
            'benchmark_summaries': [], 'fee_totals': None,
            'counts': {
                'total_periods': item_count, 'signals': 0, 'trades': 0,
                'completed_trades': 0, 'holding_periods': None,
            },
            'initial_capital': None, 'win_rate': None,
            'related_snapshot_links': _related_links(snapshot),
            'trade_summary': None, 'etf_summary': None, 'signal_summary': None,
            'indicator_snapshots': None, 'exposure_mode': None,
            'active_index': None, 'active_benchmark': None,
            'holdings_context': None, 'split': None,
        }
    start, end = _dates(frame)
    metrics = _metric_payload(snapshot, frame)
    timing = snapshot.spec.source_id in {'a_share_timing', 'us_timing', 'hk_timing', 'commodity'}
    trade_rows = _trade_rows(frame) if timing else frame.iloc[0:0]
    payload = {
        'source_id': snapshot.spec.source_id,
        'strategy_id': snapshot.spec.strategy_id,
        'variant_id': snapshot.variant.variant_id,
        'snapshot_id': snapshot.snapshot_id,
        'view_id': view_id,
        'canonical_view': view,
        'data_as_of': end,
        'cache_generated_at': snapshot.generated_at,
        'date_range': {'start': start, 'end': end, 'etf_inception_date': frame.attrs.get('etf_inception_date')},
        'metrics': metrics,
        'current_signal': published_signal(snapshot),
        'position_snapshot': _position(frame) if 'position' in snapshot.spec.capabilities else None,
        'interval_windows_link': (f'/api/r0/sources/{snapshot.spec.source_id}/strategies/{snapshot.spec.strategy_id}/snapshots/{snapshot.snapshot_id}/interval-windows?view_id={view_id}' if 'interval_windows' in snapshot.spec.capabilities else None),
        'benchmark': view['benchmark_id'],
        'benchmark_summaries': [],
        'fee_totals': fees(snapshot, frame) if 'fees' in snapshot.spec.capabilities else None,
        'counts': {
            'total_periods': len(frame),
            'signals': len(frame) if timing else 0,
            'trades': len(trade_rows),
            'completed_trades': int((trade_rows.get('signal_action') == 'sell').sum()) if timing and 'signal_action' in trade_rows.columns else 0,
            'holding_periods': len(frame) if snapshot.spec.source_id == 'selection' else None,
        },
        'initial_capital': _number(frame.attrs.get('initial_capital')) or (100000.0 if snapshot.spec.source_id == 'selection' else 50000.0),
        'win_rate': _number((pd.to_numeric(frame.get(_return_column(snapshot, frame)), errors='coerce') > 0).mean()) if _return_column(snapshot, frame) else None,
        'related_snapshot_links': _related_links(snapshot),
        'trade_summary': _trade_summary(frame) if timing else None,
        'etf_summary': _etf_summary(frame) if timing else None,
        'signal_summary': _timing_signal_summary(frame) if timing else None,
        'indicator_snapshots': ({key: _number(_value(frame.iloc[-1], key)) for key in ('close', 'ma_fast', 'ma_slow', 'momentum_long', 'momentum_short', 'trend_ma', 'breakout_high', 'exit_low', 'strength_score', 'target_exposure')} if timing else None),
        'exposure_mode': frame.attrs.get('exposure_mode') if timing else None,
        'active_index': ({'id': _value(frame.iloc[-1], 'index_id'), 'name': _value(frame.iloc[-1], 'index_name')} if timing else None),
        'active_benchmark': ({'id': view['benchmark_id'], 'name': view['benchmark_id']} if timing else None),
        'holdings_context': ({'label': 'Stored selection periods', 'date_range': {'start': start, 'end': end}} if snapshot.spec.source_id == 'selection' else None),
        'split': (frame.attrs.get('split') or {'split_date': None, 'initial_capital': _number(frame.attrs.get('initial_capital')) or 100000.0, 'train_window_id': 'train', 'test_window_id': 'test'}) if snapshot.spec.source_id == 'selection' else None,
    }
    return payload


def series(snapshot: LoadedSnapshot, frame: pd.DataFrame | None, *, kind: str, window: str, resolution: str, benchmark_id: str | None, series_id: str | None, view: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    valid_kinds = {'equity', 'capital', 'drawdown', 'period_return', 'benchmark', 'factor_nav'}
    valid_windows = {'full', 'train', 'test', 'pre_6m_history', 'recent_6m', 'recent_1q', 'recent_1m'}
    if kind not in valid_kinds or window not in valid_windows or resolution not in {'day', 'month', 'quarter', 'year'}:
        raise ApiError(400, 'invalid_series_query', 'Invalid series kind/window/resolution combination.')
    if kind == 'benchmark' and not benchmark_id:
        raise ApiError(400, 'invalid_series_query', 'benchmark_id is required for benchmark series.')
    if kind == 'factor_nav' and not series_id:
        raise ApiError(400, 'invalid_series_query', 'series_id is required for factor_nav.')
    if kind not in {'benchmark', 'factor_nav'} and (benchmark_id or series_id):
        raise ApiError(400, 'invalid_series_query', 'This series kind does not accept benchmark_id or series_id.')
    factor_payload = snapshot.document.get('content', {}) if snapshot.document.get('kind') == 'payload' else {}
    if kind == 'factor_nav':
        factors = factor_payload.get('factors') or factor_payload.get('items') or []
        match = next((item for item in factors if str(item.get('series_id') or item.get('id')) == series_id), None)
        if not match:
            return []
        rows = []
        for ordinal, (day_value, nav_value) in enumerate(zip(match.get('dates', []), match.get('nav', []))):
            day = _day(day_value)
            value = _number(nav_value)
            if day is None or value is None:
                raise ApiError(500, 'snapshot_schema_invalid', 'factor_nav contains an invalid point.')
            rows.append({'artifact_row_ordinal': ordinal, 'series_id': series_id, 'date': day, 'value': value})
        stored_rows = len(rows)
        if view is not None:
            rows = [
                item for item in rows
                if (not view.get('start') or item['date'] >= view['start'])
                and (not view.get('end') or item['date'] <= view['end'])
            ]
            if stored_rows and not rows:
                raise ApiError(422, 'empty_view_range', 'The valid date range contains no stored points.')
        if window.startswith('recent_') and rows:
            months = {'recent_6m': 6, 'recent_1q': 3, 'recent_1m': 1}[window]
            cutoff = pd.Timestamp(rows[-1]['date']) - pd.DateOffset(months=months)
            rows = [item for item in rows if pd.Timestamp(item['date']) >= cutoff]
        if resolution != 'day' and rows:
            frequency = {'month': 'M', 'quarter': 'Q', 'year': 'Y'}[resolution]
            grouped: dict[pd.Period, list[dict[str, Any]]] = defaultdict(list)
            for item in rows:
                grouped[pd.Timestamp(item['date']).to_period(frequency)].append(item)
            rows = [
                {**bucket[-1], 'artifact_row_ordinal': ordinal}
                for ordinal, (_, bucket) in enumerate(sorted(grouped.items()))
            ]
        return rows
    if frame is None or '交易日期' not in frame.columns:
        return []
    sliced = frame
    dates = pd.to_datetime(frame['交易日期'], errors='coerce')
    if window.startswith('recent_') and len(dates.dropna()):
        months = {'recent_6m': 6, 'recent_1q': 3, 'recent_1m': 1}[window]
        cutoff = dates.max() - pd.DateOffset(months=months)
        sliced = frame.loc[dates >= cutoff]
    nav_key = _nav_column(sliced)
    ret_key = _return_column(snapshot, sliced)
    rows = []
    peak = None
    for ordinal, (_, row) in enumerate(sliced.iterrows()):
        day_value = _day(_value(row, '交易日期'))
        nav = _number(_value(row, nav_key)) if nav_key else None
        ret = _number(_value(row, ret_key)) if ret_key else None
        if kind == 'equity':
            item = {'date': day_value, 'value': nav, 'return': ret, 'close': _number(_value(row, 'close')), 'etf_close': _number(_value(row, 'etf_close'))}
        elif kind == 'capital':
            item = {'date': day_value, 'value': _number(_value(row, '累计资金', _value(row, '总资金'))), 'capital_start': _number(_value(row, '当期本金')), 'pnl': _number(_value(row, '当期盈亏')), 'return': ret}
        elif kind == 'drawdown':
            peak = nav if peak is None else max(peak, nav) if nav is not None else peak
            item = {'date': day_value, 'value': nav / peak - 1 if nav is not None and peak else None}
        elif kind == 'period_return':
            item = {'date': day_value, 'year': None, 'value': ret}
        elif kind == 'benchmark':
            value = _number(_value(row, f'benchmark_{benchmark_id}', _value(row, 'etf_close')))
            item = {'benchmark_id': benchmark_id, 'name': benchmark_id, 'date': day_value, 'value': value, 'return': None}
        else:
            continue
        item['artifact_row_ordinal'] = ordinal
        rows.append(item)
    if resolution != 'day' and rows:
        usable = [item for item in rows if item.get('date')]
        if usable:
            index = pd.to_datetime([item['date'] for item in usable])
            frequency = {'month': 'ME', 'quarter': 'QE', 'year': 'YE'}[resolution]
            grouped: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
            for stamp, item in zip(index.to_period(frequency[0]).to_timestamp(how='end'), usable):
                grouped[stamp].append(item)
            reduced = []
            for ordinal, (_, bucket) in enumerate(sorted(grouped.items())):
                last = dict(bucket[-1])
                last['artifact_row_ordinal'] = ordinal
                if kind in {'equity', 'capital', 'benchmark'}:
                    last['return'] = None
                reduced.append(last)
            rows = reduced
    return rows


def signals(frame: pd.DataFrame, event: str) -> list[dict[str, Any]]:
    if event not in {'all', 'buy_sell'}:
        raise ApiError(400, 'invalid_params', 'event must be all or buy_sell.')
    rows = []
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        action = _value(row, 'signal_action')
        if event == 'buy_sell' and action not in {'buy', 'sell'}:
            continue
        reason_detail = _value(row, 'reason_detail')
        if reason_detail is None:
            reason_detail = []
        elif not isinstance(reason_detail, (list, tuple)):
            reason_detail = [str(reason_detail)]
        if len(reason_detail) > 64:
            raise ApiError(500, 'bounded_collection_overflow', 'reason_detail exceeds its hard limit.')
        rows.append({
            'artifact_row_ordinal': ordinal, 'date': _day(_value(row, '交易日期')),
            'action': action, 'position': _value(row, 'position'),
            'reason_summary': _value(row, 'reason_summary'), 'reason_detail': list(reason_detail),
            'score': _number(_value(row, 'signal_score')), 'strength_score': _number(_value(row, 'strength_score')),
            'target_exposure': _number(_value(row, 'target_exposure', _value(row, 'position'))),
            'prev_exposure': _number(_value(row, 'prev_exposure')), 'exposure_change': _number(_value(row, 'exposure_change')),
            'rebalance_action': _value(row, 'rebalance_action', action), 'close': _number(_value(row, 'close')),
            'etf_close': _number(_value(row, 'etf_close')),
        })
    return rows


def trades(frame: pd.DataFrame, projection: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if projection not in {'detail', 'compact'}:
        raise ApiError(400, 'invalid_params', 'projection must be detail or compact.')
    rows = []
    for ordinal, (_, row) in enumerate(_trade_rows(frame).iterrows()):
        action = _value(row, 'signal_action')
        if action not in {'buy', 'sell'}:
            action = 'buy' if (_number(_value(row, 'exposure_change')) or 0) > 0 else 'sell'
        full = {
            'artifact_row_ordinal': ordinal, 'date': _day(_value(row, '交易日期')), 'action': action,
            'rebalance_action': _value(row, 'rebalance_action', action),
            'target_exposure': _number(_value(row, 'target_exposure', _value(row, 'position'))),
            'prev_exposure': _number(_value(row, 'prev_exposure')), 'close': _number(_value(row, 'close')),
            'etf_code': _value(row, 'etf_code', frame.attrs.get('etf_code')), 'etf_name': _value(row, 'etf_name', frame.attrs.get('etf_name')),
            'trade_price': _number(_value(row, 'etf_open')), 'etf_close': _number(_value(row, 'etf_close')),
            'latest_price': _number(_value(row, 'etf_close')), 'cost_price': _number(_value(row, 'trade_entry_price', _value(row, 'entry_price'))),
            'quantity': _number(_value(row, 'trade_quantity')), 'trade_amount': _number(_value(row, 'trade_amount')),
            'fee_amount': _number(_value(row, 'trade_fee_amount')), 'commission': _number(_value(row, 'commission_cost')),
            'stamp': _number(_value(row, 'stamp_cost')), 'transfer': _number(_value(row, 'transfer_cost')),
            'slippage_cost': _number(_value(row, 'slippage_cost')), 'blocked_by_limit': bool(_value(row, 'blocked_by_limit', False)),
            'limit_delays': _integer(_value(row, 'limit_delays')), 'holding_value': _number(_value(row, 'holding_value')),
            'cash_balance': _number(_value(row, 'cash_balance')), 'realized_pnl': _number(_value(row, 'realized_pnl')),
            'realized_pnl_pct': _number(_value(row, 'realized_pnl_pct')), 'unrealized_pnl': _number(_value(row, 'unrealized_pnl')),
            'entry_price': _number(_value(row, 'trade_entry_price', _value(row, 'entry_price'))), 'entry_date': _day(_value(row, 'trade_entry_date', _value(row, 'entry_date'))),
            'holding_days': _integer(_value(row, 'trade_holding_days', _value(row, 'holding_days'))),
            'entry_capital': _number(_value(row, 'trade_entry_capital', _value(row, 'entry_capital'))),
            'invested_amount': _number(_value(row, 'trade_invested_amount', _value(row, 'invested_amount'))),
            'nav': _number(_value(row, _nav_column(frame))) if _nav_column(frame) else None,
            'position_label': _value(row, 'position_label'),
        }
        if projection == 'compact':
            keys = ('artifact_row_ordinal', 'date', 'action', 'rebalance_action', 'target_exposure', 'prev_exposure', 'close', 'trade_price', 'etf_close', 'latest_price', 'cost_price', 'quantity', 'trade_amount', 'fee_amount', 'nav')
            full = {key: full[key] for key in keys}
        rows.append(full)
    return rows, _trade_summary(frame)


def fees(snapshot: LoadedSnapshot, frame: pd.DataFrame) -> dict[str, Any]:
    attrs = frame.attrs
    if snapshot.spec.source_id == 'selection':
        return {
            'c_rate': _number(attrs.get('c_rate')), 't_rate': _number(attrs.get('t_rate')),
            'sell_cost': _number(attrs.get('sell_cost')), 'total_buy_fees': _number(attrs.get('total_buy_fees')),
            'total_sell_fees': _number(attrs.get('total_sell_fees')), 'total_fees': _number(attrs.get('total_fees')),
        }
    payload = {
        'buy_cost': _number(attrs.get('buy_cost')), 'sell_cost': _number(attrs.get('sell_cost')),
        'total_trade_cost': _number(pd.to_numeric(frame.get('trade_fee_amount'), errors='coerce').fillna(0).sum()) if 'trade_fee_amount' in frame.columns else None,
        'commission_total': _number(attrs.get('commission_total')), 'stamp_total': _number(attrs.get('stamp_total')),
        'transfer_total': _number(attrs.get('transfer_total')), 'slippage_total': _number(attrs.get('slippage_total')),
        'cash_interest_total': _number(attrs.get('cash_interest_total')), 'limit_block_count': _integer(attrs.get('limit_block_count')),
    }
    payload['realism_meta'] = {
        'settlement_mode': attrs.get('settlement_mode'), 'limit_pct': _number(attrs.get('limit_pct')), 'market': attrs.get('market'),
        'slippage_bps': _number(attrs.get('slippage_bps')), 'cash_interest_rate': _number(attrs.get('cash_interest_rate')),
        'commission_rate': _number(attrs.get('commission_rate')), 'commission_min': _number(attrs.get('commission_min')),
        'stamp_tax_rate': _number(attrs.get('stamp_tax_rate')), 'transfer_fee_rate': _number(attrs.get('transfer_fee_rate')),
        'limit_max_delay_days': _integer(attrs.get('limit_max_delay_days')), 'profit_lock_enabled': attrs.get('profit_lock_enabled'),
        'profit_lock_drawdown': _number(attrs.get('profit_lock_drawdown')), 'profit_lock_level_1': _number(attrs.get('profit_lock_level_1')),
        'profit_lock_level_2': _number(attrs.get('profit_lock_level_2')), 'profit_lock_level_3': _number(attrs.get('profit_lock_level_3')),
    }
    return payload


def interval_windows(snapshot: LoadedSnapshot, frame: pd.DataFrame, view_id: str) -> list[dict[str, Any]]:
    timing = snapshot.spec.source_id != 'selection'
    windows = ('pre_6m_history', 'recent_6m', 'recent_1q', 'recent_1m') if timing else ('train', 'test', 'full')
    start, end = _dates(frame)
    rows = []
    for ordinal, window_id in enumerate(windows):
        if timing:
            if window_id == 'pre_6m_history':
                cutoff = pd.Timestamp(end) - pd.DateOffset(months=6) if end else None
                part = frame[pd.to_datetime(frame['交易日期']) < cutoff] if cutoff is not None else frame.iloc[0:0]
            elif window_id.startswith('recent_'):
                months = {'recent_6m': 6, 'recent_1q': 3, 'recent_1m': 1}[window_id]
                cutoff = pd.Timestamp(end) - pd.DateOffset(months=months) if end else None
                part = frame[pd.to_datetime(frame['交易日期']) >= cutoff] if cutoff is not None else frame
            else:
                part = frame
            pstart, pend = _dates(part)
            rows.append({
                'artifact_window_ordinal': ordinal, 'window_id': window_id, 'label': window_id.replace('_', ' '),
                'rows': len(part), 'is_tradable': bool(len(part)), 'start': pstart, 'end': pend,
                'training_range': {'start': start if pstart and pstart > start else None, 'end': None},
                'test_range': {'start': pstart, 'end': pend}, 'metrics': _metric_payload(snapshot, part) if len(part) else {},
                'etf_summary': _etf_summary(part) if len(part) else {}, 'series_links': [],
            })
        else:
            rows.append({
                'artifact_window_ordinal': ordinal, 'window_id': window_id, 'label': window_id,
                'months': len(frame), 'win_rate': _number((pd.to_numeric(frame.get(_return_column(snapshot, frame)), errors='coerce') > 0).mean()) if _return_column(snapshot, frame) else None,
                'reset_capital': window_id != 'full', 'initial_capital': _number(frame.attrs.get('initial_capital')) or 100000.0,
                'final_capital': _metric_payload(snapshot, frame)['final_capital'], 'date_range': {'start': start, 'end': end},
                'metrics': _metric_payload(snapshot, frame), 'series_links': [],
                'holding_periods_link': f'/api/r0/sources/{snapshot.spec.source_id}/strategies/{snapshot.spec.strategy_id}/snapshots/{snapshot.snapshot_id}/holding-periods?view_id={view_id}&window={window_id}',
            })
    return rows


def _selection_stock(raw: Any, ordinal: int) -> dict[str, Any]:
    row = raw if isinstance(raw, dict) else {'code': raw}
    reason_detail = row.get('selection_reason_detail') or row.get('选股理由拆解') or []
    fundamentals = row.get('selection_fundamentals') or row.get('基本面') or []
    breakdown = row.get('selection_factor_breakdown') or row.get('因子拆解') or []
    if not isinstance(reason_detail, list):
        reason_detail = [str(reason_detail)] if reason_detail not in (None, '') else []
    if not isinstance(fundamentals, list):
        fundamentals = [str(fundamentals)] if fundamentals not in (None, '') else []
    if not isinstance(breakdown, list):
        breakdown = []
    for values in (reason_detail, fundamentals, breakdown):
        if len(values) > 64:
            raise ApiError(500, 'bounded_collection_overflow', 'Selection nested collection exceeds its hard limit.')
    return {
        'artifact_stock_ordinal': ordinal, 'code': row.get('code') or row.get('股票代码') or row.get('股票代码_映射'),
        'name': row.get('name') or row.get('股票名称'), 'market_label': row.get('market_label') or row.get('市场'),
        'industry_l2': row.get('industry_l2') or row.get('新版申万二级行业名称'), 'factor_score': _number(row.get('factor_score') or row.get('因子')),
        'rank': _integer(row.get('rank') or row.get('排名')), 'weight': _number(row.get('weight') or row.get('权重')),
        'position_weight': _number(row.get('position_weight') or row.get('仓位占比')), 'return': _number(row.get('return') or row.get('涨跌幅')),
        'pnl': _number(row.get('pnl') or row.get('盈亏')), 'display_pnl': row.get('display_pnl'),
        'buy_price': _number(row.get('buy_price') or row.get('买入价')), 'sell_price': _number(row.get('sell_price') or row.get('卖出价')),
        'exit_price': _number(row.get('exit_price')), 'latest_price': _number(row.get('latest_price') or row.get('最新价')),
        'is_open': row.get('is_open'), 'price_source': row.get('price_source'), 'shares': _number(row.get('shares') or row.get('数量')),
        'position_market_value': _number(row.get('position_market_value') or row.get('市值')), 'pe': _number(row.get('pe')),
        'pb': _number(row.get('pb')), 'market_cap': _number(row.get('market_cap') or row.get('总市值')),
        'selection_reason_summary': row.get('selection_reason_summary') or row.get('选股理由'),
        'selection_reason_detail': reason_detail, 'selection_fundamentals': fundamentals,
        'selection_factor_breakdown': breakdown,
    }


def holding_periods(snapshot: LoadedSnapshot, frame: pd.DataFrame, window: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    periods = []
    stocks_by_period = {}
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        date_value = _day(_value(row, '交易日期'))
        period_id = f'p_{ordinal}_{date_value or "unknown"}'
        raw_stocks = _value(row, '买入个股收益')
        if not isinstance(raw_stocks, (list, tuple)):
            raw_stocks = _value(row, '买入股票代码')
        if not isinstance(raw_stocks, (list, tuple)):
            raw_stocks = []
        stocks = [_selection_stock(item, index) for index, item in enumerate(raw_stocks)]
        stocks_by_period[period_id] = stocks
        periods.append({
            'artifact_period_ordinal': ordinal, 'period_id': period_id, 'date': date_value,
            'holding_start_date': _day(_value(row, 'holding_start_date')), 'holding_end_date': _day(_value(row, 'holding_end_date')),
            'holding_date_range_label': _value(row, 'holding_date_range_label'),
            'period_return': _number(_value(row, '选股下周期涨跌幅')), 'period_pnl': _number(_value(row, '当期盈亏')),
            'display_period_pnl': _value(row, 'display_period_pnl'), 'period_pnl_label': _value(row, 'period_pnl_label'),
            'capital': _number(_value(row, '当期本金')), 'stock_count': len(stocks),
            'benchmark_returns': [], 'stocks_url': None,
        })
    periods.sort(key=lambda item: (item['date'] or '', item['artifact_period_ordinal']), reverse=True)
    start, end = _dates(frame)
    return {'label': window, 'date_range': {'start': start, 'end': end}}, periods, stocks_by_period


def factor_metadata(snapshot: LoadedSnapshot) -> dict[str, Any]:
    content = snapshot.document.get('content', {})
    metadata = None
    if snapshot.frame is not None:
        metadata = snapshot.frame.attrs.get('factor_metadata')
    metadata = metadata or content.get('factor_metadata') or {}
    parameters = []
    for ordinal, item in enumerate(metadata.get('parameters') or []):
        if not isinstance(item, dict):
            raise ApiError(500, 'snapshot_schema_invalid', 'Factor parameter must be an object.')
        enum_values = item.get('enum_values') if 'enum_values' in item else item.get('options')
        enum_values = [] if enum_values is None else enum_values
        if not isinstance(enum_values, list) or len(enum_values) > 64 or any(
                not isinstance(value, (str, int, float, bool)) or (isinstance(value, float) and not math.isfinite(value))
                for value in enum_values):
            raise ApiError(500, 'snapshot_schema_invalid', 'Factor parameter enum_values is invalid.')
        parameters.append({
            'artifact_parameter_ordinal': ordinal, 'key': item.get('key'), 'label': item.get('label'),
            'description': item.get('description'), 'default': nullable_json_scalar(item.get('default')),
            'min': nullable_json_scalar(item.get('min')), 'max': nullable_json_scalar(item.get('max')),
            'step': nullable_json_scalar(item.get('step')), 'unit': item.get('unit'), 'type': item.get('type'),
            'enum_values': enum_values,
        })
    filters = []
    for ordinal, item in enumerate(metadata.get('filters') or []):
        if not isinstance(item, dict):
            raise ApiError(500, 'snapshot_schema_invalid', 'Factor filter must be an object.')
        filters.append({'artifact_filter_ordinal': ordinal, 'name': item.get('name'), 'description': item.get('description')})
    if len(parameters) > 64 or len(filters) > 64:
        raise ApiError(500, 'bounded_collection_overflow', 'Factor metadata exceeds its hard limit.')
    ranking = _ranking(metadata.get('ranking'))
    ranking_factor = _ranking_factor(metadata.get('ranking_factor'))
    return {
        'id': metadata.get('id'), 'name': metadata.get('name'), 'description': metadata.get('description'),
        'parameters': parameters, 'filters': filters, 'ranking': ranking,
        'ranking_factor': ranking_factor,
    }


def _ranking(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    keys = ('name', 'formula', 'direction', 'description', 'combination_method', 'normalization_method', 'components', 'weight_details')
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ApiError(500, 'snapshot_schema_invalid', 'Factor ranking is not a closed object.')
    components = value['components']
    weights = value['weight_details']
    if not isinstance(components, list) or len(components) > 64:
        raise ApiError(500, 'bounded_collection_overflow', 'Factor ranking components exceed their hard limit.')
    if weights is not None and (not isinstance(weights, list) or len(weights) > 64):
        raise ApiError(500, 'bounded_collection_overflow', 'Factor ranking weights exceed their hard limit.')
    return {key: value[key] for key in keys}


def _ranking_factor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    keys = ('name', 'direction', 'description', 'weight_details')
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ApiError(500, 'snapshot_schema_invalid', 'ranking_factor is not a closed object.')
    weights = value['weight_details']
    if weights is not None and (not isinstance(weights, list) or len(weights) > 64):
        raise ApiError(500, 'bounded_collection_overflow', 'ranking_factor weights exceed their hard limit.')
    return {key: value[key] for key in keys}


def _profile_params(value: Any) -> dict[str, Any]:
    keys = ('val_pct_cutoff', 'bias_pct', 'vol_pct', 'bull_tp', 'bear_tp', 'bull_n', 'bear_n')
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ApiError(500, 'snapshot_schema_invalid', 'Profile params is not a closed object.')
    result = {}
    for key in keys[:5]:
        number = _number(value[key])
        if number is None or number < 0 or number > 1:
            raise ApiError(500, 'snapshot_schema_invalid', 'Profile percentage parameter is invalid.')
        result[key] = number
    for key in keys[5:]:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ApiError(500, 'snapshot_schema_invalid', 'Profile count parameter is invalid.')
        result[key] = item
    return result


def _profiles(value: Any) -> list[dict[str, Any]]:
    rows = [] if value is None else value
    if not isinstance(rows, list):
        raise ApiError(500, 'snapshot_schema_invalid', 'profile_summary must be an array.')
    if len(rows) > 64:
        raise ApiError(500, 'bounded_collection_overflow', 'profile_summary exceeds its hard limit.')
    result = []
    keys = ('window', 'label', 'candidate_name', 'score', 'overall_score', 'recent_score', 'months', 'window_start', 'window_end', 'params')
    for ordinal, item in enumerate(rows):
        if not isinstance(item, dict) or set(item) != set(keys):
            raise ApiError(500, 'snapshot_schema_invalid', 'Profile summary item is not a closed object.')
        result.append({'artifact_profile_ordinal': ordinal, **{key: (_profile_params(item[key]) if key == 'params' else item[key]) for key in keys}})
    return result


def factor_overview(snapshot: LoadedSnapshot) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content = snapshot.document.get('content', {})
    raw = snapshot.frame.attrs.get('factor_overview') if snapshot.frame is not None else None
    raw = raw or content.get('factor_overview') or {}
    if not isinstance(raw, dict):
        raise ApiError(500, 'snapshot_schema_invalid', 'factor_overview must be an object.')
    items = []
    for ordinal, item in enumerate(raw.get('factors') or []):
        if not isinstance(item, dict):
            raise ApiError(500, 'snapshot_schema_invalid', 'factor overview item must be an object.')
        core_fields = item.get('core_fields')
        if core_fields is not None and (not isinstance(core_fields, str) or len(core_fields) > 512):
            raise ApiError(500, 'snapshot_schema_invalid', 'factor core_fields is invalid.')
        items.append({
            'artifact_factor_ordinal': ordinal, 'name': item.get('name'), 'core_fields': core_fields,
            'sort_direction': item.get('sort_direction'), 'long_short': item.get('long_short'),
            'double_sort': item.get('double_sort'), 'book_recommended': item.get('book_recommended'),
            'category': item.get('category'), 'single_factor_id': item.get('single_factor_id'), 'active': item.get('active'),
        })
    active_names = raw.get('active_factor_names') or []
    factor_map = raw.get('single_factor_id_map') or {}
    if not isinstance(active_names, list) or any(not isinstance(item, str) for item in active_names):
        raise ApiError(500, 'snapshot_schema_invalid', 'active_factor_names is invalid.')
    if not isinstance(factor_map, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in factor_map.items()):
        raise ApiError(500, 'snapshot_schema_invalid', 'single_factor_id_map is invalid.')
    extra = {
        'strategy_id': snapshot.spec.strategy_id, 'strategy_name': snapshot.spec.display_name,
        'active_factor_names': sorted(active_names),
        'single_factor_id_map': {key: factor_map[key] for key in sorted(factor_map)},
        'profile_summary': _profiles(raw.get('profile_summary')),
    }
    if any(len(value) > 64 for value in (extra['active_factor_names'], extra['single_factor_id_map'], extra['profile_summary'])):
        raise ApiError(500, 'bounded_collection_overflow', 'Factor overview metadata exceeds its hard limit.')
    return extra, items


def sector_heat(snapshot: LoadedSnapshot, weeks: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = snapshot.frame
    if frame is None:
        return {'weeks': [], 'weeks_partial': [], 'weeks_n_days': [], 'industries': [], 'latest_ranking': []}, []
    required = {'week_label', 'industry', 'weekly_ret_pct'}
    if not required.issubset(frame.columns):
        raise ApiError(500, 'snapshot_schema_invalid', 'Sector heat artifact is missing required fields.')
    full_weeks = sorted(set(str(value) for value in frame['week_label']))
    selected = full_weeks[-min(weeks, len(full_weeks)):]
    week_index = {value: index for index, value in enumerate(selected)}
    industries = list(dict.fromkeys(str(value) for value in frame['industry']))
    industry_index = {value: index for index, value in enumerate(industries)}
    partials = []
    n_days = []
    for week in selected:
        rows = frame[frame['week_label'].astype(str) == week]
        partial_values = set()
        for value in (rows['is_partial'] if 'is_partial' in rows.columns else [False] * len(rows)):
            value = value.item() if hasattr(value, 'item') else value
            if type(value) is not bool:
                raise ApiError(500, 'snapshot_schema_invalid', 'Sector heat is_partial must be boolean.')
            partial_values.add(value)
        nday_values = set()
        for value in (rows['n_days_in_week'] if 'n_days_in_week' in rows.columns else [5] * len(rows)):
            value = value.item() if hasattr(value, 'item') else value
            if type(value) is not int or value <= 0:
                raise ApiError(500, 'snapshot_schema_invalid', 'Sector heat n_days_in_week must be a positive integer.')
            nday_values.add(value)
        if len(partial_values) != 1 or len(nday_values) != 1 or next(iter(nday_values)) <= 0:
            raise ApiError(500, 'snapshot_schema_invalid', 'Sector heat week metadata is inconsistent.')
        partials.append(next(iter(partial_values)))
        n_days.append(next(iter(nday_values)))
    true_indexes = [index for index, value in enumerate(partials) if value]
    for index in true_indexes[:-1]:
        partials[index] = False
    items = []
    for physical_ordinal, (_, row) in enumerate(frame.iterrows()):
        week = str(row['week_label'])
        if week not in week_index:
            continue
        industry = str(row['industry'])
        value = _number(row['weekly_ret_pct'])
        if value is None:
            raise ApiError(500, 'snapshot_schema_invalid', 'Sector heat cell must be finite.')
        items.append({'artifact_cell_ordinal': physical_ordinal, 'week_index': week_index[week], 'industry_index': industry_index[industry], 'weekly_ret_pct': value})
    complete_indexes = [index for index, partial in enumerate(partials) if not partial]
    ranking_weeks = set(complete_indexes[-4:] if complete_indexes else range(max(0, len(selected) - 4), len(selected)))
    means = []
    for industry, index in industry_index.items():
        values = [item['weekly_ret_pct'] for item in items if item['industry_index'] == index and item['week_index'] in ranking_weeks]
        if values:
            means.append((industry, sum(values) / len(values), index))
    means.sort(key=lambda item: (-item[1], item[2]))
    ranking = [{'industry': industry, 'avg_ret_4w': value, 'rank': rank} for rank, (industry, value, _) in enumerate(means, 1)]
    if len(selected) > 52 or len(industries) > 512 or len(ranking) > 512:
        raise ApiError(500, 'bounded_collection_overflow', 'Sector heat axes exceed their hard limits.')
    return {'weeks': selected, 'weeks_partial': partials, 'weeks_n_days': n_days, 'industries': industries, 'latest_ranking': ranking}, items


def single_factor(snapshot: LoadedSnapshot) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content = snapshot.document.get('content', {})
    raw_items = content.get('factors') or content.get('items') or []
    expected_top_k = int(snapshot.variant.canonical_params['top_k'])
    if content.get('top_k') is None or int(content['top_k']) != expected_top_k:
        raise ApiError(500, 'snapshot_schema_invalid', 'single factor top_k does not match its catalog variant.')
    if not isinstance(raw_items, list):
        raise ApiError(500, 'snapshot_schema_invalid', 'single factor items must be an array.')
    items = []
    seen_ids = set()
    for ordinal, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ApiError(500, 'snapshot_schema_invalid', 'single factor item must be an object.')
        factor_id = item.get('id')
        if not isinstance(factor_id, str) or not factor_id or factor_id in seen_ids:
            raise ApiError(500, 'snapshot_schema_invalid', 'single factor id is missing or duplicated.')
        seen_ids.add(factor_id)
        dates = item.get('dates')
        nav = item.get('nav')
        if not isinstance(dates, list) or not isinstance(nav, list) or len(dates) != len(nav):
            raise ApiError(500, 'snapshot_schema_invalid', 'single factor dates/nav series is invalid.')
        if any(_day(value) is None for value in dates) or any(_number(value) is None for value in nav):
            raise ApiError(500, 'snapshot_schema_invalid', 'single factor series contains an invalid point.')
        regime = item.get('regime_metrics')
        if not isinstance(regime, dict) or set(regime) != {'牛市', '熊市'}:
            raise ApiError(500, 'snapshot_schema_invalid', 'single factor regime_metrics is invalid.')
        ordered_regime = {}
        for name in ('牛市', '熊市'):
            value = regime[name]
            if not isinstance(value, dict) or set(value) != {'avg_monthly_return', 'n_periods'}:
                raise ApiError(500, 'snapshot_schema_invalid', 'single factor regime item is invalid.')
            periods = value['n_periods']
            average = value['avg_monthly_return']
            if isinstance(periods, bool) or not isinstance(periods, int) or periods < 0:
                raise ApiError(500, 'snapshot_schema_invalid', 'single factor regime count is invalid.')
            if not isinstance(average, str) or not (average == 'N/A' or re.fullmatch(r'-?(?:0|[1-9]\d*)(?:\.\d+)?%', average)):
                raise ApiError(500, 'snapshot_schema_invalid', 'single factor regime return is invalid.')
            if periods == 0 and average != 'N/A':
                raise ApiError(500, 'snapshot_schema_invalid', 'empty regime must use N/A.')
            if periods > 0 and average == 'N/A':
                raise ApiError(500, 'snapshot_schema_invalid', 'nonempty regime requires a decimal-percent return.')
            ordered_regime[name] = {'avg_monthly_return': average, 'n_periods': periods}
        series_id = str(item.get('series_id') or item.get('id'))
        numeric = {}
        for key in ('annual_return', 'max_drawdown', 'calmar'):
            value = _number(item.get(key))
            if item.get(key) is not None and value is None:
                raise ApiError(500, 'snapshot_schema_invalid', f'single factor {key} must be finite.')
            numeric[key] = value
        items.append({
            'artifact_factor_ordinal': ordinal, 'id': item.get('id'), 'name': item.get('name'), 'column': item.get('column'),
            'annual_return': numeric['annual_return'], 'max_drawdown': numeric['max_drawdown'],
            'calmar': numeric['calmar'], 'regime_metrics': ordered_regime, 'series_id': series_id, 'series_url': None,
        })
    return {'version': content.get('version'), 'saved_at': content.get('saved_at'), 'top_k': expected_top_k}, items


def configuration(snapshot: LoadedSnapshot) -> dict[str, Any]:
    content = snapshot.document.get('content', {})
    attrs = snapshot.frame.attrs if snapshot.frame is not None else {}
    config = attrs.get('configuration') or content.get('configuration') or {}
    if not isinstance(config, dict):
        raise ApiError(500, 'snapshot_schema_invalid', 'configuration must be an object.')
    selection_metadata = None
    if snapshot.spec.source_id == 'selection':
        selection_metadata = factor_metadata(snapshot)
    timing_metadata = attrs.get('timing_metadata') or content.get('timing_metadata') or config.get('timing_metadata')
    if timing_metadata is not None:
        keys = ('id', 'name', 'description', 'principle_summary', 'formula_blocks', 'shared_exposure_blocks', 'parameters')
        if not isinstance(timing_metadata, dict) or set(timing_metadata) != set(keys):
            raise ApiError(500, 'snapshot_schema_invalid', 'timing_metadata is invalid.')
        timing_metadata = {key: timing_metadata.get(key) for key in keys}
        for key in ('formula_blocks', 'shared_exposure_blocks', 'parameters'):
            value = timing_metadata[key] or []
            if not isinstance(value, list) or len(value) > 64:
                raise ApiError(500, 'bounded_collection_overflow', f'{key} exceeds its hard limit.')
            timing_metadata[key] = value
    best_profile = attrs.get('best_profile') or content.get('best_profile') or config.get('best_profile')
    if best_profile is not None:
        keys = ('strategy_id', 'training_cutoff', 'generated_at', 'score', 'score_formula', 'maxdd_threshold', 'tuned_params', 'window_metrics')
        if not isinstance(best_profile, dict) or set(best_profile) != set(keys):
            raise ApiError(500, 'snapshot_schema_invalid', 'best_profile is invalid.')
        best_profile = {key: best_profile.get(key) for key in keys}
        if not isinstance(best_profile['window_metrics'] or [], list) or len(best_profile['window_metrics'] or []) > 64:
            raise ApiError(500, 'bounded_collection_overflow', 'window_metrics exceeds its hard limit.')
        best_profile['window_metrics'] = best_profile['window_metrics'] or []
    changelog = attrs.get('changelog') or content.get('changelog') or config.get('changelog')
    if changelog is not None:
        keys = ('market_group', 'changelog_title', 'changelog_summary', 'changelog_bullets', 'supersedes', 'performance_delta')
        if not isinstance(changelog, dict) or not set(changelog).issubset(set(keys)):
            raise ApiError(500, 'snapshot_schema_invalid', 'changelog is invalid.')
        changelog = {key: changelog.get(key) for key in keys}
        bullets = changelog['changelog_bullets'] or []
        if not isinstance(bullets, list) or len(bullets) > 64:
            raise ApiError(500, 'bounded_collection_overflow', 'changelog_bullets exceeds its hard limit.')
        changelog['changelog_bullets'] = bullets
    effective = attrs.get('effective_configuration') or content.get('effective_configuration') or config.get('effective_configuration')
    if effective is not None and not isinstance(effective, dict):
        raise ApiError(500, 'snapshot_schema_invalid', 'effective_configuration must be an object or null.')
    realism = fees(snapshot, snapshot.frame).get('realism_meta') if snapshot.frame is not None and snapshot.spec.source_id in {'a_share_timing', 'us_timing', 'hk_timing', 'commodity'} else None
    provenance = attrs.get('provenance_label') or content.get('provenance_label') or config.get('provenance_label')
    if provenance is None:
        provenance = 'Current code defaults / unknown'
    if not isinstance(provenance, str):
        raise ApiError(500, 'snapshot_schema_invalid', 'provenance_label must be a string.')
    return {
        'param_schema_version': snapshot.variant.param_schema['version'],
        'canonical_params': snapshot.variant.canonical_params,
        'selection_metadata': selection_metadata,
        'timing_metadata': timing_metadata,
        'best_profile': best_profile,
        'effective_configuration': effective,
        'changelog': changelog,
        'realism_meta': realism,
        'provenance_label': provenance,
    }
