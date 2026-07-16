"""Regression tests for `web.serializers._build_holdings_payload` open-snapshot dedup.

防的是 acf2ca2：同一持仓被多次以 "open snapshot"（所有 stock.sell_price=None）写入
回测结果时，UI 会出现重复的未平仓行。修复：dedup + 只保留**最后一笔** open snapshot
（earlier open snapshot 行直接跳过 `continue`），且只有最后一笔被允许标 is_open=True。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web import serializers
from web.serializers import (
    _build_holdings_payload,
    _extract_open_stock_codes,
    _is_open_snapshot_period,
)


def _stock(code, sell_price=None, *, weight=1.0, ret=0.0, buy_price=10.0):
    return {
        'code': code,
        'name': f'name_{code}',
        'weight': weight,
        'return': ret,
        'buy_price': buy_price,
        'sell_price': sell_price,
    }


def _row(date, stocks, *, capital=100000.0, period_return=0.0):
    return {
        '交易日期': pd.Timestamp(date),
        '买入个股收益': json.dumps(stocks),
        '当期本金': capital,
        '当期盈亏': 0.0,
        '选股下周期涨跌幅': period_return,
        '买入股票代码': ' '.join(s['code'] for s in stocks),
        '买入股票名称': ' '.join(s['name'] for s in stocks),
    }


def test_is_open_snapshot_period_logic():
    """sanity check 配合 dedup 用：只有"全部 sell_price=None"才算 open snapshot。"""
    assert _is_open_snapshot_period([_stock('A'), _stock('B')]) is True
    assert _is_open_snapshot_period([_stock('A', sell_price=11), _stock('B')]) is False
    assert _is_open_snapshot_period([]) is False  # 空列表不算 open snapshot


def test_multiple_open_snapshots_dedup_to_last_only():
    """3 行 open snapshot + 1 行 closed → 输出只剩 1 行 open + 1 行 closed。"""
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('AAA', sell_price=11.0)]),       # closed
        _row('2025-02-28', [_stock('BBB', sell_price=None)]),       # open #1
        _row('2025-03-31', [_stock('CCC', sell_price=None)]),       # open #2
        _row('2025-04-30', [_stock('DDD', sell_price=None)]),       # open #3 (last)
    ])

    payload = _build_holdings_payload(df, default_capital=100000.0)
    # 输出反序（newest first），所以 payload[0] 必是最后一行
    dates = [h['date'] for h in payload]
    # 2025-02-28 / 2025-03-31 这两行 open snapshot 应被 dedup 掉
    assert '2025-02-28' not in dates
    assert '2025-03-31' not in dates
    # closed 行 + 最后一笔 open 都应保留
    assert '2025-01-31' in dates
    assert '2025-04-30' in dates
    assert len(payload) == 2, f'输出应只有 2 行，实际 {dates}'

    # 验证只有最后一笔 open snapshot 内的 stock 被标记 is_open=True
    open_row = next(h for h in payload if h['date'] == '2025-04-30')
    assert len(open_row['stocks']) == 1
    assert open_row['stocks'][0]['code'] == 'DDD'
    assert open_row['stocks'][0]['is_open'] is True, '最后一笔 open snapshot 内必须保留 is_open=True'

    closed_row = next(h for h in payload if h['date'] == '2025-01-31')
    assert closed_row['stocks'][0]['is_open'] is False, 'closed 行内 stock 不能被标 open'


def test_single_open_snapshot_preserved():
    """只有一行 open snapshot 时不要被错误去掉。"""
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('AAA', sell_price=11.0)]),
        _row('2025-02-28', [_stock('BBB', sell_price=None)]),
    ])
    payload = _build_holdings_payload(df, default_capital=100000.0)
    dates = [h['date'] for h in payload]
    assert dates == ['2025-02-28', '2025-01-31'], dates
    open_row = next(h for h in payload if h['date'] == '2025-02-28')
    assert open_row['stocks'][0]['is_open'] is True


def test_closed_rows_with_distinct_dates_not_deduped():
    """negative case：多笔 closed (各自不同 sell_price) 不能被 open dedup 误伤。"""
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('A', sell_price=11.0)]),
        _row('2025-02-28', [_stock('B', sell_price=12.0)]),
        _row('2025-03-31', [_stock('C', sell_price=13.0)]),
    ])
    payload = _build_holdings_payload(df, default_capital=100000.0)
    dates = sorted(h['date'] for h in payload)
    assert dates == ['2025-01-31', '2025-02-28', '2025-03-31']
    # 没有任何 open snapshot → 所有 stock 都不应标 is_open
    for h in payload:
        for s in h['stocks']:
            assert s['is_open'] is False


def test_no_open_snapshot_returns_no_open_position_flag():
    """全是 closed 时 last_open_snapshot_idx 仍为 None，allow_open_position 不能误开。"""
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('X', sell_price=10.5), _stock('Y', sell_price=11.5)]),
    ])
    payload = _build_holdings_payload(df, default_capital=100000.0)
    assert len(payload) == 1
    for s in payload[0]['stocks']:
        assert s['is_open'] is False


def test_mixed_partial_close_not_treated_as_open_snapshot():
    """部分 sell_price=None 的混合行不属于 open snapshot，不参与 dedup。"""
    # 行内既有未平仓股，也有已平仓股 → _is_open_snapshot_period 返回 False
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('A', sell_price=11.0), _stock('B', sell_price=None)]),
        _row('2025-02-28', [_stock('C', sell_price=None), _stock('D', sell_price=None)]),  # full open snapshot
    ])
    payload = _build_holdings_payload(df, default_capital=100000.0)
    dates = sorted(h['date'] for h in payload)
    # 两行都应保留（前者不算 open snapshot，后者是唯一的 open snapshot）
    assert dates == ['2025-01-31', '2025-02-28']


def test_realtime_quote_codes_only_include_last_open_snapshot():
    """实时行情与持仓输出使用同一口径：只抓最后一笔 open snapshot。"""
    df = pd.DataFrame([
        _row('2025-01-31', [_stock('OLD_A'), _stock('OLD_B')]),
        _row('2025-02-28', [_stock('MIXED_A', sell_price=11.0), _stock('MIXED_B')]),
        _row('2025-03-31', [_stock('LATEST_B'), _stock('LATEST_A')]),
    ])

    assert _extract_open_stock_codes(df) == ['LATEST_A', 'LATEST_B']


def test_compact_payload_does_not_fetch_realtime_quotes(monkeypatch):
    """compact 首屏会删除持仓明细，不得在返回前同步抓行情。"""
    result = pd.DataFrame([
        {
            '交易日期': pd.Timestamp('2025-01-31'),
            '累积净值': 1.01,
            '选股下周期涨跌幅': 0.01,
            '买入个股收益': json.dumps([_stock('LATEST')]),
        }
    ])
    result.attrs['initial_capital'] = 100000.0

    def _unexpected_fetch(_result):
        raise AssertionError('compact payload must not fetch realtime quotes')

    monkeypatch.setattr(serializers, '_fetch_open_stock_quotes', _unexpected_fetch)
    monkeypatch.setattr(serializers, '_load_trading_calendar', lambda _benchmark=None: pd.DatetimeIndex([]))
    monkeypatch.setattr(serializers, '_build_holdings_payload', lambda *args, **kwargs: [{'date': '2025-01-31'}])
    monkeypatch.setattr(serializers, 'build_selection_interval_windows', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        serializers,
        'compute_split_metrics',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('compact payload must skip split metrics')
        ),
    )
    monkeypatch.setattr(serializers, '_get_benchmark_series', lambda _benchmark=None: ('csi1000', None))
    monkeypatch.setattr(serializers, '_compute_single_benchmark_curve_daily', lambda *args, **kwargs: [])
    monkeypatch.setattr(serializers, '_index_returns_map', lambda: {})

    payload = serializers.result_to_json(
        result,
        pd.DataFrame(),
        split_date=None,
        benchmark_id='csi1000',
        compact=True,
    )

    assert payload['has_holdings'] is True
    assert payload['holdings'] == []


def test_explicit_empty_quote_map_does_not_fall_back_to_network(monkeypatch):
    """compact 向子序列化器传递 {} 时，不得被 `or fetch()` 覆盖。"""
    def _unexpected_fetch(_result):
        raise AssertionError('explicit empty quote map must skip network fetch')

    monkeypatch.setattr(serializers, '_fetch_open_stock_quotes', _unexpected_fetch)
    assert serializers._resolve_quote_map(pd.DataFrame(), {}) == {}


def test_compact_payload_limits_benchmark_ids_to_active(monkeypatch):
    monkeypatch.setattr(
        serializers,
        '_index_returns_map',
        lambda: {'csi1000': object(), 'chinext': object(), 'star50': object()},
    )

    assert serializers._benchmark_ids_for_payload('chinext', compact=True) == ['chinext']
    assert serializers._benchmark_ids_for_payload('chinext', compact=False) == [
        'csi1000', 'chinext', 'star50',
    ]
