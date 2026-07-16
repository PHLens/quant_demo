#!/usr/bin/env python
"""离线抓 A 股宏观风险因子 → data/a_share_macro/*.csv

只产出 3 个日频文件，供 build_risk_signals.py 读取做风险面板：
  - pe_ttm.csv          : 全 A 个股 PE-TTM 中位数（含 10 年分位）
  - cn10y.csv           : 中债国债 10 年收益率
  - sse_daily.csv       : 上交所每日 流通市值 / 成交金额 / 流通换手率 / 融资买入额

注意：
  - 不计算衍生因子（ERP / 融资买入占比），那是 build_risk_signals 的责任，
    保持本脚本「只取数」、「无业务规则」。
  - 增量续抓：已存在日期跳过；首次冷启回看 365 天；其余仅补今天。
  - 周末/节假日 API 返回空属正常，不报错。

运行：
    /Users/fatcat/opt/anaconda3/bin/python scripts/fetch_a_share_macro.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE_ROOT, 'stock_trade_demo'))
from utils.atomic_io import atomic_write_csv as _atomic_write_csv
_RESOURCE_ROOT = os.path.abspath(os.environ.get('R0_RESOURCE_ROOT') or _CODE_ROOT)
_OUT_DIR = os.path.join(_RESOURCE_ROOT, 'data', 'a_share_macro')
os.makedirs(_OUT_DIR, exist_ok=True)

_PE_FILE = os.path.join(_OUT_DIR, 'pe_ttm.csv')
_CN10Y_FILE = os.path.join(_OUT_DIR, 'cn10y.csv')
_SSE_FILE = os.path.join(_OUT_DIR, 'sse_daily.csv')

_COLD_LOOKBACK_DAYS = 365


def _read_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        return df.sort_values('date').drop_duplicates('date', keep='last')
    except Exception:
        return pd.DataFrame()


def _upsert(path: str, new_rows: pd.DataFrame) -> int:
    if new_rows is None or new_rows.empty:
        return 0
    existing = _read_existing(path)
    if not existing.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date').drop_duplicates('date', keep='last')
    _atomic_write_csv(path, combined, index=False,
                      produced_by=f"scripts/fetch_a_share_macro:{os.path.basename(path)}")
    return len(new_rows)


def fetch_pe_ttm() -> int:
    """全 A 个股 PE-TTM 中位数 + 10 年分位。stock_a_ttm_lyr 一次返回全历史。"""
    import akshare as ak
    df = ak.stock_a_ttm_lyr()
    out = pd.DataFrame({
        'date': pd.to_datetime(df['date']),
        'pe_median': df['middlePETTM'].astype(float),
        'pe_mean': df['averagePETTM'].astype(float),
        'pe_median_q10y': df['quantileInRecent10YearsMiddlePeTtm'].astype(float),
    })
    out = out.dropna(subset=['date', 'pe_median'])
    _upsert(_PE_FILE, out)
    return len(out)


def fetch_cn10y() -> int:
    """中债国债 10Y。bond_china_yield 支持区间查询，按 90 天窗口分批避开服务端限流。"""
    import akshare as ak
    today = datetime.now().date()
    existing = _read_existing(_CN10Y_FILE)
    if existing.empty:
        start = today - timedelta(days=_COLD_LOOKBACK_DAYS)
    else:
        last = existing['date'].max().date()
        start = last + timedelta(days=1)
        if start >= today:
            return 0  # 已是最新

    rows = []
    cur = start
    while cur <= today:
        chunk_end = min(cur + timedelta(days=90), today)
        try:
            df = ak.bond_china_yield(
                start_date=cur.strftime('%Y%m%d'),
                end_date=chunk_end.strftime('%Y%m%d'),
            )
            # 多条曲线，挑「中债国债收益率曲线」
            df = df[df['曲线名称'] == '中债国债收益率曲线'].copy()
            if not df.empty:
                df['date'] = pd.to_datetime(df['日期'])
                df['cn10y_pct'] = pd.to_numeric(df['10年'], errors='coerce')
                rows.append(df[['date', 'cn10y_pct']].dropna())
        except Exception as exc:
            print(f"  [cn10y] {cur} ~ {chunk_end}: {exc}", file=sys.stderr)
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.3)
    if not rows:
        return 0
    return _upsert(_CN10Y_FILE, pd.concat(rows, ignore_index=True))


def _safe_get_value(df_two_col: pd.DataFrame, key: str, col: str = '股票') -> Optional[float]:
    """stock_sse_deal_daily 返回的 ['单日情况', '股票', ...] 类二维表，按行名取数。"""
    row = df_two_col[df_two_col['单日情况'] == key]
    if row.empty:
        return None
    try:
        return float(row.iloc[0][col])
    except (TypeError, ValueError):
        return None


def fetch_sse_daily(max_backfill: int = 180) -> int:
    """上交所每日概况 + 融资买入额。逐日补缺（API 只支持单日）。

    上交所流通市值单位：亿元；成交金额：亿元；流通换手率：% 直接给。
    融资买入额单位：元（注意要换算到亿元）。
    """
    import akshare as ak

    existing = _read_existing(_SSE_FILE)
    today = datetime.now().date()
    if existing.empty:
        start = today - timedelta(days=max_backfill)
    else:
        last = existing['date'].max().date()
        start = last + timedelta(days=1)
        if start >= today:
            return 0

    # 一次性把这段时间内 SSE 融资数据拉下来，按日期入字典
    try:
        margin_df = ak.stock_margin_sse(
            start_date=start.strftime('%Y%m%d'),
            end_date=today.strftime('%Y%m%d'),
        )
        margin_df['date'] = pd.to_datetime(margin_df['信用交易日期'])
        margin_map = {
            d.date(): float(amt)
            for d, amt in zip(margin_df['date'], margin_df['融资买入额'])
        }
        margin_balance_map = {
            d.date(): float(amt)
            for d, amt in zip(margin_df['date'], margin_df['融资余额'])
        }
    except Exception as exc:
        print(f"  [sse-margin] {start} ~ {today}: {exc}", file=sys.stderr)
        margin_map, margin_balance_map = {}, {}

    rows = []
    cur = start
    while cur <= today:
        # 周末直接跳过
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        ymd = cur.strftime('%Y%m%d')
        try:
            df = ak.stock_sse_deal_daily(date=ymd)
        except Exception as exc:
            # 非交易日 / 节假日 API 会返回畸形 frame → 静默跳过
            msg = str(exc)
            if 'Length mismatch' not in msg and 'columns' not in msg:
                print(f"  [sse-deal {ymd}] {exc}", file=sys.stderr)
            cur += timedelta(days=1)
            continue
        try:
            float_mcap = _safe_get_value(df, '流通市值')   # 亿元
            amount = _safe_get_value(df, '成交金额')        # 亿元
            turnover_float = _safe_get_value(df, '流通换手率')  # %
            if float_mcap is None or amount is None:
                cur += timedelta(days=1)
                continue
            margin_buy_yi = margin_map.get(cur)
            margin_bal_yi = margin_balance_map.get(cur)
            # SSE 融资数据是元，换算到亿元；缺则 None
            margin_buy_yi = margin_buy_yi / 1e8 if margin_buy_yi is not None else None
            margin_bal_yi = margin_bal_yi / 1e8 if margin_bal_yi is not None else None
            rows.append({
                'date': pd.Timestamp(cur),
                'sse_float_mcap_yi': float_mcap,
                'sse_amount_yi': amount,
                'sse_turnover_float_pct': turnover_float,
                'sse_margin_buy_yi': margin_buy_yi,
                'sse_margin_balance_yi': margin_bal_yi,
            })
        except Exception as exc:
            print(f"  [sse-deal parse {ymd}] {exc}", file=sys.stderr)
        cur += timedelta(days=1)
        time.sleep(0.25)

    if not rows:
        return 0
    return _upsert(_SSE_FILE, pd.DataFrame(rows))


def main():
    print(f"[fetch_a_share_macro] start, output dir = {_OUT_DIR}")
    n_pe = fetch_pe_ttm()
    print(f"  [pe_ttm] upserted {n_pe} rows → {_PE_FILE}")
    n_b = fetch_cn10y()
    print(f"  [cn10y]  upserted {n_b} rows → {_CN10Y_FILE}")
    n_s = fetch_sse_daily()
    print(f"  [sse]    upserted {n_s} rows → {_SSE_FILE}")

    # tail 一下让人一眼看到最新
    for path in [_PE_FILE, _CN10Y_FILE, _SSE_FILE]:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date']).sort_values('date')
            if len(df):
                print(f"  [tail {os.path.basename(path)}] last={df.iloc[-1].to_dict()}")


if __name__ == '__main__':
    main()
