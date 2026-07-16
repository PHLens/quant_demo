"""Pure JSON-serialization helpers for the Flask read-only viewer.

从 web_app.py 抽出（Pillar 1 Step 1，纯函数搬家）。本模块只依赖：
- 标准库 / numpy / pandas
- index_data 的常量与查询函数
- get_stock_info.fetch_realtime_quotes_batch（实时报价 HTTP 客户端）
- backtest.strategy_evaluate（评估指标）

INDEX_RETURNS_MAP 由 web_app 在启动时通过 set_index_returns_map_provider
注入一个 lambda；本模块不持有可变全局状态。
"""

import json

import numpy as np
import pandas as pd

from index_data import (
    INDEX_CONFIGS,
    A_SHARE_INDEX_IDS,
    get_a_share_trading_calendar,
    get_index_daily,
    get_timing_etf_daily,
    build_period_lookup,
    get_index_return_for_date,
)
from backtest import strategy_evaluate


# ─── 常量（沿用 web_app 原值） ───
SPLIT_DATE = pd.to_datetime('2026-03-31')
DEFAULT_BENCHMARK_ID = 'csi1000'


# ─── INDEX_RETURNS_MAP 依赖注入 ───
# 该 dict 在 web_app 启动时由 ensure_index_returns_loaded() 填充，
# 但函数被注入的 lambda 在每次调用时按名读取，因此对 web_app 中的重新赋值
# 也能感知到（lambda 在 module global 作用域捕获名字，运行时 LOAD_GLOBAL）。
_INDEX_RETURNS_MAP_PROVIDER = None  # type: callable[[], dict] | None


def set_index_returns_map_provider(provider):
    """由 web_app 调用一次：注入返回当前 INDEX_RETURNS_MAP 的 callable。

    Example:
        from web import serializers
        serializers.set_index_returns_map_provider(lambda: INDEX_RETURNS_MAP)
    """
    global _INDEX_RETURNS_MAP_PROVIDER
    _INDEX_RETURNS_MAP_PROVIDER = provider


def _index_returns_map():
    """返回当前 INDEX_RETURNS_MAP（若未注入则为空 dict）。"""
    if _INDEX_RETURNS_MAP_PROVIDER is None:
        return {}
    return _INDEX_RETURNS_MAP_PROVIDER() or {}


# ─── 基准 ID 归一化 / 元数据 ───
def _normalize_benchmark_id(benchmark_id):
    if benchmark_id in INDEX_CONFIGS:
        return benchmark_id
    if DEFAULT_BENCHMARK_ID in INDEX_CONFIGS:
        return DEFAULT_BENCHMARK_ID
    return next(iter(INDEX_CONFIGS.keys()), None)


def _get_benchmark_series(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None, None
    index_returns_map = _index_returns_map()
    series = index_returns_map.get(normalized_id)
    if series is not None:
        return normalized_id, series
    fallback_id = DEFAULT_BENCHMARK_ID if DEFAULT_BENCHMARK_ID in index_returns_map else None
    if fallback_id is not None:
        return fallback_id, index_returns_map[fallback_id]
    if index_returns_map:
        first_id = next(iter(index_returns_map.keys()))
        return first_id, index_returns_map[first_id]
    return normalized_id, None


def _get_benchmark_meta(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None
    cfg = INDEX_CONFIGS.get(normalized_id, {})
    return {
        'id': normalized_id,
        'name': cfg.get('name', normalized_id),
    }


def _infer_market_label(code):
    code = str(code or '').strip()
    digits = ''.join(ch for ch in code if ch.isdigit())
    if digits.startswith(('688', '689')):
        return '科创板'
    if digits.startswith(('300', '301')):
        return '创业板'
    if digits.startswith(('600', '601', '603', '605')):
        return '上证主板'
    if digits.startswith(('000', '001', '002', '003')):
        return '深证主板'
    return '其他'


# ─── 曲线重采样 / 交易日历 ───
def _resample_curve(curve, resolution):
    """将月度曲线重采样为季线/年线，返回重采样后的曲线列表。"""
    if not curve or resolution == 'month':
        return curve
    groups = {}
    for d in curve:
        dt = pd.to_datetime(d['date'])
        if resolution == 'quarter':
            key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        else:
            key = str(dt.year)
        if key not in groups:
            groups[key] = {'returns': [], 'last_date': d['date']}
        groups[key]['returns'].append(d.get('return', 0))
        groups[key]['last_date'] = d['date']
    result = []
    cum = 1.0
    for g in groups.values():
        period_ret = float(np.prod([1 + r for r in g['returns']]) - 1)
        cum *= (1 + period_ret)
        result.append({
            'date': g['last_date'],
            'value': round(cum, 4),
            'return': round(period_ret, 6),
        })
    return result


def _load_trading_calendar(benchmark_id=None):
    try:
        calendar_df = get_a_share_trading_calendar()
        if calendar_df is not None and len(calendar_df) > 0 and 'date' in calendar_df.columns:
            dates = pd.to_datetime(calendar_df['date'], errors='coerce').dropna().sort_values().unique()
            if len(dates) > 0:
                return pd.DatetimeIndex(dates)
    except Exception:
        pass

    normalized_id = _normalize_benchmark_id(benchmark_id)
    candidate_ids = []
    if normalized_id in A_SHARE_INDEX_IDS:
        candidate_ids.append(normalized_id)
    if DEFAULT_BENCHMARK_ID not in candidate_ids:
        candidate_ids.append(DEFAULT_BENCHMARK_ID)
    for index_id in A_SHARE_INDEX_IDS:
        if index_id not in candidate_ids:
            candidate_ids.append(index_id)

    best_dates = pd.DatetimeIndex([])
    for index_id in candidate_ids:
        try:
            df = get_index_daily(index_id)
        except Exception:
            continue
        if df is None or len(df) == 0 or 'date' not in df.columns:
            continue
        dates = pd.to_datetime(df['date'], errors='coerce').dropna().sort_values().unique()
        if len(dates) > len(best_dates):
            best_dates = pd.DatetimeIndex(dates)
        elif len(dates) == len(best_dates) and len(dates) > 0 and pd.to_datetime(dates[-1]) > pd.to_datetime(best_dates[-1]):
            best_dates = pd.DatetimeIndex(dates)
    return best_dates


def _resolve_period_trading_dates(trade_date, period_curve, trading_calendar):
    trade_ts = pd.to_datetime(trade_date)
    period_len = len(period_curve or [])
    if period_len <= 0:
        return []
    if trading_calendar is None or len(trading_calendar) == 0:
        return []

    future_dates = trading_calendar[trading_calendar > trade_ts]
    if len(future_dates) == 0:
        return []
    return [pd.to_datetime(d) for d in future_dates[:period_len]]


def _build_holding_date_range(trade_date, period_curve, trading_calendar):
    trade_label = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
    trading_dates = _resolve_period_trading_dates(trade_date, period_curve, trading_calendar)
    if trading_dates:
        start_label = trading_dates[0].strftime('%Y-%m-%d')
        end_label = trading_dates[-1].strftime('%Y-%m-%d')
    else:
        start_label = trade_label
        end_label = trade_label
    return {
        'holding_start_date': start_label,
        'holding_end_date': end_label,
        'holding_date_range_label': f'{start_label} → {end_label}',
    }


# ─── 持仓快报 ───
def _is_open_snapshot_period(raw_stocks):
    return bool(raw_stocks) and all(stock.get('sell_price') is None for stock in raw_stocks)


def _build_daily_curve_slice(result_df, full_daily_curve, base_value=1.0, trading_calendar=None):
    """按结果区间切分并重置日线净值基准。"""
    if not full_daily_curve or len(result_df) == 0:
        return []

    period_curves = result_df.attrs.get('period_daily_curves', [])
    if period_curves:
        daily_curve = []
        running_value = float(base_value)
        for period_idx, period_curve in enumerate(period_curves):
            trade_date = pd.to_datetime(result_df.iloc[period_idx]['交易日期'])
            trading_dates = _resolve_period_trading_dates(trade_date, period_curve, trading_calendar)
            if not period_curve:
                daily_curve.append({
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'value': round(running_value, 6),
                    'return': 0.0,
                })
                continue
            prev = 1.0
            for day_idx, period_value in enumerate(period_curve, start=1):
                day_ret = 0.0 if prev == 0 else float(period_value / prev - 1)
                running_value *= (1 + day_ret)
                curve_date = trading_dates[day_idx - 1].strftime('%Y-%m-%d') if day_idx - 1 < len(trading_dates) else (trade_date + pd.Timedelta(days=day_idx)).strftime('%Y-%m-%d')
                daily_curve.append({
                    'date': curve_date,
                    'value': round(running_value, 6),
                    'return': round(day_ret, 6),
                })
                prev = period_value
        return daily_curve

    first_value = float(full_daily_curve[0].get('value', base_value))
    if first_value == 0:
        first_value = base_value

    return [{
        'date': p.get('date'),
        'value': round(float(p.get('value', 1.0)) / first_value * base_value, 6),
        'return': round(float(p.get('return', 0.0)), 6),
    } for p in full_daily_curve]


def _safe_float_or_none(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_stock_code(code):
    text = str(code or '').strip().lower()
    if not text:
        return ''
    if text.startswith(('sh', 'sz', 'bj')):
        return text[2:]
    return text


def _extract_open_stock_codes(result):
    if result is None or len(result) == 0 or '买入个股收益' not in result.columns:
        return []
    latest_open_stocks = None
    for raw in result['买入个股收益']:
        try:
            stocks = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if _is_open_snapshot_period(stocks):
            latest_open_stocks = stocks

    codes = set()
    for stock in latest_open_stocks or []:
        code = str(stock.get('code', '')).strip()
        if code:
            codes.add(code)
    return sorted(codes)


def _fetch_open_stock_quotes(result):
    codes = _extract_open_stock_codes(result)
    if not codes:
        return {}
    # Lazy import: get_stock_info 在仓库根（不在 stock_trade_demo/），
    # 从 stock_trade_demo cwd 跑 pytest 时模块顶层 import 会失败而无法 collect。
    # 这个函数只在 /api/backtest 触发开仓快照时被调用，生产路径下 sys.path 已注入仓库根。
    try:
        from get_stock_info import fetch_realtime_quotes_batch as _fetch_realtime_quotes_batch
        return _fetch_realtime_quotes_batch(codes)
    except Exception:
        return {}


def _resolve_quote_map(result, quote_map):
    """区分“未提供”和“明确跳过”；空 dict 是 compact 的有效选择。"""
    if quote_map is not None:
        return quote_map
    return _fetch_open_stock_quotes(result)


def _build_stock_payload(raw_stock, cap_per_stock, period_capital, stock_count, quote_map=None, allow_open_position=False):
    quote_map = quote_map or {}
    ret = float(raw_stock.get('return', 0) or 0)
    buy_price = _safe_float_or_none(raw_stock.get('buy_price')) or 0.0
    sell_price = _safe_float_or_none(raw_stock.get('sell_price'))
    code = raw_stock.get('code', '')
    if buy_price > 0:
        shares = int(cap_per_stock / buy_price / 100) * 100
        actual_invested = shares * buy_price
    else:
        shares = 0
        actual_invested = cap_per_stock

    normalized_code = _normalize_stock_code(code)
    quote = quote_map.get(code) or quote_map.get(normalized_code) or {}
    realtime_price = _safe_float_or_none(quote.get('latest'))
    is_open = bool(allow_open_position and sell_price is None)
    if is_open and realtime_price is not None:
        latest_price = realtime_price
        price_source = 'realtime'
    elif sell_price is not None:
        latest_price = sell_price
        price_source = 'exit'
    elif is_open and buy_price > 0:
        latest_price = buy_price
        price_source = 'buy_fallback'
    else:
        latest_price = None
        price_source = None

    position_market_value = round(float(shares * latest_price), 2) if shares > 0 and latest_price is not None else None
    position_weight = round(position_market_value / period_capital, 6) if position_market_value is not None and period_capital > 0 else raw_stock.get('weight', round(1.0 / stock_count, 4))

    if actual_invested > 0:
        backtest_pnl = round(float(actual_invested * ret), 2)
    else:
        backtest_pnl = round(float(cap_per_stock * ret), 2)

    display_pnl = backtest_pnl
    if is_open and position_market_value is not None and actual_invested > 0:
        display_pnl = round(float(position_market_value - actual_invested), 2)

    return {
        'code': code,
        'name': raw_stock.get('name', ''),
        'weight': raw_stock.get('weight', round(1.0 / stock_count, 4)),
        'position_weight': position_weight,
        'return': ret,
        'pnl': backtest_pnl,
        'display_pnl': display_pnl,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'exit_price': sell_price,
        'latest_price': round(float(latest_price), 2) if latest_price is not None else None,
        'is_open': is_open,
        'price_source': price_source,
        'shares': shares,
        'position_market_value': position_market_value,
        'factor_score': raw_stock.get('factor_score'),
        'rank': raw_stock.get('rank'),
        'industry_l2': raw_stock.get('industry_l2', ''),
        'market_label': raw_stock.get('market_label') or _infer_market_label(code),
        'pe': raw_stock.get('pe'),
        'pb': raw_stock.get('pb'),
        'market_cap': raw_stock.get('market_cap'),
        'selection_reason_summary': raw_stock.get('selection_reason_summary', ''),
        'selection_reason_detail': raw_stock.get('selection_reason_detail', []),
        'selection_fundamentals': raw_stock.get('selection_fundamentals', []),
        'selection_factor_breakdown': raw_stock.get('selection_factor_breakdown', []),
    }


def _build_holdings_payload(df, default_capital, quote_map=None, trading_calendar=None):
    holdings = []
    if df is None or len(df) == 0:
        return holdings
    quote_map = quote_map or {}
    period_curves = df.attrs.get('period_daily_curves', [])
    last_open_snapshot_idx = None
    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_stocks = []
        if '买入个股收益' in df.columns:
            try:
                raw_stocks = json.loads(row['买入个股收益'])
            except (json.JSONDecodeError, TypeError):
                pass
        if _is_open_snapshot_period(raw_stocks):
            last_open_snapshot_idx = row_idx

    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_stocks = []
        if '买入个股收益' in df.columns:
            try:
                raw_stocks = json.loads(row['买入个股收益'])
            except (json.JSONDecodeError, TypeError):
                pass
        is_open_snapshot = _is_open_snapshot_period(raw_stocks)
        if is_open_snapshot and row_idx != last_open_snapshot_idx:
            continue
        allow_open_position = is_open_snapshot and row_idx == last_open_snapshot_idx
        period_capital = float(row.get('当期本金', default_capital))
        n = len(raw_stocks) if raw_stocks else 1
        stocks = []
        for s in raw_stocks:
            target_weight = float(s.get('weight', round(1.0 / n, 4))) if n > 0 else 0
            cap_per_stock = period_capital * target_weight
            stocks.append(_build_stock_payload(s, cap_per_stock, period_capital, n, quote_map=quote_map, allow_open_position=allow_open_position))
        if not stocks:
            codes = str(row.get('买入股票代码', '')).strip().split()
            names = str(row.get('买入股票名称', '')).strip().split()
            stocks = [{
                'code': c,
                'name': names[i] if i < len(names) else '',
                'weight': round(1.0 / len(codes), 4) if codes else 0,
                'position_weight': round(1.0 / len(codes), 4) if codes else 0,
                'return': 0,
                'pnl': 0,
                'display_pnl': 0,
                'buy_price': 0,
                'sell_price': None,
                'exit_price': None,
                'latest_price': None,
                'is_open': False,
                'price_source': None,
                'shares': 0,
                'position_market_value': None,
                'factor_score': None,
                'rank': None,
                'industry_l2': '',
                'pe': None,
                'pb': None,
                'market_cap': None,
            } for i, c in enumerate(codes)]
        has_open_position = any(bool(s.get('is_open')) for s in stocks)
        display_period_pnl = round(sum(float(s.get('display_pnl', 0) or 0) for s in stocks), 2) if stocks else round(float(row.get('当期盈亏', 0)), 2)
        period_curve = period_curves[row_idx] if row_idx < len(period_curves) else []
        holding_range = _build_holding_date_range(row['交易日期'], period_curve, trading_calendar)
        holdings.append({
            'date': row['交易日期'].strftime('%Y-%m-%d'),
            'period_return': round(float(row['选股下周期涨跌幅']), 6),
            'period_pnl': round(float(row.get('当期盈亏', 0)), 2),
            'display_period_pnl': display_period_pnl,
            'period_pnl_label': '当前浮盈亏' if has_open_position else '持仓盈亏',
            'capital': round(period_capital, 2),
            'stocks': stocks,
            'stock_count': len(stocks),
            'benchmark_returns': _build_period_benchmark_returns(row['交易日期'], _index_returns_map()),
            **holding_range,
        })
    holdings.reverse()
    return holdings


# ─── 基准曲线 ───
def _compute_single_benchmark_curve(result, index_returns):
    if index_returns is None:
        return []
    lookup = build_period_lookup(index_returns)
    bm = []
    cum = 1.0
    for _, row in result.iterrows():
        idx_ret = get_index_return_for_date(row['交易日期'], lookup)
        cum *= (1 + idx_ret)
        bm.append({
            'date': row['交易日期'].strftime('%Y-%m-%d'),
            'value': round(cum, 4),
        })
    return bm


def _compute_single_benchmark_curve_daily(result, benchmark_id, trading_calendar=None):
    """近端验证窗口的 benchmark 曲线统一改为日线口径。

    设计原因：选股页近一月/近一季/近半年若继续用 INDEX_RETURNS_MAP 月度收益序列，
    会与择时页（日线 ETF 价格口径）严重背离。例如同一区间科创50在选股页只涨 11%，
    而择时页 ETF 涨 37%，用户会误以为其中一边算错。这里统一到：
      - A股 benchmark: 指数日线 close（csi1000/chinext/star50）
      - 美股代理 benchmark: ETF 日线 qfq close（nasdaq/sp500）
    """
    daily_curve = result.attrs.get('daily_equity_curve', []) if hasattr(result, 'attrs') else []
    if not daily_curve:
        return []
    dates = [pd.to_datetime(x['date']) for x in daily_curve if x.get('date')]
    if not dates:
        return []

    if benchmark_id in A_SHARE_INDEX_IDS:
        bench_df = get_index_daily(benchmark_id)
        price_col = 'close'
    else:
        bench_df = get_timing_etf_daily(benchmark_id)
        price_col = 'close'

    if bench_df is None or len(bench_df) == 0 or price_col not in bench_df.columns:
        return []
    bench_df = bench_df.copy()
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df[price_col] = pd.to_numeric(bench_df[price_col], errors='coerce')
    bench_df = bench_df.dropna(subset=['date', price_col]).sort_values('date')
    if len(bench_df) == 0:
        return []

    # 用 asof 对齐到每个 strategy daily curve date，要求 benchmark 至少有当天或更早 close
    s = bench_df.set_index('date')[price_col]
    aligned = []
    for dt in dates:
        price = s.asof(dt)
        if pd.notna(price):
            aligned.append((dt, float(price)))
    if len(aligned) < 2:
        return []
    base = aligned[0][1]
    if not base or base <= 0:
        return []
    return [
        {'date': dt.strftime('%Y-%m-%d'), 'value': round(px / base, 4)}
        for dt, px in aligned
    ]


def _compute_benchmark_curves(result, index_returns_map):
    curves = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        curves.append({
            'id': index_id,
            'name': cfg['name'],
            'curve': _compute_single_benchmark_curve(result, series),
        })
    return curves


def _build_period_benchmark_returns(trade_date, index_returns_map=None):
    benchmark_returns = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        period_lookup = build_period_lookup(series)
        benchmark_returns.append({
            'id': index_id,
            'name': cfg['name'],
            'return': round(float(get_index_return_for_date(trade_date, period_lookup)), 6),
        })
    return benchmark_returns


def _build_etf_monthly_returns(result_df):
    if result_df is None or len(result_df) == 0 or 'etf_close' not in result_df.columns:
        return None
    etf = result_df[['交易日期', 'etf_close']].copy()
    etf['交易日期'] = pd.to_datetime(etf['交易日期'])
    etf['etf_close'] = pd.to_numeric(etf['etf_close'], errors='coerce')
    etf = etf.dropna(subset=['交易日期', 'etf_close'])
    etf = etf[etf['etf_close'] > 0].drop_duplicates(subset=['交易日期']).sort_values('交易日期')
    if len(etf) < 2:
        return None
    daily_returns = etf.set_index('交易日期')['etf_close'].pct_change().dropna()
    if len(daily_returns) == 0:
        return None
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).dropna()
    return monthly_returns if len(monthly_returns) else None


def _month_start_from_end(end_date, months):
    end_ts = pd.to_datetime(end_date)
    start_ts = (end_ts - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
    return start_ts.normalize()


# ─── 窗口摘要 / 拆分指标 / 顶级序列化 ───
def build_selection_interval_windows(result, index_returns=None, benchmark_id=None, quote_map=None):
    if len(result) == 0:
        return {}

    trading_calendar = _load_trading_calendar(benchmark_id)
    full_result = result.copy().reset_index(drop=True)
    full_start = pd.to_datetime(full_result['交易日期'].min())
    full_end = pd.to_datetime(full_result['交易日期'].max())
    recent_6m_start = _month_start_from_end(full_end, 6)

    windows = {
        'pre_6m_history': (full_start, recent_6m_start - pd.Timedelta(days=1), False),
        'recent_6m': (_month_start_from_end(full_end, 6), full_end, True),
        'recent_1q': (_month_start_from_end(full_end, 3), full_end, True),
        'recent_1m': (_month_start_from_end(full_end, 1), full_end, True),
    }

    initial_capital = float(full_result.attrs.get('initial_capital', 100000))
    period_curves = full_result.attrs.get('period_daily_curves', [])
    curve_lookup = {
        pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
        for dt, curve in zip(full_result['交易日期'], period_curves)
    } if period_curves else {}

    index_returns_map = _index_returns_map()

    summary = {}
    for name, (start_date, end_date, reset_capital) in windows.items():
        df = full_result[(full_result['交易日期'] >= pd.to_datetime(start_date)) & (full_result['交易日期'] <= pd.to_datetime(end_date))].copy()
        if len(df) == 0:
            summary[name] = {
                'label': {
                    'pre_6m_history': '半年前历史',
                    'recent_6m': '近半年',
                    'recent_1q': '近一季',
                    'recent_1m': '近一月',
                }.get(name, name),
                'months': 0,
                'reset_capital': reset_capital,
                'date_range': {'start': None, 'end': None},
                'metrics': {},
                'holdings': [],
                'benchmark_curves': [],
                'daily_equity_curve': [],
            }
            continue

        if curve_lookup:
            df.attrs['period_daily_curves'] = [
                curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
                for dt in df['交易日期']
            ]

        df['累积净值'] = (1 + df['选股下周期涨跌幅']).cumprod()
        capital = float(initial_capital)
        capitals, pnls, cum_caps = [], [], []
        for _, row in df.iterrows():
            capitals.append(capital)
            pnl = capital * row['选股下周期涨跌幅']
            pnls.append(pnl)
            capital += pnl
            cum_caps.append(capital)
        df['当期本金'] = capitals
        df['当期盈亏'] = pnls
        df['累计资金'] = cum_caps

        ev = strategy_evaluate(df, initial_capital=initial_capital, index_returns=index_returns)

        def g(metric_name):
            if metric_name not in ev.index:
                return 'N/A'
            value = ev.loc[metric_name].values[0]
            if value is None:
                return 'N/A'
            text = str(value).strip()
            return 'N/A' if text.lower() in {'nan', 'nan%', 'none', 'undefined'} else text

        holdings = []
        if reset_capital:
            holdings = _build_holdings_payload(
                df,
                initial_capital,
                quote_map=_resolve_quote_map(df, quote_map),
                trading_calendar=trading_calendar,
            )

        # 近端窗口 benchmark 统一改成日线口径（与择时页一致）
        bm_curves_raw = []
        for index_id, _series in index_returns_map.items():
            curve_daily = _compute_single_benchmark_curve_daily(df, index_id, trading_calendar=trading_calendar)
            bm_curves_raw.append({'id': index_id, 'name': INDEX_CONFIGS[index_id]['name'], 'curve': curve_daily})

        df_final_val = float(df['累积净值'].iloc[-1]) if len(df) > 0 else 1.0
        benchmark_curves = []
        for item in bm_curves_raw:
            bm_c = item.get('curve', [])
            bm_final = bm_c[-1]['value'] if bm_c else 1.0
            excess_pct = round((df_final_val / bm_final - 1) * 100, 2) if bm_final != 0 else 0
            benchmark_curves.append({
                'id': item['id'],
                'name': item['name'],
                'curve': bm_c,
                'curve_quarterly': _resample_curve(bm_c, 'quarter'),
                'curve_yearly': _resample_curve(bm_c, 'year'),
                'excess_return_pct': excess_pct,
            })

        # 持仓期结束日期：取 daily_equity_curve 末端（最后一个交易日）作为展示 end，
        # 而不是最后一次换仓 canonical date。后者是"选股日"，前者才是"持仓结束日"。
        # 例如：选股日 2026-04-30，持仓穿越 5 月，日线末端是 2026-05-26 → 应展示 5/26。
        daily_eq_this = _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar)
        # 关键：后续 benchmark helper 也要用当前窗口切出来的日线日期，而不是 full result 的 attrs
        df.attrs['daily_equity_curve'] = daily_eq_this
        if daily_eq_this:
            _holding_end = daily_eq_this[-1]['date']
        else:
            _holding_end = df['交易日期'].max().strftime('%Y-%m-%d')

        summary[name] = {
            'label': {
                'pre_6m_history': '半年前历史',
                'recent_6m': '近半年',
                'recent_1q': '近一季',
                'recent_1m': '近一月',
            }.get(name, name),
            'months': len(df),
            'win_rate': round(float((df['选股下周期涨跌幅'] > 0).mean()), 4),
            'reset_capital': reset_capital,
            'initial_capital': initial_capital,
            'final_capital': float(df['累计资金'].iloc[-1]) if len(df) else initial_capital,
            'date_range': {
                'start': df['交易日期'].min().strftime('%Y-%m-%d'),
                'end': _holding_end,
            },
            'metrics': {
                'cumulative_return': g('累积净值'),
                'annual_return': g('年化收益'),
                'max_drawdown': g('最大回撤'),
                'max_dd_start': g('最大回撤开始'),
                'max_dd_end': g('最大回撤结束'),
                'calmar_ratio': g('年化收益/回撤比'),
                'final_capital': g('最终资金'),
                'total_return_pct': g('总收益率'),
                'total_pnl': g('总盈亏'),
                'beta': g('Beta'),
                'annual_alpha': g('年化Alpha'),
                'information_ratio': g('信息比率'),
                'r_squared': g('R-squared'),
                'up_capture': g('上行捕获率'),
                'down_capture': g('下行捕获率'),
            },
            'equity_curve': [
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ],
            'equity_curve_quarterly': _resample_curve([
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ], 'quarter'),
            'equity_curve_yearly': _resample_curve([
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ], 'year'),
            'daily_equity_curve': _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar),
            'holdings': holdings,
            'benchmark_curves': benchmark_curves,
        }

    return summary


def compute_split_metrics(result, split_date=SPLIT_DATE, index_returns=None, benchmark_id=None, quote_map=None):
    """
    将回测结果拆分为训练集和测试集，分别计算指标。

    训练集: 交易日 <= split_date
    测试集: 交易日 > split_date

    如果提供 index_returns，还会计算每段的 alpha/beta 归因指标和基准曲线。
    如果 split_date 为 None，返回空的 train/test（表示不拆分）。

    返回 dict:
      train: {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      test:  {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      split_date: str
    """
    if split_date is None or pd.isna(split_date):
        return {'train': None, 'test': None, 'split_date': None}

    trading_calendar = _load_trading_calendar(benchmark_id)
    train = result[result['交易日期'] <= split_date].copy()
    test = result[result['交易日期'] > split_date].copy()

    full_period_curves = result.attrs.get('period_daily_curves', [])
    if full_period_curves:
        period_curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(result['交易日期'], full_period_curves)
        }
        train.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in train['交易日期']
        ]
        test.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in test['交易日期']
        ]

    # 确定初始本金
    initial_capital = result.attrs.get('initial_capital', 100000)
    if '当期本金' in result.columns and len(result) > 0:
        initial_capital = result['当期本金'].iloc[0]

    index_returns_map = _index_returns_map()

    def compute_period(df, start_capital, include_holdings=False):
        if len(df) == 0:
            return None, start_capital
        df = df.copy()
        df['累积净值'] = (1 + df['选股下周期涨跌幅']).cumprod()

        # 重新计算绝对资金
        capital = float(start_capital)
        capitals = []
        pnls = []
        cum_caps = []
        for _, row in df.iterrows():
            capitals.append(capital)
            pnl = capital * row['选股下周期涨跌幅']
            pnls.append(pnl)
            capital += pnl
            cum_caps.append(capital)
        df['当期本金'] = capitals
        df['当期盈亏'] = pnls
        df['累计资金'] = cum_caps

        ev = strategy_evaluate(df, initial_capital=start_capital,
                              index_returns=index_returns)
        win_rate = round(float((df['选股下周期涨跌幅'] > 0).mean()), 4)
        monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['选股下周期涨跌幅']), 6)}
                   for _, r in df.iterrows()]
        final_capital = float(df['累计资金'].iloc[-1]) if len(df) > 0 else start_capital

        period_result = {
            'metrics': {
                'cumulative_return': str(ev.loc['累积净值'].values[0]) if '累积净值' in ev.index else 'N/A',
                'annual_return': str(ev.loc['年化收益'].values[0]) if '年化收益' in ev.index else 'N/A',
                'max_drawdown': str(ev.loc['最大回撤'].values[0]) if '最大回撤' in ev.index else 'N/A',
                'max_dd_start': str(ev.loc['最大回撤开始'].values[0]) if '最大回撤开始' in ev.index else 'N/A',
                'max_dd_end': str(ev.loc['最大回撤结束'].values[0]) if '最大回撤结束' in ev.index else 'N/A',
                'calmar_ratio': str(ev.loc['年化收益/回撤比'].values[0]) if '年化收益/回撤比' in ev.index else 'N/A',
                'final_capital': str(ev.loc['最终资金'].values[0]) if '最终资金' in ev.index else 'N/A',
                'total_return_pct': str(ev.loc['总收益率'].values[0]) if '总收益率' in ev.index else 'N/A',
                'total_pnl': str(ev.loc['总盈亏'].values[0]) if '总盈亏' in ev.index else 'N/A',
            },
            'win_rate': win_rate,
            'months': len(df),
            'monthly_returns': monthly,
            'daily_equity_curve': _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar),
            'initial_capital': start_capital,
            'final_capital': final_capital,
            'date_range': {
                'start': df['交易日期'].min().strftime('%Y-%m-%d'),
                'end': df['交易日期'].max().strftime('%Y-%m-%d'),
            },
        }
        # 持股明细
        if include_holdings:
            period_result['holdings'] = _build_holdings_payload(
                df,
                start_capital,
                quote_map=_resolve_quote_map(df, quote_map),
                trading_calendar=trading_calendar,
            )

        # ── 归因指标 ──
        if index_returns is not None:
            attr_keys_map = {
                'Beta': 'beta', '年化Alpha': 'annual_alpha',
                '信息比率': 'information_ratio', 'R-squared': 'r_squared',
                '上行捕获率': 'up_capture', '下行捕获率': 'down_capture',
            }
            for ev_key, json_key in attr_keys_map.items():
                if ev_key in ev.index:
                    period_result['metrics'][json_key] = str(ev.loc[ev_key].values[0])

        bm_curve = _compute_single_benchmark_curve(df, index_returns)
        period_result['benchmark_curve'] = bm_curve
        period_result['benchmark_curve_quarterly'] = _resample_curve(bm_curve, 'quarter')
        period_result['benchmark_curve_yearly'] = _resample_curve(bm_curve, 'year')

        bm_curves_raw = _compute_benchmark_curves(df, index_returns_map)
        df_final_val = float(df['累积净值'].iloc[-1]) if len(df) > 0 else 1.0
        bm_curves = []
        for item in bm_curves_raw:
            bm_c = item.get('curve', [])
            bm_final = bm_c[-1]['value'] if bm_c else 1.0
            excess_pct = round((df_final_val / bm_final - 1) * 100, 2) if bm_final != 0 else 0
            bm_curves.append({
                'id': item['id'],
                'name': item['name'],
                'curve': bm_c,
                'curve_quarterly': _resample_curve(bm_c, 'quarter'),
                'curve_yearly': _resample_curve(bm_c, 'year'),
                'excess_return_pct': excess_pct,
            })
        period_result['benchmark_curves'] = bm_curves
        period_result['active_benchmark'] = _get_benchmark_meta(benchmark_id)

        return period_result, final_capital

    train_result, train_final_cap = compute_period(train, initial_capital)
    test_result, _ = compute_period(test,
                                    train_final_cap if train_final_cap is not None else initial_capital,
                                    include_holdings=True)

    return {
        'train': train_result,
        'test': test_result,
        'split_date': split_date.strftime('%Y-%m-%d'),
        'initial_capital': initial_capital,
    }


def result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=None, compact=False):
    """将回测结果 DataFrame 转为前端 JSON，包含训练/测试集拆分"""
    # 资金曲线（倍数）
    equity_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['累积净值']), 4),
                     'return': round(float(r['选股下周期涨跌幅']), 6)}
                    for _, r in result.iterrows()]

    # 绝对资金曲线
    capital_curve = []
    if '累计资金' in result.columns:
        capital_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                          'value': round(float(r['累计资金']), 2),
                          'capital_start': round(float(r.get('当期本金', 0)), 2),
                          'pnl': round(float(r.get('当期盈亏', 0)), 2),
                          'return': round(float(r['选股下周期涨跌幅']), 6)}
                         for _, r in result.iterrows()]

    # 回撤
    cum = result['累积净值'].values
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    drawdown = [{'date': result.iloc[i]['交易日期'].strftime('%Y-%m-%d'),
                 'value': round(float(dd[i]), 6)}
                for i in range(len(result))]

    # 年度收益
    yr = result.copy()
    yr['年份'] = yr['交易日期'].dt.year
    yearly = yr.groupby('年份')['选股下周期涨跌幅'].apply(
        lambda x: (1 + x).prod() - 1)
    yearly_returns = [{'year': int(y), 'value': round(float(v), 6)}
                      for y, v in yearly.items()]

    # 月度收益
    monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                'value': round(float(r['选股下周期涨跌幅']), 6)}
               for _, r in result.iterrows()]
    trading_calendar = _load_trading_calendar(benchmark_id)
    daily_equity_curve = _build_daily_curve_slice(result, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar)
    # 顶部主 payload 的 benchmark 也必须使用当前过滤结果的日线曲线日期，而不是 full attrs
    result.attrs['daily_equity_curve'] = daily_equity_curve

    # 预计算各分辨率曲线（月线为原始数据，季线/年线由后端重采样）
    equity_curve_quarterly = _resample_curve(equity_curve, 'quarter')
    equity_curve_yearly = _resample_curve(equity_curve, 'year')

    # 持仓明细（含个股仓位占比和盈亏）
    # compact 是首屏图表请求，后面会删掉持仓明细，不应为了即将删除的
    # 字段同步等待实时行情网络请求。full payload 仍保留最新未平仓持仓的行情。
    holdings_quote_map = {} if compact else _fetch_open_stock_quotes(result)
    holdings = _build_holdings_payload(
        result,
        float(result.attrs.get('initial_capital', 100000)),
        quote_map=holdings_quote_map,
        trading_calendar=trading_calendar,
    ) if '买入个股收益' in result.columns else []

    def g(m):
        if m not in ev.index:
            return 'N/A'
        value = ev.loc[m].values[0]
        if value is None:
            return 'N/A'
        text = str(value).strip()
        if text.lower() in {'nan', 'nan%', 'none', 'undefined'}:
            return 'N/A'
        return text

    # 初始本金和费率信息
    initial_capital = float(result.attrs.get('initial_capital', 100000))

    active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

    # 统一区间窗口摘要
    interval_windows = build_selection_interval_windows(
        result,
        index_returns=active_benchmark_series,
        benchmark_id=active_benchmark_id,
        quote_map=holdings_quote_map,
    )

    # 兼容旧结构的临时拆分摘要
    split = compute_split_metrics(result, split_date,
                                  index_returns=active_benchmark_series,
                                  benchmark_id=active_benchmark_id,
                                  quote_map=holdings_quote_map)

    # 分别构建训练集和测试集的资金曲线（各自从 1 开始）
    train_curve = []
    test_curve = []
    train_capital_curve = []
    test_capital_curve = []
    if split_date and split and split.get('train') and split.get('test'):
        train_df = result[result['交易日期'] <= split_date].copy()
        train_df['累积净值'] = (1 + train_df['选股下周期涨跌幅']).cumprod()
        train_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                        'value': round(float(r['累积净值']), 4),
                        'return': round(float(r['选股下周期涨跌幅']), 6)}
                       for _, r in train_df.iterrows()]

        test_df = result[result['交易日期'] > split_date].copy()
        test_df['累积净值'] = (1 + test_df['选股下周期涨跌幅']).cumprod()
        test_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                       'value': round(float(r['累积净值']), 4),
                       'return': round(float(r['选股下周期涨跌幅']), 6)}
                      for _, r in test_df.iterrows()]

        # 训练/测试集的绝对资金曲线
        if split['train'] and 'final_capital' in split['train']:
            train_initial = split['train'].get('initial_capital', initial_capital)
            cap = float(train_initial)
            for _, r in train_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                train_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

        if split['test'] and 'initial_capital' in split['test']:
            test_initial = split['test'].get('initial_capital', initial_capital)
            cap = float(test_initial)
            for _, r in test_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                test_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

    # 预计算训练/测试集各分辨率曲线
    train_curve_quarterly = _resample_curve(train_curve, 'quarter')
    train_curve_yearly = _resample_curve(train_curve, 'year')
    test_curve_quarterly = _resample_curve(test_curve, 'quarter')
    test_curve_yearly = _resample_curve(test_curve, 'year')

    profile_summary = []
    strategy_meta = result.attrs.get('strategy_meta', {}) if hasattr(result, 'attrs') else {}
    if isinstance(strategy_meta, dict):
        profile_summary = strategy_meta.get('profile_summary', []) or []

    # 基准曲线及超额收益
    # 顶部 benchmark summary / 主图的 active benchmark 也统一成日线口径
    benchmark_curve = _compute_single_benchmark_curve_daily(result, active_benchmark_id, trading_calendar=trading_calendar)
    benchmark_curves_raw = []
    for index_id, _series in _index_returns_map().items():
        benchmark_curves_raw.append({
            'id': index_id,
            'name': INDEX_CONFIGS[index_id]['name'],
            'curve': _compute_single_benchmark_curve_daily(result, index_id, trading_calendar=trading_calendar),
        })
    strategy_final = equity_curve[-1]['value'] if equity_curve else 1.0
    benchmark_curves = []
    for item in benchmark_curves_raw:
        bm_curve = item.get('curve', [])
        bm_final = bm_curve[-1]['value'] if bm_curve else 1.0
        excess_pct = round((strategy_final / bm_final - 1) * 100, 2) if bm_final != 0 else 0
        benchmark_curves.append({
            'id': item['id'],
            'name': item['name'],
            'curve': bm_curve,
            'curve_quarterly': _resample_curve(bm_curve, 'quarter'),
            'curve_yearly': _resample_curve(bm_curve, 'year'),
            'excess_return_pct': excess_pct,
        })

    start_label = result['交易日期'].min().strftime('%Y-%m-%d') if len(result) > 0 else None
    # 结束标签用持仓期最后一个交易日（daily_equity_curve 末端），而非最后换仓日；
    # 换仓日是"选股时点"，持仓末端才是用户看到的实际区间结束。
    end_label = (daily_equity_curve[-1]['date'] if daily_equity_curve
                 else result['交易日期'].max().strftime('%Y-%m-%d') if len(result) > 0 else None)
    if len(result) <= 1:
        holdings_label = '近一月持股明细 & 仓位'
    else:
        holdings_label = '当前区间持股明细 & 仓位'
    holdings_context = {
        'label': holdings_label,
        'date_range': {
            'start': start_label,
            'end': end_label,
        },
        'holdings': holdings,
    } if len(result) > 0 else None

    payload = {
        'equity_curve': equity_curve,
        'equity_curve_quarterly': equity_curve_quarterly,
        'equity_curve_yearly': equity_curve_yearly,
        'daily_equity_curve': daily_equity_curve,
        'capital_curve': capital_curve,
        'train_equity_curve': train_curve,
        'train_equity_curve_quarterly': train_curve_quarterly,
        'train_equity_curve_yearly': train_curve_yearly,
        'test_equity_curve': test_curve,
        'test_equity_curve_quarterly': test_curve_quarterly,
        'test_equity_curve_yearly': test_curve_yearly,
        'train_daily_equity_curve': split.get('train', {}).get('daily_equity_curve', []) if split and split.get('train') else [],
        'test_daily_equity_curve': split.get('test', {}).get('daily_equity_curve', []) if split and split.get('test') else [],
        'train_capital_curve': train_capital_curve,
        'test_capital_curve': test_capital_curve,
        'drawdown': drawdown,
        'yearly_returns': yearly_returns,
        'monthly_returns': monthly,
        'holdings': holdings,
        'holdings_context': holdings_context,
        'metrics': {
            'cumulative_return': g('累积净值'),
            'annual_return': g('年化收益'),
            'max_drawdown': g('最大回撤'),
            'max_dd_start': g('最大回撤开始'),
            'max_dd_end': g('最大回撤结束'),
            'calmar_ratio': g('年化收益/回撤比'),
            'final_capital': g('最终资金'),
            'total_return_pct': g('总收益率'),
            'total_pnl': g('总盈亏'),
            'beta': g('Beta'),
            'annual_alpha': g('年化Alpha'),
            'information_ratio': g('信息比率'),
            'r_squared': g('R-squared'),
            'up_capture': g('上行捕获率'),
            'down_capture': g('下行捕获率'),
        },
        'initial_capital': initial_capital,
        'fee_info': {
            'c_rate': result.attrs.get('c_rate', 1.0 / 10000),
            't_rate': result.attrs.get('t_rate', 1 / 1000),
            'sell_cost': result.attrs.get('sell_cost', 1.0 / 10000 + 1 / 1000),
            'total_buy_fees': round(result.attrs.get('total_buy_fees', 0), 2),
            'total_sell_fees': round(result.attrs.get('total_sell_fees', 0), 2),
            'total_fees': round(result.attrs.get('total_fees', 0), 2),
        },
        'win_rate': round(float((result['选股下周期涨跌幅'] > 0).mean()), 4),
        'date_range': {
            'start': result['交易日期'].min().strftime('%Y-%m-%d'),
            # 结束日用持仓期末端（daily_equity_curve 最后一个交易日），不用换仓选股日
            'end': (daily_equity_curve[-1]['date'] if daily_equity_curve
                    else result['交易日期'].max().strftime('%Y-%m-%d')),
        },
        'total_months': len(result),
        'split': split,
        'interval_windows': interval_windows,
        'profile_summary': profile_summary,
        'active_benchmark': _get_benchmark_meta(active_benchmark_id),
        'benchmark_curve': benchmark_curve,
        'benchmark_curve_quarterly': _resample_curve(benchmark_curve, 'quarter'),
        'benchmark_curve_yearly': _resample_curve(benchmark_curve, 'year'),
        'benchmark_curves': benchmark_curves,
    }

    if compact:
        # 顶层持仓：只保留 has-holdings meta，详细内容点击时再拉 full payload
        payload['has_holdings'] = bool(holdings)
        payload['holdings'] = []
        if payload.get('holdings_context'):
            payload['holdings_context'] = {
                'label': payload['holdings_context'].get('label'),
                'date_range': payload['holdings_context'].get('date_range'),
                'has_holdings': bool(holdings),
            }

        # 删掉首屏完全不消费的大字段
        for k in ('split', 'train_equity_curve', 'train_equity_curve_quarterly', 'train_equity_curve_yearly',
                  'test_equity_curve', 'test_equity_curve_quarterly', 'test_equity_curve_yearly',
                  'train_daily_equity_curve', 'test_daily_equity_curve',
                  'train_capital_curve', 'test_capital_curve',
                  'capital_curve', 'drawdown', 'yearly_returns', 'monthly_returns', 'profile_summary'):
            payload.pop(k, None)

        # interval_windows 裁剪：保留 pre_6m_history / recent_6m 完整曲线；1q/1m 只保留 metrics/date_range
        iw = payload.get('interval_windows') or {}
        for name, w in list(iw.items()):
            if name in ('recent_1q', 'recent_1m') and isinstance(w, dict):
                iw[name] = {
                    'label': w.get('label'),
                    'months': w.get('months'),
                    'win_rate': w.get('win_rate'),
                    'reset_capital': w.get('reset_capital'),
                    'initial_capital': w.get('initial_capital'),
                    'final_capital': w.get('final_capital'),
                    'date_range': w.get('date_range'),
                    'metrics': w.get('metrics'),
                    'has_holdings': bool(w.get('holdings')),
                }
            elif isinstance(w, dict):
                w['has_holdings'] = bool(w.get('holdings'))
                w['holdings'] = []
                # benchmark_curves 只保留当前 active benchmark
                if w.get('benchmark_curves'):
                    active_id = active_benchmark_id
                    filtered = [b for b in w['benchmark_curves'] if b.get('id') == active_id]
                    w['benchmark_curves'] = filtered or w['benchmark_curves'][:1]
        payload['interval_windows'] = iw

        # 顶层 benchmark_curves 只保留当前 benchmark，减少重复
        if payload.get('benchmark_curves'):
            active_only = [b for b in payload['benchmark_curves'] if b.get('id') == active_benchmark_id]
            payload['benchmark_curves'] = active_only or payload['benchmark_curves'][:1]

    return payload
