#!/usr/bin/env python
"""离线计算实时风险因子 → strategy/risk_signals.json

把 VIX / 美债利率 / 收益率曲线 / 高收益债利差 / 指数距 200dma 偏离度 /
波动率分位 / 60 日涨幅 等因子按阈值翻译成中文风险/机会条目，按策略归类。
web_app 通过 _load_risk_signals() 只读消费。

运行：
    /Users/fatcat/opt/anaconda3/bin/python scripts/build_risk_signals.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
_DEMO_ROOT = os.path.join(_CODE_ROOT, 'stock_trade_demo')
sys.path.insert(0, _DEMO_ROOT)
from utils.atomic_io import atomic_write_json as _atomic_write_json
_RESOURCE_ROOT = os.path.abspath(os.environ.get('R0_RESOURCE_ROOT') or _CODE_ROOT)
_DATA_DIR = os.path.join(_RESOURCE_ROOT, 'data')
_DAILY_DIR = os.path.join(_RESOURCE_ROOT, '.cache')
_A_SHARE_MACRO_DIR = os.path.join(_DATA_DIR, 'a_share_macro')
_OUTPUT = os.path.join(_RESOURCE_ROOT, 'strategy', 'risk_signals.json')
os.makedirs(os.path.dirname(_OUTPUT), exist_ok=True)

# 指数 ID → daily csv 文件名（不含路径）
_INDEX_DAILY_FILES = {
    'csi1000': 'csi1000_daily.csv',
    'star50': 'star50_daily.csv',
    'chinext': 'chinext_daily.csv',
    'nasdaq': 'nasdaq_daily.csv',
    'sp500': 'sp500_daily.csv',
}

# 策略 → 用于「指数因子」的 daily 文件 key
_STRATEGY_TO_INDEX = {
    'csi1000_timing': 'csi1000',
    'star50_timing': 'star50',
    'chinext_timing': 'chinext',
    'sp500_timing': 'sp500',
    'macro_v32_timing': 'nasdaq',  # 宏观策略主要驱动美股纳指
}

_US_STRATEGIES = {'sp500_timing', 'macro_v32_timing'}
_A_SHARE_STRATEGIES = {'csi1000_timing', 'chinext_timing', 'star50_timing'}


def _load_fred(name: str) -> pd.DataFrame:
    """读取 data/fred_{name}.csv → DataFrame，index 为 DATE，值列改名为 'value'。"""
    fp = os.path.join(_DATA_DIR, f'fred_{name}.csv')
    df = pd.read_csv(fp, parse_dates=['DATE']).dropna().sort_values('DATE')
    df = df.set_index('DATE')
    df.columns = ['value']
    return df


def _load_index_daily(idx_key: str) -> pd.DataFrame:
    """读取指数 daily csv → index 为 date，包含 open/high/low/close/volume。"""
    fp = os.path.join(_DAILY_DIR, _INDEX_DAILY_FILES[idx_key])
    df = pd.read_csv(fp, parse_dates=['date']).dropna(subset=['close']).sort_values('date')
    return df.set_index('date')


def _percentile_1y(series: pd.Series, value: float) -> float:
    """value 在 series 近 252 个交易日（约 1 年）中的分位（0~1）。"""
    recent = series.dropna().iloc[-252:]
    if len(recent) < 30:
        return float('nan')
    return float((recent <= value).mean())


def _load_a_share_macro_csv(fname: str) -> pd.DataFrame:
    fp = os.path.join(_A_SHARE_MACRO_DIR, fname)
    if not os.path.exists(fp):
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp, parse_dates=['date']).sort_values('date')
        return df
    except Exception as exc:
        print(f"  [a_share_macro] {fname}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def _compute_a_share_snapshot() -> Optional[dict]:
    """读 data/a_share_macro/*.csv → ERP / 融资买入占比 / 流通换手率 + 各自 1y 分位。

    缺文件返回 None，让上层跳过 A 股因子条目（前端会优雅降级）。
    """
    pe = _load_a_share_macro_csv('pe_ttm.csv')
    bond = _load_a_share_macro_csv('cn10y.csv')
    sse = _load_a_share_macro_csv('sse_daily.csv')
    if pe.empty or bond.empty or sse.empty:
        return None

    # 取每个序列的最新有效值
    pe_last = pe.dropna(subset=['pe_median']).iloc[-1]
    bond_last = bond.dropna(subset=['cn10y_pct']).iloc[-1]
    sse_last = sse.dropna(subset=['sse_amount_yi']).iloc[-1]

    pe_med = float(pe_last['pe_median'])
    pe_q10y = float(pe_last['pe_median_q10y']) if pd.notna(pe_last.get('pe_median_q10y')) else None
    cn10y = float(bond_last['cn10y_pct'])

    # ERP = 1/PE - CN10Y/100，单位 % → 转成百分点显示
    erp_pct = (1.0 / pe_med - cn10y / 100.0) * 100.0

    # 构造 ERP 历史序列做 1y 分位（按 date 对齐 pe + bond）
    pe_idx = pe.set_index('date')['pe_median'].dropna()
    bond_idx = bond.set_index('date')['cn10y_pct'].dropna()
    common = pe_idx.index.intersection(bond_idx.index)
    if len(common) >= 30:
        erp_series = (1.0 / pe_idx.loc[common] - bond_idx.loc[common] / 100.0) * 100.0
        erp_pct_1y = round(_percentile_1y(erp_series, erp_pct), 3)
    else:
        erp_pct_1y = None

    turnover = float(sse_last['sse_turnover_float_pct']) if pd.notna(sse_last['sse_turnover_float_pct']) else None
    turnover_series = sse.set_index('date')['sse_turnover_float_pct'].dropna()
    turnover_q1y = round(_percentile_1y(turnover_series, turnover), 3) if turnover is not None and len(turnover_series) >= 30 else None

    # 融资买入占比 = 当日融资买入额 / 当日成交金额（两者均上交所，单位都已折算亿元）
    # 上交所融资数据通常 T+1 才发布，所以单独取「最近一个 margin 非空」的交易日
    margin_sse = sse.dropna(subset=['sse_margin_buy_yi', 'sse_amount_yi']).copy()
    margin_sse = margin_sse[margin_sse['sse_amount_yi'] > 0]
    if not margin_sse.empty:
        m_last = margin_sse.iloc[-1]
        margin_buy_pct = float(m_last['sse_margin_buy_yi']) / float(m_last['sse_amount_yi']) * 100.0
        margin_date = m_last['date'].strftime('%Y-%m-%d')
        ratio_series = margin_sse['sse_margin_buy_yi'] / margin_sse['sse_amount_yi'] * 100.0
        ratio_series.index = margin_sse['date']
        margin_buy_pct_q1y = round(_percentile_1y(ratio_series, margin_buy_pct), 3) if len(ratio_series) >= 30 else None
    else:
        margin_buy_pct = None
        margin_buy_pct_q1y = None
        margin_date = None

    return {
        'pe_median': round(pe_med, 2),
        'pe_q10y': round(pe_q10y, 3) if pe_q10y is not None else None,
        'cn10y_pct': round(cn10y, 3),
        'erp_pct': round(erp_pct, 3),
        'erp_q1y': erp_pct_1y,
        'sse_turnover_float_pct': round(turnover, 3) if turnover is not None else None,
        'sse_turnover_q1y': turnover_q1y,
        'sse_margin_buy_pct': round(margin_buy_pct, 2) if margin_buy_pct is not None else None,
        'sse_margin_buy_pct_q1y': margin_buy_pct_q1y,
        'pe_date': pe_last['date'].strftime('%Y-%m-%d'),
        'cn10y_date': bond_last['date'].strftime('%Y-%m-%d'),
        'sse_date': sse_last['date'].strftime('%Y-%m-%d'),
        'margin_date': margin_date,
    }


def compute_factor_snapshot() -> dict:
    snap: dict = {}

    # --- 宏观因子 ---
    vix = _load_fred('VIX')['value']
    snap['vix'] = {
        'value': round(float(vix.iloc[-1]), 2),
        'percentile_1y': round(_percentile_1y(vix, float(vix.iloc[-1])), 3),
        'date': vix.index[-1].strftime('%Y-%m-%d'),
    }

    ust10y = _load_fred('Treasury10Y')['value']
    last_10y = float(ust10y.iloc[-1])
    chg_60d_bp = (last_10y - float(ust10y.iloc[-min(60, len(ust10y))])) * 100  # 单位 bp
    snap['ust10y'] = {
        'value': round(last_10y, 2),
        'chg_60d_bp': round(chg_60d_bp, 1),
        'date': ust10y.index[-1].strftime('%Y-%m-%d'),
    }

    curve = _load_fred('YieldCurve_10Y2Y')['value']
    last_curve = float(curve.iloc[-1])
    # 过去 60 日是否出现过倒挂
    recent_60 = curve.iloc[-60:]
    was_inverted = bool((recent_60 < 0).any())
    snap['yield_curve'] = {
        'value': round(last_curve, 2),
        'was_inverted_in_60d': was_inverted,
        'now_positive': last_curve > 0,
        'date': curve.index[-1].strftime('%Y-%m-%d'),
    }

    hy = _load_fred('HighYieldSpread')['value']
    last_hy = float(hy.iloc[-1])
    snap['hy_spread'] = {
        'value': round(last_hy, 2),
        'percentile_1y': round(_percentile_1y(hy, last_hy), 3),
        'date': hy.index[-1].strftime('%Y-%m-%d'),
    }

    # --- 各指数因子 ---
    indexes = {}
    for key in _INDEX_DAILY_FILES:
        df = _load_index_daily(key)
        close = df['close']
        if len(close) < 220:
            continue
        last = float(close.iloc[-1])
        ma200 = float(close.iloc[-200:].mean())
        dist_pct = (last - ma200) / ma200 * 100

        rets = close.pct_change().dropna()
        vol20 = rets.iloc[-20:].std()
        vol20_1y = rets.rolling(20).std().iloc[-252:]
        vol20_pct = float((vol20_1y <= vol20).mean()) if len(vol20_1y.dropna()) >= 30 else float('nan')

        ret60 = (last / float(close.iloc[-60]) - 1) * 100 if len(close) >= 60 else float('nan')

        indexes[key] = {
            'close': round(last, 3),
            'dist_200dma_pct': round(dist_pct, 2),
            'vol20_pct_1y': round(vol20_pct, 3) if vol20_pct == vol20_pct else None,
            'ret60d_pct': round(ret60, 2) if ret60 == ret60 else None,
            'date': close.index[-1].strftime('%Y-%m-%d'),
        }
    snap['indexes'] = indexes

    # --- A 股估值与情绪因子（仅前端展示，不影响策略仓位）---
    a_share = _compute_a_share_snapshot()
    if a_share is not None:
        snap['a_share'] = a_share
    return snap


# ---------- 规则评估 ----------

def _us_macro_bullish_risks(snap: dict, strategy_id: str) -> list[str]:
    out: list[str] = []
    vix = snap['vix']
    if vix['value'] > 25 or vix['percentile_1y'] >= 0.80:
        out.append(
            f"VIX 当前 {vix['value']}，处近 1 年 {vix['percentile_1y']*100:.0f}% 分位，"
            f"市场恐慌情绪升温，进场后短期波动放大风险高。"
        )
    elif vix['value'] < 13:
        out.append(
            f"VIX 当前 {vix['value']} 偏低，历史上极低波动率常预示均值回归，"
            f"VIX 反弹时美股回撤会被放大。"
        )

    ust = snap['ust10y']
    if ust['chg_60d_bp'] > 50:
        out.append(
            f"10 年期美债收益率近 60 日上行 {ust['chg_60d_bp']:.0f}bp 至 {ust['value']}%，"
            f"对成长股估值压制明显（{'纳指尤甚' if strategy_id != 'sp500_timing' else '关注估值溢价回落'}）。"
        )

    curve = snap['yield_curve']
    if curve['value'] < 0:
        out.append(
            f"美债 10Y-2Y 收益率曲线倒挂（{curve['value']}%），历史上倒挂后 6–18 个月常出现衰退。"
        )

    hy = snap['hy_spread']
    if hy['value'] > 5.0 or hy['percentile_1y'] >= 0.80:
        out.append(
            f"高收益债利差 {hy['value']}%，处近 1 年 {hy['percentile_1y']*100:.0f}% 分位，"
            f"信用利差走阔通常领先股市风险偏好下行。"
        )
    return out


def _us_macro_bearish_opps(snap: dict) -> list[str]:
    out: list[str] = []
    if snap['vix']['value'] > 30:
        out.append(
            f"VIX 当前 {snap['vix']['value']} > 30 处极端恐慌区，历史上往往对应中期底部，"
            f"可考虑分批配置长债 ETF（TLT）/ 黄金 ETF（GLD）。"
        )
    ust = snap['ust10y']
    if ust['chg_60d_bp'] < -50:
        out.append(
            f"10 年期美债收益率近 60 日下行 {abs(ust['chg_60d_bp']):.0f}bp 至 {ust['value']}%，"
            f"降息预期升温，长债 ETF（TLT）和黄金 ETF（GLD）有补涨空间。"
        )
    curve = snap['yield_curve']
    if curve['was_inverted_in_60d'] and curve['now_positive']:
        out.append(
            f"美债收益率曲线近期由倒挂转正（当前 {curve['value']}%），"
            f"历史上多为衰退确认信号，防御资产（TLT/GLD/SGOV）配置价值提升。"
        )
    return out


def _index_bullish_risks(idx_stats: dict, idx_label: str) -> list[str]:
    out: list[str] = []
    if idx_stats is None:
        return out
    dist = idx_stats.get('dist_200dma_pct')
    vol_pct = idx_stats.get('vol20_pct_1y')
    ret60 = idx_stats.get('ret60d_pct')
    if dist is not None and dist > 15:
        out.append(
            f"{idx_label} 已较 200 日均线高出 {dist:.1f}%，短期超买，回归均值压力较大。"
        )
    if vol_pct is not None and vol_pct >= 0.90:
        out.append(
            f"{idx_label} 20 日波动率处近 1 年 {vol_pct*100:.0f}% 分位，"
            f"波动显著放大，建议适度降低单次加仓比例。"
        )
    if ret60 is not None and ret60 > 30:
        out.append(
            f"{idx_label} 近 60 日累计涨幅 {ret60:.1f}%，短期斜率较陡，注意获利盘抛压。"
        )
    return out


def _index_bearish_opps(idx_stats: dict, idx_label: str) -> list[str]:
    out: list[str] = []
    if idx_stats is None:
        return out
    dist = idx_stats.get('dist_200dma_pct')
    ret60 = idx_stats.get('ret60d_pct')
    if dist is not None and dist < -15:
        out.append(
            f"{idx_label} 较 200 日均线低 {abs(dist):.1f}%，处于深度超卖区，"
            f"可分批观察反弹机会而非一味回避。"
        )
    if ret60 is not None and ret60 < -20:
        out.append(
            f"{idx_label} 近 60 日累计下跌 {abs(ret60):.1f}%，下行空间收窄，"
            f"可关注情绪修复后的反弹窗口。"
        )
    return out


def _a_share_macro_risks(snap: dict) -> list[str]:
    """A 股估值与情绪类风险（看多视角下显示）。仅做提示，不进策略仓位。"""
    a = snap.get('a_share')
    if not a:
        return []
    out: list[str] = []

    erp = a.get('erp_pct')
    erp_q = a.get('erp_q1y')
    pe = a.get('pe_median')
    pe_q10y = a.get('pe_q10y')
    cn10y = a.get('cn10y_pct')
    erp_date = a.get('pe_date')

    if erp is not None and cn10y is not None and pe is not None:
        if erp < 1.0:
            out.append(
                f"A 股 ERP 仅 {erp:.2f}pct（1/PE中位数 − 10Y国债 {cn10y:.2f}%，{erp_date}），"
                f"估值-利率剪刀差接近历史极端，看多需对回撤更谨慎。"
            )
        elif erp < 2.0:
            out.append(
                f"A 股 ERP {erp:.2f}pct（1/PE中位数 − 10Y国债 {cn10y:.2f}%），"
                f"股票相对债券的吸引力低于中位水平，注意估值压力。"
            )
        elif erp_q is not None and erp_q <= 0.10:
            out.append(
                f"A 股 ERP {erp:.2f}pct 已落入近 1 年 {erp_q*100:.0f}% 分位，"
                f"相对债券的安全垫处低位，加仓节奏宜放慢。"
            )

    if pe_q10y is not None and pe_q10y >= 0.85:
        out.append(
            f"全 A 个股 PE-TTM 中位数 {pe:.1f}x，处近 10 年 {pe_q10y*100:.0f}% 分位，"
            f"估值百分位偏高，结构性风险积累。"
        )

    turn = a.get('sse_turnover_float_pct')
    turn_q = a.get('sse_turnover_q1y')
    sse_date = a.get('sse_date')
    if turn is not None:
        if turn >= 3.0:
            out.append(
                f"上交所流通换手率 {turn:.2f}%（{sse_date}）已升至历史过热区（≥3%），"
                f"放量见顶后常伴随缩量阴跌，注意仓位管理。"
            )
        elif turn_q is not None and turn_q >= 0.90:
            out.append(
                f"上交所流通换手率 {turn:.2f}%，处近 1 年 {turn_q*100:.0f}% 分位，"
                f"短期情绪偏热，加仓建议分批。"
            )

    margin_pct = a.get('sse_margin_buy_pct')
    margin_q = a.get('sse_margin_buy_pct_q1y')
    margin_date = a.get('margin_date')
    if margin_pct is not None:
        if margin_pct >= 11.0:
            out.append(
                f"上交所融资买入额占当日成交额 {margin_pct:.1f}%（{margin_date}），"
                f"杠杆资金参与度接近 2021 顶部（12–13%）水平，回调时易放大跌幅。"
            )
        elif margin_q is not None and margin_q >= 0.90:
            out.append(
                f"上交所融资买入额占比 {margin_pct:.1f}%（{margin_date}），近 1 年 {margin_q*100:.0f}% 分位，"
                f"杠杆资金近期持续涌入，需警惕情绪反转。"
            )
    return out


def _a_share_macro_opps(snap: dict) -> list[str]:
    """A 股估值与情绪类机会（看空视角下显示）。"""
    a = snap.get('a_share')
    if not a:
        return []
    out: list[str] = []

    erp = a.get('erp_pct')
    erp_q = a.get('erp_q1y')
    cn10y = a.get('cn10y_pct')
    erp_date = a.get('pe_date')
    pe_q10y = a.get('pe_q10y')

    if erp is not None and erp >= 4.0:
        out.append(
            f"A 股 ERP {erp:.2f}pct（1/PE中位数 − 10Y国债 {cn10y:.2f}%，{erp_date}）处历史高位，"
            f"股相对债的安全垫厚，可考虑分批建仓宽基 ETF。"
        )
    elif erp_q is not None and erp_q >= 0.90:
        out.append(
            f"A 股 ERP 处近 1 年 {erp_q*100:.0f}% 分位（{erp:.2f}pct），"
            f"股票性价比相对历史明显改善。"
        )

    if pe_q10y is not None and pe_q10y <= 0.20:
        out.append(
            f"全 A 个股 PE-TTM 中位数处近 10 年 {pe_q10y*100:.0f}% 分位，"
            f"估值已回到便宜区，中长期赔率上修。"
        )

    turn = a.get('sse_turnover_float_pct')
    if turn is not None and turn <= 0.8:
        out.append(
            f"上交所流通换手率仅 {turn:.2f}%，市场极度冷清，"
            f"历史上低换手区常出现政策催化或基本面拐点带来的反弹窗口。"
        )
    return out


def build_strategy_signals(snap: dict) -> dict:
    by_strategy = {}
    for strategy_id, idx_key in _STRATEGY_TO_INDEX.items():
        idx_stats = snap['indexes'].get(idx_key)
        idx_label = {
            'csi1000': '中证1000', 'star50': '科创50', 'chinext': '创业板',
            'nasdaq': '纳指', 'sp500': '标普500',
        }.get(idx_key, idx_key)

        bullish = _index_bullish_risks(idx_stats, idx_label)
        bearish = _index_bearish_opps(idx_stats, idx_label)
        if strategy_id in _US_STRATEGIES:
            bullish = _us_macro_bullish_risks(snap, strategy_id) + bullish
            bearish = _us_macro_bearish_opps(snap) + bearish
        if strategy_id in _A_SHARE_STRATEGIES:
            # A 股估值/情绪因子放在最前面，单只指数的技术因子在后
            bullish = _a_share_macro_risks(snap) + bullish
            bearish = _a_share_macro_opps(snap) + bearish

        by_strategy[strategy_id] = {
            'bullish_risks_dynamic': bullish,
            'bearish_opportunities_dynamic': bearish,
        }
    return by_strategy


def main():
    snap = compute_factor_snapshot()
    by_strategy = build_strategy_signals(snap)

    # as_of 取所有实际消费到的底层数据中的最早日期，确保文件头时间戳不会比任一子因子“虚新”。
    all_dates = [snap['vix']['date'], snap['ust10y']['date'],
                 snap['yield_curve']['date'], snap['hy_spread']['date']]
    all_dates += [v['date'] for v in snap['indexes'].values()]
    a_share = snap.get('a_share') or {}
    all_dates += [a_share.get(k) for k in ('pe_date', 'cn10y_date', 'sse_date', 'margin_date') if a_share.get(k)]
    as_of = min(all_dates)

    payload = {
        'as_of': as_of,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'factor_snapshot': snap,
        'by_strategy': by_strategy,
    }

    os.makedirs(os.path.dirname(_OUTPUT), exist_ok=True)
    _atomic_write_json(_OUTPUT, payload, ensure_ascii=False, indent=2,
                       produced_by="scripts/build_risk_signals")

    # stdout 摘要
    print(f"[OK] {_OUTPUT}")
    print(f"     as_of = {as_of}")
    print(f"     VIX={snap['vix']['value']} ({snap['vix']['percentile_1y']*100:.0f}%ile)  "
          f"10Y={snap['ust10y']['value']}% (60d {snap['ust10y']['chg_60d_bp']:+.0f}bp)  "
          f"10Y-2Y={snap['yield_curve']['value']}%  HY={snap['hy_spread']['value']}%")
    for strategy_id, sigs in by_strategy.items():
        nb, no = len(sigs['bullish_risks_dynamic']), len(sigs['bearish_opportunities_dynamic'])
        print(f"     {strategy_id:20s}  bullish_risks={nb}  bearish_opps={no}")


if __name__ == '__main__':
    main()
