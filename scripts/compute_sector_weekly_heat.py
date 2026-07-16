"""
按申万一级行业计算周级别涨跌幅热度。

原理：
  - stock_data 是月度频率，每行包含 `下周期每天涨跌幅`（下一月内每日收益列表）
  - 把每月的日线按 5 个交易日一周分组，计算该周的累计收益
  - 按流通市值加权聚合到申万一级行业
  - 输出 strategy/sector_weekly_heat.csv

用法：
  python scripts/compute_sector_weekly_heat.py
"""

import ast, math, os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

CODE_ROOT = Path(__file__).parent.parent
RESOURCE_ROOT = Path(os.environ.get('R0_RESOURCE_ROOT') or CODE_ROOT).expanduser().resolve()
sys.path.insert(0, str(CODE_ROOT / 'stock_trade_demo'))
from utils.atomic_io import atomic_write_csv as _atomic_write_csv
DATA_PATH = RESOURCE_ROOT / "stock_trade_demo" / "stock_data.parquet"
OUT_PATH = RESOURCE_ROOT / "strategy" / "sector_weekly_heat.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

INDUSTRY_COL = "新版申万一级行业名称"
DAYS_PER_WEEK = 5
MIN_STOCKS = 5          # 一个 (行业, 周) 最少需要几只股票才纳入

def parse_daily(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

def week_chunk(daily_returns, week_idx: int):
    """取第 week_idx 周（0-based）的日收益 chunk；最后一周可能不足 5 天。"""
    start = week_idx * DAYS_PER_WEEK
    return daily_returns[start : start + DAYS_PER_WEEK]


def chunk_cumret(chunk) -> float:
    if not chunk:
        return np.nan
    return float(np.prod([1 + r for r in chunk]) - 1)

def month_offset(date: pd.Timestamp, months: int) -> pd.Timestamp:
    """向后推 months 个月（返回该月最后一天，用 period 做对齐）"""
    p = date.to_period("M") + months
    return p.to_timestamp("M")  # 月末

def main():
    print("读取数据...")
    df = pd.read_parquet(DATA_PATH)

    # 只保留有日线数据且行业标签有效的行
    df = df[df["是否交易"] == 1].copy()
    df["daily_rets"] = df["下周期每天涨跌幅"].apply(parse_daily)
    # 至少要有 1 天日线才能算 partial 周（之前是 >= DAYS_PER_WEEK，会丢掉当月 partial 周）
    df = df[df["daily_rets"].apply(len) >= 1].copy()
    df = df[df[INDUSTRY_COL].notna() & (df[INDUSTRY_COL] != "")].copy()

    # 下周期 = 交易日期的下一个月（兼容 string / Timestamp 两种 dtype）
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    df["next_month"] = df["交易日期"].apply(lambda d: month_offset(d, 1))
    # 最大有几周（含 partial 末周）
    max_weeks = int(math.ceil(df["daily_rets"].apply(len).max() / DAYS_PER_WEEK))
    print(f"月度条数: {len(df):,}，每月最多 {max_weeks} 周（含 partial 末周）")

    # 展开：每行 → 最多 max_weeks 行（对应该下周期的各周）
    records = []
    for row in df.itertuples():
        daily = row.daily_rets
        mktcap = row.流通市值 if not np.isnan(row.流通市值) else 0.0
        total_chunks = int(math.ceil(len(daily) / DAYS_PER_WEEK))
        for w in range(total_chunks):
            chunk = week_chunk(daily, w)
            if not chunk:
                continue
            cr = chunk_cumret(chunk)
            if np.isnan(cr):
                continue
            records.append({
                "year_month": row.next_month.to_period("M"),
                "week_in_month": w + 1,           # 1-based
                "stock_code": row.股票代码,
                "industry": getattr(row, INDUSTRY_COL),
                "mktcap": mktcap,
                "weekly_ret": cr,
                "n_days_in_week": len(chunk),
                "is_partial": len(chunk) < DAYS_PER_WEEK,
            })

    long_df = pd.DataFrame(records)
    long_df["year_month_str"] = long_df["year_month"].astype(str)

    # 构建全局唯一的周标签，如 "2025-04 W2"
    long_df["week_label"] = long_df["year_month_str"] + " W" + long_df["week_in_month"].astype(str)

    print(f"展开后条数: {len(long_df):,}")

    # 按 (周标签, 行业) 聚合 → 流通市值加权平均收益
    def wavg(g):
        w = g["mktcap"].clip(lower=0)
        if w.sum() == 0:
            w = pd.Series(np.ones(len(g)), index=g.index)
        return np.average(g["weekly_ret"], weights=w)

    grp = long_df.groupby(["year_month", "week_in_month", "week_label", "industry"])
    agg = grp.apply(
        lambda g: pd.Series({
            "weekly_ret": wavg(g),
            "n_stocks": len(g),
        })
    ).reset_index()

    # 全局 (year_month, week_in_month) 的 partial 判定：用"主流股票当周天数"的众数。
    # max 会被「数据稍旧的老股票恰好凑够 5 天」误判 full，
    # min 会被 IPO 当周拉低；众数最接近市场真实的 partial 状态。
    week_meta = (
        long_df.groupby(["year_month", "week_in_month"])["n_days_in_week"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index()
        .rename(columns={"n_days_in_week": "n_days_in_week_typ"})
    )
    week_meta["is_partial"] = week_meta["n_days_in_week_typ"] < DAYS_PER_WEEK
    agg = agg.merge(week_meta, on=["year_month", "week_in_month"], how="left")
    agg = agg.rename(columns={"n_days_in_week_typ": "n_days_in_week"})

    agg = agg[agg["n_stocks"] >= MIN_STOCKS].copy()
    agg["weekly_ret_pct"] = (agg["weekly_ret"] * 100).round(2)
    agg["is_partial"] = agg["is_partial"].astype(bool)
    agg["n_days_in_week"] = agg["n_days_in_week"].astype(int)
    agg.sort_values(["year_month", "week_in_month", "weekly_ret"], ascending=[True, True, False], inplace=True)

    # 保存
    out_cols = ["year_month", "week_in_month", "week_label", "industry",
                "weekly_ret_pct", "n_stocks", "is_partial", "n_days_in_week"]
    _atomic_write_csv(OUT_PATH, agg[out_cols], index=False, encoding="utf-8-sig",
                      produced_by="scripts/compute_sector_weekly_heat")
    print(f"已保存 → {OUT_PATH}  (行数: {len(agg):,})")

    # ── 最近 8 周行业热度预览 ──
    latest_weeks = sorted(agg["week_label"].unique())[-8:]
    pivot = (
        agg[agg["week_label"].isin(latest_weeks)]
        .pivot_table(index="industry", columns="week_label", values="weekly_ret_pct")
    )
    # 排列列顺序
    pivot = pivot[sorted(pivot.columns)]
    # 按最新一周排序
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    print("\n===== 最近 8 周行业涨跌幅（%，市值加权）=====")
    print(pivot.to_string())

    # 热度汇总：最近 4 周平均收益 × 行业排名
    recent_4w = sorted(agg["week_label"].unique())[-4:]
    heat = (
        agg[agg["week_label"].isin(recent_4w)]
        .groupby("industry")["weekly_ret_pct"]
        .mean()
        .sort_values(ascending=False)
        .rename("近4周平均周收益(%)")
    )
    print("\n===== 近 4 周行业热度排名 =====")
    print(heat.to_string())

if __name__ == "__main__":
    main()
