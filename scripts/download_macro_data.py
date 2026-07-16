"""
美股宏观因子 + ETF价格历史数据下载脚本
数据源: FRED (St. Louis Fed) + Yahoo Finance
保存路径: /Users/fatcat/Desktop/quant/data/
"""
import argparse
import os
import sys
import time
import pandas as pd
from datetime import datetime
from pandas_datareader import data as pdr
import yfinance as yf

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCE_ROOT = os.path.abspath(os.environ.get('R0_RESOURCE_ROOT') or _CODE_ROOT)
DATA_DIR = os.path.join(_RESOURCE_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")

# FRED 宏观指标代码映射
FRED_SERIES = {
    "CPIAUCSL":   "CPI_headline",         # 整体CPI
    "CPILFESL":   "CPI_core",             # 核心CPI
    "PCEPI":      "PCE_headline",         # PCE
    "PCEPILFE":   "PCE_core",             # 核心PCE
    "UNRATE":     "Unemployment",         # 失业率
    "PAYEMS":     "NonFarmPayrolls",      # 非农就业人数
    "FEDFUNDS":   "FedFundsRate",         # 联邦基金利率
    "DGS10":      "Treasury10Y",          # 10年期美债收益率
    "DGS2":       "Treasury2Y",           # 2年期美债收益率
    "T10Y2Y":     "YieldCurve_10Y2Y",     # 期限利差
    "DTWEXBGS":   "DollarIndex",          # 美元指数(广义)
    "VIXCLS":     "VIX",                  # 恐慌指数
    "DCOILWTICO": "WTI_Oil",              # WTI原油
    "GDPC1":      "GDP_Real",             # 实际GDP
    "INDPRO":     "IndustrialProduction", # 工业生产指数
    "UMCSENT":    "ConsumerSentiment",    # 密歇根消费者信心
    "M2SL":       "M2_MoneySupply",       # M2货币供应
    "BAMLH0A0HYM2": "HighYieldSpread",    # 高收益债利差
}

# ETF / 指数代码
TICKERS = {
    "QQQ":  "Nasdaq100_ETF",
    "SPY":  "SP500_ETF",
    "TQQQ": "Nasdaq100_3xETF",
    "DIA":  "DowJones_ETF",
    "IWM":  "Russell2000_ETF",
    "TLT":  "Treasury20Y_ETF",
    "GLD":  "Gold_ETF",
    "^VIX": "VIX_Index",
}

def _load_existing_fred(name):
    fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
    if not os.path.exists(fp):
        return None
    try:
        return pd.read_csv(fp, parse_dates=['DATE']).set_index('DATE')
    except Exception:
        return None


def download_fred(strict=False):
    print("=" * 60)
    print("下载 FRED 宏观数据")
    print("=" * 60)
    summary = []
    failures = []
    for code, name in FRED_SERIES.items():
        fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
        success = False
        for attempt in range(3):
            try:
                df = pdr.DataReader(code, "fred", START, END)
                df.columns = [name]
                df.to_csv(fp)
                n = len(df)
                first = df.index.min().strftime("%Y-%m-%d")
                last  = df.index.max().strftime("%Y-%m-%d")
                print(f"  ✓ {name:25s} {n:5d} rows  {first} → {last}")
                summary.append({"name": name, "code": code, "rows": n,
                                "start": first, "end": last,
                                "latest_value": float(df.iloc[-1, 0])})
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ✗ {name}: {e}")
                else:
                    print(f"  … {name}: 第 {attempt + 1} 次失败，1s 后重试 ({e})")
                    time.sleep(1)
        if success:
            continue
        old = _load_existing_fred(name)
        if old is not None and len(old) > 0:
            first = old.index.min().strftime("%Y-%m-%d")
            last = old.index.max().strftime("%Y-%m-%d")
            latest_val = old.iloc[-1, 0]
            latest_val = None if pd.isna(latest_val) else float(latest_val)
            print(f"  ↺ {name:25s} 保留旧缓存  {first} → {last}")
            summary.append({"name": name, "code": code, "rows": len(old),
                            "start": first, "end": last,
                            "latest_value": latest_val, 'stale_cache_retained': True})
        else:
            failures.append(name)
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "_fred_summary.csv"), index=False)
    if strict and failures:
        raise RuntimeError(f"FRED 数据缺失且无旧缓存可保留: {failures}")
    return failures

def download_yf():
    print("=" * 60)
    print("下载 Yahoo Finance ETF/指数数据")
    print("=" * 60)
    summary = []
    for ticker, name in TICKERS.items():
        try:
            df = yf.download(ticker, start=START, end=END,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  ✗ {name}: empty")
                continue
            # 处理多层列
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            fp = os.path.join(DATA_DIR, f"yf_{name}.csv")
            df.to_csv(fp)
            n = len(df)
            first = df.index.min().strftime("%Y-%m-%d")
            last  = df.index.max().strftime("%Y-%m-%d")
            close_col = "Close" if "Close" in df.columns else df.columns[0]
            latest = float(df[close_col].iloc[-1])
            print(f"  ✓ {name:25s} {n:5d} rows  {first} → {last}  last={latest:.2f}")
            summary.append({"name": name, "ticker": ticker, "rows": n,
                            "start": first, "end": last, "latest_close": latest})
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "_etf_summary.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-yf', action='store_true', help='只刷新 FRED 宏观，不下载 Yahoo Finance 研究数据')
    parser.add_argument('--strict-fred', action='store_true', help='若 FRED 某序列抓取失败且本地无旧缓存，则返回非 0')
    args = parser.parse_args()

    failures = download_fred(strict=args.strict_fred)
    if failures:
        print(f"\n[FRED] 以下序列本次抓取失败，但已保留旧缓存: {failures}")
    if not args.skip_yf:
        download_yf()
    print("\n全部完成。数据保存于:", DATA_DIR)
    if args.strict_fred and failures:
        sys.exit(1)
