"""
ETF/指数数据下载 - 使用stooq作为备选源(免费,免key)
"""
import os
import pandas as pd
from datetime import datetime
from pandas_datareader import data as pdr

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
START = datetime(2005, 1, 1)
END = datetime.today()

# stooq 代码: 美股代码后加 .us
INDEX_TICKERS = {
    "spy.us":   "SP500_ETF",
    "qqq.us":   "Nasdaq100_ETF",
    "tqqq.us":  "Nasdaq100_3xETF",
    "dia.us":   "DowJones_ETF",
    "iwm.us":   "Russell2000_ETF",
    "tlt.us":   "Treasury20Y_ETF",
    "gld.us":   "Gold_ETF",
    "^vix":     "VIX_Index",
    "^spx":     "SP500_Index",
    "^ndx":     "Nasdaq100_Index",
}

# 同时从FRED补充 SP500 / NASDAQ100 指数
FRED_INDEX = {
    "SP500":      "SP500_FRED",      # 10年标普500
    "NASDAQ100":  "NASDAQ100_FRED",  # 纳指100
    "NASDAQCOM":  "NasdaqComposite_FRED",
    "DJIA":       "DowJones_FRED",
}

def download_stooq():
    print("=" * 60); print("Stooq 指数/ETF 数据"); print("=" * 60)
    summary = []
    for code, name in INDEX_TICKERS.items():
        try:
            df = pdr.DataReader(code, "stooq", START, END)
            if df.empty:
                print(f"  ✗ {name}: empty"); continue
            df = df.sort_index()  # stooq倒序
            fp = os.path.join(DATA_DIR, f"idx_{name}.csv")
            df.to_csv(fp)
            n = len(df)
            print(f"  ✓ {name:25s} {n:5d} rows  {df.index.min().date()} → {df.index.max().date()}  last={df['Close'].iloc[-1]:.2f}")
            summary.append({"name": name, "ticker": code, "rows": n,
                            "start": str(df.index.min().date()),
                            "end": str(df.index.max().date()),
                            "latest_close": float(df['Close'].iloc[-1])})
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "_idx_summary.csv"), index=False)

def download_fred_index():
    print("=" * 60); print("FRED 指数补充"); print("=" * 60)
    for code, name in FRED_INDEX.items():
        try:
            df = pdr.DataReader(code, "fred", START, END)
            df.columns = [name]
            fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
            df.to_csv(fp)
            print(f"  ✓ {name:25s} {len(df):5d} rows  {df.index.min().date()} → {df.index.max().date()}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

if __name__ == "__main__":
    download_stooq()
    download_fred_index()
    print("\n完成")
