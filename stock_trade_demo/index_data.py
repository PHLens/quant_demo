"""
Benchmark index data fetcher.

Fetches daily K-line data from Sina Finance API, computes monthly returns,
and caches them under .cache/ to avoid re-fetching on every run.
"""

import os
import sys
import json
import logging
import socket
import urllib.error
import urllib.request

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from utils.atomic_io import atomic_write_csv, sweep_dangling_tmps
from schemas.index_panel import INDEX_DAILY_SCHEMA

# 网络/解析类异常：允许 fall back to cache（短暂故障，不该让 batch 整体挂掉）。
# pandera 的 SchemaError / SchemaErrors 不能并入这里 —— 那意味着上游数据脏，
# 必须 raise，否则会静默把脏数据吞掉，反过来阻塞新数据落盘（Worker A 之前血泪教训）。
_NETWORK_FETCH_EXCEPTIONS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    socket.timeout,
    json.JSONDecodeError,
    ConnectionError,
    TimeoutError,
    OSError,  # 关键：requests.exceptions.ConnectionError 继承 IOError（=OSError），
              #       而 builtins.ConnectionError 跟 requests.ConnectionError 是平行类，
              #       不加 OSError 会让 akshare 抛的瞬断异常穿透所有 fallback 直接挂掉
    ValueError,  # _fetch_daily_kline 在 raw JSON 异常时 float() 会 raise ValueError
    RuntimeError,  # _fetch_etf_daily_akshare 自抛 RuntimeError（空 frame / 缺列）
)


def _stringify_date_column(df):
    """Return a shallow copy with the 'date' column rendered as 'YYYY-MM-DD' str.

    The on-disk CSV always stores date as a string (CSV has no native date type);
    this just makes the in-memory DataFrame match what pandera expects so the
    declared INDEX_DAILY_SCHEMA can validate before any tmp file is written.
    """
    if 'date' not in df.columns:
        return df
    out = df.copy()
    out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
    return out

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
TIMING_ETF_CACHE_DIR = os.path.join(CACHE_DIR, 'timing_etf')
A_SHARE_CALENDAR_CACHE_FILE = os.path.join(CACHE_DIR, 'a_share_calendar_daily.csv')


def configure_cache_dir(cache_dir):
    """Route all index/ETF reads and writes to the configured data cache."""
    global CACHE_DIR, TIMING_ETF_CACHE_DIR, A_SHARE_CALENDAR_CACHE_FILE
    CACHE_DIR = os.fspath(cache_dir)
    TIMING_ETF_CACHE_DIR = os.path.join(CACHE_DIR, 'timing_etf')
    A_SHARE_CALENDAR_CACHE_FILE = os.path.join(CACHE_DIR, 'a_share_calendar_daily.csv')

INDEX_CONFIGS = {
    'csi1000': {
        'name': 'CSI 1000',
        'symbol': 'sh000852',
        'cache_file': 'csi1000_monthly.csv',
        'daily_cache_file': 'csi1000_daily.csv',
        'series_name': 'csi1000_return',
    },
    'chinext': {
        'name': '创业板指数',
        'symbol': 'sz399006',
        'cache_file': 'chinext_monthly.csv',
        'daily_cache_file': 'chinext_daily.csv',
        'series_name': 'chinext_return',
    },
    'star50': {
        'name': '科创50指数',
        'symbol': 'sh000688',
        'cache_file': 'star50_monthly.csv',
        'daily_cache_file': 'star50_daily.csv',
        'series_name': 'star50_return',
    },
    'nasdaq': {
        'name': '纳指ETF',
        'symbol': 'sz159941',
        'cache_file': 'nasdaq_monthly.csv',
        'daily_cache_file': 'nasdaq_daily.csv',
        'series_name': 'nasdaq_return',
    },
    'sp500': {
        'name': '标普500ETF',
        'symbol': 'sh513500',
        'cache_file': 'sp500_monthly.csv',
        'daily_cache_file': 'sp500_daily.csv',
        'series_name': 'sp500_return',
    },
    'gold': {
        'name': '黄金ETF',
        'symbol': 'sh518880',
        'cache_file': 'gold_monthly.csv',
        'daily_cache_file': 'gold_daily.csv',
        'series_name': 'gold_return',
    },
    'hsi': {
        'name': '恒生ETF',
        'symbol': 'sz159920',
        'cache_file': 'hsi_monthly.csv',
        'daily_cache_file': 'hsi_daily.csv',
        'series_name': 'hsi_return',
    },
    'hstech': {
        'name': '恒生科技ETF',
        'symbol': 'sh513180',
        'cache_file': 'hstech_monthly.csv',
        'daily_cache_file': 'hstech_daily.csv',
        'series_name': 'hstech_return',
    },
}

# A股指数（不含美股ETF代理，仅用于A股benchmark对比）
A_SHARE_INDEX_IDS = {'csi1000', 'chinext', 'star50'}

TIMING_ETF_CONFIGS = {
    'csi1000': {
        'name': '中证1000ETF',
        'code': '510980',
        'symbol': 'sh510980',
        'daily_cache_file': 'csi1000_etf_daily.csv',
        # T+1 交收：A股 ETF 当日买入次日才能卖出（信号 t -> 成交 t+1 open -> 标记 t+1 close）。
        'settlement': 'T+1',
        # 涨跌停幅度（沪市中证1000ETF 为 10%）。
        'limit_pct': 0.10,
        'market': 'SH',
        # 复权方式：使用 akshare 的前复权（qfq）拿日线，避免分红日被误判为跳水。
        'dividend_adjust_method': 'qfq',
    },
    'chinext': {
        'name': '创业板ETF',
        'code': '159205',
        'symbol': 'sz159205',
        'daily_cache_file': 'chinext_etf_daily.csv',
        'settlement': 'T+1',
        # 创业板 ETF 跟踪创业板成份，单日涨跌停 20%。
        'limit_pct': 0.20,
        'market': 'SZ',
        'dividend_adjust_method': 'qfq',
    },
    'star50': {
        'name': '科创50ETF',
        'code': '589850',
        'symbol': 'sh589850',
        'daily_cache_file': 'star50_etf_daily.csv',
        'settlement': 'T+1',
        # 科创板 ETF 单日涨跌停 20%。
        'limit_pct': 0.20,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
    'nasdaq': {
        'name': '纳指ETF',
        'code': '159941',
        'symbol': 'sz159941',
        'daily_cache_file': 'nasdaq_etf_daily.csv',
        # 跨境 ETF（QDII）按 T+0 回转交易；无 A 股式涨跌停限制（盘中可能因溢价熔断停牌但非固定百分比）。
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SZ',
        'dividend_adjust_method': 'qfq',
    },
    'sp500': {
        'name': '标普500ETF',
        'code': '513500',
        'symbol': 'sh513500',
        'daily_cache_file': 'sp500_etf_daily.csv',
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
    'gold': {
        'name': '黄金ETF',
        'code': '518880',
        'symbol': 'sh518880',
        'daily_cache_file': 'gold_etf_daily.csv',
        'settlement': 'T+1',
        'limit_pct': 0.10,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
    'hsi': {
        'name': '恒生ETF',
        'code': '159920',
        'symbol': 'sz159920',
        'daily_cache_file': 'hsi_etf_daily.csv',
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SZ',
        'dividend_adjust_method': 'qfq',
    },
    'hstech': {
        'name': '恒生科技ETF',
        'code': '513180',
        'symbol': 'sh513180',
        'daily_cache_file': 'hstech_etf_daily.csv',
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
}

# ── 大宗商品 ETF 配置（黄金等，独立于股票指数页面） ──
COMMODITY_ETF_CONFIGS = {
    'gold': {
        'name': '黄金ETF',
        'code': '518880',
        'symbol': 'sh518880',
        'daily_cache_file': 'gold_etf_daily.csv',
        'settlement': 'T+1',
        'limit_pct': 0.10,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
}
COMMODITY_INDEX_IDS = {'gold'}


def _cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['cache_file'])


def _daily_cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['daily_cache_file'])


def _a_share_calendar_cache_path():
    return A_SHARE_CALENDAR_CACHE_FILE


def _ensure_cache_dir(path):
    os.makedirs(path, exist_ok=True)


def _etf_daily_cache_path(index_id, adjust='qfq'):
    """ETF daily cache path. Cache key 必须包含 adjust，避免与未复权旧缓存撞键。"""
    base = TIMING_ETF_CONFIGS[index_id]['daily_cache_file']
    if adjust:
        root, ext = os.path.splitext(base)
        base = f'{root}_{adjust}{ext}'
    return os.path.join(TIMING_ETF_CACHE_DIR, base)


def _legacy_etf_daily_cache_path(index_id):
    """旧的未复权缓存路径（Sina 抓取产物），仅用于 fallback。"""
    return os.path.join(CACHE_DIR, TIMING_ETF_CONFIGS[index_id]['daily_cache_file'])


def _legacy_etf_daily_cache_path_in_subdir(index_id):
    """位于 timing_etf 子目录的旧未复权缓存（升级前的中间态）。"""
    return os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[index_id]['daily_cache_file'])


def _sina_url(symbol):
    return (
        'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
        f'CN_MarketData.getKLineData?symbol={symbol}&scale=240&datalen=8000'
    )


# ── 东方财富直连作为新浪的备选线路 ──
# Sina 经常被限流或瞬断；东方财富走完全不同的 CDN（push2his.eastmoney.com），
# 跟新浪共失败的概率极低。endpoint 不需要 API key、可直接 curl 验证。
# symbol 形如 sh000852 / sz399006 → East Money secid 1.000852 / 0.399006
def _em_secid_from_sina_symbol(symbol):
    s = symbol.lower().strip()
    if s.startswith('sh'):
        return f'1.{s[2:]}'
    if s.startswith('sz'):
        return f'0.{s[2:]}'
    raise ValueError(f'Unknown symbol format for East Money mapping: {symbol}')


def _fetch_daily_kline_eastmoney(symbol):
    """从东方财富 push2his 拉日 K（未复权，跟 Sina 同等口径）。

    返回与 _fetch_daily_kline 完全一致的 DataFrame，可直接当 drop-in 替代。
    """
    secid = _em_secid_from_sina_symbol(symbol)
    # klt=101 日线，fqt=0 未复权（跟 Sina 一致）
    url = (
        'https://push2his.eastmoney.com/api/qt/stock/kline/get'
        f'?secid={secid}&fields1=f1,f2,f3,f4,f5,f6'
        '&fields2=f51,f52,f53,f54,f55,f56,f57,f58'  # date, open, close, high, low, volume, amount, amplitude
        '&klt=101&fqt=0&beg=20050101&end=20500101'
    )
    req = urllib.request.Request(url)
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36')
    req.add_header('Referer', 'https://quote.eastmoney.com/')
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    j = json.loads(raw)
    klines = ((j.get('data') or {}).get('klines') or [])
    if not klines:
        raise RuntimeError(f'East Money returned no klines for secid={secid}')

    records = []
    for line in klines:
        parts = line.split(',')
        if len(parts) < 6:
            continue
        records.append({
            'date': parts[0],
            'open': float(parts[1]),
            'close': float(parts[2]),
            'high': float(parts[3]),
            'low': float(parts[4]),
            'volume': float(parts[5]),
        })
    if not records:
        raise RuntimeError(f'East Money parsed 0 valid rows for secid={secid}')
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def _fetch_daily_kline_with_fallback(symbol):
    """对指数/ETF 日 K 的多源 fetch：
       第 1 条线：Sina (money.finance.sina.com.cn)
       第 2 条线：East Money (push2his.eastmoney.com)
    任一成功即返回；都失败时 raise，让调用方走 disk cache。
    """
    sources = [
        ('Sina', lambda: _fetch_daily_kline(symbol)),
        ('East Money', lambda: _fetch_daily_kline_eastmoney(symbol)),
    ]
    last_err = None
    for name, fn in sources:
        try:
            df = fn()
            if df is not None and len(df) > 0:
                if last_err is not None:
                    print(f"[index_data] {name} 救回了 {symbol}（上一条线 {type(last_err).__name__} 已 fail）",
                          file=sys.stderr)
                return df
        except _NETWORK_FETCH_EXCEPTIONS as exc:
            print(f"[index_data] {name} fetch failed for {symbol}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            last_err = exc
            continue
    # 全部线路都炸 → 抛最后一个错给上层
    raise last_err if last_err else RuntimeError(f'all daily-kline sources failed for {symbol}')


def _fetch_daily_kline(symbol):
    """Fetch daily K-line data from Sina API (未复权, 用于指数 / fallback)."""
    req = urllib.request.Request(_sina_url(symbol))
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36')
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    data = json.loads(raw)
    records = [{
        'date': item['day'],
        'open': float(item['open']),
        'high': float(item['high']),
        'low': float(item['low']),
        'close': float(item['close']),
        'volume': float(item['volume']),
    } for item in data]

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def _fetch_etf_daily_akshare(code, adjust='qfq'):
    """通过 akshare 抓取 ETF 日线（默认前复权 qfq）。

    akshare.fund_etf_hist_em 返回中文列：'日期','开盘','收盘','最高','最低',
    '成交量','成交额','振幅','涨跌幅','涨跌额','换手率'。这里重命名为现有
    调用方期望的英文 schema，保持向后兼容。

    严格遵守 CLAUDE.md 第 11 条：不做 forward/backfill，不补 ETF 上市前数据；
    akshare 返回什么就用什么。
    """
    import akshare as ak  # 延迟导入，避免无 akshare 环境下加载本模块即失败

    raw = ak.fund_etf_hist_em(symbol=code, period='daily', adjust=adjust)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f'akshare returned empty frame for ETF {code} (adjust={adjust})')

    rename_map = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
    }
    keep = [c for c in rename_map if c in raw.columns]
    df = raw[keep].rename(columns=rename_map).copy()

    required = {'date', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'akshare ETF frame missing columns {missing} for {code}')

    df['date'] = pd.to_datetime(df['date'])
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close']).sort_values('date').reset_index(drop=True)
    return df


def get_index_daily(index_id='csi1000', force_refetch=False):
    """Get cached daily K-line data for a configured benchmark index."""
    if index_id not in INDEX_CONFIGS:
        raise ValueError(f'Unknown index_id: {index_id}')

    cfg = INDEX_CONFIGS[index_id]
    cache_file = _daily_cache_path(index_id)
    cached_df = None

    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file, parse_dates=['date']).sort_values('date').reset_index(drop=True)
        if not force_refetch:
            return cached_df

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        print(f"[index_data] Fetching {cfg['name']} daily K-line (Sina → East Money fallback)...")
        df_daily = _fetch_daily_kline_with_fallback(cfg['symbol']).sort_values('date').reset_index(drop=True)
        fetched_max = pd.to_datetime(df_daily['date']).max() if len(df_daily) > 0 else None
        cached_max = pd.to_datetime(cached_df['date']).max() if cached_df is not None and len(cached_df) > 0 else None

        if cached_max is not None and fetched_max is not None and fetched_max < cached_max:
            print(f"[index_data] WARN fetched {cfg['name']} daily data is stale ({fetched_max.strftime('%Y-%m-%d')} < cached {cached_max.strftime('%Y-%m-%d')}); keep existing cache")
            return cached_df

        if cached_max is not None and fetched_max is not None and fetched_max == cached_max and cached_df is not None and len(cached_df) >= len(df_daily):
            print(f"[index_data] {cfg['name']} daily cache already fresh through {cached_max.strftime('%Y-%m-%d')}; skip overwrite")
            return cached_df

        atomic_write_csv(cache_file, _stringify_date_column(df_daily), index=False, schema=INDEX_DAILY_SCHEMA,
                         produced_by=f"index_data.get_index_daily:{index_id}")
        print(f"[index_data] Cached daily K-line to {cache_file}")
        return df_daily
    except (SchemaError, SchemaErrors):
        # 上游数据脏（open<=0 / date 格式错 / volume 越界 …）—— 硬失败，
        # 绝不写脏数据进缓存，也不静默回退到旧缓存掩盖问题。
        print(f"[index_data] FATAL fetched {cfg['name']} daily K-line failed schema validation; refusing to write cache")
        raise
    except _NETWORK_FETCH_EXCEPTIONS as e:
        print(f"[index_data] ERROR fetching {cfg['name']} daily K-line: {e}")
        if cached_df is not None:
            print(f"[index_data] Falling back to cached daily K-line for {cfg['name']}")
            return cached_df
        raise RuntimeError(f"Failed to fetch {cfg['name']} daily data and no cache exists: {e}")


def get_a_share_trading_calendar(force_refetch=False):
    """Get a full-history A-share trading calendar from broad market index daily bars."""
    cache_file = _a_share_calendar_cache_path()
    cached_df = None

    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file, parse_dates=['date']).sort_values('date').reset_index(drop=True)
        if not force_refetch:
            return cached_df

    _ensure_cache_dir(CACHE_DIR)

    for symbol in ('sh000001', 'sz399001'):
        try:
            # 用多源 fetcher：每个 symbol 都先 Sina 后 East Money，两条线全断才放弃
            df_daily = _fetch_daily_kline_with_fallback(symbol).sort_values('date').reset_index(drop=True)
        except (SchemaError, SchemaErrors):
            # 与主 fetch 路径一致：上游数据脏 → 硬失败，不要悄悄写脏日历回 cache。
            print(f"[index_data] FATAL trading-calendar fetch {symbol} failed schema validation; refusing to write cache")
            raise
        except _NETWORK_FETCH_EXCEPTIONS:
            # 该 symbol 的所有线路都断 → 试下一个 symbol；都失败时下方再回退 cached_df。
            continue
        fetched_max = pd.to_datetime(df_daily['date']).max() if len(df_daily) > 0 else None
        cached_max = pd.to_datetime(cached_df['date']).max() if cached_df is not None and len(cached_df) > 0 else None
        if cached_max is not None and fetched_max is not None and fetched_max < cached_max:
            continue
        if cached_max is not None and fetched_max is not None and fetched_max == cached_max and cached_df is not None and len(cached_df) >= len(df_daily):
            return cached_df
        atomic_write_csv(cache_file, df_daily[['date']], index=False,
                         produced_by="index_data.get_a_share_trading_calendar")
        return df_daily[['date']]

    if cached_df is not None:
        return cached_df[['date']] if 'date' in cached_df.columns else cached_df
    raise RuntimeError('Failed to fetch A-share trading calendar and no cache exists.')


def _select_preferred_timing_etf_cache(rows):
    """Pick the runtime ETF cache without silently switching away from qfq.

    运行时口径必须稳定：若 qfq 缓存存在且可读，就优先用 qfq，哪怕 legacy 更“新”。
    fresher legacy 只能作为诊断信号，不应静默压过 qfq，否则会把未复权价格混进
    回测、持仓估值和 benchmark 对比里。
    """
    preferred = next((item for item in rows if item.get('label') == 'preferred_qfq_cache'), None)
    if preferred and preferred.get('exists') and not preferred.get('read_error'):
        return preferred['path']
    for item in rows:
        if item.get('exists') and not item.get('read_error'):
            return item['path']
    return None


def describe_timing_etf_cache(index_id='csi1000', adjust=None):
    """Describe all known cache paths for a timing ETF and the current preferred order."""
    if index_id not in TIMING_ETF_CONFIGS:
        raise ValueError(f'Unknown ETF timing index_id: {index_id}')

    cfg = TIMING_ETF_CONFIGS[index_id]
    if adjust is None:
        adjust = cfg.get('dividend_adjust_method', 'qfq')

    candidates = [
        ('preferred_qfq_cache', _etf_daily_cache_path(index_id, adjust=adjust)),
        ('legacy_subdir_cache', _legacy_etf_daily_cache_path_in_subdir(index_id)),
        ('legacy_root_cache', _legacy_etf_daily_cache_path(index_id)),
    ]
    rows = []
    for label, path in candidates:
        item = {
            'label': label,
            'path': path,
            'exists': os.path.exists(path),
            'adjust': adjust if label == 'preferred_qfq_cache' else 'legacy_unadjusted',
            'rows': 0,
            'start_date': None,
            'end_date': None,
            'columns': [],
            'read_error': None,
        }
        if item['exists']:
            try:
                df = pd.read_csv(path)
                item['rows'] = int(len(df))
                item['columns'] = list(df.columns)
                if 'date' in df.columns and len(df) > 0:
                    s = pd.to_datetime(df['date'], errors='coerce').dropna()
                    if len(s) > 0:
                        item['start_date'] = s.min().strftime('%Y-%m-%d')
                        item['end_date'] = s.max().strftime('%Y-%m-%d')
            except Exception as exc:
                # describe_timing_etf_cache 是 admin 诊断接口，目的就是把"这个缓存文件能不能读"
                # 的真实情况完整地塞到返回结构里给运维看；这里故意保留 bare Exception，
                # 不区分 SchemaError / IOError / UnicodeDecodeError —— 任何读失败都应原样上报，
                # 不能 raise（会让整张诊断表无法返回）也不需要 fall back（这里不消费数据）。
                item['read_error'] = repr(exc)
        rows.append(item)

    preferred_runtime_path = _select_preferred_timing_etf_cache(rows)

    return {
        'index_id': index_id,
        'code': cfg.get('code'),
        'symbol': cfg.get('symbol'),
        'default_adjust': adjust,
        'preferred_runtime_path': preferred_runtime_path,
        'candidates': rows,
    }


def get_timing_etf_daily(index_id='csi1000', force_refetch=False, adjust=None):
    """Get cached daily K-line data for the user-specified ETF mapped to a timing index.

    主路径：akshare.fund_etf_hist_em(adjust='qfq')，前复权，分红日不会被误判为下跌。
    Fallback：原 Sina 未复权路径（仅在 akshare 抓取失败时使用，并在 logger 中 warn）。
    缓存 key 包含 adjust，避免与旧未复权缓存撞键。
    """
    if index_id not in TIMING_ETF_CONFIGS:
        raise ValueError(f'Unknown ETF timing index_id: {index_id}')

    cfg = TIMING_ETF_CONFIGS[index_id]
    if adjust is None:
        adjust = cfg.get('dividend_adjust_method', 'qfq')

    cache_file = _etf_daily_cache_path(index_id, adjust=adjust)
    legacy_cache_file = _legacy_etf_daily_cache_path(index_id)
    legacy_sub_cache_file = _legacy_etf_daily_cache_path_in_subdir(index_id)

    if not force_refetch:
        cache_info = describe_timing_etf_cache(index_id=index_id, adjust=adjust)
        preferred_runtime_path = (cache_info or {}).get('preferred_runtime_path')
        if preferred_runtime_path and os.path.exists(preferred_runtime_path):
            print(f"[index_data] Using preferred ETF cache for {cfg['name']}: {preferred_runtime_path}")
            df = pd.read_csv(preferred_runtime_path, parse_dates=['date'])
            return df.sort_values('date').reset_index(drop=True)

    _ensure_cache_dir(CACHE_DIR)
    _ensure_cache_dir(TIMING_ETF_CACHE_DIR)

    # 主路径：akshare 前复权
    try:
        print(f"[index_data] Fetching {cfg['name']} ({cfg['code']}) daily K-line via akshare (adjust={adjust})...")
        df_daily = _fetch_etf_daily_akshare(cfg['code'], adjust=adjust)
        atomic_write_csv(cache_file, _stringify_date_column(df_daily), index=False, schema=INDEX_DAILY_SCHEMA,
                         produced_by=f"index_data.get_timing_etf_daily:akshare:{cfg['code']}:{adjust}")
        print(f"[index_data] Cached ETF daily K-line to {cache_file} ({len(df_daily)} rows)")
        return df_daily
    except (SchemaError, SchemaErrors):
        # akshare 返回数据脏（应该几乎不会发生）—— 硬失败，绝不静默回退到 Sina 掩盖
        print(f"[index_data] FATAL akshare ETF data for {cfg['name']} ({cfg['code']}) failed schema validation; refusing fallback")
        raise
    except _NETWORK_FETCH_EXCEPTIONS as ak_err:
        logger.warning(
            "[index_data] akshare fetch failed for %s (%s, adjust=%s): %s; falling back to Sina (un-adjusted).",
            cfg['name'], cfg['code'], adjust, ak_err,
        )
        print(f"[index_data] WARN akshare failed for {cfg['name']} ({cfg['code']}): {ak_err}; "
              f"falling back to Sina un-adjusted feed")

    # Fallback：未复权抓取（Sina → East Money 双线）。注意：未复权数据在分红日会有
    # 跳水缺口，只在 akshare 不可用时作为 best-effort。
    try:
        print(f"[index_data] Fetching {cfg['name']} ({cfg['code']}) daily K-line via Sina → East Money fallback (un-adjusted)...")
        df_daily = _fetch_daily_kline_with_fallback(cfg['symbol'])
        # 不写入 qfq 的 cache_file，避免污染前复权缓存；写到未复权 legacy path 以便复用
        atomic_write_csv(legacy_sub_cache_file, _stringify_date_column(df_daily), index=False, schema=INDEX_DAILY_SCHEMA,
                         produced_by=f"index_data.get_timing_etf_daily:sina_em_fallback:{cfg['symbol']}")
        print(f"[index_data] Cached un-adjusted ETF daily K-line to {legacy_sub_cache_file}")
        return df_daily
    except (SchemaError, SchemaErrors):
        # 上游数据脏 —— 硬失败，绝不回到磁盘上不知道多旧的缓存掩盖问题
        print(f"[index_data] FATAL Sina/East Money fallback ETF data for {cfg['name']} failed schema validation; refusing disk fallback")
        raise
    except _NETWORK_FETCH_EXCEPTIONS as fb_err:
        logger.warning(
            "[index_data] Sina/East Money fallback also failed for %s: %s",
            cfg['name'], fb_err,
        )
        print(f"[index_data] ERROR Sina/East Money fallback failed for {cfg['name']}: {fb_err}")

    # Fallback 2：磁盘上任何已有缓存
    for candidate in (cache_file, legacy_sub_cache_file, legacy_cache_file):
        if os.path.exists(candidate):
            print(f"[index_data] Falling back to existing cached ETF daily K-line: {candidate}")
            df = pd.read_csv(candidate, parse_dates=['date'])
            return df.sort_values('date').reset_index(drop=True)

    raise RuntimeError(
        f"Failed to fetch {cfg['name']} ETF daily data via akshare and Sina, and no cache exists."
    )


def refresh_all_timing_etf_daily():
    sweep_dangling_tmps(CACHE_DIR, recursive=True)
    refreshed = {}
    for index_id in TIMING_ETF_CONFIGS:
        refreshed[index_id] = get_timing_etf_daily(index_id=index_id, force_refetch=True)
    return refreshed


def build_index_panel(index_ids=None, force_refetch=False):
    """Build a daily wide panel for A-share indexes (excludes US ETF proxies by default)."""
    index_ids = index_ids or list(A_SHARE_INDEX_IDS)
    panel = None

    for index_id in index_ids:
        cfg = INDEX_CONFIGS[index_id]
        daily = get_index_daily(index_id=index_id, force_refetch=force_refetch).copy()
        renamed = daily.rename(columns={
            'open': f'{index_id}_open',
            'high': f'{index_id}_high',
            'low': f'{index_id}_low',
            'close': f'{index_id}_close',
            'volume': f'{index_id}_volume',
        })
        renamed[f'{index_id}_name'] = cfg['name']
        keep_cols = [
            'date',
            f'{index_id}_open', f'{index_id}_high', f'{index_id}_low',
            f'{index_id}_close', f'{index_id}_volume', f'{index_id}_name',
        ]
        renamed = renamed[keep_cols]
        if panel is None:
            panel = renamed
        else:
            panel = panel.merge(renamed, on='date', how='outer')

    if panel is None:
        return pd.DataFrame(columns=['date'])

    panel = panel.sort_values('date').reset_index(drop=True)
    panel['交易日期'] = pd.to_datetime(panel['date'])
    return panel


def _compute_monthly_returns(df_daily, series_name):
    """Compute monthly returns from daily close prices."""
    df = df_daily.set_index('date').sort_index()
    daily_returns = df['close'].pct_change()
    monthly = (1 + daily_returns).resample('M').prod() - 1
    monthly = monthly.dropna()
    monthly.name = series_name
    return monthly


def _compute_weekly_returns(df_daily, series_name):
    """Compute weekly returns from daily close prices using each week's last trading day."""
    df = df_daily.set_index('date').sort_index()
    weekly_close = df['close'].resample('W-FRI').last().dropna()
    weekly = weekly_close.pct_change().dropna()
    weekly.name = series_name
    return weekly


def get_index_returns(index_id='csi1000', force_refetch=False, frequency='monthly'):
    """Get benchmark returns for a configured index."""
    if index_id not in INDEX_CONFIGS:
        raise ValueError(f'Unknown index_id: {index_id}')
    if frequency not in {'monthly', 'weekly'}:
        raise ValueError(f'Unsupported frequency: {frequency}')

    cfg = INDEX_CONFIGS[index_id]
    cache_file = _cache_path(index_id)
    if frequency == 'weekly':
        cache_file = cache_file.replace('_monthly.csv', '_weekly.csv')

    if not force_refetch and os.path.exists(cache_file):
        s = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = cfg['series_name']
        print(f"[index_data] Loaded {cfg['name']} {len(s)} {frequency} returns from cache: {cache_file}")
        return s

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        df_daily = get_index_daily(index_id=index_id, force_refetch=force_refetch)
        print(f"[index_data] Fetched {len(df_daily)} daily bars "
              f"({df_daily['date'].min().strftime('%Y-%m-%d')} to "
              f"{df_daily['date'].max().strftime('%Y-%m-%d')})")

        returns = _compute_monthly_returns(df_daily, cfg['series_name']) if frequency == 'monthly' else _compute_weekly_returns(df_daily, cfg['series_name'])
        print(f"[index_data] Computed {len(returns)} {frequency} returns "
              f"({returns.index.min().strftime('%Y-%m-%d')} to "
              f"{returns.index.max().strftime('%Y-%m-%d')})")

        atomic_write_csv(cache_file, returns, header=True,
                         produced_by=f"index_data.get_index_returns:{index_id}:{frequency}")
        print(f"[index_data] Cached to {cache_file}")
        return returns

    except (SchemaError, SchemaErrors):
        raise
    except _NETWORK_FETCH_EXCEPTIONS as e:
        print(f"[index_data] ERROR fetching {cfg['name']}: {e}")
        if os.path.exists(cache_file):
            print(f"[index_data] Falling back to cached {cfg['name']} data")
            s = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = cfg['series_name']
            return s
        raise RuntimeError(f"Failed to fetch {cfg['name']} data and no cache exists: {e}")


US_INDEX_IDS = {'nasdaq', 'sp500'}


def build_us_index_panel(force_refetch=False):
    """Build a daily wide panel for US ETF proxies (nasdaq, sp500)."""
    return build_index_panel(index_ids=list(US_INDEX_IDS), force_refetch=force_refetch)


def build_commodity_index_panel(force_refetch=False):
    """Build a daily wide panel for commodity ETFs (gold, etc.)."""
    return build_index_panel(index_ids=list(COMMODITY_INDEX_IDS), force_refetch=force_refetch)


HK_INDEX_IDS = {'hsi', 'hstech'}


def build_hk_index_panel(force_refetch=False):
    """Build a daily wide panel for HK ETFs (HSI, HSTech)."""
    return build_index_panel(index_ids=list(HK_INDEX_IDS), force_refetch=force_refetch)


# ═══════════════════════════════════════════════════════════════════
# Utility: build index-by-period lookup for date alignment
# ═══════════════════════════════════════════════════════════════════


def build_period_lookup(index_returns):
    """Build a dict mapping (year, month) → index return for fast lookup."""
    lookup = {}
    for idx, val in index_returns.items():
        ts = pd.to_datetime(idx)
        lookup[(ts.year, ts.month)] = float(val)
    return lookup


def get_index_return_for_date(date, period_lookup):
    """Look up the monthly return for a given date's calendar month."""
    ts = pd.to_datetime(date)
    return period_lookup.get((ts.year, ts.month), 0.0)
