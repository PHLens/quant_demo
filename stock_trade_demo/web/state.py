"""Module-level state + helper functions for the Flask read-only viewer.

Pillar 1 Step 4: 把原 web_app.py 中的全局可变状态（DATA_DF / BACKTEST_CACHE / ...）
和所有非路由辅助函数搬到这里。所有 blueprint 通过 `from web import state` 访问，
保证 `_run_data_update` 等地方重新赋值 module-level dict / DataFrame 时，
其他模块能感知到变化（因为始终通过 `state.XXX` 按名访问）。

绝对不要 `from web.state import DATA_DF` —— 那样拿到的是 import 时刻的快照。
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import threading
import warnings
import inspect

import numpy as np
import pandas as pd

# 把 stock_trade_demo/ 加入 sys.path（让 from get_stock_info / strategies / ... 等绝对 import 工作）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import csv as _csv
from datetime import datetime as _datetime
from utils.atomic_io import atomic_write_parquet
from get_stock_info import (
    supplement_csv as _supplement_csv,
    supplement_csv_incremental as _supplement_csv_incremental,
    fetch_realtime_quotes_batch as _fetch_realtime_quotes_batch,
)

# 必须先 import 所有策略模块，让 BaseStrategy / BaseTimingStrategy 的
# __init_subclass__ 钩子把它们写入 STRATEGY_REGISTRY / TIMING_REGISTRY /
# US_TIMING_REGISTRY。不要把这些 import 当成 “冗余可删”。
from strategies.original import OriginalStrategy  # noqa: F401
from strategies.original_ensemble import OriginalEnsembleStrategy  # noqa: F401
from strategies.chan_enhanced import ChanEnhancedStrategy  # noqa: F401
from strategies.chan_only import ChanOnlyStrategy  # noqa: F401
from strategies.method_a import MethodAStrategy  # noqa: F401
from strategies.quality_value import QualityValueStrategy  # noqa: F401
from strategies.sector_heat import SectorHeatStrategy  # noqa: F401
from strategies.base import (
    STRATEGY_REGISTRY,
    TIMING_REGISTRY,
    US_TIMING_REGISTRY,
)
from strategies.registry import COMMODITY_REGISTRY, HK_TIMING_REGISTRY
from backtest import load_data, select_and_backtest, strategy_evaluate, compute_alpha_beta
from index_data import (
    INDEX_CONFIGS, TIMING_ETF_CONFIGS, A_SHARE_INDEX_IDS,
    get_index_daily, get_timing_etf_daily, get_a_share_trading_calendar,
    get_index_returns, build_index_panel, build_us_index_panel, build_commodity_index_panel,
    build_hk_index_panel,
    build_period_lookup, get_index_return_for_date, refresh_all_timing_etf_daily,
)
# 触发 timing 策略类注册（NasdaqTimingStrategy 设了 registry=None，会自动跳过）
from timing import (  # noqa: F401
    CSI1000TimingStrategy, Star50TimingStrategy, ChiNextTimingStrategy,
    SP500TimingStrategy, MacroV32TimingStrategy, GoldTimingStrategy,
    HSITimingStrategy, HSTechTimingStrategy,
    run_timing_backtest, evaluate_timing_result, timing_result_to_json,
    filter_timing_result, summarize_timing_windows,
)

from web import serializers as _serializers
from web.serializers import (
    SPLIT_DATE,
    DEFAULT_BENCHMARK_ID,
    _normalize_benchmark_id,
    _get_benchmark_series,
    _get_benchmark_meta,
    _infer_market_label,
    _resample_curve,
    _load_trading_calendar,
    _resolve_period_trading_dates,
    _build_holding_date_range,
    _is_open_snapshot_period,
    _build_daily_curve_slice,
    _safe_float_or_none,
    _normalize_stock_code,
    _extract_open_stock_codes,
    _fetch_open_stock_quotes,
    _build_stock_payload,
    _build_holdings_payload,
    _compute_single_benchmark_curve,
    _compute_benchmark_curves,
    _build_period_benchmark_returns,
    _build_etf_monthly_returns,
    _month_start_from_end,
    build_selection_interval_windows,
    compute_split_metrics,
    result_to_json,
)

from services import live_trades as _live_trades_service
from services import cache_store as _cache_store

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)


# ═════════════════════════════════════════════════════════════════
# 全局可变状态（通过 state.XXX 引用，保证重新赋值后所有 blueprint 可见）
# ═════════════════════════════════════════════════════════════════
DATA_DF = None
INDEX_RETURNS = None                 # CSI 1000 月度收益 Series（用于归因主基准）
INDEX_RETURNS_MAP = {}               # key: index_id -> monthly returns series
BACKTEST_CACHE = {}                  # key: strategy_name → (result_df, eval_df)
TIMING_PANEL = None
TIMING_CACHE = {}                    # key: strategy_name -> result_df
US_TIMING_PANEL = None
US_TIMING_CACHE = {}                 # key: strategy_name -> result_df
COMMODITY_PANEL = None
COMMODITY_CACHE = {}                 # key: strategy_name -> result_df
HK_PANEL = None
HK_CACHE = {}                        # key: strategy_name -> result_df
_PROFILE_SUMMARY_CACHE = {}          # key: strategy_name -> profile_summary list
FACTOR_BACKTEST_CACHE = {}           # key: "top_k=N" -> factor backtest payload
CSI1000_SIGNAL_SERIES = None         # CSI1000 择时策略日线仓位信号

# 让 serializers 能拿到当前最新的 INDEX_RETURNS_MAP（包括之后被整 dict 替换）
_serializers.set_index_returns_map_provider(lambda: INDEX_RETURNS_MAP)


# ═════════════════════════════════════════════════════════════════
# 常量 / 配置
# ═════════════════════════════════════════════════════════════════
TRAINING_CUTOFF = '2025-11-30'
HOLDOUT_START = '2025-12-01'

_BEST_PROFILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'strategy'))
_BEST_PROFILE_CACHE = {}
_RISK_SIGNALS_FILE = os.path.join(_BEST_PROFILE_DIR, 'risk_signals.json')
_RISK_SIGNALS_CACHE = {'mtime': 0, 'data': None}
_HOLDOUT_REPORT_CACHE = {}

_LIVE_INITIAL_CAPITAL = 50000.0
_LIVE_LOT_SIZE = {
    'csi1000_timing': 100, 'star50_timing': 100, 'chinext_timing': 100,
    'sp500_timing': 1, 'macro_v32_timing': 1,
}
_LIVE_CURRENCY = {
    'csi1000_timing': 'CNY', 'star50_timing': 'CNY', 'chinext_timing': 'CNY',
    'sp500_timing': 'USD', 'macro_v32_timing': 'USD',
}

_STRATEGY_RULE14_STATUS = {
    'csi1000_timing': 'research',
    'star50_timing': 'production',
    'chinext_timing': 'research',
    'sp500_timing': 'research',
    'macro_v32_timing': 'production',
}

_REBALANCE_ACTION_LABELS = {
    'hold': '继续持有', 'enter': '建仓买入', 'add': '加仓',
    'trim': '减仓', 'exit': '清仓', 'flat': '空仓观望',
}

_BULLISH_RISKS_GENERAL = [
    "ETF 跟踪误差与滑点：实盘开盘成交价与策略基准 ETF 收盘价偏离（盘中波动 + 申赎冲击）会带来 0.1~0.3%/月级别的成本，不应被回测净值完全覆盖。",
    "策略历史样本有限：择时模型在极端 regime（熔断 / 政策急转 / 黑天鹅）下未必有可靠表现，遇到首次出现的环境建议主动降仓避险。",
]
_BULLISH_RISKS_BY_STRATEGY = {
    'csi1000_timing': [
        "中证1000 是小盘股代表，流动性弱于沪深300，缩量阶段反向时回撤会被放大。",
        "突破型信号易出现「假突破」：若 T+1 高开低走，建仓首日就可能回吐 1~2%。",
    ],
    'star50_timing': [
        "科创板个股集中度高（前 10 大权重接近一半），单只权重股波动会直接影响整体净值。",
        "科创板 IPO / 解禁 / 减持节奏对市场情绪冲击显著，需关注公告日临近的回撤风险。",
    ],
    'chinext_timing': [
        "创业板成长股估值对利率和流动性变化敏感，央行行长讲话 / 社融数据日易出现急跌。",
        "若指数已从底部反弹 ≥15%，继续追涨会面临短期获利盘抛压。",
    ],
    'sp500_timing': [
        "标普500 在牛市末期常出现「最后的拉升」后快速回落，注意 VIX 持续走低后的反转风险。",
        "美股财报季个股波动放大，板块轮动可能令 ETF 表现与个别龙头股脱钩。",
    ],
    'macro_v32_timing': [
        "宏观多因子模型依赖月频数据（PMI / 失业率 / CPI），对突发事件（地缘 / 银行流动性）反应滞后。",
        "Fed regime 切换时因子权重重排，策略短期信号可能反复，需容忍 1~2 周方向噪声。",
    ],
}
_BEARISH_OPPS_BY_REGION = {
    'cn': [
        "防御资产：10 年期国债 ETF（511260）或黄金 ETF（518880），历史上与小盘成长 ETF 呈负相关。",
        "现金管理：货币 ETF（511990）/ 银行 T+0 理财可作为等待信号期间的资金停车位。",
    ],
    'us': [
        "防御资产：长债 ETF（TLT）或黄金 ETF（GLD），在风险偏好下行时通常受益。",
        "现金管理：短端美债 ETF（SGOV / BIL），当前美债短端收益率仍处于历史较高水平。",
    ],
}
_REGION_OF_STRATEGY = {
    'csi1000_timing': 'cn', 'star50_timing': 'cn', 'chinext_timing': 'cn',
    'sp500_timing': 'us', 'macro_v32_timing': 'us',
}
_STRATEGY_DISPLAY = {
    'csi1000_timing': '中证1000',
    'star50_timing': '科创50',
    'chinext_timing': '创业板',
    'sp500_timing': '标普500',
    'macro_v32_timing': '纳指宏观 v3.3',
}
_SIGNAL_ACTION_LABELS = {
    'buy': '买入信号', 'sell': '卖出信号', 'hold': '维持', 'flat': '空仓',
}

_CACHE_DIR = _cache_store.CACHE_DIR
_CACHE_FILE = _cache_store.WEB_CACHE_FILE
FACTOR_BACKTEST_CACHE_FILE = _cache_store.FACTOR_BACKTEST_CACHE_FILE
FACTOR_BACKTEST_BUILD_SCRIPT = _cache_store.FACTOR_BACKTEST_BUILD_SCRIPT


# ═════════════════════════════════════════════════════════════════
# 策略 MAP（自动注册）
# ─────────────────────────────────────────────────────────────────
# Pillar 1 Step 5：原来这里是三个硬编码字典，现在直接 alias 到
# strategies.base 的三个 registry。同一对象引用，下游 import 不变。
# 新增策略只要：
#   1. 在策略类上声明 strategy_id + registry
#   2. 在本文件顶部 import 该模块（让 __init_subclass__ 触发）
# 不再需要在多处同步 MAP。
# ═════════════════════════════════════════════════════════════════
STRATEGY_MAP = STRATEGY_REGISTRY
TIMING_STRATEGY_MAP = TIMING_REGISTRY
US_TIMING_STRATEGY_MAP = US_TIMING_REGISTRY
COMMODITY_STRATEGY_MAP = COMMODITY_REGISTRY
HK_STRATEGY_MAP = HK_TIMING_REGISTRY

US_TIMING_PAGE_STRATEGY_IDS = [
    'macro_v32_timing',
    'sp500_timing',
]


def _collect_changelog_meta(registry):
    """从 registry 里收集每个策略类的 changelog_meta，返回 {strategy_id: meta}。

    Pillar 1 Step 5：原来 web/state.py 维护两份硬编码 dict
    (TIMING_CHANGELOG_META / US_TIMING_CHANGELOG_META)。现在 meta 直接挂在
    策略类上 (cls.changelog_meta)，blueprint 在请求时按需取。
    """
    return {sid: dict(cls.changelog_meta) for sid, cls in registry.items()
            if getattr(cls, 'changelog_meta', None)}


# 兼容老 import 路径：blueprint 仍可写 state.TIMING_CHANGELOG_META[sid]
TIMING_CHANGELOG_META = _collect_changelog_meta(TIMING_REGISTRY)
US_TIMING_CHANGELOG_META = _collect_changelog_meta(US_TIMING_REGISTRY)

FACTOR_OVERVIEW = [
    {'name': '市场因子', 'core_fields': '市场组合收益率、无风险收益率', 'sort_direction': '不适用', 'long_short': '不适用', 'double_sort': '不适用', 'book_recommended': '是', 'category': '风险归因'},
    {'name': '规模因子', 'core_fields': '总市值', 'sort_direction': '从低到高', 'long_short': 'Small - Big', 'double_sort': '否', 'book_recommended': '是', 'category': '核心选股'},
    {'name': '价值因子', 'core_fields': 'BM', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '是', 'category': '核心选股'},
    {'name': '动量因子', 'core_fields': '过去 11 个月累计收益', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，A 股不稳', 'category': '核心选股'},
    {'name': '盈利因子', 'core_fields': 'ROE(TTM)', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '是', 'category': '核心选股'},
    {'name': '投资因子', 'core_fields': '总资产同比增长率', 'sort_direction': '从低到高', 'long_short': 'Low - High', 'double_sort': '是，但仍受污染', 'book_recommended': '否，证据较弱', 'category': '核心选股'},
    {'name': '换手率因子', 'core_fields': '异常换手率', 'sort_direction': '从低到高', 'long_short': 'Low - High', 'double_sort': '是', 'book_recommended': '是，A 股很强', 'category': '交易行为'},
    {'name': '缠论背驰因子', 'core_fields': '收盘价、MACD柱', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，缠论扩展', 'category': '缠论扩展'},
    {'name': '缠论中枢位置因子', 'core_fields': '最高价、最低价、收盘价', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，缠论扩展', 'category': '缠论扩展'},
    {'name': '缠论分型因子', 'core_fields': '最高价、最低价', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，缠论扩展', 'category': '缠论扩展'},
    {'name': '缠论笔强度因子', 'core_fields': '收盘价、涨跌幅_20', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，缠论扩展', 'category': '缠论扩展'},
    {'name': '缠论买卖点信号因子', 'core_fields': '收盘价、最高价、最低价、MACD', 'sort_direction': '从低到高', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '否，缠论扩展', 'category': '缠论扩展'},
    {'name': 'BIAS偏离因子', 'core_fields': 'bias_20 (20日偏离率)', 'sort_direction': '从低到高（超跌优先）', 'long_short': 'Low - High', 'double_sort': '是', 'book_recommended': '否，技术指标', 'category': '技术指标', 'single_factor_id': 'bias'},
    {'name': 'KDJ超卖因子', 'core_fields': 'J值 (KDJ随机指标)', 'sort_direction': '从低到高（超卖优先）', 'long_short': 'Low - High', 'double_sort': '否', 'book_recommended': '否，技术指标', 'category': '技术指标', 'single_factor_id': 'kdj_j'},
    {'name': '市盈率倒数因子', 'core_fields': '市盈率倒数 (EP)', 'sort_direction': '从高到低（高EP优先）', 'long_short': 'High - Low', 'double_sort': '是', 'book_recommended': '是，与BM有一定互补', 'category': '核心选股', 'single_factor_id': 'pe_inv'},
    {'name': '成交额波动因子', 'core_fields': '成交额std_10 (10日成交额标准差)', 'sort_direction': '从低到高（低波动优先）', 'long_short': 'Low - High', 'double_sort': '否', 'book_recommended': '否，流动性衍生', 'category': '交易行为', 'single_factor_id': 'vol_stab'},
]

SINGLE_FACTOR_ID_TO_OVERVIEW_NAME = {
    'size': '规模因子', 'pb': '价值因子', 'profit': '盈利因子', 'momentum': '动量因子',
    'bias': 'BIAS偏离因子', 'kdj_j': 'KDJ超卖因子', 'pe_inv': '市盈率倒数因子',
    'vol_stab': '成交额波动因子', 'turn_abn': '换手率因子',
    'chan_div': '缠论背驰因子', 'chan_zs': '缠论中枢位置因子',
    'chan_fr': '缠论分型因子', 'chan_str': '缠论笔强度因子', 'chan_sig': '缠论买卖点信号因子',
}

FOCUSED_STRATEGY_ID = 'original_ensemble'


# ═════════════════════════════════════════════════════════════════
# 加载状态
# ═════════════════════════════════════════════════════════════════
_LOAD_STATUS = {
    'loading': False,
    'start_time': None,
    'end_time': None,
    'message': '等待启动',
    'stage': 'idle',
}
_DATA_READY = threading.Event()

_UPDATE_DATA_STATUS = {
    'running': False,
    'stage': 'idle',
    'message': '',
    'progress_pct': 0,
    'error': None,
}

_INDEX_UPDATE_STATUS = {'stage': 'idle', 'message': '', 'progress': 0, 'warning': None, 'details': None}

# 辅助数据流水线状态：FRED 宏观 + A股估值/情绪 + risk_signals 汇总
_AUX_UPDATE_STATUS = {
    'running': False, 'stage': 'idle', 'message': '', 'progress_pct': 0, 'error': None,
}

# 衍生因子流水线状态：当前仅 sector_weekly_heat（行业周度热度）
_FACTOR_UPDATE_STATUS = {
    'running': False, 'stage': 'idle', 'message': '', 'progress_pct': 0, 'error': None,
}


# ═════════════════════════════════════════════════════════════════
# 数据加载 / 缓存初始化
# ═════════════════════════════════════════════════════════════════
def _save_disk_cache():
    _cache_store.save_web_cache(
        backtest_cache=BACKTEST_CACHE,
        timing_cache=TIMING_CACHE,
        profile_summary_cache=_PROFILE_SUMMARY_CACHE,
    )


def _load_disk_cache():
    return _cache_store.load_web_cache(
        backtest_cache=BACKTEST_CACHE,
        timing_cache=TIMING_CACHE,
        profile_summary_cache=_PROFILE_SUMMARY_CACHE,
    )


def _load_factor_backtest_cache():
    return _cache_store.load_factor_cache(FACTOR_BACKTEST_CACHE)


def ensure_index_returns_loaded():
    global INDEX_RETURNS, INDEX_RETURNS_MAP
    if INDEX_RETURNS_MAP:
        return
    index_returns_map = {}
    for index_id, cfg in INDEX_CONFIGS.items():
        try:
            series = get_index_returns(index_id=index_id)
            index_returns_map[index_id] = series
            print(f"[init] {cfg['name']} 指数收益加载完成，{len(series)} 个月")
        except Exception as e:
            print(f"[WARN] 无法加载 {cfg['name']} 指数收益: {e}")
    INDEX_RETURNS_MAP = index_returns_map
    INDEX_RETURNS = INDEX_RETURNS_MAP.get('csi1000')
    if INDEX_RETURNS is None:
        print('[WARN] CSI 1000 不可用，业绩归因功能将不可用')


def ensure_stock_data_loaded():
    global DATA_DF
    if DATA_DF is not None:
        return
    if _LOAD_STATUS.get('loading'):
        _DATA_READY.wait()
        if DATA_DF is not None:
            return
    csv_path = os.path.join(_PROJECT_ROOT, 'stock_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'数据文件不存在: {csv_path}')
    print('[init] 加载数据中 (823MB)...')
    DATA_DF = load_data(csv_path)
    print('[init] 数据加载完成')


def init_cache():
    """Load the prebuilt stock-selection cache; never calculate in Flask."""
    if BACKTEST_CACHE:
        return
    if not _load_disk_cache():
        print(
            f'[init] 选股缓存缺失或不兼容: {_cache_store.WEB_CACHE_FILE}; '
            'R0 保持 unavailable，不在 Web 进程中重建。'
        )


def ensure_timing_panel_loaded():
    global TIMING_PANEL
    ensure_index_returns_loaded()
    if TIMING_PANEL is not None:
        return
    try:
        TIMING_PANEL = build_index_panel()
    except Exception as e:
        print(f"[WARN] 无法加载指数日线面板: {e}")
        TIMING_PANEL = None
        raise


def ensure_us_timing_panel_loaded(force_reload=False):
    global US_TIMING_PANEL
    if US_TIMING_PANEL is not None and not force_reload:
        return
    try:
        US_TIMING_PANEL = build_us_index_panel()
        print(f"[init] 美股ETF面板加载完成: {len(US_TIMING_PANEL)} 行")
    except Exception as e:
        print(f"[WARN] 无法加载美股ETF面板: {e}")
        US_TIMING_PANEL = None
        raise


def ensure_commodity_panel_loaded(force_reload=False):
    global COMMODITY_PANEL
    if COMMODITY_PANEL is not None and not force_reload:
        return
    try:
        COMMODITY_PANEL = build_commodity_index_panel()
        print(f"[init] 大宗商品ETF面板加载完成: {len(COMMODITY_PANEL)} 行")
    except Exception as e:
        print(f"[WARN] 无法加载大宗商品ETF面板: {e}")
        COMMODITY_PANEL = None
        raise


def init_commodity_cache():
    """R0 has no commodity cache builder or request-time fallback."""
    if not COMMODITY_CACHE:
        print('[init] 大宗商品预构建缓存不可用；R0 不在 Web 进程中计算。')


def ensure_hk_panel_loaded(force_reload=False):
    global HK_PANEL
    if HK_PANEL is not None and not force_reload:
        return
    try:
        HK_PANEL = build_hk_index_panel()
        print(f"[init] 港股ETF面板加载完成: {len(HK_PANEL)} 行")
    except Exception as e:
        print(f"[WARN] 无法加载港股ETF面板: {e}")
        HK_PANEL = None
        raise


def init_hk_cache():
    """R0 has no Hong Kong cache builder or request-time fallback."""
    if not HK_CACHE:
        print('[init] 港股预构建缓存不可用；R0 不在 Web 进程中计算。')


def build_us_timing_strategy(strategy_name='macro_v32_timing', **params):
    strat_cls = US_TIMING_STRATEGY_MAP.get(strategy_name, MacroV32TimingStrategy)
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = dict(_US_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})

    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}

    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        if k == 'profit_lock_enabled':
            v = bool(v)
        elif k == 'limit_max_delay_days':
            v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate',
                   'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
                   'profit_lock_drawdown'}:
            v = max(float(v or 0.0), 0.0)
        elif k == 'base_floor':
            # floor 是 0~1 的仓位比例，best_profile 误写 >1 会让 target_exposure 超过 100%。
            v = min(max(float(v or 0.0), 0.0), 1.0)
        else:
            v = float(v)
        setattr(instance, k, v)
    return instance


def init_us_timing_cache():
    """从离线缓存加载美股择时回测结果。"""
    global US_TIMING_CACHE
    if US_TIMING_CACHE:
        return
    cache_dir = os.path.join(_CACHE_DIR, 'us_timing')
    US_TIMING_CACHE = {}
    missing = []
    for sid in US_TIMING_STRATEGY_MAP.keys():
        pkl_path = os.path.join(cache_dir, f"{sid}.pkl")
        if not os.path.exists(pkl_path):
            missing.append(sid)
            print(f"[init] {sid} 美股择时缓存缺失: {pkl_path}")
            continue
        try:
            with open(pkl_path, 'rb') as f:
                US_TIMING_CACHE[sid] = pickle.load(f)
            print(f"[init] {sid} 美股择时缓存载入, 累积净值: {US_TIMING_CACHE[sid]['累积净值'].iloc[-1]:.4f}")
        except Exception as e:
            missing.append(sid)
            print(f"[init] {sid} 美股择时缓存载入失败: {e}")
    if missing:
        print(f"[init] 请运行: python scripts/build_us_timing_cache.py 重建缓存 (缺失: {missing})")


def init_timing_cache():
    """Load the prebuilt China timing cache; never calculate in Flask."""
    if TIMING_CACHE:
        return
    if not _load_disk_cache():
        print(
            f'[init] A 股择时缓存缺失或不兼容: {_cache_store.WEB_CACHE_FILE}; '
            'R0 保持 unavailable，不在 Web 进程中重建。'
        )


def _get_csi1000_timing_gate():
    return CSI1000_SIGNAL_SERIES if CSI1000_SIGNAL_SERIES is not None and len(CSI1000_SIGNAL_SERIES) > 0 else None


def build_strategy(strategy_name='original', **params):
    strat_cls = STRATEGY_MAP.get(strategy_name, OriginalStrategy)
    sig = inspect.signature(strat_cls.__init__)
    valid_params = {k: v for k, v in params.items()
                    if k in sig.parameters and v is not None}
    valid_params.pop('self', None)
    return strat_cls(**valid_params)


def _load_best_profile(strategy_name):
    if strategy_name in _BEST_PROFILE_CACHE:
        return _BEST_PROFILE_CACHE[strategy_name]
    fp = os.path.join(_BEST_PROFILE_DIR, f'best_profile_{strategy_name}.json')
    if not os.path.exists(fp):
        _BEST_PROFILE_CACHE[strategy_name] = None
        return None
    try:
        with open(fp) as f:
            profile = json.load(f)
        # 读入时立即清洗 NaN，确保缓存中不含非法 JSON 字面量（防止磁盘文件已损坏时传播）
        profile = _deep_sanitize_nan(profile)
        _BEST_PROFILE_CACHE[strategy_name] = profile
        return profile
    except Exception as exc:
        print(f"[WARN] 读取 {fp} 失败: {exc}")
        _BEST_PROFILE_CACHE[strategy_name] = None
        return None


def _load_holdout_report(strategy_name):
    fp = os.path.join(_BEST_PROFILE_DIR, f'holdout_report_{strategy_name}.json')
    if not os.path.exists(fp):
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': None, 'mtime': 0}
        return None
    mtime = os.path.getmtime(fp)
    cache = _HOLDOUT_REPORT_CACHE.get(strategy_name)
    if cache and cache.get('mtime') == mtime and cache.get('data') is not None:
        return cache['data']
    try:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        # 读入时立即清洗 NaN（holdout_report.json 中的 metrics 也可能含 NaN）
        data = _deep_sanitize_nan(data)
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': data, 'mtime': mtime}
        return data
    except Exception as exc:
        print(f"[WARN] 读取 {fp} 失败: {exc}")
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': None, 'mtime': 0}
        return None


def _load_risk_signals():
    if not os.path.exists(_RISK_SIGNALS_FILE):
        _RISK_SIGNALS_CACHE['data'] = None
        _RISK_SIGNALS_CACHE['mtime'] = 0
        return None
    mtime = os.path.getmtime(_RISK_SIGNALS_FILE)
    if mtime == _RISK_SIGNALS_CACHE['mtime'] and _RISK_SIGNALS_CACHE['data'] is not None:
        return _RISK_SIGNALS_CACHE['data']
    try:
        with open(_RISK_SIGNALS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        _RISK_SIGNALS_CACHE['data'] = data
        _RISK_SIGNALS_CACHE['mtime'] = mtime
        return data
    except Exception as exc:
        print(f"[WARN] 读取 {_RISK_SIGNALS_FILE} 失败: {exc}")
        _RISK_SIGNALS_CACHE['data'] = None
        return None


def _nan_to_none(v):
    """将 float NaN/Inf 转 None（标量版本，保留以向后兼容）。

    NaN 判断用 v != v（IEEE 754 标准：只有 NaN 不等于自身），无需额外 import。
    """
    if isinstance(v, float) and v != v:
        return None
    return v


def _deep_sanitize_nan(obj):
    """递归将所有 float NaN/Inf 替换为 None，用于在磁盘 JSON 读入时一次性清洗。

    Python json.load 默认 allow_nan=True，所以磁盘上若存在字面量 NaN（非法 JSON 但
    Python 容忍）会被读成 float('nan')，进而在 jsonify 时重新输出非法 JSON。
    在缓存入口处调用此函数，确保 _BEST_PROFILE_CACHE / _HOLDOUT_REPORT_CACHE
    中永远不含 NaN 值，即使磁盘文件已损坏也能安全运行。
    """
    if isinstance(obj, float) and obj != obj:  # NaN check (obj != obj is True only for NaN)
        return None
    if isinstance(obj, dict):
        return {k: _deep_sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_sanitize_nan(v) for v in obj]
    return obj


def get_best_profile_view(strategy_name):
    profile = _load_best_profile(strategy_name)
    if profile is None:
        return None
    return {
        'strategy_id': strategy_name,
        'training_cutoff': profile.get('training_cutoff'),
        'generated_at': profile.get('generated_at'),
        'score': _nan_to_none(profile.get('score')),
        'score_formula': profile.get('score_formula'),
        'maxdd_threshold': _nan_to_none(profile.get('maxdd_threshold')),
        'tuned_params': profile.get('tuned_params', {}),
        'window_metrics': profile.get('window_metrics', {}),
    }


def build_timing_strategy(strategy_name='csi1000_timing', **params):
    strat_cls = TIMING_STRATEGY_MAP.get(strategy_name, CSI1000TimingStrategy)
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = _get_timing_default_params(strategy_name)
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})

    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}

    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        if k == 'profit_lock_enabled':
            v = bool(v)
        elif k == 'limit_max_delay_days':
            v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate',
                   'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
                   'profit_lock_drawdown'}:
            v = max(float(v or 0.0), 0.0)
        elif k == 'base_floor':
            # floor 是 0~1 的仓位比例，best_profile 误写 >1 会让 target_exposure 超过 100%。
            v = min(max(float(v or 0.0), 0.0), 1.0)
        else:
            v = float(v)
        setattr(instance, k, v)
    return instance


def build_commodity_strategy(strategy_name='gold_timing', **params):
    strat_cls = COMMODITY_STRATEGY_MAP.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f'未知大宗商品策略: {strategy_name}')
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = {}
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})
    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}
    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        if k == 'profit_lock_enabled':
            v = bool(v)
        elif k == 'limit_max_delay_days':
            v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate',
                   'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
                   'profit_lock_drawdown'}:
            v = max(float(v or 0.0), 0.0)
        elif k == 'base_floor':
            v = min(max(float(v or 0.0), 0.0), 1.0)
        else:
            v = float(v)
        setattr(instance, k, v)
    return instance


def build_hk_strategy(strategy_name='hsi_timing', **params):
    strat_cls = HK_STRATEGY_MAP.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f'未知港股策略: {strategy_name}')
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = {}
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})
    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}
    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        if k == 'profit_lock_enabled': v = bool(v)
        elif k == 'limit_max_delay_days': v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate', 'commission_min', 'stamp_tax_rate', 'transfer_fee_rate', 'profit_lock_drawdown'}: v = max(float(v or 0.0), 0.0)
        elif k == 'base_floor': v = min(max(float(v or 0.0), 0.0), 1.0)
        else: v = float(v)
        setattr(instance, k, v)
    return instance


def run_timing_backtest_fresh(strategy_name='csi1000_timing', benchmark_id=None, **params):
    """**Offline-only.** 跑一次完整的 A 股择时回测并返回 (result, metrics, strategy)。

    Pillar 1 Step 6 之后，blueprint 请求路径不允许再调用本函数 —— `/api/timing/backtest`
    在缓存缺失或参数偏离默认时直接返回 HTTP 400 `cache_miss`，让用户去跑
    `scripts/build_timing_cache.py`。本函数继续保留是因为：
      - 离线脚本（build_timing_cache、walk-forward、explore_compare debug）需要现算入口
      - `state.init_cache()` 启动时预跑 CSI1000 信号（短路径，不写 backtest_cache）
    不要在新代码里把它接进 Flask 请求路径。
    """
    ensure_timing_panel_loaded()
    strategy = build_timing_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    signal_df = strategy.run(TIMING_PANEL.copy())
    result = run_timing_backtest(signal_df, strategy, benchmark_returns=benchmark_series)
    metrics = evaluate_timing_result(result, benchmark_returns=benchmark_series)
    return result, metrics, strategy


def _build_timing_compare_payload(result_df, metrics, benchmark_series, params=None, start_date=None, end_date=None):
    payload = {
        'metrics': metrics,
        'interval_windows': summarize_timing_windows(result_df, benchmark_returns=benchmark_series),
    }
    if params is not None:
        payload['params'] = params
    if start_date or end_date:
        sliced = filter_timing_result(result_df, start_date=start_date, end_date=end_date)
        payload['current_range'] = {
            'start': start_date,
            'end': end_date,
            'rows': len(sliced),
            'metrics': evaluate_timing_result(sliced, benchmark_returns=benchmark_series) if len(sliced) else {},
        }
    return payload


def get_focused_strategy_id():
    return FOCUSED_STRATEGY_ID if FOCUSED_STRATEGY_ID in STRATEGY_MAP else 'original'


def build_factor_overview_payload(strategy_name=None):
    strategy_id = strategy_name or get_focused_strategy_id()
    strategy = build_strategy(strategy_id)

    if strategy_id in _PROFILE_SUMMARY_CACHE:
        setattr(strategy, '_profile_summary', _PROFILE_SUMMARY_CACHE[strategy_id])
    elif hasattr(strategy, '_resolve_profiles') and DATA_DF is not None and not getattr(strategy, '_profile_summary', None):
        try:
            strategy._resolve_profiles(DATA_DF)
        except Exception:
            pass
    active_tags = set(strategy.get_factor_overview_tags() or [])
    items = []
    for row in FACTOR_OVERVIEW:
        item = dict(row)
        item['active'] = item['name'] in active_tags
        items.append(item)
    payload = {
        'strategy_id': strategy_id,
        'strategy_name': strategy.get_display_name(),
        'active_factor_names': sorted(active_tags),
        'items': items,
        'single_factor_id_map': SINGLE_FACTOR_ID_TO_OVERVIEW_NAME,
    }
    if hasattr(strategy, '_profile_summary'):
        payload['profile_summary'] = getattr(strategy, '_profile_summary', [])
    return payload


def run_backtest_fresh(strategy_name='original', benchmark_id=None, **params):
    """**Offline-only.** 跑一次完整的月度选股回测并返回 (result, eval_df)。

    Pillar 1 Step 6 之后，blueprint 请求路径不允许再调用本函数 —— `/api/backtest`
    在缓存缺失或参数偏离默认时直接返回 HTTP 400 `cache_miss`，让用户去跑
    `scripts/build_select_cache.py`。本函数继续保留是因为：
      - 离线脚本（build_select_cache、compare_strategies、walk-forward）需要现算入口
      - `state.init_cache()` 启动时在缓存空白的兜底场景里临时预跑（短路径）
    不要在新代码里把它接进 Flask 请求路径。
    """
    if DATA_DF is None:
        raise RuntimeError("数据未加载，请确认 stock_data.csv 存在")

    strategy = build_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)

    df = strategy.run(DATA_DF.copy())
    result = select_and_backtest(df, strategy,
                                 c_rate=strategy.c_rate, t_rate=strategy.t_rate,
                                 bull_tp=strategy.bull_tp, bear_tp=strategy.bear_tp,
                                 bull_n=strategy.bull_n, bear_n=strategy.bear_n,
                                 initial_capital=strategy.initial_capital)
    if hasattr(strategy, '_profile_summary'):
        result.attrs['strategy_meta'] = {
            'profile_summary': getattr(strategy, '_profile_summary', []),
        }
    ev = strategy_evaluate(result, index_returns=benchmark_series)
    return result, ev


def filter_by_date(result, start_date, end_date, benchmark_id=None):
    original = result.copy()
    original_dates = list(original['交易日期'])
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    if start_date:
        result = result[result['交易日期'] >= pd.to_datetime(start_date)].copy()
    if end_date:
        result = result[result['交易日期'] <= pd.to_datetime(end_date)].copy()
    if len(result) == 0:
        return None, None

    result['累积净值'] = (1 + result['选股下周期涨跌幅']).cumprod()
    result['资金曲线'] = result['累积净值']

    start_capital = float(result.attrs.get('initial_capital', 100000))

    capital = float(start_capital)
    capitals = []
    pnls = []
    cum_caps = []
    for _, row in result.iterrows():
        capitals.append(capital)
        pnl = capital * row['选股下周期涨跌幅']
        pnls.append(pnl)
        capital += pnl
        cum_caps.append(capital)

    result['当期本金'] = capitals
    result['当期盈亏'] = pnls
    result['累计资金'] = cum_caps
    result.attrs['initial_capital'] = start_capital

    if 'period_daily_curves' in original.attrs:
        curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(original_dates, original.attrs.get('period_daily_curves', []))
        }
        result.attrs['period_daily_curves'] = [
            curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in result['交易日期']
        ]
    if 'daily_equity_curve' in original.attrs:
        result.attrs['daily_equity_curve'] = _build_daily_curve_slice(result, original.attrs.get('daily_equity_curve', []))

    ev = strategy_evaluate(result, initial_capital=start_capital,
                           index_returns=benchmark_series)
    return result, ev


# ═════════════════════════════════════════════════════════════════
# 真实交易规则参数 / 缓存默认值
# ═════════════════════════════════════════════════════════════════
_CACHE_DEFAULTS = {
    'original': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78},
    'original_ensemble': {
        'weight_3y': 0.5, 'weight_5y': 0.3, 'weight_full': 0.2,
        'vote_top_k': 12, 'board_tilt_strength': 0.4,
        'growth_timing_mode': 'off', 'growth_hold_days': 4, 'growth_top_n': 2,
    },
    'chan_enhanced': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.03},
    'chan_only': {'chan_weight': 0.70},
    'method_a': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.05},
    'quality_value': {
        'size_weight': 0.50, 'bm_weight': 0.25, 'roe_weight': 0.15,
        'turnover_weight': 0.10, 'min_market_cap': 20, 'min_turnover': 0.5,
        'select_stock_num': 3, 'bias_pct': 0.52, 'vol_pct': 0.78,
    },
}

_SHARED_REALISM_DEFAULTS = {
    'profit_lock_enabled': False,
    'profit_lock_drawdown': 0.04,
    'profit_lock_level_1': 0.10,
    'profit_lock_level_2': 0.18,
    'profit_lock_level_3': 0.28,
    'slippage_bps': 5.0,
    'cash_interest_rate': 0.015,
    'commission_rate': 0.0001,
    'commission_min': 5.0,
    'stamp_tax_rate': 0.0,
    'transfer_fee_rate': 0.00001,
    'limit_max_delay_days': 5,
    'base_floor': 0.0,
}

_TIMING_CACHE_DEFAULTS = {
    'csi1000_timing': {
        'breakout_window': 15, 'exit_window': 7, 'trend_window': 50,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.38, 'exit_threshold': 0.18, 'confirm_days': 1,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
    'star50_timing': {
        'breakout_window': 10, 'exit_window': 5, 'trend_window': 40,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
    'chinext_timing': {
        'momentum_short_window': 20, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.02,
        'exposure_mode': 'staged', 'enter_threshold': 0.6, 'add_threshold': 0.8,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 2,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
}

_US_TIMING_CACHE_DEFAULTS = {
    'macro_v32_timing': {
        'sigmoid_k': 1.2, 'max_leverage': 1.4, 'base_position': 0.45,
        'inertia': 0.05, 'crisis_vix': 40.0,
        'fed_block_weight': 0.25, 'restrictive_threshold': 0.40, 'pivot_relief': 0.60,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 1.0,
        **_SHARED_REALISM_DEFAULTS,
    },
    'sp500_timing': {
        'fast_window': 20, 'slow_window': 125, 'momentum_window': 100,
        'exposure_mode': 'staged', 'enter_threshold': 0.5, 'add_threshold': 0.72,
        'trim_threshold': 0.32, 'exit_threshold': 0.14, 'confirm_days': 2,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
}

_REALISM_BOOL_KEYS = {'profit_lock_enabled'}
_REALISM_INT_KEYS = {'limit_max_delay_days'}
_REALISM_FLOAT_KEYS = {
    'profit_lock_drawdown', 'profit_lock_level_1', 'profit_lock_level_2', 'profit_lock_level_3',
    'slippage_bps', 'cash_interest_rate',
    'commission_rate', 'commission_min',
    'stamp_tax_rate', 'transfer_fee_rate',
    'base_floor',
}
_REALISM_ALL_KEYS = _REALISM_BOOL_KEYS | _REALISM_INT_KEYS | _REALISM_FLOAT_KEYS


def _parse_realism_bool(raw):
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {'on', 'true', '1', 'yes', 'y'}:
        return True
    if s in {'off', 'false', '0', 'no', 'n', ''}:
        return False
    return None


def _collect_realism_params(args):
    params = {}
    for key in _REALISM_BOOL_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        val = _parse_realism_bool(raw)
        if val is not None:
            params[key] = val
    for key in _REALISM_INT_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        try:
            params[key] = int(float(raw))
        except (TypeError, ValueError):
            continue
    for key in _REALISM_FLOAT_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        try:
            params[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return params


def _get_timing_default_params(strategy_name):
    return dict(_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))


# ── 缓存生效默认值（用于 blueprint 判断「请求参数与离线缓存等价」）──
# 缓存由 build_timing_strategy / build_us_timing_strategy 跑出来，那里实际用的是
# `_TIMING_CACHE_DEFAULTS + best_profile.all_params`。所以前端 /api/timing/params
# 返回的 default（来自 strategy 实例属性 = 上面这套 merged 值）和裸的 _TIMING_CACHE_DEFAULTS
# 是不一致的——如果 blueprint 拿裸 defaults 去比较，前端按其默认值回放就直接 cache_miss。
# 下面两个函数返回的就是与缓存等价的「effective defaults」。

_EFFECTIVE_TIMING_DEFAULTS_CACHE = {}
_EFFECTIVE_US_TIMING_DEFAULTS_CACHE = {}


def _extract_effective_from_instance(instance, base_defaults):
    """从 build_*_strategy 出来的实例上提取所有可调 timing 参数的实际取值。

    缓存是用这个 instance 跑出来的，所以它的属性 = 缓存的「真实生效默认值」。
    覆盖关系：base_defaults（_*_CACHE_DEFAULTS 静态默认） ⊂ 实例属性（叠加 best_profile + 类硬编码）。

    ⚠ 关键 snap：HTML `<input type=range step="0.05">` 的初始 value 若不在 step 网格上
       （例如 best_profile 的 trim_threshold=0.38, step=0.05），浏览器读 `el.value`
       会自动 round 到最近 step（0.40）。前端默认页就是这样把 0.38 发成 0.40。
       后端 effective_defaults 必须按同一 step 做 snap，否则严格比较会必然 miss。
       snap 不会放大 user 真改动 tolerance（用户拖到 0.45 仍会与 snap 后的 0.40 不等）。
    """
    # 从 TimingParams 已知字段（与前端 slider 的 key 集合一致）里抓
    from web.params import _BOOL_KEYS, _INT_KEYS, _FLOAT_KEYS, _STR_KEYS
    known = _BOOL_KEYS | _INT_KEYS | _FLOAT_KEYS | _STR_KEYS
    eff = dict(base_defaults)
    for key in known:
        if hasattr(instance, key):
            val = getattr(instance, key)
            if val is None:
                continue
            eff[key] = val

    # snap 到前端 slider step
    try:
        meta_params = instance.get_signal_metadata().get('parameters', []) or []
    except Exception:
        meta_params = []
    for p in meta_params:
        key = p.get('key')
        step = p.get('step')
        if not key or key not in eff:
            continue
        if step is None:
            continue
        try:
            step_f = float(step)
        except (TypeError, ValueError):
            continue
        if step_f <= 0:
            continue
        val = eff[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        # 按 step 四舍五入；再 round 到 10 位避免浮点尾数
        snapped = round(round(float(val) / step_f) * step_f, 10)
        eff[key] = snapped
    return eff


def get_effective_timing_defaults(strategy_name):
    """返回 A 股择时缓存实际生效的参数。

    口径：与 build_timing_strategy(strategy_name) 出来的实例属性一致 —— 这就是
    init_timing_cache / build_timing_cache 跑出来的那份缓存所使用的参数。
    覆盖来源：strategy 类 __init__ 硬编码 默认 < _TIMING_CACHE_DEFAULTS < best_profile.all_params
    （由 build_timing_strategy 内部 merge）。
    """
    if strategy_name in _EFFECTIVE_TIMING_DEFAULTS_CACHE:
        return dict(_EFFECTIVE_TIMING_DEFAULTS_CACHE[strategy_name])
    base = dict(_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    try:
        instance = build_timing_strategy(strategy_name)
        eff = _extract_effective_from_instance(instance, base)
    except Exception as exc:
        print(f'[WARN] get_effective_timing_defaults({strategy_name}) build_strategy failed: {exc}')
        eff = base
        profile = _load_best_profile(strategy_name)
        if profile is not None:
            for k, v in (profile.get('all_params') or {}).items():
                if v is not None:
                    eff[k] = v
    _EFFECTIVE_TIMING_DEFAULTS_CACHE[strategy_name] = dict(eff)
    return eff


def get_effective_us_timing_defaults(strategy_name):
    """美股择时版本，口径同 get_effective_timing_defaults。"""
    if strategy_name in _EFFECTIVE_US_TIMING_DEFAULTS_CACHE:
        return dict(_EFFECTIVE_US_TIMING_DEFAULTS_CACHE[strategy_name])
    base = dict(_US_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    try:
        instance = build_us_timing_strategy(strategy_name)
        eff = _extract_effective_from_instance(instance, base)
    except Exception as exc:
        print(f'[WARN] get_effective_us_timing_defaults({strategy_name}) build_strategy failed: {exc}')
        eff = base
        profile = _load_best_profile(strategy_name)
        if profile is not None:
            for k, v in (profile.get('all_params') or {}).items():
                if v is not None:
                    eff[k] = v
    _EFFECTIVE_US_TIMING_DEFAULTS_CACHE[strategy_name] = dict(eff)
    return eff


def get_effective_select_defaults(strategy_name):
    """选股策略目前未走 best_profile，effective 等于裸 defaults。预留扩展点。"""
    return dict(_CACHE_DEFAULTS.get(strategy_name, {}))


# ═════════════════════════════════════════════════════════════════
# 实盘 helpers
# ═════════════════════════════════════════════════════════════════
def _latest_live_position(strategy_id):
    return _live_trades_service.latest_position(strategy_id)


def _derive_action_from_delta(live_pos, target, tol=0.005):
    delta = target - live_pos
    if abs(delta) < tol:
        return 'flat' if target <= tol else 'hold'
    if delta > 0:
        return 'enter' if live_pos <= tol else 'add'
    return 'exit' if target <= tol else 'trim'


def _format_money(v, currency):
    sym = '$' if currency == 'USD' else '¥'
    return f'{sym}{v:,.2f}'


def _build_action_rationale(action, live_pos, target, ref_open, ref_close, strategy_id):
    capital = _LIVE_INITIAL_CAPITAL
    currency = _LIVE_CURRENCY.get(strategy_id, 'CNY')
    lot = _LIVE_LOT_SIZE.get(strategy_id, 1)
    price = ref_open or ref_close
    delta = target - live_pos
    cap_str = _format_money(capital, currency)

    def _shares_for(pos):
        if not price or price <= 0:
            return None
        return int((capital * pos) / price / lot) * lot

    target_shares = _shares_for(target)
    live_shares = _shares_for(live_pos)
    delta_shares = (target_shares - live_shares) if (target_shares is not None and live_shares is not None) else None
    px_txt = f'{price:.3f}' if price else '—'

    if action == 'enter':
        return (
            f'首次建仓。策略目标 {target*100:.1f}%，实盘当前 0%（{cap_str} 全部空仓）。'
            f'建议 T+1 开盘按参考价 {px_txt} 一次性买入约 {target_shares or "—"} 股，'
            f'完成后实盘仓位 ≈ {target*100:.1f}%。每手 {lot} 股已按整手取整。'
        )
    if action == 'add':
        return (
            f'加仓补齐。实盘当前 {live_pos*100:.1f}%（约 {live_shares or "—"} 股），策略目标 {target*100:.1f}%，'
            f'差 +{delta*100:.1f}%。建议 T+1 开盘按 {px_txt} 补买约 {delta_shares if delta_shares is not None else "—"} 股。'
        )
    if action == 'trim':
        return (
            f'减仓回落。实盘当前 {live_pos*100:.1f}%（约 {live_shares or "—"} 股），策略目标降到 {target*100:.1f}%，'
            f'差 {delta*100:.1f}%。建议 T+1 开盘按 {px_txt} 卖出约 {abs(delta_shares) if delta_shares is not None else "—"} 股。'
        )
    if action == 'exit':
        return (
            f'清仓离场。策略已切换为空仓信号（目标 0%），实盘当前 {live_pos*100:.1f}%。'
            f'建议 T+1 开盘按 {px_txt} 全部卖出，共约 {live_shares or "—"} 股，资金回到现金。'
        )
    if action == 'hold':
        return (
            f'与目标一致。实盘 {live_pos*100:.1f}% 与策略目标 {target*100:.1f}% 基本对齐（差 {abs(delta)*100:.2f}%），'
            f'无需操作。等待下一次信号变化（每日收盘后由 T+1 信号决定）。'
        )
    if action == 'flat':
        return (
            f'空仓观望。策略目标 0%，实盘已为 0%，无持仓需要管理。'
            f'等待 ETF 给出新的入市信号（如均线突破 + 成交量放量配合）再建仓。'
        )
    return ''


def _build_decision_context(strategy_id, target, live_pos, action):
    if target >= 0.5 or action in ('enter', 'add'):
        view_bias = 'bullish'
    elif target <= 0.3 or action in ('exit', 'trim', 'flat'):
        view_bias = 'bearish'
    else:
        view_bias = 'neutral'

    risks = []
    opportunities = []

    dynamic = _load_risk_signals()
    dyn_by_strat = (dynamic or {}).get('by_strategy', {}).get(strategy_id, {})
    dyn_risks = list(dyn_by_strat.get('bullish_risks_dynamic', []))
    dyn_opps = list(dyn_by_strat.get('bearish_opportunities_dynamic', []))
    risk_as_of = (dynamic or {}).get('as_of')
    risk_generated_at = (dynamic or {}).get('generated_at')

    if view_bias == 'bullish':
        risks = dyn_risks \
              + list(_BULLISH_RISKS_BY_STRATEGY.get(strategy_id, [])) \
              + list(_BULLISH_RISKS_GENERAL)
    elif view_bias == 'bearish':
        cross = []
        for other_id, _other_cls in (
            list(TIMING_STRATEGY_MAP.items()) + list(US_TIMING_STRATEGY_MAP.items())
        ):
            if other_id == strategy_id:
                continue
            cache = TIMING_CACHE if other_id in TIMING_STRATEGY_MAP else US_TIMING_CACHE
            df = cache.get(other_id)
            if df is None or len(df) == 0:
                continue
            try:
                other_target = float(df.iloc[-1].get('target_exposure', 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if other_target >= 0.5:
                cross.append(
                    f'{_STRATEGY_DISPLAY.get(other_id, other_id)} 当前目标 {other_target*100:.0f}%，'
                    f'可考虑把部分资金分散到该方向。'
                )
        if dyn_opps:
            opportunities.extend(dyn_opps)
        if cross:
            opportunities.extend(cross)
        region = _REGION_OF_STRATEGY.get(strategy_id, 'cn')
        opportunities.extend(_BEARISH_OPPS_BY_REGION.get(region, []))

    return view_bias, risks, opportunities, risk_as_of, risk_generated_at


def _compute_bullish_score(signal_latest_row, strategy_id):
    """把当期信号转换成 0‒100 的看多强度分。

    设计意图（CLAUDE.md §10）：100 = 极度看多，0 = 极度看空，50 = 中性。
    算法按策略类型分两路：
      - A 股 / SP500（breakout + momentum 二元状态机）：
          strength_score（已经是 sigmoid 处理后的 0~1）× 80 + signal_action 修正项。
          strength_score 反映了突破/趋势/MACD 多重因子叠加后的强度。
      - macro_v32（ContScore 连续因子）：
          用 cont_score（加权宏观因子，已扣除 crisis/risk_off penalty）做 sigmoid 映射。
          ContScore ≈ 0 → 50 分（中性），+1 → ~88 分，-1 → ~12 分。
    """
    try:
        action = str(signal_latest_row.get('signal_action') or 'hold')
        if strategy_id in ('macro_v32_timing',):
            # macro_v32 用 cont_score（连续因子）
            cs = float(signal_latest_row.get('cont_score', 0.0) or 0.0)
            if pd.isna(cs):
                cs = 0.0
            # sigmoid(cs * 2) × 100，让 ±1 映射到 ~88/12，±2 映射到 ~98/2
            import math
            raw = 1.0 / (1.0 + math.exp(-cs * 2.0))
            return int(round(min(100, max(0, raw * 100))))
        else:
            # A 股 + SP500：strength_score（sigmoid 0~1）
            ss = float(signal_latest_row.get('strength_score', 0.5) or 0.5)
            if pd.isna(ss):
                ss = 0.5
            action_bonus = {'buy': 20, 'hold': 10, 'flat': -10, 'sell': -30}.get(action, 0)
            score = ss * 80 + action_bonus
            return int(round(min(100, max(0, score))))
    except Exception:
        return None


def _compute_current_signal_core(strategy_id, strategy, result_df):
    latest_settled = result_df.iloc[-1]
    settled_as_of_date_str = pd.to_datetime(latest_settled['交易日期']).strftime('%Y-%m-%d')
    prev_settled = result_df.iloc[-2] if len(result_df) >= 2 else latest_settled
    target = float(latest_settled.get('target_exposure', 0.0) or 0.0)
    prev_exp = float(prev_settled.get('target_exposure', 0.0) or 0.0)
    rebalance = str(latest_settled.get('rebalance_action') or 'hold')
    signal = str(latest_settled.get('signal_action') or 'hold')
    current_reason = latest_settled.get('reason_summary') or ''
    as_of_date_str = settled_as_of_date_str
    ref_close_val = float(latest_settled.get('etf_close', float('nan')))
    ref_close = None if pd.isna(ref_close_val) else ref_close_val
    ref_open_val = float(latest_settled.get('etf_open', float('nan')))
    ref_open = None if pd.isna(ref_open_val) else ref_open_val
    bullish_score = _compute_bullish_score(latest_settled, strategy_id)
    etf_code = latest_settled.get('etf_code') or None
    etf_name = latest_settled.get('etf_name') or (strategy.get_index_name() if strategy else None)

    # ── 设计意图（CLAUDE.md §10）：当前信号 = 用截至今日收盘全部已知数据推断的仓位建议。
    # result_df.iloc[-1] 是最后一个“已结算”行（需要 T+1 ETF 价格），比最新信号日滞后 1 天。
    # 真正的“当前信号”来自 TIMING_PANEL / US_TIMING_PANEL / COMMODITY_PANEL 最新行跑 strategy.run()，
    # 不依赖 T+1 结算价；但参考价格与净值仍保留最近已结算口径。
    try:
        if strategy_id in HK_STRATEGY_MAP:
            panel = HK_PANEL
        elif strategy_id in COMMODITY_STRATEGY_MAP:
            panel = COMMODITY_PANEL
        elif strategy_id in TIMING_STRATEGY_MAP:
            panel = TIMING_PANEL
        else:
            panel = US_TIMING_PANEL
        if panel is not None and len(panel) > 0 and strategy is not None:
            signal_df = strategy.run(panel.copy())
            if signal_df is not None and len(signal_df) > 0:
                signal_latest = signal_df.iloc[-1]
                target = float(signal_latest.get('target_exposure', 0.0) or 0.0)
                signal_prev = signal_df.iloc[-2] if len(signal_df) >= 2 else signal_latest
                prev_exp = float(signal_prev.get('target_exposure', 0.0) or 0.0)
                as_of_date_str = pd.to_datetime(signal_latest['交易日期']).strftime('%Y-%m-%d')
                rebalance = str(signal_latest.get('rebalance_action') or 'hold')
                signal = str(signal_latest.get('signal_action') or 'hold')
                current_reason = signal_latest.get('reason_summary') or current_reason
                bullish_score = _compute_bullish_score(signal_latest, strategy_id)
                etf_code = signal_latest.get('etf_code') or etf_code
                etf_name = signal_latest.get('etf_name') or etf_name
            else:
                raise ValueError('signal_df empty')
        else:
            raise ValueError('panel not available')
    except Exception as _sig_err:
        print(f'[_build_latest_signal] {strategy_id} live signal from panel failed: {_sig_err}, falling back to result_df',
              file=sys.stderr)

    exposure_delta = round(target - prev_exp, 4)
    settled_nav = round(float(latest_settled.get('累积净值', 1.0)), 4) if pd.notna(latest_settled.get('累积净值', None)) else None
    current_position = int(target > 1e-8)
    # data_stale_warning：信号日（as_of，来自新鲜指数面板）与已结算日（settled_as_of，
    # 依赖 T+1 ETF 价）正常相差 ≤1 天。相差 >2 天说明 ETF/结算价长期未刷新（如 qfq 停在
    # 数周前），此时 nav/ref_close 等标记并非最新，必须显式提示，避免交易员把陈旧 P&L 当今日。
    data_stale_warning = None
    try:
        _gap_days = (pd.Timestamp(as_of_date_str) - pd.Timestamp(settled_as_of_date_str)).days
        if _gap_days > 2:
            data_stale_warning = (
                f'ETF/结算数据滞后于信号日 {_gap_days} 天（信号日 {as_of_date_str}，'
                f'已结算日 {settled_as_of_date_str}），P&L 标记可能非最新，请刷新指数/ETF 数据。'
            )
    except Exception:
        data_stale_warning = None
    return {
        'strategy_id': strategy_id,
        'name': strategy.get_display_name() if strategy else strategy_id,
        'index_name': strategy.get_index_name() if strategy else etf_name,
        'etf_code': etf_code,
        'etf_name': etf_name,
        'as_of_date': as_of_date_str,
        'settled_as_of_date': settled_as_of_date_str,
        'data_stale_warning': data_stale_warning,
        'target_exposure': round(target, 4),
        'prev_exposure': round(prev_exp, 4),
        'exposure_delta': exposure_delta,
        'rebalance_action': rebalance,
        'rebalance_label': _REBALANCE_ACTION_LABELS.get(rebalance, rebalance),
        'signal_action': signal,
        'signal_label': _SIGNAL_ACTION_LABELS.get(signal, signal),
        'current_action': signal,
        'current_position': current_position,
        'current_reason': current_reason,
        'reason_summary': current_reason,
        'bullish_score': bullish_score,
        'ref_close': ref_close,
        'ref_open': ref_open,
        'nav': settled_nav,
        'settled_nav': settled_nav,
    }


def _build_latest_signal(strategy_id, strategy, result_df, profile=None):
    if result_df is None or len(result_df) == 0:
        return {
            'strategy_id': strategy_id,
            'name': strategy.get_display_name() if strategy else strategy_id,
            'status': _STRATEGY_RULE14_STATUS.get(strategy_id, 'research'),
            'as_of_date': None,
            'error': '缓存未加载',
        }

    core = _compute_current_signal_core(strategy_id, strategy, result_df)
    target = core['target_exposure']
    status = _STRATEGY_RULE14_STATUS.get(strategy_id, 'research')
    live_position = _latest_live_position(strategy_id)
    live_action = _derive_action_from_delta(live_position, target)
    live_delta = round(target - live_position, 4)
    action_rationale = _build_action_rationale(live_action, live_position, target, core['ref_open'], core['ref_close'], strategy_id)
    view_bias, risks, opportunities, risk_signals_as_of, risk_signals_generated_at = _build_decision_context(strategy_id, target, live_position, live_action)
    profile_window = (profile or {}).get('window_metrics', {}) if isinstance(profile, dict) else {}
    if not isinstance(profile_window, dict):
        profile_window = {}
    win = profile_window.get('recent_6m', {}) if isinstance(profile_window, dict) else {}
    holdout_payload = _load_holdout_report(strategy_id) or {}
    holdout_metrics = holdout_payload.get('holdout_metrics') if isinstance(holdout_payload, dict) else None
    training_metrics = holdout_payload.get('training_window_metrics') or profile_window or {}
    if not isinstance(training_metrics, dict):
        training_metrics = {}
    experiment_meta = {
        'training_cutoff': (profile or {}).get('training_cutoff') if isinstance(profile, dict) else None,
        'holdout_start': holdout_payload.get('holdout_start'),
        'holdout_end': holdout_payload.get('holdout_end'),
        'holdout_bars': holdout_payload.get('holdout_bars'),
        'training_recent_6m': training_metrics.get('recent_6m'),
        'training_full_pre_cutoff': training_metrics.get('full_pre_cutoff'),
        'holdout_metrics': holdout_metrics,
    } if (profile or holdout_payload) else None
    if (not win) and isinstance(holdout_metrics, dict):
        win = {
            'total_return': holdout_metrics.get('final_nav', 1.0) - 1.0,
            'max_drawdown': holdout_metrics.get('max_drawdown', 0.0),
        }
    return {
        **core,
        'status': status,
        'passes_rule14': status == 'production',
        'live_position': round(live_position, 4),
        'live_exposure_delta': live_delta,
        'live_rebalance_action': live_action,
        'live_rebalance_label': _REBALANCE_ACTION_LABELS.get(live_action, live_action),
        'action_rationale': action_rationale,
        'view_bias': view_bias,
        'risks': risks,
        'opportunities': opportunities,
        'risk_signals_as_of': risk_signals_as_of,
        'risk_signals_generated_at': risk_signals_generated_at,
        # 看多强度分（0‒100）：100=极度看多，50=中性，0=极度看空
        # A股/SP500 = strength_score×80 + signal_action 修正；macro_v32 = sigmoid(ContScore×2)×100
        'exec_basis': '信号基于 T 日收盘生成；T+1 交易日按开盘价执行；当日盈亏按 T+1 收盘价标记。',
        'profile_recent_6m': {
            'strategy_total_return_pct': round(float(win.get('total_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'total_return' in win else None,
            'etf_total_return_pct': round(float(win.get('etf_total_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'etf_total_return' in win else None,
            'excess_return_pct': round(float(win.get('excess_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'excess_return' in win else None,
            'max_drawdown_pct': round(float(win.get('max_drawdown', 0.0)) * 100, 2) if isinstance(win, dict) and 'max_drawdown' in win else None,
            'etf_max_drawdown_pct': round(float(win.get('etf_max_drawdown', 0.0)) * 100, 2) if isinstance(win, dict) and 'etf_max_drawdown' in win else None,
        } if isinstance(win, dict) and win else None,
        'experiment_meta': experiment_meta,
    }


# ═════════════════════════════════════════════════════════════════
# 指数 vs ETF 对齐检查 / 后台更新流程
# ═════════════════════════════════════════════════════════════════
def _check_a_share_index_etf_alignment():
    mismatches = []
    for index_id in A_SHARE_INDEX_IDS:
        try:
            idx_df = get_index_daily(index_id)
            etf_df = get_timing_etf_daily(index_id)
        except Exception as exc:
            mismatches.append({
                'index_id': index_id,
                'error': repr(exc),
            })
            continue
        idx_max = pd.to_datetime(idx_df['date']).max() if idx_df is not None and len(idx_df) > 0 else None
        etf_max = pd.to_datetime(etf_df['date']).max() if etf_df is not None and len(etf_df) > 0 else None
        if idx_max is None or etf_max is None:
            continue
        # 指数是 A 股交易日历真值源。
        # - idx_max < etf_max：指数被旧数据污染（危险，CLAUDE.md 明确必须报）。
        # - etf_max < idx_max 且滞后 > STALE_TOLERANCE_DAYS：ETF 长期未刷新（典型场景：
        #   非 qfq 已更新但 qfq 停留数周），会让择时信号/结算价停在旧日期，必须报告；
        #   同日/次日的 1 天小滞后属正常刷新节奏，不报，避免误报。
        STALE_TOLERANCE_DAYS = 2
        if idx_max < etf_max:
            mismatches.append({
                'index_id': index_id,
                'direction': 'index_behind',
                'index_max_date': idx_max.strftime('%Y-%m-%d'),
                'etf_max_date': etf_max.strftime('%Y-%m-%d'),
            })
        elif (idx_max - etf_max).days > STALE_TOLERANCE_DAYS:
            mismatches.append({
                'index_id': index_id,
                'direction': 'etf_behind',
                'index_max_date': idx_max.strftime('%Y-%m-%d'),
                'etf_max_date': etf_max.strftime('%Y-%m-%d'),
                'lag_days': (idx_max - etf_max).days,
            })
    return mismatches


def _latest_a_share_trading_day():
    """从 a_share_calendar_daily.csv 读出"最近一个**已完结**的 A 股交易日"。

    口径：strictly < today。今天即使是交易日也不算（盘中或刚收盘的数据不稳定）。
    所有 freshness check 都用 A 股交易日历作权威：A 股月度数据 / A 股指数 / A 股 ETF /
    跨境 ETF（159941、513500，均按 A 股交易日结算）都用同一份日历。

    Refetch 策略：先读 cache；如果 cache 的 max_date 仍 >= 昨天，就用 cache（避免每次
    按钮 click 都打 8s 网络）；只有 cache 落后 >=1 天时才 force_refetch。这能保证：
      - 缓存当天刷过一次后，后续 check 全部走本地，<10ms 返回
      - 缓存停在很久之前的情况下，自动刷一次到最新
    """
    today = pd.Timestamp(_datetime.now().date())
    cal = None
    try:
        cal = get_a_share_trading_calendar()
    except Exception as exc:
        print(f'[freshness] 读取 cached 日历失败：{exc}', file=sys.stderr)

    def _max_dt(df):
        if df is None or len(df) == 0 or 'date' not in df.columns:
            return None
        s = pd.to_datetime(df['date'], errors='coerce').dropna()
        return s.max() if len(s) else None

    cached_max = _max_dt(cal)
    # cache 比"昨天"还旧才刷网络。昨天 = today - 1 day
    yesterday = today - pd.Timedelta(days=1)
    if cached_max is None or cached_max < yesterday:
        try:
            cal = get_a_share_trading_calendar(force_refetch=True)
        except Exception as exc:
            print(f'[freshness] 强制刷新日历失败，退回 cached：{exc}', file=sys.stderr)
            # 退回 cached（cal 已加载或仍是 None）

    if cal is None or len(cal) == 0 or 'date' not in cal.columns:
        return None
    dates = pd.to_datetime(cal['date'], errors='coerce').dropna()
    if len(dates) == 0:
        return None
    # strictly <：今天的 bar 不算"已完结"，避免盘中报"最新交易日=今天"导致误判
    eligible = dates[dates < today]
    if len(eligible) == 0:
        return None
    return eligible.max().normalize()


def _check_stock_data_freshness():
    """只读判定 stock_data.csv 是否需要 supplement。返回 dict，不写盘。

    判定逻辑：取 csv 内最大「月份」字段（YYYY-MM），与最近一个 A 股交易日所在月做比较。
    若 csv 最新月份 < 最新交易日所在月 → 需要更新；否则视作已最新（同月内的 daily 增量
    由 _supplement_csv_incremental 处理，但只要 csv 已包含当月行就不强求每个交易日都触发）。
    """
    result = {
        'needs_update': False,
        'current_local_date': None,
        'latest_market_date': None,
        'reason': '',
        'checked_at': _datetime.now().isoformat(timespec='seconds'),
    }
    csv_path = os.path.join(_PROJECT_ROOT, 'stock_data.csv')
    if not os.path.exists(csv_path):
        result['needs_update'] = True
        result['reason'] = 'stock_data.csv 不存在，首次需要初始化'
        return result
    latest_trading = _latest_a_share_trading_day()
    if latest_trading is None:
        result['reason'] = '交易日历不可用，无法判定，建议手动刷新一次'
        result['needs_update'] = True
        return result
    result['latest_market_date'] = latest_trading.strftime('%Y-%m-%d')

    # 优先用 in-memory DATA_DF（启动 eager_load 已加载，~ms 级），否则回退读 CSV（~8s）
    max_date = None
    if DATA_DF is not None and '交易日期' in DATA_DF.columns and len(DATA_DF) > 0:
        try:
            max_date = pd.to_datetime(DATA_DF['交易日期'], errors='coerce').dropna().max()
        except Exception as exc:
            print(f'[freshness] DATA_DF 读最大日期失败，回退 CSV：{exc}', file=sys.stderr)
            max_date = None
    if max_date is None or pd.isna(max_date):
        try:
            df_tail = pd.read_csv(csv_path, encoding='gbk', usecols=['交易日期'], low_memory=False)
            max_date = pd.to_datetime(df_tail['交易日期'].astype(str), errors='coerce').dropna().max()
        except Exception as exc:
            result['reason'] = f'读取 stock_data.csv 交易日期列失败: {exc}'
            result['needs_update'] = True
            return result
    if pd.isna(max_date):
        result['reason'] = 'stock_data.csv 中无可解析的交易日期'
        result['needs_update'] = True
        return result

    # 日级比较：stock_data.csv 虽然是月度聚合面板，但 _supplement_csv_incremental
    # 会重抓当月所有 daily 再聚合，所以"本地最大交易日 < 市场最新交易日"就需要刷一次。
    local_ts = pd.Timestamp(max_date)
    local_ymd = local_ts.strftime('%Y-%m-%d')
    market_ymd = latest_trading.strftime('%Y-%m-%d')
    result['current_local_date'] = local_ymd
    if local_ts.normalize() < latest_trading:
        result['needs_update'] = True
        result['reason'] = f'本地 stock_data.csv 最新交易日 {local_ymd} 落后于最新交易日 {market_ymd}'
    else:
        result['reason'] = f'本地 stock_data.csv 已更新至 {local_ymd}（不早于最新交易日 {market_ymd}）'
    return result


def _check_index_etf_freshness():
    """只读判定所有指数 / ETF 日线缓存是否落后于最新 A 股交易日。

    覆盖：5 个择时 ETF（csi1000 / chinext / star50 / nasdaq / sp500，全部 A 股上市）
    + 3 个 A 股指数日线（csi1000 / chinext / star50）。任一缓存 max_date < 最新交易日 →
    needs_update=true。
    """
    result = {
        'needs_update': False,
        'current_local_date': None,
        'latest_market_date': None,
        'reason': '',
        'lagging_items': [],
        'checked_at': _datetime.now().isoformat(timespec='seconds'),
    }
    latest_trading = _latest_a_share_trading_day()
    if latest_trading is None:
        result['reason'] = '交易日历不可用，无法判定，建议手动刷新一次'
        result['needs_update'] = True
        return result
    result['latest_market_date'] = latest_trading.strftime('%Y-%m-%d')

    cache_max_dates = []
    lagging = []

    # A 股指数日线
    for index_id in A_SHARE_INDEX_IDS:
        try:
            df = get_index_daily(index_id)
        except Exception as exc:
            lagging.append({'kind': 'index', 'id': index_id, 'reason': f'读取失败: {exc}'})
            continue
        if df is None or len(df) == 0:
            lagging.append({'kind': 'index', 'id': index_id, 'reason': '日线缓存为空'})
            continue
        max_date = pd.to_datetime(df['date'], errors='coerce').dropna().max()
        if pd.isna(max_date):
            lagging.append({'kind': 'index', 'id': index_id, 'reason': '无可解析日期'})
            continue
        cache_max_dates.append(max_date)
        if max_date < latest_trading:
            lagging.append({
                'kind': 'index', 'id': index_id,
                'local_max_date': max_date.strftime('%Y-%m-%d'),
            })

    # 所有 ETF 日线（A 股 + 跨境 QDII 都走 A 股日历）
    for etf_id in TIMING_ETF_CONFIGS.keys():
        try:
            df = get_timing_etf_daily(etf_id)
        except Exception as exc:
            lagging.append({'kind': 'etf', 'id': etf_id, 'reason': f'读取失败: {exc}'})
            continue
        if df is None or len(df) == 0:
            lagging.append({'kind': 'etf', 'id': etf_id, 'reason': '日线缓存为空'})
            continue
        max_date = pd.to_datetime(df['date'], errors='coerce').dropna().max()
        if pd.isna(max_date):
            lagging.append({'kind': 'etf', 'id': etf_id, 'reason': '无可解析日期'})
            continue
        cache_max_dates.append(max_date)
        if max_date < latest_trading:
            lagging.append({
                'kind': 'etf', 'id': etf_id,
                'local_max_date': max_date.strftime('%Y-%m-%d'),
            })

    if cache_max_dates:
        result['current_local_date'] = max(cache_max_dates).strftime('%Y-%m-%d')

    if lagging:
        result['needs_update'] = True
        result['lagging_items'] = lagging
        names = ', '.join(f"{x['kind']}:{x['id']}" for x in lagging)
        result['reason'] = f'以下缓存落后于最新交易日 {latest_trading.strftime("%Y-%m-%d")}：{names}'
    else:
        result['reason'] = f'所有指数/ETF 缓存均已更新至 {latest_trading.strftime("%Y-%m-%d")}'
    return result


def _check_aux_data_freshness():
    """检查 FRED 宏观 / A股估值/情绪 / risk_signals 是否落后于最新 A 股交易日。

    覆盖 macro_v32_timing 用到的 FRED 因子（VIX/FedFunds/UST10Y 等）+ live 风险面板
    用的 A 股估值/情绪因子（PE-TTM/CN10Y/SSE 成交融资）+ 二者合成的 risk_signals.json。
    任一文件 max_date < 最新交易日（容忍 1 天，FRED 美股发布通常滞后 1 个工作日）→
    needs_update=true。
    """
    result = {
        'needs_update': False,
        'current_local_date': None,
        'latest_market_date': None,
        'reason': '',
        'lagging_items': [],
        'checked_at': _datetime.now().isoformat(timespec='seconds'),
    }
    latest_trading = _latest_a_share_trading_day()
    if latest_trading is None:
        result['reason'] = '交易日历不可用'
        result['needs_update'] = True
        return result
    result['latest_market_date'] = latest_trading.strftime('%Y-%m-%d')

    # FRED 数据：data/fred_VIX.csv 是高频典型代表（每日发布）。允许滞后 1 天。
    # A 股估值/情绪：data/a_share_macro/pe_ttm.csv 是每日。允许 0 天滞后。
    repo_root = os.path.dirname(_PROJECT_ROOT)
    targets = [
        ('fred_vix', os.path.join(repo_root, 'data', 'fred_VIX.csv'), 1),
        ('fred_ust10y', os.path.join(repo_root, 'data', 'fred_Treasury10Y.csv'), 1),
        ('fred_curve_10y2y', os.path.join(repo_root, 'data', 'fred_YieldCurve_10Y2Y.csv'), 1),
        ('fred_hy_spread', os.path.join(repo_root, 'data', 'fred_HighYieldSpread.csv'), 1),
        ('a_share_pe_ttm', os.path.join(repo_root, 'data', 'a_share_macro', 'pe_ttm.csv'), 0),
        ('a_share_cn10y', os.path.join(repo_root, 'data', 'a_share_macro', 'cn10y.csv'), 0),
        ('a_share_sse_daily', os.path.join(repo_root, 'data', 'a_share_macro', 'sse_daily.csv'), 0),
        ('risk_signals', os.path.join(repo_root, 'strategy', 'risk_signals.json'), 1),
    ]
    cache_max_dates = []
    lagging = []
    for label, path, tolerance_days in targets:
        if not os.path.exists(path):
            lagging.append({'id': label, 'reason': '文件不存在'})
            continue
        try:
            if path.endswith('.json'):
                import json as _json
                with open(path, 'r') as f:
                    j = _json.load(f)
                as_of = j.get('as_of') or j.get('generated_at', '').split('T')[0]
                max_date = pd.to_datetime(as_of, errors='coerce') if as_of else None
            else:
                # CSV 第一列是日期（date / 日期）
                df = pd.read_csv(path, nrows=None)
                date_col = next((c for c in df.columns if c.lower() in {'date', '日期', '交易日期'}), df.columns[0])
                max_date = pd.to_datetime(df[date_col], errors='coerce').dropna().max()
        except Exception as exc:
            lagging.append({'id': label, 'reason': f'读取失败: {exc}'})
            continue
        if pd.isna(max_date):
            lagging.append({'id': label, 'reason': '无可解析日期'})
            continue
        cache_max_dates.append(max_date)
        # 容忍带：cache 比"最新交易日 - tolerance" 还旧才算落后
        threshold = latest_trading - pd.Timedelta(days=tolerance_days)
        if max_date < threshold:
            lagging.append({
                'id': label,
                'local_max_date': max_date.strftime('%Y-%m-%d'),
                'tolerance_days': tolerance_days,
            })

    if cache_max_dates:
        result['current_local_date'] = max(cache_max_dates).strftime('%Y-%m-%d')

    if lagging:
        result['needs_update'] = True
        result['lagging_items'] = lagging
        names = ', '.join(x['id'] for x in lagging)
        result['reason'] = f'以下辅助数据落后于最新交易日 {latest_trading.strftime("%Y-%m-%d")}：{names}'
    else:
        result['reason'] = f'辅助数据（FRED / A 股估值 / risk_signals）均不落后于 {latest_trading.strftime("%Y-%m-%d")}'
    return result


def _run_aux_data_update():
    """串跑 3 个辅助脚本：FRED 宏观 → A股估值/情绪 → 汇总 risk_signals.json。

    Web 进程只能读这些文件，刷新需要外部脚本（涉及 akshare/yfinance/pandas_datareader
    等可能有 SSL/网络阻塞的库）。subprocess.run 跑出来，每段 ~30s-2min。
    """
    import subprocess
    status = _AUX_UPDATE_STATUS
    status['running'] = True
    status['error'] = None
    status['stage'] = 'starting'
    status['progress_pct'] = 0
    status['message'] = '准备启动辅助数据刷新...'

    repo_root = os.path.dirname(_PROJECT_ROOT)
    scripts_dir = os.path.join(repo_root, 'scripts')
    py = sys.executable
    pipeline = [
        ('fred_macro', os.path.join(scripts_dir, 'download_macro_data.py'),
         'FRED 宏观（VIX / Fed / 10Y / CPI / …）', 10, 45),
        ('a_share_macro', os.path.join(scripts_dir, 'fetch_a_share_macro.py'),
         'A 股估值/情绪（PE-TTM / CN10Y / 成交融资）', 45, 80),
        ('risk_signals', os.path.join(scripts_dir, 'build_risk_signals.py'),
         '汇总 risk_signals.json', 80, 100),
    ]

    try:
        for stage_id, script_path, label, pct_from, pct_to in pipeline:
            status['stage'] = stage_id
            status['progress_pct'] = pct_from
            status['message'] = f'正在跑 {label}...'
            print(f'[aux_update] {stage_id} → {script_path}', file=sys.stderr)
            if not os.path.exists(script_path):
                raise RuntimeError(f'脚本不存在: {script_path}')
            # 脚本里有 `from utils.atomic_io import ...`，但 utils 模块在 stock_trade_demo
            # 下，脚本本身不会主动 sys.path.insert。这里给 subprocess 显式注入 PYTHONPATH，
            # 让 stock_trade_demo 在 import path 上，否则会报 ModuleNotFoundError: 'utils'。
            sub_env = os.environ.copy()
            existing_pp = sub_env.get('PYTHONPATH', '')
            sub_env['PYTHONPATH'] = _PROJECT_ROOT + (os.pathsep + existing_pp if existing_pp else '')
            cmd = [py, script_path]
            if stage_id == 'fred_macro':
                # Yahoo Finance `data/yf_*.csv` 仅研究用途，不参与 live 风险面板或生产策略；
                # 辅助数据流水线只需要 FRED 宏观，跳过 yfinance 可显著降低超时概率。
                cmd.append('--skip-yf')
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=sub_env,
                timeout=900,  # FRED 上游偶发慢响应，留 15 分钟缓冲
            )
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or '')[-500:]
                raise RuntimeError(f'{stage_id} 失败 (rc={proc.returncode}): {tail}')
            status['progress_pct'] = pct_to
            print(f'[aux_update] {stage_id} OK', file=sys.stderr)

        status['stage'] = 'done'
        status['progress_pct'] = 100
        status['message'] = '辅助数据全部刷新完成'
        print('[aux_update] 全部完成', file=sys.stderr)
    except subprocess.TimeoutExpired as exc:
        status['stage'] = 'error'
        status['message'] = f'超时（>10 min）: {exc.cmd}'
        status['error'] = str(exc)
        print(f'[aux_update ERROR] timeout: {exc}', file=sys.stderr)
    except Exception as exc:
        status['stage'] = 'error'
        status['message'] = f'更新失败: {exc}'
        status['error'] = str(exc)
        print(f'[aux_update ERROR] {exc}', file=sys.stderr)
    finally:
        status['running'] = False


def _run_factor_update():
    """跑衍生因子重算脚本（目前只有 compute_sector_weekly_heat）。

    放在「更新数据」按钮的最后一段，因为：
      - 它读 stock_data.parquet 的 `下周期每天涨跌幅`
      - stock 阶段 supplement 会回填上月的这一列
      - 因此 factor 必须在 stock 之后跑，不然会少算最近一个月的周热度
    """
    import subprocess
    status = _FACTOR_UPDATE_STATUS
    status['running'] = True
    status['error'] = None
    status['stage'] = 'starting'
    status['progress_pct'] = 0
    status['message'] = '准备启动衍生因子重算...'

    repo_root = os.path.dirname(_PROJECT_ROOT)
    scripts_dir = os.path.join(repo_root, 'scripts')
    py = sys.executable
    pipeline = [
        ('sector_weekly_heat', os.path.join(scripts_dir, 'compute_sector_weekly_heat.py'),
         '行业周度热度（含当周 partial）', 0, 100),
    ]

    try:
        for stage_id, script_path, label, pct_from, pct_to in pipeline:
            status['stage'] = stage_id
            status['progress_pct'] = pct_from
            status['message'] = f'正在跑 {label}...'
            print(f'[factor_update] {stage_id} → {script_path}', file=sys.stderr)
            if not os.path.exists(script_path):
                raise RuntimeError(f'脚本不存在: {script_path}')
            sub_env = os.environ.copy()
            existing_pp = sub_env.get('PYTHONPATH', '')
            sub_env['PYTHONPATH'] = _PROJECT_ROOT + (os.pathsep + existing_pp if existing_pp else '')
            proc = subprocess.run(
                [py, script_path],
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=sub_env,
                timeout=600,
            )
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or '')[-500:]
                raise RuntimeError(f'{stage_id} 失败 (rc={proc.returncode}): {tail}')
            status['progress_pct'] = pct_to
            print(f'[factor_update] {stage_id} OK', file=sys.stderr)

        # 重算完显式清空 sector_heat 的 mtime 缓存，确保下次 /api/sector_heat 读新文件
        try:
            _SECTOR_HEAT_CACHE['data'] = None
            _SECTOR_HEAT_CACHE['mtime'] = 0
        except Exception:
            pass

        status['stage'] = 'done'
        status['progress_pct'] = 100
        status['message'] = '衍生因子全部刷新完成'
        print('[factor_update] 全部完成', file=sys.stderr)
    except subprocess.TimeoutExpired as exc:
        status['stage'] = 'error'
        status['message'] = f'超时（>10 min）: {exc.cmd}'
        status['error'] = str(exc)
        print(f'[factor_update ERROR] timeout: {exc}', file=sys.stderr)
    except Exception as exc:
        status['stage'] = 'error'
        status['message'] = f'更新失败: {exc}'
        status['error'] = str(exc)
        print(f'[factor_update ERROR] {exc}', file=sys.stderr)
    finally:
        status['running'] = False


def _check_factor_data_freshness():
    """检查 sector_weekly_heat.csv 是否落后于 stock_data 的最新已完结月。

    sector_weekly_heat 的 `year_month` 表示「下周期」（被预测/被切周的月份）。
    stock_data 最新行的 `交易日期` 是月度 snapshot，其下月就是当前在切周的月份。
    所以最新可期望的 `year_month` = stock_data_max_month + 1（用 Period 算）。
    如果 sector_weekly_heat 的 max year_month 小于这个，就是落后。
    """
    result = {
        'needs_update': False,
        'current_local_date': None,
        'latest_market_date': None,
        'reason': '',
        'lagging_items': [],
        'checked_at': _datetime.now().isoformat(timespec='seconds'),
    }
    heat_path = _SECTOR_HEAT_FILE
    if not os.path.exists(heat_path):
        result['needs_update'] = True
        result['reason'] = f'{heat_path} 不存在，需首次生成'
        result['lagging_items'] = [{'id': 'sector_weekly_heat', 'reason': '文件不存在'}]
        return result

    try:
        df_heat = pd.read_csv(heat_path, encoding='utf-8-sig', usecols=['year_month'])
        heat_max_ym = pd.Period(df_heat['year_month'].max(), freq='M')
    except Exception as exc:
        result['needs_update'] = True
        result['reason'] = f'读取 {heat_path} 失败: {exc}'
        result['lagging_items'] = [{'id': 'sector_weekly_heat', 'reason': str(exc)}]
        return result

    result['current_local_date'] = str(heat_max_ym)

    expected_ym = None
    try:
        from backtest import load_data as _load_data
        df_stock = _load_data()
        if df_stock is not None and len(df_stock):
            stock_max = pd.to_datetime(df_stock['交易日期']).max()
            # sector_weekly_heat 的 year_month = (snapshot 月份) + 1（因为读的是 "下周期每天涨跌幅"）。
            # 但最新 snapshot 月那行的下周期日线永远是空的（[]，要等下个月数据），
            # 所以 CSV 实际能产出的最大 year_month = (max_month - 1) + 1 = max_month。
            expected_ym = pd.Period(stock_max, freq='M')
    except Exception as exc:
        print(f'[factor_check] 无法读 stock_data 推断 expected_ym: {exc}', file=sys.stderr)

    if expected_ym is None:
        latest_trading = _latest_a_share_trading_day()
        if latest_trading is not None:
            expected_ym = pd.Period(latest_trading, freq='M')
            result['latest_market_date'] = latest_trading.strftime('%Y-%m-%d')
    else:
        result['latest_market_date'] = str(expected_ym)

    if expected_ym is not None and heat_max_ym < expected_ym:
        result['needs_update'] = True
        result['lagging_items'] = [{
            'id': 'sector_weekly_heat',
            'local_max_ym': str(heat_max_ym),
            'expected_ym': str(expected_ym),
        }]
        result['reason'] = f'sector_weekly_heat 最新月 {heat_max_ym} 落后于期望月 {expected_ym}'
    else:
        result['reason'] = f'sector_weekly_heat 已更新至 {heat_max_ym}'
    return result


def _schedule_self_restart():
    """异步调度自重启：先落盘 cache，再 spawn detached subprocess + os._exit 当前进程。

    为什么不用 os.execv：execv 会保留原进程的 socket FD（Werkzeug 的 listening socket
    没有 FD_CLOEXEC），新进程映像里 app.run() 重新 bind 8080 时会拿到 EADDRINUSE。
    实测一次重启后旧 web 已退出 + 新 web bind 失败 = 服务彻底没了。

    新方案：spawn 一个 detached shell（new session，独立于父 stdio），让它 sleep 2 秒
    等父进程释放 socket，再 exec python web_app.py；父进程立刻 os._exit(0) 让 socket
    立即被内核回收（SO_REUSEADDR 是 Werkzeug 默认，2s 后绝对可以 bind）。
    """
    import subprocess

    def _do_restart():
        try:
            _save_disk_cache()
            print('[restart] disk cache flushed', file=sys.stderr)
        except Exception as exc:
            print(f'[restart] save cache failed: {exc}', file=sys.stderr)
        script_path = os.path.join(_PROJECT_ROOT, 'web_app.py')
        log_path = '/tmp/quant_web_restart_auto.log'
        # /bin/sh 里 exec 把 shell 替换成 python，避免多留一个 shell 进程
        shell_cmd = f'sleep 2 && exec {sys.executable} {script_path}'
        try:
            log_fp = open(log_path, 'a')
            subprocess.Popen(
                ['/bin/sh', '-c', shell_cmd],
                cwd=_PROJECT_ROOT,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # detach session → 父进程死了也不会带走
            )
            print(f'[restart] detached shell scheduled (will exec {script_path} after 2s)', file=sys.stderr)
        except Exception as exc:
            print(f'[restart] failed to spawn restart shell: {exc}', file=sys.stderr)
            return
        # 给 Popen 一点时间 fork 出来，然后硬退出释放 socket
        time.sleep(0.3)
        print('[restart] os._exit(0) → releasing 8080 socket', file=sys.stderr)
        os._exit(0)

    timer = threading.Timer(1.5, _do_restart)
    timer.daemon = True
    timer.start()


def _retry_network(label, fn, status=None, attempts=3, backoff=(2, 5, 10)):
    """对带网络副作用的 fetch 函数做有限重试，只对瞬断类异常重试。

    transient（重试）：ConnectionError / Timeout / RemoteDisconnected / OSError / IncompleteRead
    permanent（直抛）：SchemaError / 数据格式问题 / 业务异常 —— 立刻 fail，重试也没用
    """
    transient_keywords = (
        'ConnectionError', 'Timeout', 'TimeoutError', 'RemoteDisconnected',
        'IncompleteRead', 'ProtocolError', 'ReadTimeout', 'ConnectTimeout',
        'ConnectionAborted', 'ConnectionReset', 'OSError', 'ChunkedEncodingError',
    )
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            name = type(exc).__name__
            msg = repr(exc)
            is_transient = (name in transient_keywords
                            or any(k in msg for k in transient_keywords))
            last_err = exc
            if not is_transient or i == attempts - 1:
                if not is_transient:
                    print(f'[retry] {label} 非瞬断异常，不重试：{name}: {exc}', file=sys.stderr)
                else:
                    print(f'[retry] {label} 已重试 {attempts} 次仍失败：{name}: {exc}', file=sys.stderr)
                raise
            wait = backoff[min(i, len(backoff) - 1)]
            print(f'[retry] {label} 第 {i+1}/{attempts} 次失败 ({name})，{wait}s 后重试...', file=sys.stderr)
            if status is not None:
                orig_msg = status.get('message', '')
                status['message'] = f'{orig_msg} (上游瞬断，{wait}s 后重试 {i+2}/{attempts}...)'
            time.sleep(wait)
    if last_err:
        raise last_err


def _run_index_data_update():
    global INDEX_RETURNS, INDEX_RETURNS_MAP, TIMING_PANEL, TIMING_CACHE, US_TIMING_PANEL, US_TIMING_CACHE
    status = _INDEX_UPDATE_STATUS
    status['stage'] = 'running'
    status['progress'] = 0
    status['warning'] = None
    status['details'] = None

    index_ids = list(INDEX_CONFIGS.keys())
    total = len(index_ids)

    try:
        for i, index_id in enumerate(index_ids):
            cfg = INDEX_CONFIGS[index_id]
            name = cfg.get('name', index_id)

            status['message'] = f'正在拉取日K：{name}...'
            status['progress'] = int(i / total * 45)
            _retry_network(f'get_index_daily({index_id})',
                           lambda: get_index_daily(index_id, force_refetch=True),
                           status=status)

            status['message'] = f'正在计算月度收益：{name}...'
            status['progress'] = int((i + 0.5) / total * 45)
            series = _retry_network(f'get_index_returns({index_id})',
                                    lambda: get_index_returns(index_id, force_refetch=True),
                                    status=status)
            INDEX_RETURNS_MAP[index_id] = series

        status['message'] = '正在刷新择时 ETF 日线缓存...'
        status['progress'] = 55
        _retry_network('refresh_all_timing_etf_daily',
                       lambda: refresh_all_timing_etf_daily(),
                       status=status)

        INDEX_RETURNS = INDEX_RETURNS_MAP.get('csi1000')

        # ── 盘中对齐：若 ETF 拿到了今天（intraday bar），但 index 当天 bar 还没出，
        # 把 ETF cache 末尾的今天那行砍掉，让两者对齐到 index 的最新已完结日。
        # 这种 case 在盘中 13:00–15:00 之间常见：ETF 是连续撮合即时报价，
        # 指数日 K 上游延迟到收盘后才出，导致 ETF 抢跑 1 天。
        today = pd.Timestamp(_datetime.now().date())
        for index_id in A_SHARE_INDEX_IDS:
            try:
                idx_df = get_index_daily(index_id)
                etf_df = get_timing_etf_daily(index_id)
            except Exception:
                continue
            idx_max = pd.to_datetime(idx_df['date']).max() if idx_df is not None and len(idx_df) else None
            etf_max = pd.to_datetime(etf_df['date']).max() if etf_df is not None and len(etf_df) else None
            if (idx_max is not None and etf_max is not None
                    and etf_max > idx_max and etf_max.normalize() >= today.normalize()):
                # ETF 抢跑了今天（intraday）；砍掉 ETF 中 > idx_max 的所有行。
                trimmed = etf_df[pd.to_datetime(etf_df['date']) <= idx_max].copy()
                if len(trimmed) >= len(etf_df) - 5 and len(trimmed) > 0:
                    # 写回磁盘（用 atomic_write_csv 避免污染）
                    from index_data import (
                        TIMING_ETF_CONFIGS as _ETF_CFG,
                        _etf_daily_cache_path,
                        _stringify_date_column,
                    )
                    from utils.atomic_io import atomic_write_csv as _aw
                    adjust = _ETF_CFG[index_id].get('dividend_adjust_method', 'qfq')
                    path = _etf_daily_cache_path(index_id, adjust=adjust)
                    _aw(path, _stringify_date_column(trimmed), index=False,
                        produced_by=f'index_update:intraday_trim_etf_to_index:{index_id}')
                    print(f'[index_update] ETF {index_id} 砍掉今天 intraday {etf_max.date()} → 对齐 index {idx_max.date()}',
                          file=sys.stderr)

        mismatches = _check_a_share_index_etf_alignment()
        if mismatches:
            status['warning'] = 'A股指数日线与对应ETF日线最新日期不一致'
            status['details'] = mismatches
            raise RuntimeError(f'A股指数/ETF日线不同步: {mismatches}')

        status['message'] = '正在重建指数日线面板...'
        status['progress'] = 70
        TIMING_PANEL = None
        ensure_timing_panel_loaded()

        # TIMING_CACHE 必须强制重算，不能走 init_timing_cache()：
        # init_timing_cache 里有 `if TIMING_CACHE: return`，而它会先 _load_disk_cache() 把
        # 旧版重新装进来，然后直接 return——这会让 timing cache 停在旧的日期。
        status['message'] = '正在重建 A 股择时回测缓存...'
        status['progress'] = 80
        TIMING_CACHE.clear()
        for sid, cls in TIMING_STRATEGY_MAP.items():
            try:
                strategy = build_timing_strategy(sid)
                signal_df = strategy.run(TIMING_PANEL.copy())
                if sid == 'csi1000_timing':
                    CSI1000_SIGNAL_SERIES = pd.Series(
                        pd.to_numeric(signal_df['target_exposure'], errors='coerce').fillna(0.0).values,
                        index=pd.to_datetime(signal_df['交易日期']),
                    ).sort_index()
                result = run_timing_backtest(
                    signal_df, strategy,
                    benchmark_returns=INDEX_RETURNS_MAP.get(strategy.get_index_id()),
                )
                TIMING_CACHE[sid] = result
                nav = float(result['累积净值'].iloc[-1]) if len(result) else float('nan')
                last = result['交易日期'].iloc[-1] if len(result) else 'N/A'
                print(f'[index_update] {sid} A股 timing rebuilt, nav={nav:.4f}, last={last}',
                      file=sys.stderr)
            except Exception as cn_err:
                print(f'[index_update] {sid} A股 timing rebuild failed: {cn_err}', file=sys.stderr)

        # ── 美股择时缓存重建（用最新 ETF 日线）──
        # US_TIMING_CACHE 之前只在 web 启动时从 pkl 读入；ETF 日线更新后必须重算，
        # 否则 /api/us_timing/backtest 的 interval_windows.end 会停在上次 build 的日期。
        status['message'] = '正在重建美股择时回测缓存（纳指 / 标普500）...'
        status['progress'] = 88
        try:
            US_TIMING_PANEL = None
            US_TIMING_CACHE.clear()
            _EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()
            ensure_us_timing_panel_loaded(force_reload=True)
            us_cache_dir = os.path.join(_CACHE_DIR, 'us_timing')
            os.makedirs(us_cache_dir, exist_ok=True)
            for sid, cls in US_TIMING_STRATEGY_MAP.items():
                try:
                    strategy = build_us_timing_strategy(sid)
                    if US_TIMING_PANEL is None or len(US_TIMING_PANEL) == 0:
                        print(f'[index_update] {sid}: US_TIMING_PANEL 为空，跳过', file=sys.stderr)
                        continue
                    bm_id = strategy.get_index_id()
                    bm_returns = INDEX_RETURNS_MAP.get(bm_id)
                    result = run_timing_backtest(strategy.run(US_TIMING_PANEL.copy()), strategy,
                                                benchmark_returns=bm_returns)
                    US_TIMING_CACHE[sid] = result
                    pkl_path = os.path.join(us_cache_dir, f'{sid}.pkl')
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(result, f, protocol=4)
                    nav = float(result['累积净值'].iloc[-1]) if len(result) else float('nan')
                    last_date = result['交易日期'].iloc[-1] if len(result) else 'N/A'
                    print(f'[index_update] {sid} US timing rebuilt, nav={nav:.4f}, last={last_date}',
                          file=sys.stderr)
                except Exception as us_err:
                    print(f'[index_update] {sid} US timing rebuild failed: {us_err}', file=sys.stderr)
        except Exception as us_panel_err:
            print(f'[index_update] 美股 timing panel 加载失败，跳过 US timing rebuild: {us_panel_err}',
                  file=sys.stderr)

        _save_disk_cache()

        status['stage'] = 'done'
        status['message'] = '指数与ETF数据刷新完成（含美股择时缓存）'
        status['progress'] = 100
        print('[index_update] 指数与ETF数据更新完成（含美股 timing cache）', file=sys.stderr)

    except Exception as e:
        status['stage'] = 'error'
        err_name = type(e).__name__
        # 给上游瞬断类异常一个友好的、可操作的中文提示
        transient_hint = any(k in repr(e) for k in (
            'RemoteDisconnected', 'ConnectionAborted', 'ConnectionReset',
            'ProtocolError', 'Timeout', 'ConnectionError', 'IncompleteRead',
        ))
        if transient_hint:
            status['message'] = f'上游数据源（akshare/新浪/东财）瞬断，已重试 3 次仍失败。请稍后再试。原始：{err_name}'
        else:
            status['message'] = f'更新失败 ({err_name}): {e}'
        status['progress'] = 0
        if status.get('details') is None:
            status['details'] = {'error': repr(e), 'error_class': err_name}
        print(f'[index_update ERROR] {e}', file=sys.stderr)


def _run_data_update():
    global DATA_DF, INDEX_RETURNS, INDEX_RETURNS_MAP, TIMING_PANEL
    status = _UPDATE_DATA_STATUS
    status['running'] = True
    status['error'] = None

    try:
        now = _datetime.now()
        target_year = now.year
        target_month = now.month
        target_prefix = f"{target_year}-{target_month:02d}"
        prev_month = target_month - 1
        prev_year = target_year
        if prev_month == 0:
            prev_month = 12
            prev_year -= 1
        prev_prefix = f"{prev_year}-{prev_month:02d}"

        csv_path = os.path.join(_PROJECT_ROOT, 'stock_data.csv')
        repo_root = os.path.dirname(_PROJECT_ROOT)
        cache_dir = os.path.join(repo_root, '.cache')

        status['stage'] = 'fetching'
        status['message'] = '正在增量获取最新行情数据（首次约3-5分钟，之后只补差值）...'
        status['progress_pct'] = 10
        print(f'[update] 开始增量拉取 {target_prefix} 数据', file=sys.stderr)

        count = _supplement_csv_incremental(
            csv_path,
            target_year=target_year,
            target_month=target_month,
            cache_dir=cache_dir,
        )
        print(f'[update] 增量拉取完成，本月 upsert {count} 行', file=sys.stderr)
        status['progress_pct'] = 70

        parquet_path = csv_path.replace('.csv', '.parquet')
        try:
            status['message'] = '正在重建 Parquet 缓存...'
            status['progress_pct'] = 72
            print('[update] 重建 stock_data.parquet', file=sys.stderr)
            _df_for_parquet = pd.read_csv(csv_path, encoding='gbk', low_memory=False)
            atomic_write_parquet(parquet_path, _df_for_parquet, engine='pyarrow', compression='snappy', index=False,
                                 produced_by='web.state._run_data_update:stock_data_parquet_rebuild')
            del _df_for_parquet
            print('[update] parquet 重建完成', file=sys.stderr)
        except Exception as parquet_err:
            print(f'[update] parquet 重建失败 ({parquet_err})，删除旧 parquet 强制 CSV fallback', file=sys.stderr)
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

        status['stage'] = 'rebuilding_cache'
        status['message'] = '正在重建回测缓存...'
        status['progress_pct'] = 75

        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)

        BACKTEST_CACHE.clear()
        TIMING_CACHE.clear()
        _PROFILE_SUMMARY_CACHE.clear()
        INDEX_RETURNS_MAP.clear()
        DATA_DF = None
        INDEX_RETURNS = None
        TIMING_PANEL = None

        status['message'] = '正在重新加载股票数据...'
        status['progress_pct'] = 80
        DATA_DF = load_data(csv_path)

        status['message'] = '正在加载指数数据...'
        status['progress_pct'] = 85
        ensure_index_returns_loaded()

        status['message'] = '正在运行选股策略回测...'
        status['progress_pct'] = 90
        init_cache()

        status['message'] = '正在运行择时策略回测...'
        status['progress_pct'] = 95
        init_timing_cache()

        status['stage'] = 'done'
        status['message'] = f'更新完成！新增 {count} 行数据'
        status['progress_pct'] = 100
        print(f'[update] 全部完成', file=sys.stderr)

    except Exception as e:
        status['stage'] = 'error'
        status['message'] = f'更新失败: {e}'
        status['error'] = str(e)
        print(f'[update ERROR] {e}', file=sys.stderr)
    finally:
        status['running'] = False


# ═════════════════════════════════════════════════════════════════
# 单因子回测（仅作为离线脚本的入口；web 层只读 FACTOR_BACKTEST_CACHE）
# ═════════════════════════════════════════════════════════════════
_SINGLE_FACTOR_CONFIGS = [
    {'id': 'size',      'name': '总市值(小市值)',    'column': '总市值',           'ascending': True},
    {'id': 'pb',        'name': '市净率倒数(高BM)',  'column': '市净率倒数',       'ascending': False},
    {'id': 'profit',    'name': '净利润TTM(高盈利)', 'column': '归母净利润_ttm',   'ascending': False},
    {'id': 'vol_stab',  'name': '成交额波动(低波动)','column': '成交额std_10',     'ascending': True},
    {'id': 'bias',      'name': 'BIAS20(超跌反弹)',  'column': 'bias_20',          'ascending': True},
    {'id': 'kdj_j',     'name': 'KDJ-J(超卖)',       'column': 'J',                'ascending': True},
    {'id': 'momentum',  'name': '动量(20日涨幅)',     'column': '涨跌幅_20',        'ascending': False},
    {'id': 'pe_inv',    'name': '市盈率倒数(低PE)',   'column': '市盈率倒数',       'ascending': False},
    {'id': 'turn_abn',  'name': '异常换手率(冷门)',   'column': '异常换手率',       'ascending': True},
    {'id': 'chan_div',  'name': '缠论背驰强度',       'column': 'chan_div_strength','ascending': False},
    {'id': 'chan_zs',   'name': '缠论中枢位置(下方)','column': 'chan_zs_position', 'ascending': True},
    {'id': 'chan_fr',   'name': '缠论底分型',         'column': 'chan_bottom_fractal','ascending': False},
    {'id': 'chan_str',  'name': '缠论笔强度',         'column': 'chan_stroke_strength','ascending': False},
    {'id': 'chan_sig',  'name': '缠论买卖点综合得分',  'column': 'chan_signal_score','ascending': False},
]


class _MockStrategy:
    name = 'single_factor'
    c_rate = 1.0 / 10000
    t_rate = 1.0 / 1000
    bull_tp = 0.30
    bear_tp = 0.22
    bull_n = 6
    bear_n = 4
    initial_capital = 100000

    def build_position_weights(self, selected_df):
        if selected_df is None or len(selected_df) == 0:
            return []
        n = len(selected_df)
        return [round(1.0 / n, 6)] * n

    def build_selection_reason(self, row, rank, total):
        return {'summary': f'单因子排名第{rank}/{total}', 'details': [], 'fundamentals': {}, 'factor_breakdown': []}


_EXTRA_FACTOR_COLS_READY = False


def _ensure_extra_factor_columns():
    global DATA_DF, _EXTRA_FACTOR_COLS_READY
    if _EXTRA_FACTOR_COLS_READY:
        return
    if DATA_DF is None:
        return

    if '异常换手率' not in DATA_DF.columns:
        if {'成交额', '流通市值'}.issubset(DATA_DF.columns):
            turn = DATA_DF['成交额'] / DATA_DF['流通市值'].replace(0, np.nan)
            DATA_DF['_turn_raw'] = turn
            g = DATA_DF.sort_values(['股票代码', '交易日期']).groupby('股票代码')['_turn_raw']
            DATA_DF['_turn_ma20'] = g.transform(lambda s: s.rolling(20, min_periods=5).mean())
            DATA_DF['异常换手率'] = DATA_DF['_turn_raw'] / DATA_DF['_turn_ma20'].replace(0, np.nan) - 1
            DATA_DF.drop(columns=['_turn_raw', '_turn_ma20'], inplace=True)
            print('[init] 已计算 异常换手率 列')

    needed_chan = ['chan_div_strength', 'chan_zs_position', 'chan_bottom_fractal',
                   'chan_stroke_strength', 'chan_signal_score']
    if not all(c in DATA_DF.columns for c in needed_chan):
        try:
            from chan_factors import compute_chan_factors
            DATA_DF = compute_chan_factors(DATA_DF)
            print(f"[init] 缠论因子列已生成: {[c for c in needed_chan if c in DATA_DF.columns]}")
        except Exception as e:
            print(f'[init] 缠论因子计算失败: {e}', file=sys.stderr)

    _EXTRA_FACTOR_COLS_READY = True


def _run_single_factor_backtest(top_k=5):
    ensure_stock_data_loaded()
    if DATA_DF is None:
        raise RuntimeError("数据未加载")

    _ensure_extra_factor_columns()

    required_cols = ['交易日期', '股票代码', '股票名称', '市场状态', '下周期每天涨跌幅']
    regime_map = DATA_DF.groupby('交易日期')['市场状态'].first()

    results = []
    for fc in _SINGLE_FACTOR_CONFIGS:
        col = fc['column']
        if col not in DATA_DF.columns:
            continue
        df = DATA_DF[required_cols + [col]].dropna(subset=[col]).copy()
        df['因子'] = df[col] if fc['ascending'] else -df[col]

        try:
            result = select_and_backtest(df, _MockStrategy(), select_stock_num=top_k)
        except Exception as e:
            print(f"[factor backtest] {fc['id']} failed: {e}", file=sys.stderr)
            continue

        ev = strategy_evaluate(result)

        result2 = result.copy()
        result2['regime'] = result2['交易日期'].map(regime_map)
        regime_metrics = {}
        for regime_key, regime_label in [('bull', '牛市'), ('bear', '熊市')]:
            sub = result2[result2['regime'] == regime_key]
            if len(sub) < 2:
                regime_metrics[regime_label] = {'avg_monthly_return': 'N/A', 'n_periods': 0}
            else:
                avg_ret = float(sub['选股下周期涨跌幅'].mean())
                regime_metrics[regime_label] = {
                    'avg_monthly_return': f"{round(avg_ret * 100, 2)}%",
                    'n_periods': len(sub),
                }

        results.append({
            'id': fc['id'],
            'name': fc['name'],
            'column': col,
            'dates': [str(d)[:10] for d in result['交易日期'].tolist()],
            'nav': [round(float(v), 4) for v in result['累积净值'].tolist()],
            'annual_return': str(ev.loc['年化收益', 0]),
            'max_drawdown': str(ev.loc['最大回撤', 0]),
            'calmar': float(ev.loc['年化收益/回撤比', 0]),
            'regime_metrics': regime_metrics,
        })

    return results


# ── 行业热度 ──
_SECTOR_HEAT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'strategy', 'sector_weekly_heat.csv')
)
_SECTOR_HEAT_CACHE = {'mtime': 0, 'data': None}


def _load_sector_heat():
    try:
        mtime = os.path.getmtime(_SECTOR_HEAT_FILE)
    except FileNotFoundError:
        return None
    if _SECTOR_HEAT_CACHE['data'] is None or mtime != _SECTOR_HEAT_CACHE['mtime']:
        df = pd.read_csv(_SECTOR_HEAT_FILE, encoding='utf-8-sig')
        _SECTOR_HEAT_CACHE['data'] = df
        _SECTOR_HEAT_CACHE['mtime'] = mtime
    return _SECTOR_HEAT_CACHE['data']
