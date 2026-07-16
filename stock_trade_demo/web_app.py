"""量化策略 Web 可视化 — 启动 shim。

Pillar 1 Step 4 之后，所有路由 / 业务逻辑都搬到了 web/blueprints/ 和 web/state.py 下。
本文件只剩三件事：
  1. 把 stock_trade_demo/ 与仓库根加入 sys.path（方便 `python web_app.py` 直接跑）；
  2. 通过 web.app.create_app() 拿到 Flask app；
  3. 启动后台预加载线程 + app.run(port=8080)。

为了不破坏现有 import 路径（tests / scripts 依赖 `from web_app import ...` 拿全局状态），
这里继续 re-export 关键符号——所有 re-export 都是 module-attribute 委托，不要把可变 dict
按值复制过来。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import state as _state
from web.app import create_app, start_eager_load_thread

# ── 向后兼容 re-export ──
# 测试 (test_index_etf_alignment.py) 会 monkeypatch.setattr(web_app, 'get_index_daily', ...)
# 所以必须把这些名字真正放到本 module 的命名空间。
from web.state import (  # noqa: F401
    TRAINING_CUTOFF, HOLDOUT_START,
    FACTOR_BACKTEST_BUILD_SCRIPT,
    ensure_stock_data_loaded, ensure_us_timing_panel_loaded, ensure_timing_panel_loaded,
    ensure_index_returns_loaded, init_cache, init_timing_cache, init_us_timing_cache,
    run_timing_backtest_fresh, run_backtest_fresh,
    build_us_timing_strategy, build_timing_strategy, build_strategy,
    _run_single_factor_backtest,
)
# A 股 ETF 校验 / index 拉取入口：测试通过 monkeypatch.setattr(web_app, ...) 替换
from index_data import (  # noqa: F401
    get_index_daily, get_timing_etf_daily, A_SHARE_INDEX_IDS,
)
import pandas as _pd


def _check_a_share_index_etf_alignment():
    """A 股指数日线 vs ETF 日线 max_date 一致性检查。

    定义在 web_app 模块里（不仅仅是从 state re-export），是为了让
    test_index_etf_alignment.py 的 `monkeypatch.setattr(web_app, 'get_index_daily', _idx)`
    能真正生效——函数读到的是当前 web_app 模块命名空间下的 get_index_daily / get_timing_etf_daily。
    生产路径（state._run_index_data_update）调用 state 自己的同名实现，互不影响。
    """
    mismatches = []
    for index_id in A_SHARE_INDEX_IDS:
        try:
            idx_df = get_index_daily(index_id)
            etf_df = get_timing_etf_daily(index_id)
        except Exception as exc:
            mismatches.append({'index_id': index_id, 'error': repr(exc)})
            continue
        idx_max = _pd.to_datetime(idx_df['date']).max() if idx_df is not None and len(idx_df) > 0 else None
        etf_max = _pd.to_datetime(etf_df['date']).max() if etf_df is not None and len(etf_df) > 0 else None
        if idx_max is None or etf_max is None:
            continue
        # 与 web/state.py:_check_a_share_index_etf_alignment 保持一致（容差双向）：
        # idx < etf 始终报；etf < idx 且滞后 >2 天才报，避免同日刷新小滞后误报。
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


def __getattr__(name):
    """对未显式 re-export 的名字，转发到 web.state；保持下游 `web_app.XXX` 读到最新值。"""
    if hasattr(_state, name):
        return getattr(_state, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


app = create_app()


if __name__ == '__main__':
    print('=' * 60)
    print('  量化策略 Web 可视化')
    print('  访问 http://localhost:8080')
    print('=' * 60)
    start_eager_load_thread()
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
