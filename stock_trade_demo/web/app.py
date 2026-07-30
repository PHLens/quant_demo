"""Flask application factory + blueprint 注册。

启动入口现在是 stock_trade_demo/web_app.py（≤30 行 shim），它只调用 create_app() + run。
所有路由按职责拆分在 web/blueprints/ 下：
  - pages          → 4 个模板路由 (/ /timing /us_timing /live)
  - select_api     → 选股策略 API
  - timing_api     → A 股择时 API
  - us_timing_api  → 美股择时 API
  - live_api       → 实盘记录 API（唯一写 live_trades.csv 的入口，走文件锁）
  - data_admin_api → 数据刷新 API
  - factor_explore_api → 行业热度 + 单因子回测只读 API
  - lab_api        → Learn/Lab-1 严格命令与只读状态 API（独立 artifact）
"""
from __future__ import annotations

import math
import os
import time
import threading
from flask import Flask
from flask.json.provider import DefaultJSONProvider

from web import state
from web.blueprints import (
    pages, select_api, timing_api, us_timing_api,
    live_api, data_admin_api, factor_explore_api, commodity_api,
    hk_timing_api, lab_api,
)
from lab.service import LabService, default_artifact_root


def _sanitize_nan_for_json(obj):
    """递归将 float NaN / Inf 替换为 None，保证 jsonify 输出合法 JSON（RFC 8259）。

    Python json.dumps 默认 allow_nan=True，会把 float('nan') / float('inf') 写成
    字面量 NaN / Infinity，这不是合法 JSON，浏览器 JSON.parse 会抛 SyntaxError。

    这里在 Flask JSON provider 层做全局拦截，作为最后一道防线——即使上游代码忘记
    清洗，HTTP 响应也不会输出非法字面量。
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan_for_json(v) for v in obj]
    return obj


class _NaNSafeJSONProvider(DefaultJSONProvider):
    """Flask JSON provider：序列化前自动清洗 float NaN/Inf → None。

    替换默认 DefaultJSONProvider，使所有 jsonify(...) 调用都经过 NaN 清洗，
    不依赖各个 blueprint 手动处理。Flask 2.x 通过 app.json_provider_class 配置。
    """

    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize_nan_for_json(obj), **kwargs)


def create_app(test_config=None) -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app = Flask(__name__, template_folder=template_dir,
                static_folder=static_dir, static_url_path='/static')

    # 全局 NaN 安全 JSON provider：防止任何端点意外序列化 float NaN → 非法 JSON
    app.json_provider_class = _NaNSafeJSONProvider
    app.json = _NaNSafeJSONProvider(app)
    app.config.from_mapping(
        LAB_ARTIFACT_ROOT=str(default_artifact_root()),
        LAB_EXECUTOR_KIND='process',
    )
    if test_config:
        app.config.update(test_config)

    lab_service = app.config.get('LAB_SERVICE')
    if lab_service is None:
        lab_service = LabService(
            app.config['LAB_ARTIFACT_ROOT'],
            executor_kind=app.config['LAB_EXECUTOR_KIND'],
        )
    app.extensions['lab_service'] = lab_service

    app.register_blueprint(pages.bp)
    app.register_blueprint(select_api.bp)
    app.register_blueprint(timing_api.bp)
    app.register_blueprint(us_timing_api.bp)
    app.register_blueprint(live_api.bp)
    app.register_blueprint(data_admin_api.bp)
    app.register_blueprint(factor_explore_api.bp)
    app.register_blueprint(commodity_api.bp)
    app.register_blueprint(hk_timing_api.bp)
    app.register_blueprint(lab_api.bp)

    return app


def start_eager_load_thread() -> None:
    """启动后台预加载线程：与原 web_app.py 的 _eager_load 行为等价。"""

    def _eager_load():
        state._LOAD_STATUS['loading'] = True
        state._LOAD_STATUS['start_time'] = time.time()
        try:
            # Stage 1: stock_data
            state._LOAD_STATUS['stage'] = 'stock_data'
            state._LOAD_STATUS['message'] = '正在加载股票数据 (823MB CSV)...'
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock_data.csv')
            if os.path.exists(csv_path) and state.DATA_DF is None:
                from backtest import load_data
                state.DATA_DF = load_data(csv_path)
            # Stage 2: index returns
            state._LOAD_STATUS['stage'] = 'index_data'
            state._LOAD_STATUS['message'] = '正在加载指数收益数据...'
            state.ensure_index_returns_loaded()
            # Stage 3: stock selection cache
            state._LOAD_STATUS['stage'] = 'strategy_cache'
            state._LOAD_STATUS['message'] = '正在预热选股策略回测缓存...'
            state.init_cache()
            # Stage 4: timing cache
            state._LOAD_STATUS['stage'] = 'timing_cache'
            state._LOAD_STATUS['message'] = '正在预热择时策略回测缓存...'
            state.init_timing_cache()
            # Stage 5: 单因子缓存
            state._LOAD_STATUS['stage'] = 'factor_cache'
            state._LOAD_STATUS['message'] = '正在加载单因子回测缓存...'
            if state._load_factor_backtest_cache():
                print('[init] 单因子回测缓存加载成功')
            else:
                print(
                    f'[init] 未找到 {state.FACTOR_BACKTEST_CACHE_FILE}，'
                    f'/api/factor_single_backtest 将返回 503。\n'
                    f'  请离线运行: python3 {state.FACTOR_BACKTEST_BUILD_SCRIPT}'
                )
            state._LOAD_STATUS['end_time'] = time.time()
            state._LOAD_STATUS['loading'] = False
            state._LOAD_STATUS['message'] = '数据加载完成'
            state._LOAD_STATUS['stage'] = 'ready'
            print(f'[init] 全部数据加载完成，耗时 {state._LOAD_STATUS["end_time"] - state._LOAD_STATUS["start_time"]:.1f}s')
        except Exception as e:
            state._LOAD_STATUS['loading'] = False
            state._LOAD_STATUS['message'] = f'加载失败: {e}'
            state._LOAD_STATUS['stage'] = 'error'
            print(f'[init ERROR] {e}')
        finally:
            state._DATA_READY.set()

    threading.Thread(target=_eager_load, daemon=True).start()
