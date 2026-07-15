"""R0 Snapshot / Legacy Viewer Flask application factory.

Only the page shell and the cache-only R0 API are registered. Historical API
blueprints remain in the repository for offline/legacy work, but they are not
reachable from this application.
"""
from __future__ import annotations

import math
import os
import time
import threading
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider

from web import state
from web.blueprints import pages, r0_viewer_api


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


def create_app() -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app = Flask(__name__, template_folder=template_dir,
                static_folder=static_dir, static_url_path='/static')

    # 全局 NaN 安全 JSON provider：防止任何端点意外序列化 float NaN → 非法 JSON
    app.json_provider_class = _NaNSafeJSONProvider
    app.json = _NaNSafeJSONProvider(app)

    app.register_blueprint(pages.bp)
    app.register_blueprint(r0_viewer_api.bp)

    @app.before_request
    def _enforce_r0_read_only():
        """Fail closed for every mutation, including unknown legacy paths."""
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            return jsonify({
                'error': 'read_only_viewer',
                'message': 'R0 only exposes read-only snapshot access.',
            }), 405
        return None

    @app.after_request
    def _r0_security_headers(response):
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('Referrer-Policy', 'same-origin')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Cache-Control', 'no-store')
        return response

    return app


def start_eager_load_thread() -> None:
    """Load existing cache files in the background without building anything."""

    def _eager_load():
        state._LOAD_STATUS['loading'] = True
        state._LOAD_STATUS['start_time'] = time.time()
        try:
            # R0 deliberately skips raw-data loading, external refresh and any
            # strategy calculation. Only already-built cache files are read.
            state._LOAD_STATUS['stage'] = 'strategy_cache'
            state._LOAD_STATUS['message'] = '正在读取选股快照缓存...'
            state.init_cache()
            state._LOAD_STATUS['stage'] = 'timing_cache'
            state._LOAD_STATUS['message'] = '正在读取择时快照缓存...'
            state.init_timing_cache()
            state.init_us_timing_cache()
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
