"""quant_demo v0.1 Snapshot/Legacy Viewer application factory."""
from __future__ import annotations

import math
import os
import time
import threading
from pathlib import Path
from typing import Any, Mapping
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider

from web import state
from web.blueprints import pages, v01_api
from web.v01.actions import init_runtime
from web.v01.contracts import ApiError
from web.v01.operation_log import append_request_result


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


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app = Flask(__name__, template_folder=template_dir,
                static_folder=static_dir, static_url_path='/static')

    # 全局 NaN 安全 JSON provider：防止任何端点意外序列化 float NaN → 非法 JSON
    app.json_provider_class = _NaNSafeJSONProvider
    app.json = _NaNSafeJSONProvider(app)
    # v0.1 closed schemas declare canonical response key order. Preserve the
    # insertion order built by each projection instead of Flask's lexical sort.
    app.json.sort_keys = False

    if config:
        app.config.update(config)
    app.config.setdefault('R0_OPERATION_LOG_PATH', Path(app.instance_path) / 'r0-operation-log.jsonl')

    init_runtime(app)
    app.register_blueprint(pages.bp)
    app.register_blueprint(v01_api.bp)

    @app.errorhandler(ApiError)
    def _stable_api_error(error):
        return error.response()

    @app.before_request
    def _enforce_v01_surface():
        """Only canonical v0.1 mutation routes may change local state."""
        if request.method in {'POST', 'DELETE'} and not (
            request.path.startswith('/api/r0/actions/')
            or request.path == '/api/r0/manual-records'
            or request.path.startswith('/api/r0/manual-records/')
        ):
            return jsonify({'error': 'route_not_available', 'message': 'Legacy mutation routes are not available.'}), 405
        return None

    @app.after_request
    def _r0_security_headers(response):
        mutation = {
            ('POST', '/api/r0/actions/data-update'): 'data-update',
            ('POST', '/api/r0/actions/cache-recover'): 'cache-recover',
            ('POST', '/api/r0/actions/restart'): 'restart',
            ('POST', '/api/r0/manual-records'): 'manual-create',
        }.get((request.method, request.path))
        if mutation is None and request.method == 'DELETE' and request.path.startswith('/api/r0/manual-records/'):
            mutation = 'manual-delete'
        if mutation is None and request.method == 'POST' and request.path.startswith('/api/r0/actions/') and request.path.endswith('/retry'):
            mutation = 'action-retry'
        if mutation is not None:
            payload = request.get_json(silent=True)
            scope = None
            if mutation == 'data-update' and isinstance(payload, dict):
                scope = payload.get('scopes')
            elif mutation == 'cache-recover' and isinstance(payload, dict):
                scope = '/'.join(str(payload.get(key) or '') for key in ('source_id', 'strategy_id', 'variant_id'))
            elif mutation == 'manual-create' and isinstance(payload, dict):
                scope = payload.get('strategy')
            elif mutation == 'restart':
                scope = 'service'
            elif mutation == 'manual-delete':
                scope = 'manual-record'
            elif mutation == 'action-retry':
                scope = 'existing-operation'
            error_payload = response.get_json(silent=True) if response.status_code >= 400 else None
            append_request_result(
                action=mutation, scope=scope, source_ip=request.remote_addr or 'unknown',
                status=response.status_code,
                error_code=error_payload.get('error') if isinstance(error_payload, dict) else None,
            )
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('Referrer-Policy', 'same-origin')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Cache-Control', 'no-store')
        return response

    return app


def start_eager_load_thread() -> None:
    """Open only canonical generation pointers; never deserialize legacy pickle.

    Canonical generation payloads are validated lazily by the production
    loader.  Fixed recovery/update POST workers are the only paths allowed to
    consume legacy builders and publish a new generation.
    """

    def _eager_load():
        state._LOAD_STATUS['loading'] = True
        state._LOAD_STATUS['start_time'] = time.time()
        try:
            state._LOAD_STATUS['stage'] = 'canonical_snapshot_store'
            state._LOAD_STATUS['message'] = 'Canonical snapshot store ready'
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
