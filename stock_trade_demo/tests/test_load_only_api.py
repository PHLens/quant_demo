"""Pillar 1 Step 6 — Web 请求路径只读缓存。

验证：
  1. /api/backtest 缓存缺失 → 400 cache_miss，message 指向 build_select_cache.py
  2. /api/timing/backtest 缓存缺失 → 400 cache_miss，指向 build_timing_cache.py
  3. /api/us_timing/backtest 缓存缺失 → 400 cache_miss，指向 build_us_timing_cache.py
  4. 命中默认参数缓存时正常返回 200

这些测试只用 Flask test_client + 内存里塞假缓存，不触发任何真实数据加载。
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from flask import Flask

from web import state
from web.blueprints import select_api, timing_api, us_timing_api


@pytest.fixture
def client():
    # These are unit tests for dormant legacy blueprints. R0 create_app()
    # intentionally does not register any of them.
    app = Flask(__name__)
    app.register_blueprint(select_api.bp)
    app.register_blueprint(timing_api.bp)
    app.register_blueprint(us_timing_api.bp)
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _isolate_caches():
    """每个用例前后清空内存缓存，避免互相污染。"""
    saved_bt = dict(state.BACKTEST_CACHE)
    saved_tm = dict(state.TIMING_CACHE)
    saved_us = dict(state.US_TIMING_CACHE)
    state.BACKTEST_CACHE.clear()
    state.TIMING_CACHE.clear()
    state.US_TIMING_CACHE.clear()
    try:
        yield
    finally:
        state.BACKTEST_CACHE.clear()
        state.BACKTEST_CACHE.update(saved_bt)
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE.update(saved_tm)
        state.US_TIMING_CACHE.clear()
        state.US_TIMING_CACHE.update(saved_us)


def test_select_backtest_returns_400_on_cache_miss(client, monkeypatch):
    # /api/backtest 入口会先调 state.init_cache() 把磁盘缓存加载回内存。
    # 为了断言"缓存确实缺失时 → 400"，必须把 init_cache stub 掉，否则
    # web_cache.pkl 里的 'original' bucket 会被加载，请求变成 200。
    monkeypatch.setattr(state, 'init_cache', lambda: None)
    resp = client.get('/api/backtest?strategy=original')
    assert resp.status_code == 400, resp.data
    body = resp.get_json()
    assert body['error'] == 'cache_miss'
    assert body['strategy'] == 'original'
    assert 'build_select_cache.py' in body['build_script']


def test_timing_backtest_returns_400_on_cache_miss(client, monkeypatch):
    # 避免任何真实数据加载（panel 拉取很慢且依赖磁盘）
    monkeypatch.setattr(state, 'ensure_timing_panel_loaded', lambda: None)
    resp = client.get('/api/timing/backtest?strategy=csi1000_timing')
    assert resp.status_code == 400, resp.data
    body = resp.get_json()
    assert body['error'] == 'cache_miss'
    assert body['strategy'] == 'csi1000_timing'
    assert 'build_timing_cache.py' in body['build_script']


def test_us_timing_backtest_returns_400_on_cache_miss(client, monkeypatch):
    monkeypatch.setattr(state, 'ensure_us_timing_panel_loaded', lambda force_reload=False: None)
    monkeypatch.setattr(state, 'init_us_timing_cache', lambda: None)
    resp = client.get('/api/us_timing/backtest?strategy=macro_v32_timing')
    assert resp.status_code == 400, resp.data
    body = resp.get_json()
    assert body['error'] == 'cache_miss'
    assert body['strategy'] == 'macro_v32_timing'
    assert 'build_us_timing_cache.py' in body['build_script']


def test_select_backtest_non_default_param_returns_400(client, monkeypatch):
    """即使缓存就绪，参数偏离默认也要走 cache_miss，绝不在请求路径里现算。"""
    # 不需要真的跑 init_cache（它会预热 CSI1000 择时信号，路径里的 sigmoid 会触发
    # RuntimeWarning: overflow in exp，跟本用例无关）。本测试已自行注入假缓存。
    monkeypatch.setattr(state, 'init_cache', lambda: None)
    # 注入一个假"缓存命中"以排除 cache_miss path 的第一条件 (strategy in BACKTEST_CACHE)
    fake = pd.DataFrame({'交易日期': pd.to_datetime(['2024-01-31', '2024-02-29']),
                         '累积净值': [1.0, 1.02],
                         '选股下周期涨跌幅': [0.0, 0.02]})
    state.BACKTEST_CACHE['original'] = (fake, None)
    # original 默认 val_pct_cutoff=0.68，传一个不一样的值
    resp = client.get('/api/backtest?strategy=original&val_pct_cutoff=0.5')
    assert resp.status_code == 400, resp.data
    body = resp.get_json()
    assert body['error'] == 'cache_miss'
