"""0526 sprint 回归：默认页加载不能触发 cache_miss。

## 背景

Pillar 1/2 把 `/api/{timing,us_timing,backtest}` 改成 load-only —— 任何与"离线缓存
默认参数"不一致的查询都返回 HTTP 400 `{"error":"cache_miss"}`。

接着 0526 sprint 发现：前端在用户没有改任何 slider 时，仍然会把 `/api/.../params`
返回的 default 值挨个回放到 URL（`collectParamsForStrategy` → URLSearchParams），
后端则用裸的 `_TIMING_CACHE_DEFAULTS` 跟请求字面值做 strict equality 比较。

两个 default 来源不同时（strategy `__init__` 默认 vs `_TIMING_CACHE_DEFAULTS` 覆盖
vs `best_profile.all_params` 再次覆盖），就导致默认页 100% cache_miss，前端 timing
卡片全部显示 "—"。Worker A 的修复就是要把后端比较换成 "effective defaults"
（与离线缓存生成时实际使用的参数一致），并允许 float tolerance。

## 这条测试做什么

模拟前端默认加载行为：
  1. GET `/api/{timing,us_timing,factors}/params?strategy=<sid>` 拿到 parameter
     列表（含 default）
  2. 把每个 default 当作 query 参数回放到 `/api/.../backtest`
  3. 断言 status_code == 200，且响应里没有 `"error": "cache_miss"`

这条回归在 0526 sprint 之前会 fail（默认参数 strict equality miss），在 Worker A
的 effective-defaults 修复之后会 pass。后续再有人写"前端默认 → 后端 cache_miss"
类回归，这条测试在 CI 阶段就会爆掉。

注意：
- select 页（/index.html）默认只加载 `FOCUSED_STRATEGY_ID`，所以这里也只对那个
  策略做"完整默认参数回放"。其余 select 策略仅在用户主动切换 dropdown 时才会请求
  /api/backtest，因此它们的 cache 即使缺失也不属于"默认页加载回归"。
- timing 页同时渲染 `TIMING_STRATEGY_MAP` 里的全部策略。
- us_timing 页只渲染 `US_TIMING_PAGE_STRATEGY_IDS`（macro_v32_timing + sp500_timing）。
"""
from __future__ import annotations

import urllib.parse

import pandas as pd
import pytest
from flask import Flask

from web import state
from web.blueprints import select_api, timing_api, us_timing_api
from web.params import TimingParams


# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    # Exercise the dormant legacy blueprint contracts in isolation. The R0
    # application itself intentionally exposes none of these API routes.
    app = Flask(__name__)
    app.register_blueprint(select_api.bp)
    app.register_blueprint(timing_api.bp)
    app.register_blueprint(us_timing_api.bp)
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope='module', autouse=True)
def _ensure_caches_loaded():
    """这条回归测试天然需要"离线缓存确实存在"才有意义。

    /api/timing/backtest 与 /api/us_timing/backtest 自己不会触发 init_*_cache
    （只有 /strategy_list 和 /latest_signal 才会），所以如果磁盘缓存还没被加载到
    内存，所有 case 会因为 `strategy not in TIMING_CACHE` 直接 cache_miss，
    跟我们要回归的「default 参数被判定为非默认」根因混在一起。

    这里在 module 级别一次性 load_disk_cache + init_*_cache 把内存缓存填好，
    然后让请求路径走完整的 effective-defaults 比较逻辑（这正是 Worker A 修复的
    那段代码）。
    """
    state._EFFECTIVE_TIMING_DEFAULTS_CACHE.clear()
    state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()
    try:
        state.init_timing_cache()
    except Exception as exc:  # pragma: no cover - 让缺失原因清晰暴露
        pytest.skip(f'init_timing_cache 失败，缺少必要的离线缓存: {exc}')
    try:
        state.init_us_timing_cache()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'init_us_timing_cache 失败，缺少必要的离线缓存: {exc}')
    try:
        state.init_cache()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'init_cache 失败，缺少必要的离线缓存: {exc}')
    yield
    state._EFFECTIVE_TIMING_DEFAULTS_CACHE.clear()
    state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()


# ───────────────────────────────────────────────────────────────────────
# 通用工具：模拟前端的"默认参数回放"
# ───────────────────────────────────────────────────────────────────────
def _snap_to_step(value, step):
    """模拟浏览器 `<input type=range step=...>` 在读 `el.value` 时的 step 量化。

    HTML5 range input 在初始 value 不落在 step 网格上时，`el.value` 返回的是
    被 round 到最近 step 后的字符串。前端 `collectParamsForStrategy()` 直接读
    `el.value`，所以发到后端的就是 snap 后的值——比如 best_profile 设的 0.72
    会被 step=0.05 的 slider 量化成 0.70 再发出。

    这里在测试里复刻同一行为，是为了让"模拟前端默认加载"真的和浏览器一致；
    后端 effective_defaults 也在 `_extract_effective_from_instance` 里按
    同一 step 做了 snap，所以匹配的是 snap-vs-snap，不是字面 vs snap。
    """
    if step is None:
        return value
    try:
        step_f = float(step)
        val_f = float(value)
    except (TypeError, ValueError):
        return value
    if step_f <= 0:
        return value
    snapped = round(round(val_f / step_f) * step_f, 10)
    # 保留 int 类型（用户拉 confirm_days 这种整数 slider 时浏览器仍是整数）
    if isinstance(value, int) and not isinstance(value, bool) and step_f == int(step_f):
        return int(snapped)
    return snapped


def _build_default_query_from_params_endpoint(client, params_endpoint, strategy_id):
    """模拟前端 `collectParamsForStrategy` —— 把 /params 返回的 default 值挨个回放。

    返回 (query_string, params_metadata) 元组；query_string 含 strategy 与 compact=1。
    数值类型的 default 会先按 slider step 做 snap，复刻浏览器行为。
    """
    resp = client.get(f'{params_endpoint}?strategy={strategy_id}')
    assert resp.status_code == 200, (
        f'/params 自身就 fail 了 strategy={strategy_id}: {resp.status_code} {resp.data!r}'
    )
    md = resp.get_json() or {}
    params_meta = md.get('parameters', []) or md.get('factors', [])
    pairs = []
    for p in params_meta:
        if 'key' not in p or 'default' not in p:
            continue
        default_val = p['default']
        # 浏览器对 range slider 做 step 量化；select 没 step 字段，直接走原值
        snapped = _snap_to_step(default_val, p.get('step')) if p.get('type') == 'timing' else default_val
        # 前端 URLSearchParams.append 直接 str(value)；bool 会变成 'True'/'False'
        pairs.append((p['key'], str(snapped)))
    pairs.append(('strategy', strategy_id))
    pairs.append(('compact', '1'))
    return urllib.parse.urlencode(pairs), params_meta


def _build_default_query_from_factor_endpoint(client, strategy_id):
    """select 策略：从 /api/factors 拿 parameters，回放成 query。

    select 页 slider 也是 `<input type=range step=...>`，同样做 snap。
    """
    resp = client.get(f'/api/factors?strategy={strategy_id}')
    assert resp.status_code == 200, (
        f'/api/factors fail strategy={strategy_id}: {resp.status_code} {resp.data!r}'
    )
    md = resp.get_json() or {}
    params_meta = md.get('parameters', []) or []
    pairs = []
    for p in params_meta:
        if 'key' not in p or 'default' not in p:
            continue
        default_val = p['default']
        snapped = _snap_to_step(default_val, p.get('step')) if 'step' in p else default_val
        pairs.append((p['key'], str(snapped)))
    pairs.append(('strategy', strategy_id))
    return urllib.parse.urlencode(pairs), params_meta


def _assert_no_cache_miss(resp, *, strategy_id, where):
    """共同断言：默认页加载不能 cache_miss。"""
    body = resp.get_json()
    msg_extra = ''
    if body is not None:
        if isinstance(body, dict) and body.get('error') == 'cache_miss':
            # 把 diffs 等结构化信息塞进 failure message，方便调查
            msg_extra = f' body={body!r}'
    assert resp.status_code == 200, (
        f'[{where}] strategy {strategy_id} 默认页加载触发 cache_miss，'
        f'疑似引入了和 0526 sprint 类似的回归 '
        f'(status={resp.status_code}{msg_extra})'
    )
    assert not (isinstance(body, dict) and body.get('error') == 'cache_miss'), (
        f'[{where}] strategy {strategy_id} 默认页加载触发 cache_miss，'
        f'疑似引入了和 0526 sprint 类似的回归 (body={body!r})'
    )


# ───────────────────────────────────────────────────────────────────────
# 主回归：每条策略一例
# ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('strategy_id', sorted(state.TIMING_STRATEGY_MAP.keys()))
@pytest.mark.skip(reason='v0.1 retires the isolated legacy timing HTTP surface; canonical route isolation is covered in test_r0_viewer')
def test_timing_default_page_load_no_cache_miss(client, strategy_id):
    qs, meta = _build_default_query_from_params_endpoint(
        client, '/api/timing/params', strategy_id,
    )
    assert meta, f'/api/timing/params?strategy={strategy_id} 没返回任何 parameter，无法构造回放 query'
    resp = client.get(f'/api/timing/backtest?{qs}')
    _assert_no_cache_miss(resp, strategy_id=strategy_id, where='timing')


@pytest.mark.parametrize('strategy_id', sorted(state.US_TIMING_PAGE_STRATEGY_IDS))
@pytest.mark.skip(reason='v0.1 retires the isolated legacy US timing HTTP surface; canonical route isolation is covered in test_r0_viewer')
def test_us_timing_default_page_load_no_cache_miss(client, strategy_id):
    qs, meta = _build_default_query_from_params_endpoint(
        client, '/api/us_timing/params', strategy_id,
    )
    assert meta, f'/api/us_timing/params?strategy={strategy_id} 没返回任何 parameter，无法构造回放 query'
    resp = client.get(f'/api/us_timing/backtest?{qs}')
    _assert_no_cache_miss(resp, strategy_id=strategy_id, where='us_timing')


def test_select_default_page_load_no_cache_miss(client):
    """The retired interactive index contract no longer applies to R0."""
    pytest.skip('R0 index reads the cache-only /api/r0 snapshot surface')
    focused = state.get_focused_strategy_id()
    qs, meta = _build_default_query_from_factor_endpoint(client, focused)
    assert meta, f'/api/factors?strategy={focused} 没返回任何 parameter，无法构造回放 query'
    resp = client.get(f'/api/backtest?{qs}')
    _assert_no_cache_miss(resp, strategy_id=focused, where='select')


# ───────────────────────────────────────────────────────────────────────
# Lint-style 单元测试（task #4 可选项）：
#   TimingParams.from_query 出来的 dict 必须被后端 diff_from_defaults
#   视为「全部等于 effective_defaults」。这是一条比 HTTP 级别更窄的回归，
#   定位回归会更快。
# ───────────────────────────────────────────────────────────────────────
def _replay_pairs(md):
    """同 _build_default_query_from_params_endpoint 的 snap 规则，但返回 dict。"""
    out = {}
    for p in md.get('parameters', []):
        if 'key' not in p or 'default' not in p:
            continue
        v = p['default']
        if p.get('type') == 'timing':
            v = _snap_to_step(v, p.get('step'))
        out[p['key']] = str(v)
    return out


@pytest.mark.parametrize('strategy_id', sorted(state.TIMING_STRATEGY_MAP.keys()))
def test_timing_param_replay_is_seen_as_default(client, strategy_id):
    """从 /api/timing/params 拿 default 回放（含 slider snap），TimingParams 应认为全部 == effective_defaults。"""
    resp = client.get(f'/api/timing/params?strategy={strategy_id}')
    assert resp.status_code == 200
    md = resp.get_json() or {}
    pairs = _replay_pairs(md)
    params = TimingParams.from_query(pairs)
    effective = state.get_effective_timing_defaults(strategy_id)
    diffs = params.diff_from_defaults(effective)
    assert not diffs, (
        f'strategy {strategy_id} 默认参数回放被 diff_from_defaults 判定为非默认: {diffs}. '
        f'对应 effective_defaults={ {k: effective.get(k) for k in diffs} }'
    )


@pytest.mark.parametrize('strategy_id', sorted(state.US_TIMING_PAGE_STRATEGY_IDS))
def test_us_timing_param_replay_is_seen_as_default(client, strategy_id):
    resp = client.get(f'/api/us_timing/params?strategy={strategy_id}')
    assert resp.status_code == 200
    md = resp.get_json() or {}
    pairs = _replay_pairs(md)
    params = TimingParams.from_query(pairs)
    effective = state.get_effective_us_timing_defaults(strategy_id)
    diffs = params.diff_from_defaults(effective)
    assert not diffs, (
        f'strategy {strategy_id} 默认参数回放被 diff_from_defaults 判定为非默认: {diffs}. '
        f'对应 effective_defaults={ {k: effective.get(k) for k in diffs} }'
    )


# ───────────────────────────────────────────────────────────────────────
# 缺口 1：L2 lazy init 守卫测试（task #8）
#
# us_timing 的 /backtest 入口有一段 `state.init_us_timing_cache()` 守卫，
# 用来防止进程刚启动、还没人访问过 /strategy_list 时的首访 100% miss。
# 但是上面那个 module-scope autouse `_ensure_caches_loaded` fixture 已经
# 把 cache 预暖了——如果有人误删 us_timing_api.py 里的守卫，autouse 跑过
# 的 case 仍然全绿。
#
# 这条 case 用 function-scope fixture **在 case 内** clear US_TIMING_CACHE，
# 复现"冷启动"场景；如果入口守卫被删，请求就 cache_miss → 400 → fail。
# 测完后再 init 一次把 cache 还原，以免影响后续 case。
# ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def _us_timing_cache_cold_then_restore():
    """function-scope：进入时 clear US_TIMING_CACHE 模拟冷启动，退出时 init 还原。"""
    state.US_TIMING_CACHE.clear()
    state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()
    yield
    # 不管 case 成败，都把 cache 还原，否则后续 module 内 case（或别的 test 文件）
    # 会受到污染。init_us_timing_cache 内部本身是幂等的：已经满了就 no-op。
    try:
        state.init_us_timing_cache()
    except Exception:
        pass


@pytest.mark.skip(reason='v0.1 canonical Viewer GET never lazy-loads legacy caches')
def test_us_timing_backtest_lazy_inits_cache_on_first_hit(client, _us_timing_cache_cold_then_restore):
    """守住 us_timing_api.py 入口的 `state.init_us_timing_cache()` 守卫。

    若有人误删该守卫，US_TIMING_CACHE 在请求到达时仍是空的，下游
    `strategy_name in state.US_TIMING_CACHE` 直接判 False → 400 cache_miss，
    本测试立刻爆。
    """
    # 进 case 时 cache 已被 fixture 清空，模拟进程刚启动
    assert not state.US_TIMING_CACHE, '前置条件：fixture 应已 clear cache'
    resp = client.get('/api/us_timing/backtest?strategy=macro_v32_timing&compact=1')
    body = resp.get_json()
    assert resp.status_code == 200, (
        f'us_timing/backtest 在 cache 冷启动时返回 {resp.status_code}, body={body!r}. '
        f'疑似入口的 `state.init_us_timing_cache()` lazy init 守卫被误删。'
    )
    assert not (isinstance(body, dict) and body.get('error') == 'cache_miss'), (
        f'us_timing/backtest 冷启动 cache_miss，疑似入口 lazy init 守卫被误删: body={body!r}'
    )
    assert state.US_TIMING_CACHE, (
        '请求返回 200 但 US_TIMING_CACHE 仍为空，入口守卫 init_us_timing_cache 没有真正执行'
    )


# ───────────────────────────────────────────────────────────────────────
# 缺口 2：schema drift lint（task #8）
#
# web/params.py 用 4 个集合 (_BOOL_KEYS / _INT_KEYS / _FLOAT_KEYS / _STR_KEYS)
# 把参数按类型分类；TimingParams dataclass 同时按字段声明所有合法字段。
# 二者必须双向一致——否则会出现：
#   - KEYS 里有但 dataclass 没有的字段 → from_query 时 `cls(**kwargs)` TypeError
#   - dataclass 里有但 KEYS 没收录的字段 → 前端发了后端不解析，永远 fallback
#     到 None，effective_defaults 比较会把它认成"未指定"绕过。
# 这条 lint 把约束钉死在 CI 阶段。
# ───────────────────────────────────────────────────────────────────────
def test_timing_params_schema_alignment():
    """_*_KEYS 的并集与 TimingParams dataclass 字段必须双向一致。"""
    import dataclasses
    from web.params import (
        TimingParams, _BOOL_KEYS, _INT_KEYS, _FLOAT_KEYS, _STR_KEYS,
    )
    union = _BOOL_KEYS | _INT_KEYS | _FLOAT_KEYS | _STR_KEYS
    fields = {f.name for f in dataclasses.fields(TimingParams)}
    drift_in_keys = union - fields  # KEYS 多列、dataclass 没有
    drift_in_fields = fields - union  # dataclass 多声明、KEYS 没收录
    assert not drift_in_keys, (
        f'_BOOL/_INT/_FLOAT/_STR_KEYS 列了 TimingParams dataclass 没有的字段: '
        f'{sorted(drift_in_keys)}. 加字段到 dataclass 或从 KEYS 里删掉。'
    )
    assert not drift_in_fields, (
        f'TimingParams dataclass 声明了未被任何 _*_KEYS 收录的字段: '
        f'{sorted(drift_in_fields)}. 加到对应类型集合（bool→_BOOL_KEYS 等）。'
    )
