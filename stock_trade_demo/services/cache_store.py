"""Disk cache persistence for the web app.

Owns ALL serialization to/from `.cache/web_cache.pkl` and
`.cache/single_factor_results.pkl`. web_app.py just hands in dicts to fill;
this module never touches Flask, strategy classes, or DataFrame logic.

Migration note (Protection 3 from plan_protections.md):
The pickle payload now optionally carries a `fingerprint` field — a sha256 of
the source files that influence cache content (backtest engine, strategies,
serializers, timing engine). On load, a mismatched fingerprint invalidates
the cache even when the manual `version` integer still matches. Existing
payloads without `fingerprint` are accepted once (gentle migration) and the
next save will populate it.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from typing import Dict, Optional, Tuple

# ── paths ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)  # stock_trade_demo/
CACHE_DIR: str = os.path.join(_PROJECT_DIR, '.cache')
WEB_CACHE_FILE: str = os.path.join(CACHE_DIR, 'web_cache.pkl')
FACTOR_BACKTEST_CACHE_FILE: str = os.path.join(CACHE_DIR, 'single_factor_results.pkl')
FACTOR_BACKTEST_BUILD_SCRIPT: str = 'stock_trade_demo/build_single_factor_cache.py'


def configure_cache_dir(cache_dir: str) -> None:
    """Route mutable web caches to the server-owned persistent data root."""
    global CACHE_DIR, WEB_CACHE_FILE, FACTOR_BACKTEST_CACHE_FILE
    CACHE_DIR = os.fspath(cache_dir)
    WEB_CACHE_FILE = os.path.join(CACHE_DIR, 'web_cache.pkl')
    FACTOR_BACKTEST_CACHE_FILE = os.path.join(CACHE_DIR, 'single_factor_results.pkl')

# ── version + fingerprint ────────────────────────────────────────────
# CACHE_VERSION: manual lever — bump only for breaking pickle-payload schema
# changes (added/removed top-level keys, type changes). Code-content shifts
# that affect cache *values* are caught by FINGERPRINT below, not by this.
CACHE_VERSION: int = 14  # last manual bump: load_data dedup → canonical date

# Files whose content affects what gets cached. Comments-only edits to these
# files will still invalidate the cache — that's accepted noise; the gain is
# that genuine semantic edits (factor changes, scoring tweaks, timing fixes)
# can never silently reuse stale results.
_FINGERPRINT_SOURCES = (
    'backtest.py',
    'index_data.py',
    'strategies/base.py',
    'strategies/original.py',
    'strategies/original_ensemble.py',
    'strategies/chan_enhanced.py',
    'strategies/chan_only.py',
    'strategies/method_a.py',
    'strategies/quality_value.py',
    'strategies/sector_heat.py',
    'timing/backtest.py',
    'web/serializers.py',
)
_SCHEMA_TAG = 'v1'  # bumped only for pickle layout changes (see CACHE_VERSION)


def compute_fingerprint() -> str:
    """sha256 over the source files that influence cache content."""
    h = hashlib.sha256()
    h.update(_SCHEMA_TAG.encode())
    for rel in _FINGERPRINT_SOURCES:
        p = os.path.join(_PROJECT_DIR, rel)
        if not os.path.exists(p):
            h.update(b'__missing__')
            continue
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


FINGERPRINT: str = compute_fingerprint()


# ── helpers ──────────────────────────────────────────────────────────
def _get_data_mtime(data_project_dir: str | None = None) -> float:
    """Latest mtime of stock_data.{csv,parquet}; treats missing files as 0."""
    project_dir = data_project_dir or _PROJECT_DIR
    max_mtime = 0.0
    for fname in ('stock_data.parquet', 'stock_data.csv'):
        fpath = os.path.join(project_dir, fname)
        if os.path.exists(fpath):
            max_mtime = max(max_mtime, os.path.getmtime(fpath))
    return max_mtime


def _ensure_cache_dir() -> None:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


# ── web_cache.pkl (strategy + timing results) ─────────────────────────
def save_web_cache(
    *,
    backtest_cache: Dict,
    timing_cache: Dict,
    profile_summary_cache: Dict,
    data_project_dir: str | None = None,
) -> bool:
    """Pickle the three runtime caches; returns True on success.

    Always overwrites WEB_CACHE_FILE. The caller passes the live dicts (NOT
    snapshots); this function takes shallow copies so concurrent mutations
    during pickle don't corrupt the file.
    """
    _ensure_cache_dir()
    payload = {
        'version': CACHE_VERSION,
        'fingerprint': FINGERPRINT,
        'data_mtime': (
            _get_data_mtime(data_project_dir)
            if data_project_dir is not None else _get_data_mtime()
        ),
        'backtest': dict(backtest_cache),
        'timing': dict(timing_cache),
        'profile_summary': dict(profile_summary_cache),
        'saved_at': time.time(),
    }
    # 原子写：tmp + fsync + os.replace，与 services/live_trades.py 同款。
    # 避免进程在 pickle 写到一半时被中断而留下残缺 web_cache.pkl，下次 load 失败。
    tmp_path = WEB_CACHE_FILE + '.tmp'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, WEB_CACHE_FILE)
        size_mb = os.path.getsize(WEB_CACHE_FILE) / (1024 * 1024)
        print(f'[cache] 磁盘缓存已保存 ({size_mb:.1f}MB)')
        return True
    except Exception as e:
        # 清理半截 tmp 文件，避免下次误读
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        print(f'[cache] 保存失败: {e}')
        return False


def load_web_cache(
    *,
    backtest_cache: Dict,
    timing_cache: Dict,
    profile_summary_cache: Dict,
    data_project_dir: str | None = None,
) -> bool:
    """Load WEB_CACHE_FILE into the supplied dicts in place.

    Returns True iff the file existed, parsed, AND passed version /
    fingerprint / data_mtime checks. Otherwise the dicts are left untouched
    and False is returned (caller should recompute).
    """
    if not os.path.exists(WEB_CACHE_FILE):
        return False
    try:
        with open(WEB_CACHE_FILE, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f'[cache] 读取磁盘缓存失败: {e}')
        return False

    if payload.get('version') != CACHE_VERSION:
        print('[cache] 版本不匹配，需要重新计算')
        return False

    # Fingerprint check: legacy payloads without the field are accepted once
    # (gentle migration); next save will populate it.
    cached_fp = payload.get('fingerprint')
    if cached_fp is None:
        print('[cache] 旧版缓存无 fingerprint，单次接受；下次保存将补齐')
    elif cached_fp != FINGERPRINT:
        print(f'[cache] fingerprint 失配 ({cached_fp} != {FINGERPRINT})，需要重新计算')
        return False

    current_data_mtime = (
        _get_data_mtime(data_project_dir)
        if data_project_dir is not None else _get_data_mtime()
    )
    if payload.get('data_mtime', 0) < current_data_mtime:
        print('[cache] 数据文件已更新，需要重新计算')
        return False

    backtest_cache.update(payload.get('backtest', {}))
    timing_cache.update(payload.get('timing', {}))
    profile_summary_cache.update(payload.get('profile_summary', {}))
    age = time.time() - payload.get('saved_at', 0)
    print(f'[cache] 磁盘缓存加载成功 (缓存保存于 {age:.0f}s 前)')
    return True


# ── single_factor_results.pkl (read-only product of offline script) ──
def load_factor_cache(factor_backtest_cache: Dict) -> bool:
    """Load the offline-produced single-factor backtest cache in place.

    Returns True iff the file existed and parsed; caller decides whether to
    serve 503 with build instructions otherwise. This module NEVER writes
    factor cache — that's exclusively `build_single_factor_cache.py`'s job.
    """
    if not os.path.exists(FACTOR_BACKTEST_CACHE_FILE):
        return False
    try:
        with open(FACTOR_BACKTEST_CACHE_FILE, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f'[cache] 读取单因子回测缓存失败: {e}')
        return False
    factors = payload.get('factors')
    top_k = payload.get('top_k', 5)
    if not isinstance(factors, list):
        print('[cache] 单因子回测缓存格式异常，已忽略')
        return False
    factor_backtest_cache[f'top_k={top_k}'] = {'factors': factors, 'top_k': top_k}
    age = time.time() - payload.get('saved_at', 0)
    print(f'[cache] 单因子回测缓存加载成功 ({len(factors)} 个因子，保存于 {age:.0f}s 前)')
    return True
