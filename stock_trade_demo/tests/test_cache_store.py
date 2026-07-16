"""Regression tests for services.cache_store.

Covers Protection 3 from plan_protections.md:
  - round-trip save → load preserves dict content
  - version mismatch invalidates cache
  - fingerprint mismatch invalidates cache
  - legacy payload (no 'fingerprint' field) is accepted ONCE (gentle migration)
  - data_mtime guard: cache stale if underlying data file is newer
"""
from __future__ import annotations

import os
import pickle
import time

import pytest

from services import cache_store as cs


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    fake_file = tmp_path / 'web_cache.pkl'
    monkeypatch.setattr(cs, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(cs, 'WEB_CACHE_FILE', str(fake_file))
    # Pin data_mtime to a constant so the guard never trips unless we move it
    monkeypatch.setattr(cs, '_get_data_mtime', lambda: 1000.0)
    return fake_file


def test_round_trip(temp_cache):
    bt = {'sid_a': ('result_obj', 'eval_obj')}
    tm = {'csi1000_timing': {'foo': 'bar'}}
    ps = {'sid_a': [{'profile': 'p1'}]}
    assert cs.save_web_cache(backtest_cache=bt, timing_cache=tm, profile_summary_cache=ps)
    assert temp_cache.exists()

    bt2, tm2, ps2 = {}, {}, {}
    assert cs.load_web_cache(backtest_cache=bt2, timing_cache=tm2, profile_summary_cache=ps2)
    assert bt2 == bt
    assert tm2 == tm
    assert ps2 == ps


def test_version_mismatch_rejected(temp_cache, monkeypatch):
    cs.save_web_cache(backtest_cache={'x': 1}, timing_cache={}, profile_summary_cache={})
    # Bump in-memory version → existing payload should be rejected
    monkeypatch.setattr(cs, 'CACHE_VERSION', cs.CACHE_VERSION + 99)
    bt = {}
    assert not cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})
    assert bt == {}


def test_fingerprint_mismatch_rejected(temp_cache, monkeypatch):
    cs.save_web_cache(backtest_cache={'x': 1}, timing_cache={}, profile_summary_cache={})
    monkeypatch.setattr(cs, 'FINGERPRINT', 'deadbeef' * 2)
    bt = {}
    assert not cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})
    assert bt == {}


def test_legacy_payload_without_fingerprint_accepted(temp_cache):
    """旧版 pickle 不带 fingerprint → 应该被一次性接受（gentle migration）。"""
    legacy = {
        'version': cs.CACHE_VERSION,
        'data_mtime': 1000.0,
        'backtest': {'sid_legacy': 'value'},
        'timing': {},
        'profile_summary': {},
        'saved_at': time.time(),
        # NOTE: no 'fingerprint' key
    }
    with open(temp_cache, 'wb') as f:
        pickle.dump(legacy, f, protocol=pickle.HIGHEST_PROTOCOL)

    bt = {}
    assert cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})
    assert bt == {'sid_legacy': 'value'}


def test_stale_data_mtime_rejected(temp_cache, monkeypatch):
    cs.save_web_cache(backtest_cache={'x': 1}, timing_cache={}, profile_summary_cache={})
    # Data file became newer than the cache → reject
    monkeypatch.setattr(cs, '_get_data_mtime', lambda: 9999999999.0)
    bt = {}
    assert not cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})
    assert bt == {}


def test_missing_file_returns_false(temp_cache):
    assert not temp_cache.exists()
    bt = {}
    assert not cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})


def test_corrupt_pickle_returns_false(temp_cache):
    with open(temp_cache, 'wb') as f:
        f.write(b'not a valid pickle')
    bt = {}
    assert not cs.load_web_cache(backtest_cache=bt, timing_cache={}, profile_summary_cache={})


def test_fingerprint_is_stable():
    """连续调用 compute_fingerprint() 必须返回相同值（无副作用）。"""
    fp1 = cs.compute_fingerprint()
    fp2 = cs.compute_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 16  # 16 hex chars (sha256 truncated)


def test_configured_data_project_dir_controls_cache_mtime(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    cache_file = cache_dir / 'web_cache.pkl'
    data_project = tmp_path / 'persistent-data/stock_trade_demo'
    data_project.mkdir(parents=True)
    stock_csv = data_project / 'stock_data.csv'
    stock_csv.write_text('date,value\n2026-01-01,1\n', encoding='utf-8')
    os.utime(stock_csv, (2000, 2000))
    monkeypatch.setattr(cs, 'CACHE_DIR', str(cache_dir))
    monkeypatch.setattr(cs, 'WEB_CACHE_FILE', str(cache_file))

    assert cs.save_web_cache(
        backtest_cache={'x': 1}, timing_cache={}, profile_summary_cache={},
        data_project_dir=str(data_project),
    )
    with cache_file.open('rb') as stream:
        payload = pickle.load(stream)
    assert payload['data_mtime'] == 2000

    os.utime(stock_csv, (3000, 3000))
    assert not cs.load_web_cache(
        backtest_cache={}, timing_cache={}, profile_summary_cache={},
        data_project_dir=str(data_project),
    )
