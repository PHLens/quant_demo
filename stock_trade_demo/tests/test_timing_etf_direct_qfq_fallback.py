from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pandas as pd

import index_data


def _daily_frame(dates):
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'open': [1.0] * len(dates),
        'high': [1.1] * len(dates),
        'low': [0.9] * len(dates),
        'close': [1.0] * len(dates),
        'volume': [100.0] * len(dates),
    })


def test_eastmoney_direct_fetch_maps_qfq_to_fqt_1(monkeypatch):
    captured = {}
    payload = {
        'data': {
            'klines': [
                '2026-07-20,1.00,1.01,1.02,0.99,100,1000,3.0',
                '2026-07-21,1.01,1.03,1.04,1.00,120,1200,4.0',
            ],
        },
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode('utf-8')

    def _urlopen(request, timeout):
        captured['url'] = request.full_url
        captured['timeout'] = timeout
        captured['referer'] = request.get_header('Referer')
        return _Response()

    monkeypatch.setattr(index_data.urllib.request, 'urlopen', _urlopen)

    df = index_data._fetch_daily_kline_eastmoney('sh510980', adjust='qfq')

    query = parse_qs(urlparse(captured['url']).query)
    assert query['secid'] == ['1.510980']
    assert query['fqt'] == ['1']
    assert captured['timeout'] == 30
    assert captured['referer'] == 'https://quote.eastmoney.com/'
    assert list(df['date']) == [pd.Timestamp('2026-07-20'), pd.Timestamp('2026-07-21')]
    assert list(df['close']) == [1.01, 1.03]


def test_timing_etf_uses_direct_qfq_before_unadjusted_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(index_data, 'TIMING_ETF_CACHE_DIR', str(tmp_path / 'timing_etf'))

    def _akshare_failed(code, adjust):
        raise ConnectionError('akshare connection closed')

    calls = []

    def _direct_qfq(symbol, adjust=''):
        calls.append((symbol, adjust))
        return _daily_frame(['2026-07-20', '2026-07-21'])

    def _unadjusted_must_not_run(symbol):
        raise AssertionError('unadjusted fallback must not run when direct qfq succeeds')

    monkeypatch.setattr(index_data, '_fetch_etf_daily_akshare', _akshare_failed)
    monkeypatch.setattr(index_data, '_fetch_daily_kline_eastmoney', _direct_qfq)
    monkeypatch.setattr(index_data, '_fetch_daily_kline_with_fallback', _unadjusted_must_not_run)

    result = index_data.get_timing_etf_daily('csi1000', force_refetch=True)

    assert calls == [('sh510980', 'qfq')]
    assert result['date'].max() == pd.Timestamp('2026-07-21')
    qfq_path = tmp_path / 'timing_etf' / 'csi1000_etf_daily_qfq.csv'
    assert qfq_path.exists()
    assert pd.read_csv(qfq_path, parse_dates=['date'])['date'].max() == pd.Timestamp('2026-07-21')
    assert not (tmp_path / 'timing_etf' / 'csi1000_etf_daily.csv').exists()


def test_timing_etf_keeps_unadjusted_cache_separate_when_both_qfq_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(index_data, 'TIMING_ETF_CACHE_DIR', str(tmp_path / 'timing_etf'))

    monkeypatch.setattr(
        index_data,
        '_fetch_etf_daily_akshare',
        lambda code, adjust: (_ for _ in ()).throw(ConnectionError('akshare down')),
    )
    monkeypatch.setattr(
        index_data,
        '_fetch_daily_kline_eastmoney',
        lambda symbol, adjust='': (_ for _ in ()).throw(ConnectionError('direct qfq down')),
    )
    monkeypatch.setattr(
        index_data,
        '_fetch_daily_kline_with_fallback',
        lambda symbol: _daily_frame(['2026-07-20', '2026-07-21']),
    )

    result = index_data.get_timing_etf_daily('csi1000', force_refetch=True)

    assert result['date'].max() == pd.Timestamp('2026-07-21')
    assert not (tmp_path / 'timing_etf' / 'csi1000_etf_daily_qfq.csv').exists()
    legacy_path = tmp_path / 'timing_etf' / 'csi1000_etf_daily.csv'
    assert legacy_path.exists()
    assert pd.read_csv(legacy_path, parse_dates=['date'])['date'].max() == pd.Timestamp('2026-07-21')
