"""Regression coverage for rebuilding the original strategy after data refresh."""

from __future__ import annotations

import pandas as pd

from strategies.original import OriginalStrategy


def test_compute_factors_replaces_existing_val_pct_column():
    """A persisted derived val_pct must not turn the recomputed column into _x/_y."""
    dates = pd.to_datetime([
        '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30',
        '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31',
        '2024-09-30', '2024-10-31', '2024-11-30', '2024-12-31',
    ])
    df = pd.DataFrame({
        '新版申万二级行业名称': ['测试行业'] * len(dates),
        '交易日期': dates,
        '市盈率倒数': [0.05 + i * 0.001 for i in range(len(dates))],
        '市净率倒数': [0.10 + i * 0.001 for i in range(len(dates))],
        'val_pct': [0.123] * len(dates),
    })

    result = OriginalStrategy().compute_factors(df)

    assert list(result.columns).count('val_pct') == 1
    assert 'val_pct_x' not in result.columns
    assert 'val_pct_y' not in result.columns
    assert result['val_pct'].notna().all()
    assert result['val_pct'].iloc[-1] != 0.123
