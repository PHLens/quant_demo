"""
SectorHeatStrategy — 在原版策略基础上增加行业动量过滤层。

核心思路：
  - 使用 strategy/sector_weekly_heat.csv（由 scripts/compute_sector_weekly_heat.py 离线生成）
  - 每个选股月对应一个"行业热度月"：stock at 2025-04-30 → year_month 2025-04
  - 计算该月各行业平均周收益的跨截面分位数，得到 sector_heat_pct（0~1，越高越热）
  - 支持两种过滤方向：
      'top'    — 只保留热度 >= cutoff 的行业（动量跟随：热者恒热）
      'bottom' — 只保留热度 <= 1-cutoff 的行业（冷板块寻底：冷者均值回归）
  - 以上过滤叠加在原版三步过滤之后，不替换任何原有因子
"""

import os
import pandas as pd
import numpy as np
from strategies.original import OriginalStrategy

_HEAT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'strategy', 'sector_weekly_heat.csv')
)

_heat_cache = {'mtime': 0, 'monthly': None}


def configure_resource_root(root):
    """Point the strategy's derived-data reader at the persistent root."""
    global _HEAT_FILE
    _HEAT_FILE = os.path.join(os.fspath(root), 'strategy', 'sector_weekly_heat.csv')
    _heat_cache.update(mtime=0, monthly=None)


def _load_monthly_heat():
    """懒加载 sector_weekly_heat.csv 并聚合为月度分位（带文件变更检测）。"""
    try:
        mtime = os.path.getmtime(_HEAT_FILE)
    except FileNotFoundError:
        return None

    if _heat_cache['monthly'] is not None and mtime == _heat_cache['mtime']:
        return _heat_cache['monthly']

    df = pd.read_csv(_HEAT_FILE, encoding='utf-8-sig')
    # 过滤 partial 周：当月未满 5 个交易日的周仅供 UI 展示，
    # 不参与策略选股，避免引入半周噪声影响历史回测可比性。
    if 'is_partial' in df.columns:
        df = df[~df['is_partial'].astype(bool)].copy()
    # 每个 (year_month, industry) 取该月所有周的平均收益
    monthly = (
        df.groupby(['year_month', 'industry'])['weekly_ret_pct']
        .mean()
        .reset_index()
    )
    monthly.columns = ['ym', 'l1_industry', 'heat_score']
    # 跨截面分位：每个月内对行业热度打分位
    monthly['heat_pct'] = monthly.groupby('ym')['heat_score'].rank(pct=True)

    _heat_cache['monthly'] = monthly
    _heat_cache['mtime'] = mtime
    return monthly


class SectorHeatStrategy(OriginalStrategy):
    """
    原版策略 + 行业热度过滤。

    新增参数：
      sector_heat_cutoff  — 热度阈值（0~1），默认 0.3
      sector_heat_mode    — 'top' | 'bottom'（默认 'top'，动量跟随）
        'top':    只保留热度分位 >= cutoff 的行业
        'bottom': 只保留热度分位 <= 1-cutoff 的行业
    """

    strategy_id = 'sector_heat'
    display_name = '行业热度增强'
    strategy_description = (
        '在原版小市值策略三步过滤后，叠加"申万一级行业月度热度"过滤，'
        '仅持有热度最高（或最低）的行业中的小市值股票。'
    )

    def __init__(self, sector_heat_cutoff=0.30, sector_heat_mode='top', **kwargs):
        super().__init__(**kwargs)
        self.sector_heat_cutoff = float(sector_heat_cutoff)
        self.sector_heat_mode = sector_heat_mode

    def get_parameter_definitions(self):
        base = super().get_parameter_definitions()
        return base + [
            {
                'key': 'sector_heat_cutoff',
                'label': '行业热度阈值',
                'description': (
                    '行业热度分位阈值。mode=top 时保留分位≥此值的行业；'
                    'mode=bottom 时保留分位≤(1−此值)的行业。'
                ),
                'default': self.sector_heat_cutoff,
                'min': 0.0, 'max': 0.7, 'step': 0.05,
                'unit': '', 'type': 'filter',
            },
        ]

    def get_filter_descriptions(self):
        return super().get_filter_descriptions() + [
            {
                'name': '行业热度过滤',
                'description': (
                    f'按当月申万一级行业平均周涨跌幅排序，'
                    f'mode={self.sector_heat_mode}，阈值={self.sector_heat_cutoff:.2f}。'
                    f'热度数据由 scripts/compute_sector_weekly_heat.py 离线生成。'
                ),
            },
        ]

    def get_factor_overview_tags(self):
        return ['规模因子', '行业动量']

    def compute_factors(self, df):
        """先调用原版因子，再附加行业热度分位列。"""
        df = super().compute_factors(df)

        monthly_heat = _load_monthly_heat()
        if monthly_heat is None:
            df['sector_heat_pct'] = 0.5
            return df

        df['_ym'] = df['交易日期'].dt.to_period('M').astype(str)
        df = df.merge(
            monthly_heat[['ym', 'l1_industry', 'heat_pct']],
            left_on=['_ym', '新版申万一级行业名称'],
            right_on=['ym', 'l1_industry'],
            how='left',
        )
        df['sector_heat_pct'] = df['heat_pct'].fillna(0.5)
        df.drop(columns=['_ym', 'ym', 'l1_industry', 'heat_pct'], inplace=True, errors='ignore')
        return df

    def apply_filters(self, df):
        """原版三步过滤后，叠加行业热度过滤。"""
        df = super().apply_filters(df)

        if 'sector_heat_pct' not in df.columns:
            return df

        cutoff = self.sector_heat_cutoff
        if self.sector_heat_mode == 'top':
            df = df[df['sector_heat_pct'] >= cutoff]
        else:
            df = df[df['sector_heat_pct'] <= (1.0 - cutoff)]
        return df

    def build_selection_reason(self, row, rank, total):
        base = super().build_selection_reason(row, rank, total)
        heat_pct = getattr(row, 'sector_heat_pct', None)
        if heat_pct is not None and not (isinstance(heat_pct, float) and np.isnan(heat_pct)):
            base['details'].append(
                f'行业热度分位: {heat_pct:.0%}（越高说明行业近期表现越强，mode={self.sector_heat_mode}）'
            )
        return base
