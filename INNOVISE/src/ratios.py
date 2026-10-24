"""Pandas ratio calculator. All figures stay unrounded."""

from __future__ import annotations

from typing import Optional

import pandas as pd

RAW_METRICS = [
    'revenue',
    'operating_income',
    'net_income',
    'total_debt',
    'total_equity',
]


def load_financials(path: str) -> pd.DataFrame:
    """Load the financials CSV and index rows by period."""
    df = pd.read_csv(path)
    if df.empty or df['period'].duplicated().any():
        raise ValueError('Financial periods must be nonempty and unique.')
    for metric in RAW_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='raise')
        if not df[metric].map(lambda x: __import__('math').isfinite(x)).all():
            raise ValueError(f'Invalid or missing numbers in {metric}')
    return df.set_index('period').sort_index()


def revenue_growth(df: pd.DataFrame) -> pd.Series:
    return df['revenue'].pct_change()


def operating_margin(df: pd.DataFrame) -> pd.Series:
    return df['operating_income'] / df['revenue']


def net_margin(df: pd.DataFrame) -> pd.Series:
    return df['net_income'] / df['revenue']


def debt_to_equity(df: pd.DataFrame) -> pd.Series:
    return df['total_debt'] / df['total_equity']


RATIO_FUNCS = {
    'revenue_growth': revenue_growth,
    'operating_margin': operating_margin,
    'net_margin': net_margin,
    'debt_to_equity': debt_to_equity,
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of raw amounts and ratios. The first period has no growth row."""
    rows: list[dict] = []
    for metric in RAW_METRICS:
        for period, value in df[metric].items():
            rows.append({'metric': metric, 'period': period, 'value': float(value)})
    for metric, func in RATIO_FUNCS.items():
        for period, value in func(df).items():
            if not __import__('math').isfinite(value):
                continue
            rows.append({'metric': metric, 'period': period, 'value': float(value)})
    return pd.DataFrame(rows, columns=['metric', 'period', 'value'])


def lookup(metrics_df: pd.DataFrame, metric: str, period: str) -> Optional[float]:
    matched = metrics_df[
        (metrics_df['metric'] == metric) & (metrics_df['period'] == period)
    ]
    if matched.empty:
        return None
    return float(matched.iloc[0]['value'])
