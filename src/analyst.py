"""Builds claims from computed numbers. The LLM only explains, never calculates."""

from __future__ import annotations

import json
import os

import pandas as pd

from src import config
from src.llm import write_narrative
from src.ratios import compute_all, load_financials, lookup
from src.schema import UNVERIFIED, Claim

FILING_SOURCE_ID = 'acme_10k_fy2024'
FILING_DATE = '2025-02-28'
COMPUTED_METRICS = [
    'revenue_growth',
    'net_margin',
    'operating_margin',
    'debt_to_equity',
]


def _claim_text(metric: str, period: str, value: float) -> str:
    if metric == 'revenue_growth':
        return f'Revenue grew {value * 100:.1f}% in {period}'
    if metric == 'net_margin':
        return f'Net margin was {value * 100:.1f}% in {period}'
    if metric == 'operating_margin':
        return f'Operating margin was {value * 100:.1f}% in {period}'
    return f'Debt-to-equity was {value:.3f} in {period}'


def build_computed_claims(metrics_df: pd.DataFrame) -> list[Claim]:
    """One unverified claim per ratio that has a value, sourced from the 10-K."""
    periods = list(metrics_df['period'].drop_duplicates())
    claims: list[Claim] = []
    for metric in COMPUTED_METRICS:
        for period in periods:
            value = lookup(metrics_df, metric, period)
            if value is None:
                continue
            claim = Claim(
                claim_id=f'{metric}_{period}',
                claim=_claim_text(metric, period, value),
                metric=metric,
                period=period,
                value=value,
                source_id=FILING_SOURCE_ID,
                page_section='',
                date=FILING_DATE,
                status=UNVERIFIED,
            )
            claim.add_trail('Analyst Agent', 'computed')
            claims.append(claim)
    return claims


def load_narrative_claims(path: str) -> list[Claim]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    return [Claim.from_dict(item) for item in raw]


def build_draft_claims() -> list[Claim]:
    """Computed ratio claims plus the planted narrative claims, unique ids."""
    financials = load_financials(os.path.join(config.DATA_DIR, 'financials.csv'))
    metrics_df = compute_all(financials)
    claims = build_computed_claims(metrics_df)
    claims.extend(
        load_narrative_claims(os.path.join(config.DATA_DIR, 'narrative_claims.json'))
    )
    seen: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        if not claim.claim_id or claim.claim_id in seen:
            claim.claim_id = f'c{index}'
        seen.add(claim.claim_id)
    return claims


def make_narrative(metrics_df: pd.DataFrame) -> str:
    """Pass already-computed numbers to the LLM; do not ask it to calculate."""
    lines = ['Computed financial metrics (use these numbers only; do not calculate):']
    for _, row in metrics_df.iterrows():
        lines.append(f"{row['metric']} in {row['period']} = {row['value']}")
    return write_narrative('\n'.join(lines))
