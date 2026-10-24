"""Checks Acme demo data and the three planted issues."""

from __future__ import annotations

import json
import os
import re
from datetime import date

import pandas as pd

from src.config import DATA_DIR, DOCS_DIR
from src.schema import Claim

CSV_COLUMNS = [
    'period',
    'revenue',
    'operating_income',
    'net_income',
    'total_debt',
    'total_equity',
]
METRIC_COLS = CSV_COLUMNS[1:]
PERCENT_RE = re.compile(r'(\d+(?:\.\d+)?)%')
TEXT_PP_TOLERANCE = 0.1  # percentage points; 10-K wording vs CSV growth


def _financials() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'financials.csv')
    return pd.read_csv(path)


def _metadata_entries_with_files() -> list[dict]:
    path = os.path.join(DOCS_DIR, 'metadata.json')
    with open(path, encoding='utf-8') as handle:
        entries = json.load(handle)
    existing = []
    for entry in entries:
        file_path = os.path.join(DOCS_DIR, entry['file'])
        if os.path.isfile(file_path):
            existing.append(entry)
    return existing


def _yoy_growth_percentages(df: pd.DataFrame) -> list[float]:
    ordered = df.sort_values('period')
    rates: list[float] = []
    for col in METRIC_COLS:
        values = ordered[col].to_numpy()
        for i in range(1, len(values)):
            prior = values[i - 1]
            current = values[i]
            rates.append((current - prior) / prior * 100)
    return rates


def test_financials_csv_loads():
    df = _financials()
    assert list(df.columns) == CSV_COLUMNS
    assert list(df['period']) == ['FY2022', 'FY2023', 'FY2024']
    fy2024 = df.loc[df['period'] == 'FY2024'].iloc[0]
    assert fy2024['revenue'] == 99.9
    assert fy2024['operating_income'] == 14.0
    assert fy2024['net_income'] == 10.4
    assert fy2024['total_debt'] == 38
    assert fy2024['total_equity'] == 58


def test_metadata_dates_parse_for_existing_files():
    entries = _metadata_entries_with_files()
    source_ids = {entry['source_id'] for entry in entries}
    assert 'acme_10k_fy2024' in source_ids
    assert 'real_10k_risk_factors' in source_ids
    by_id = {entry['source_id']: entry for entry in entries}
    assert by_id['real_10k_risk_factors']['date'] == '2025-10-31'
    for entry in entries:
        parsed = date.fromisoformat(entry['date'])
        assert parsed.isoformat() == entry['date']


def test_narrative_claims_load_into_claim_objects():
    path = os.path.join(DATA_DIR, 'narrative_claims.json')
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    claims = [Claim.from_dict(item) for item in raw]
    by_id = {claim.claim_id: claim for claim in claims}

    n1 = by_id['n1']
    assert n1.claim == 'Revenue grew 18% in FY2024'
    assert n1.metric == 'revenue_growth'
    assert n1.period == 'FY2024'
    assert n1.value == 0.18
    assert n1.source_id == 'press_release_fy2024'
    assert n1.page_section == 'p.1'
    assert n1.date == '2025-02-10'

    n2 = by_id['n2']
    assert n2.claim == 'Debt levels are declining'
    assert n2.metric == 'total_debt_trend'
    assert n2.period == 'FY2024'
    assert n2.value is None
    assert n2.source_id is None
    assert n2.date is None

    n3 = by_id['n3']
    assert n3.claim == 'Management expects operating margin to remain stable'
    assert n3.metric == 'operating_margin_outlook'
    assert n3.period == 'FY2024'
    assert n3.value is None
    assert n3.source_id == 'analyst_note_fy2022'
    assert n3.date == '2023-01-15'

    n4 = by_id['n4']
    assert n4.claim == 'Net income was $10.4 million in FY2024'
    assert n4.metric == 'net_income'
    assert n4.period == 'FY2024'
    assert n4.value == 10.4
    assert n4.source_id == 'acme_10k_fy2024'
    assert n4.page_section == 'Item 7'
    assert n4.date == '2025-02-28'


def test_10k_percentages_match_csv_within_0_1_pp():
    df = _financials()
    expected = _yoy_growth_percentages(df)
    path = os.path.join(DOCS_DIR, 'acme_10k_fy2024.txt')
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    written = [float(match) for match in PERCENT_RE.findall(text)]
    assert written
    for pct in written:
        closest = min(abs(pct - rate) for rate in expected)
        assert closest <= TEXT_PP_TOLERANCE, f'{pct}% is not within 0.1 pp of CSV growth'
