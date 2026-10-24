"""Ratio calculator and analyst draft-claim tests."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src import config
from src.analyst import build_draft_claims
from src.ratios import (
    compute_all,
    debt_to_equity,
    load_financials,
    lookup,
    net_margin,
    revenue_growth,
)


def _financials() -> pd.DataFrame:
    return load_financials(os.path.join(config.DATA_DIR, 'financials.csv'))


def _metrics() -> pd.DataFrame:
    return compute_all(_financials())


def test_fy2024_revenue_growth_is_0_11():
    df = _financials()
    assert revenue_growth(df).loc['FY2024'] == pytest.approx(0.11)
    assert lookup(_metrics(), 'revenue_growth', 'FY2024') == pytest.approx(0.11)


def test_fy2023_revenue_growth_is_0_125():
    df = _financials()
    assert revenue_growth(df).loc['FY2023'] == pytest.approx(0.125)
    assert lookup(_metrics(), 'revenue_growth', 'FY2023') == pytest.approx(0.125)


def test_fy2024_net_margin_is_about_0_1041():
    df = _financials()
    assert net_margin(df).loc['FY2024'] == pytest.approx(0.1041, abs=1e-4)
    assert lookup(_metrics(), 'net_margin', 'FY2024') == pytest.approx(0.1041, abs=1e-4)


def test_fy2024_debt_to_equity_is_about_0_655():
    df = _financials()
    assert debt_to_equity(df).loc['FY2024'] == pytest.approx(0.655, abs=1e-3)
    assert lookup(_metrics(), 'debt_to_equity', 'FY2024') == pytest.approx(0.655, abs=1e-3)


def test_build_draft_claims_includes_four_narrative_claims():
    claims = build_draft_claims()
    by_id = {claim.claim_id: claim for claim in claims}
    assert 'n1' in by_id
    assert 'n2' in by_id
    assert 'n3' in by_id
    assert 'n4' in by_id
    assert by_id['n1'].claim == 'Revenue grew 18% in FY2024'
    assert by_id['n1'].value == pytest.approx(0.18)
    assert by_id['n2'].claim == 'Debt levels are declining'
    assert by_id['n3'].claim == 'Management expects operating margin to remain stable'
    assert by_id['n4'].claim == 'Net income was $10.4 million in FY2024'
    assert by_id['n4'].value == pytest.approx(10.4)
    ids = [claim.claim_id for claim in claims]
    assert len(ids) == len(set(ids))
