"""Verifier tests use a stub judge so Ollama is not required."""

from __future__ import annotations

import json
import os

import pytest

from src import config
from src.analyst import build_draft_claims
from src.ratios import compute_all, load_financials
from src.retriever import Retriever
from src.schema import (
    CONFLICT,
    CONTRADICTED,
    OUTDATED,
    SUPPORTED,
    UNSUPPORTED,
    Claim,
)
from src.verifier import verify_all

RESULTS_PATH = os.path.join(
    os.path.dirname(config.DATA_DIR), 'eval', 'results_verifier.json'
)


def stub_judge(claim_text: str, passage_text: str) -> dict:
    return {'supported': True, 'reason': 'stub'}


@pytest.fixture(scope='module')
def retriever() -> Retriever:
    loaded = Retriever()
    loaded.load_from_metadata(config.DOCS_DIR)
    return loaded


@pytest.fixture(scope='module')
def metrics_df():
    return compute_all(
        load_financials(os.path.join(config.DATA_DIR, 'financials.csv'))
    )


def _by_id(claims: list[Claim]) -> dict[str, Claim]:
    return {claim.claim_id: claim for claim in claims}


def _injected_claims() -> list[Claim]:
    return [
        Claim(
            claim_id='inj_arith',
            claim='Revenue grew 35% in FY2024',
            metric='revenue_growth',
            period='FY2024',
            value=0.35,
            source_id='acme_10k_fy2024',
            date='2025-02-28',
        ),
        Claim(
            claim_id='inj_year',
            claim='Revenue grew 12.5% in FY2024',
            metric='revenue_growth',
            period='FY2024',
            value=0.125,
            source_id='acme_10k_fy2024',
            date='2025-02-28',
        ),
        Claim(
            claim_id='inj_nosource',
            claim='Debt levels are declining',
            metric='total_debt_trend',
            period='FY2024',
            value=None,
            source_id=None,
            date=None,
        ),
        Claim(
            claim_id='inj_stale',
            claim='Management expects operating margin to remain stable',
            metric='operating_margin_outlook',
            period='FY2024',
            value=None,
            source_id='analyst_note_fy2022',
            date='2023-06-30',
        ),
    ]


def _print_table(rows: list[dict]) -> None:
    header = f"{'error type':<28} {'injected':>8} {'caught':>8}"
    print('\n' + header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{row['error_type']:<28} {row['injected']:>8} {row['caught']:>8}"
        )


def test_verifier_planted_and_injected(retriever: Retriever, metrics_df):
    draft = verify_all(build_draft_claims(), metrics_df, retriever, judge=stub_judge)
    found = _by_id(draft)

    n1 = found['n1']
    growth = found['revenue_growth_FY2024']
    n2 = found['n2']
    n3 = found['n3']
    n4 = found['n4']

    injected = verify_all(
        _injected_claims(), metrics_df, retriever, judge=stub_judge
    )
    extra = _by_id(injected)

    rows = [
        {
            'error_type': 'cross-source conflict',
            'injected': 1,
            'caught': int(
                n1.status == CONFLICT and growth.status == CONFLICT
            ),
        },
        {
            'error_type': 'unsupported claim',
            'injected': 2,
            'caught': int(n2.status == UNSUPPORTED)
            + int(extra['inj_nosource'].status == UNSUPPORTED),
        },
        {
            'error_type': 'outdated source',
            'injected': 2,
            'caught': int(n3.status == OUTDATED)
            + int(extra['inj_stale'].status == OUTDATED),
        },
        {
            'error_type': 'wrong year (near-miss)',
            'injected': 1,
            'caught': int(extra['inj_year'].status == CONTRADICTED),
        },
        {
            'error_type': 'wrong arithmetic',
            'injected': 1,
            'caught': int(extra['inj_arith'].status == CONTRADICTED),
        },
    ]
    _print_table(rows)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2)
    print(f'\nWrote {RESULTS_PATH}')

    assert n1.status == CONFLICT
    assert growth.status == CONFLICT
    assert n2.status == UNSUPPORTED
    assert n2.reason == 'no evidence found'
    assert n3.status == OUTDATED
    assert n4.status == SUPPORTED

    computed_ids = [
        claim.claim_id for claim in draft
        if claim.claim_id not in {'n1', 'n2', 'n3', 'n4'}
    ]
    for claim_id in computed_ids:
        if claim_id == 'revenue_growth_FY2024':
            continue
        assert found[claim_id].status == SUPPORTED, claim_id

    assert extra['inj_arith'].status == CONTRADICTED
    assert extra['inj_year'].status == CONTRADICTED
    assert extra['inj_nosource'].status == UNSUPPORTED
    assert extra['inj_stale'].status == OUTDATED
    assert len(draft) == len(build_draft_claims())
    assert all(claim.trail for claim in draft)
    assert all(isinstance(claim.evidence, list) for claim in draft)
    assert all(claim.reason for claim in draft)
