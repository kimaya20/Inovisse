"""Risk flags, report banner, non-binding considerations, and audit log."""

from __future__ import annotations

import os
import re

import pytest

from src import config
from src.analyst import build_draft_claims
from src.audit import get_log, log_action
from src.ratios import compute_all, load_financials
from src.report import BANNER, CONSIDERATIONS_TITLE, build_report, render_markdown
from src.retriever import Retriever
from src.risk import assess
from src.verifier import verify_all

TRADE_RE = re.compile(r'\b(buy|sell|hold|trade)\b', re.IGNORECASE)


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


@pytest.fixture(scope='module')
def verified_claims(retriever: Retriever, metrics_df):
    return verify_all(
        build_draft_claims(), metrics_df, retriever, judge=stub_judge
    )


@pytest.fixture(scope='module')
def risk_result(verified_claims, metrics_df, retriever: Retriever):
    return assess(verified_claims, metrics_df, retriever)


def test_de_and_net_margin_flags_fire_and_confidence_is_low(risk_result):
    metrics = {item['metric'] for item in risk_result['financial']}
    assert 'debt_to_equity' in metrics
    assert 'net_margin' in metrics
    assert risk_result['overall_confidence']['level'] == 'Low'


def test_banner_first_and_considerations_have_no_trade_words(
    verified_claims, risk_result, metrics_df
):
    report = build_report(
        query='Evaluate Acme Corp FY2024',
        claims=verified_claims,
        risk=risk_result,
        narrative='Numbers were computed in pandas from the Acme financials.',
        metrics_df=metrics_df,
    )
    markdown = render_markdown(report)
    assert markdown.startswith(BANNER)
    assert report['banner'] == BANNER
    assert report['considerations']['title'] == CONSIDERATIONS_TITLE
    blob = report['considerations']['title'] + ' ' + ' '.join(
        report['considerations']['items']
    )
    assert TRADE_RE.search(blob) is None
    assert any('18%' in item and '11%' in item for item in report['considerations']['items'])


def test_audit_log_writes_and_reads_with_temp_db(tmp_path):
    db_path = str(tmp_path / 'audit.db')
    row_id = log_action(
        report_id='rep1',
        claim_id='n1',
        action='correct',
        old_value=0.18,
        new_value=0.11,
        note='reconciled to filing',
        analyst='purva',
        db_path=db_path,
    )
    log_action(
        report_id='rep2',
        claim_id=None,
        action='approve_report',
        old_value=None,
        new_value=None,
        note='ok after review',
        analyst='purva',
        db_path=db_path,
    )
    assert row_id >= 1
    rows = get_log(report_id='rep1', db_path=db_path)
    assert len(rows) == 1
    assert rows[0]['action'] == 'correct'
    assert rows[0]['old_value'] == '0.18'
    assert rows[0]['new_value'] == '0.11'
    assert rows[0]['analyst'] == 'purva'
    all_rows = get_log(db_path=db_path)
    assert len(all_rows) == 2
    assert all_rows[0]['report_id'] == 'rep1'
    assert all_rows[1]['action'] == 'approve_report'
