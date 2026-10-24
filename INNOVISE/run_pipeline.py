"""Runs the research pipeline end to end."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src import config
from src.analyst import build_draft_claims, make_narrative
from src.report import build_report, render_markdown
from src.retriever import Retriever
from src.risk import assess
from src.ratios import compute_all, load_financials
from src.schema import Claim
from src.verifier import verify_all

DEFAULT_QUERY = 'Evaluate Acme Corp FY2024 performance and risks'
LAST_RUN_PATH = os.path.join(config.CACHE_DIR, 'last_run.json')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_trail(state: dict, agent: str, action: str) -> None:
    state.setdefault('trail', []).append({
        'agent': agent,
        'action': action,
        'timestamp': _now(),
    })


def _ensure_unique_ids(claims: list[Claim]) -> None:
    seen: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        if not claim.claim_id or claim.claim_id in seen:
            claim.claim_id = f'c{index}'
        seen.add(claim.claim_id)


def _claims_to_dicts(claims: list[Claim]) -> list[dict]:
    return [claim.to_dict() for claim in claims]


def _claims_from_state(state: dict) -> list[Claim]:
    return [Claim.from_dict(item) for item in state['claims']]


def _metrics_records(metrics_df: pd.DataFrame) -> list[dict]:
    records = metrics_df.to_dict(orient='records')
    for row in records:
        for key, value in list(row.items()):
            if hasattr(value, 'item'):
                row[key] = value.item()
    return records


def save_state(state: dict) -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    payload = {
        key: value
        for key, value in state.items()
        if key not in {'retriever', 'metrics_df'}
    }
    with open(LAST_RUN_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _attach_runtime(state: dict, retriever: Retriever, metrics_df: pd.DataFrame) -> dict:
    state['retriever'] = retriever
    state['metrics_df'] = metrics_df
    save_state(state)
    return state


def _finish_report(state: dict, claims: list[Claim], metrics_df: pd.DataFrame, retriever: Retriever) -> dict:
    state['narrative'] = ' '.join(c.claim.rstrip('.') + '.' for c in claims if c.status == 'supported') or 'No claims currently meet all verification checks.'
    risk = assess(claims, metrics_df, retriever)
    _append_trail(state, 'Risk Agent', 'assessed data-quality, financial and business risks')
    report = build_report(
        state['query'],
        claims,
        risk,
        state.get('narrative') or '',
        metrics_df,
    )
    _append_trail(state, 'Report Agent', 'built review report')
    state['claims'] = _claims_to_dicts(claims)
    state['risk'] = risk
    state['report'] = report
    state['markdown'] = render_markdown(report)
    state['metrics'] = _metrics_records(metrics_df)
    return _attach_runtime(state, retriever, metrics_df)


def run(query: str, extra_claims: Optional[list] = None) -> dict:
    """Load data, retrieve, draft, verify, assess, narrate and report."""
    state: dict = {
        'query': query,
        'report_id': uuid.uuid4().hex,
        'trail': [],
    }
    financials = load_financials(os.path.join(config.DATA_DIR, 'financials.csv'))
    metrics_df = compute_all(financials)
    _append_trail(state, 'Analyst Agent', 'loaded financials and computed metrics')

    retriever = Retriever()
    retriever.load_from_metadata(config.DOCS_DIR)
    _append_trail(state, 'Research Agent', 'indexed documents from metadata')

    claims = build_draft_claims()
    if extra_claims:
        for item in extra_claims:
            claim = item if isinstance(item, Claim) else Claim.from_dict(item)
            claims.append(claim)
        _ensure_unique_ids(claims)
    _append_trail(state, 'Analyst Agent', 'built draft claims')

    verify_all(claims, metrics_df, retriever)
    _append_trail(state, 'Verifier Agent', 'verified all claims')

    narrative = ' '.join(c.claim.rstrip('.') + '.' for c in claims if c.status == 'supported') or 'No claims currently meet all verification checks.'
    state['narrative'] = narrative
    _append_trail(state, 'Analyst Agent', 'wrote narrative from computed numbers')

    return _finish_report(state, claims, metrics_df, retriever)


def rerun_verification(state: dict, retriever: Retriever, metrics_df: pd.DataFrame) -> dict:
    """Re-verify every claim, then refresh risks and the report. Keep claim_ids."""
    claims = _claims_from_state(state)
    verify_all(claims, metrics_df, retriever)
    _append_trail(state, 'Verifier Agent', 're-verified all claims')
    return _finish_report(state, claims, metrics_df, retriever)


def _print_claim_table(state: dict) -> None:
    rows = [
        {
            'claim_id': item['claim_id'],
            'status': item['status'],
            'reason': item['reason'],
        }
        for item in state['claims']
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Run the Acme research pipeline.')
    parser.add_argument('--query', default=DEFAULT_QUERY)
    args = parser.parse_args(argv)
    state = run(args.query)
    _print_claim_table(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
