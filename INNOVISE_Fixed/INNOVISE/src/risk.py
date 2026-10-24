"""Rule-based risk flags and overall confidence. No LLM opinions."""

from __future__ import annotations

from src import config
from src.ratios import lookup
from src.schema import CONFLICT, CONTRADICTED, OUTDATED, UNSUPPORTED, Claim

DATA_QUALITY_STATUSES = (CONFLICT, CONTRADICTED, UNSUPPORTED, OUTDATED)
BUSINESS_QUERIES = (
    'liquidity and debt interest leverage',
    'customer concentration major customers',
    'competition competitive markets',
    'regulation legal regulatory compliance',
)
BUSINESS_DOC_TYPES = ['filing']  # Demo scope: exclude unrelated Apple reference filing


def _sorted_periods(metrics_df) -> list[str]:
    return sorted(metrics_df['period'].drop_duplicates())


def _data_quality(claims: list[Claim]) -> list[dict]:
    items = []
    for claim in claims:
        if claim.status not in DATA_QUALITY_STATUSES:
            continue
        items.append({
            'claim_id': claim.claim_id,
            'status': claim.status,
            'claim': claim.claim,
            'reason': claim.reason,
            'source_id': claim.source_id,
            'date': claim.date,
        })
        claim.add_trail('Risk Agent', f'listed data-quality risk ({claim.status})')
    return items


def _consecutive_pairs(periods: list[str]) -> list[tuple[str, str]]:
    return list(zip(periods, periods[1:]))


def _financial_flags(metrics_df) -> list[dict]:
    flags: list[dict] = []
    periods = _sorted_periods(metrics_df)
    for prior, current in _consecutive_pairs(periods):
        prior_de = lookup(metrics_df, 'debt_to_equity', prior)
        current_de = lookup(metrics_df, 'debt_to_equity', current)
        if prior_de is not None and current_de is not None:
            change = current_de - prior_de
            if change > config.DE_RISE:
                flags.append({
                    'kind': 'debt_to_equity_rise',
                    'metric': 'debt_to_equity',
                    'from_period': prior,
                    'to_period': current,
                    'from_value': prior_de,
                    'to_value': current_de,
                    'change': change,
                    'threshold': config.DE_RISE,
                    'note': (
                        f'debt-to-equity rose from {prior_de} in {prior} '
                        f'to {current_de} in {current}'
                    ),
                })
        for metric in ('operating_margin', 'net_margin'):
            prior_m = lookup(metrics_df, metric, prior)
            current_m = lookup(metrics_df, metric, current)
            if prior_m is None or current_m is None:
                continue
            drop = prior_m - current_m
            if drop > config.MARGIN_DROP:
                flags.append({
                    'kind': 'margin_fall',
                    'metric': metric,
                    'from_period': prior,
                    'to_period': current,
                    'from_value': prior_m,
                    'to_value': current_m,
                    'change': -drop,
                    'threshold': config.MARGIN_DROP,
                    'note': (
                        f'{metric} fell from {prior_m} in {prior} '
                        f'to {current_m} in {current}'
                    ),
                })
        prior_g = lookup(metrics_df, 'revenue_growth', prior)
        current_g = lookup(metrics_df, 'revenue_growth', current)
        if prior_g is not None and current_g is not None and current_g < prior_g:
            flags.append({
                'kind': 'revenue_growth_slowing',
                'metric': 'revenue_growth',
                'from_period': prior,
                'to_period': current,
                'from_value': prior_g,
                'to_value': current_g,
                'change': current_g - prior_g,
                'note': (
                    f'revenue growth slowed from {prior_g} in {prior} '
                    f'to {current_g} in {current}'
                ),
            })
    return flags


def _business_risks(retriever) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    items: list[dict] = []
    for query in BUSINESS_QUERIES:
        hits = retriever.search(query, doc_types=BUSINESS_DOC_TYPES)
        for hit in hits:
            if hit['score'] < config.SIM_THRESHOLD:
                continue
            key = (hit['source_id'], hit['text'])
            if key in seen:
                continue
            seen.add(key)
            items.append({
                'source': hit['source_id'],
                'title': hit['title'],
                'date': hit['date'],
                'text': hit['text'],
            })
    return items


def _confidence(claims: list[Claim]) -> dict:
    if not claims or any(c.status == 'unverified' for c in claims):
        return {'level': 'Low', 'reason': 'No complete verification is available.'}
    statuses = {claim.status for claim in claims}
    if CONFLICT in statuses or CONTRADICTED in statuses:
        return {
            'level': 'Low',
            'reason': 'At least one claim is in conflict or contradicted.',
        }
    if UNSUPPORTED in statuses or OUTDATED in statuses:
        return {
            'level': 'Medium',
            'reason': 'Some claims are unsupported or outdated.',
        }
    return {
        'level': 'High',
        'reason': 'No conflict, contradicted, unsupported, or outdated claims.',
    }


def assess(claims: list[Claim], metrics_df, retriever) -> dict:
    """Return data-quality, financial and business risks plus overall confidence."""
    return {
        'data_quality': _data_quality(claims),
        'financial': _financial_flags(metrics_df),
        'business': _business_risks(retriever),
        'overall_confidence': _confidence(claims),
    }
