"""Numeric, textual, conflict, and staleness checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import pandas as pd

from src import config
from src.llm import judge_support
from src.ratios import RAW_METRICS, lookup
from src.schema import (
    CONFLICT,
    CONTRADICTED,
    OUTDATED,
    SUPPORTED,
    UNSUPPORTED,
    Claim,
)

STATUS_PRECEDENCE = [
    UNSUPPORTED,
    CONFLICT,
    CONTRADICTED,
    OUTDATED,
    SUPPORTED,
]
Judge = Callable[[str, str], dict]


def _is_raw_amount(metric: str) -> bool:
    return metric in RAW_METRICS


def within_tolerance(metric: str, left: float, right: float) -> bool:
    """Compare unrounded values with the config tolerance for this metric type."""
    delta = abs(left - right)
    if _is_raw_amount(metric):
        return delta <= config.REL_TOLERANCE * abs(right)
    return delta <= config.TOLERANCE_PP


def numeric_check(claim: Claim, metrics_df: pd.DataFrame) -> Optional[dict]:
    """Recompute (metric, period). None means this claim cannot be checked numerically."""
    if claim.value is None:
        return None
    recomputed = lookup(metrics_df, claim.metric, claim.period)
    if recomputed is None:
        return None
    match = within_tolerance(claim.metric, claim.value, recomputed)
    note = (
        f'claimed {claim.value} vs recomputed {recomputed}'
        if not match
        else f'numeric match: claimed {claim.value}, recomputed {recomputed}'
    )
    return {'recomputed': recomputed, 'match': match, 'note': note}


def _top_cited_hit(claim: Claim, retriever) -> Optional[dict]:
    if not claim.source_id:
        return None
    hits = retriever.search(claim.claim, source_id=claim.source_id)
    if not hits:
        return None
    return hits[0]


def _above_threshold(hit: Optional[dict]) -> bool:
    return hit is not None and hit['score'] >= config.SIM_THRESHOLD


def _evidence_from(hit: dict) -> dict:
    return {
        'source_id': hit['source_id'],
        'page_section': hit['page_section'],
        'date': hit['date'],
        'text': hit['text'],
        'score': hit['score'],
    }


def textual_check(claim: Claim, retriever, judge: Judge) -> Optional[dict]:
    """Judge the cited source. None if there is no source_id to search."""
    if not claim.source_id:
        return None
    hit = _top_cited_hit(claim, retriever)
    if not _above_threshold(hit):
        return {'supported': False, 'evidence': []}
    verdict = judge(claim.claim, hit['text'])
    supported = bool(verdict.get('supported')) if isinstance(verdict, dict) else bool(verdict)
    return {
        'supported': supported,
        'evidence': [_evidence_from(hit)],
        'judge_reason': verdict.get('reason', '') if isinstance(verdict, dict) else '',
    }


def staleness_check(claim: Claim) -> bool:
    """True if the source date is older than the report period minus STALE_MONTHS."""
    if not claim.date:
        return False
    source_date = pd.Timestamp(claim.date)
    cutoff = pd.Timestamp(config.REPORT_PERIOD_END) - pd.DateOffset(
        months=config.STALE_MONTHS
    )
    return source_date < cutoff


def detect_conflicts(claims: list[Claim]) -> dict[str, str]:
    """Flag groups that share (metric, period) but disagree across sources."""
    groups: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.value is None or not claim.source_id:
            continue
        groups[(claim.metric, claim.period)].append(claim)

    reasons: dict[str, str] = {}
    for (metric, _period), group in groups.items():
        sources = {claim.source_id for claim in group}
        if len(sources) < 2:
            continue
        values = [float(claim.value) for claim in group]
        disagrees = any(
            not within_tolerance(metric, values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        )
        if not disagrees:
            continue
        for claim in group:
            others = [
                other for other in group if other.source_id != claim.source_id
            ]
            named = ', '.join(
                f'{other.source_id} value {other.value}' for other in others
            )
            reasons[claim.claim_id] = f'conflict with {named}'
    return reasons


def _pick_status(flags: set[str]) -> str:
    for status in STATUS_PRECEDENCE:
        if status in flags:
            return status
    return SUPPORTED


def _apply_numeric(claim: Claim, metrics_df: pd.DataFrame, retriever) -> tuple[set[str], list[dict], dict[str, str]]:
    result = numeric_check(claim, metrics_df)
    if result is None:
        return set(), [], {}
    flags = {SUPPORTED if result['match'] else CONTRADICTED}
    reasons = {list(flags)[0]: result['note']}
    hit = _top_cited_hit(claim, retriever)
    evidence = [_evidence_from(hit)] if _above_threshold(hit) else []
    return flags, evidence, reasons


def _apply_textual(claim: Claim, retriever, judge: Judge) -> tuple[set[str], list[dict], dict[str, str]]:
    result = textual_check(claim, retriever, judge)
    if result is None:
        return {UNSUPPORTED}, [], {UNSUPPORTED: 'no evidence found'}
    if not result['supported']:
        return (
            {UNSUPPORTED},
            result['evidence'],
            {UNSUPPORTED: 'no evidence found in the cited source'},
        )
    return {SUPPORTED}, result['evidence'], {SUPPORTED: 'cited source supports the claim'}


def verify_all(
    claims: list[Claim],
    metrics_df: pd.DataFrame,
    retriever,
    judge: Judge = judge_support,
) -> list[Claim]:
    """Run every check on the full claim list. Never drop a claim."""
    flags_by_id: dict[str, set[str]] = {}
    reasons_by_id: dict[str, dict[str, str]] = {}

    for claim in claims:
        numeric_flags, evidence, reasons = _apply_numeric(claim, metrics_df, retriever)
        if numeric_flags:
            flags_by_id[claim.claim_id] = numeric_flags
            reasons_by_id[claim.claim_id] = reasons
            claim.evidence = evidence
        else:
            text_flags, evidence, reasons = _apply_textual(claim, retriever, judge)
            flags_by_id[claim.claim_id] = text_flags
            reasons_by_id[claim.claim_id] = reasons
            claim.evidence = evidence

        if staleness_check(claim):
            flags_by_id[claim.claim_id].add(OUTDATED)
            reasons_by_id[claim.claim_id][OUTDATED] = (
                f'source date {claim.date} is older than '
                f'{config.STALE_MONTHS} months before {config.REPORT_PERIOD_END}'
            )

    conflict_reasons = detect_conflicts(claims)
    for claim_id, reason in conflict_reasons.items():
        flags_by_id[claim_id].add(CONFLICT)
        reasons_by_id[claim_id][CONFLICT] = reason

    for claim in claims:
        status = _pick_status(flags_by_id[claim.claim_id])
        claim.status = status
        claim.reason = reasons_by_id[claim.claim_id].get(status, '')
        claim.add_trail('Verifier Agent', f'set status to {status}')
    return claims
