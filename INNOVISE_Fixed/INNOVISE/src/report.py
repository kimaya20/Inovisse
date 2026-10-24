"""Report builder. Banner first; no trading language."""

from __future__ import annotations

from collections import defaultdict

from src.schema import (
    CONFLICT,
    CONTRADICTED,
    OUTDATED,
    UNSUPPORTED,
    Claim,
)

BANNER = 'AI-generated, requires human review'
CONSIDERATIONS_TITLE = 'Considerations for the analyst (non-binding)'
FLAGGED = (CONFLICT, CONTRADICTED, UNSUPPORTED, OUTDATED)


def _pct_label(value: float) -> str:
    text = f'{value * 100:.1f}'.rstrip('0').rstrip('.')
    return f'{text}%'


def _summary(query: str, claims: list[Claim], risk: dict, narrative: str) -> str:
    level = risk['overall_confidence']['level']
    reason = risk['overall_confidence']['reason']
    bits = []
    if query:
        bits.append(f'Query: {query}.')
    if narrative:
        bits.append(narrative.strip())
    bits.append(f'Overall confidence: {level}. {reason}')
    return ' '.join(bits)


def _metrics_table(metrics_df) -> list[dict]:
    return metrics_df.to_dict(orient='records')


def _findings(claims: list[Claim]) -> list[dict]:
    return [
        {
            'claim_id': claim.claim_id,
            'claim': claim.claim,
            'status': claim.status,
            'reason': claim.reason,
            'source': claim.source_id,
            'date': claim.date,
        }
        for claim in claims
    ]


def _limitations(claims: list[Claim]) -> list[str]:
    items = [
        'The company data is synthetic (Acme Corp) plus one real EDGAR filing (Item 1A risk factors).',
        'A small local model gives only a second opinion on whether a passage supports a claim; numbers are computed in pandas.',
    ]
    for claim in claims:
        if claim.status == OUTDATED:
            items.append(
                f'Outdated source dated {claim.date}: {claim.claim}'
            )
    return items


def _unresolved_questions(claims: list[Claim]) -> list[str]:
    questions = []
    for claim in claims:
        if claim.status not in FLAGGED:
            continue
        questions.append(
            f'What is the resolution for "{claim.claim}" ({claim.status}: {claim.reason})?'
        )
    return questions


def _conflict_considerations(claims: list[Claim]) -> list[str]:
    groups: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.status == CONFLICT:
            groups[(claim.metric, claim.period)].append(claim)
    items = []
    for (metric, period), group in groups.items():
        values = [claim.value for claim in group if claim.value is not None]
        if metric == 'revenue_growth' and len(values) >= 2:
            labels = [_pct_label(value) for value in sorted(values, reverse=True)]
            items.append(
                'Consider asking management to reconcile the '
                f'{labels[0]} and {labels[1]} revenue growth figures'
            )
        else:
            items.append(
                f'Consider asking management to reconcile disagreeing {metric} figures for {period}'
            )
    return items


def _consideration_items(claims: list[Claim], risk: dict) -> list[str]:
    items = _conflict_considerations(claims)
    for claim in claims:
        if claim.status == CONTRADICTED:
            items.append(
                f'Consider reviewing the contradicted figure: {claim.claim}'
            )
        elif claim.status == UNSUPPORTED:
            items.append(
                f'Consider locating a source for: {claim.claim}'
            )
        elif claim.status == OUTDATED:
            items.append(
                f'Consider obtaining a more recent source than {claim.date} for: {claim.claim}'
            )
    for flag in risk.get('financial', []):
        kind = flag.get('kind')
        if kind == 'debt_to_equity_rise':
            items.append(
                'Consider reviewing why debt-to-equity rose from '
                f"{flag['from_value']} in {flag['from_period']} "
                f"to {flag['to_value']} in {flag['to_period']}"
            )
        elif kind == 'margin_fall':
            items.append(
                'Consider reviewing the decline in '
                f"{flag['metric']} from {flag['from_value']} in {flag['from_period']} "
                f"to {flag['to_value']} in {flag['to_period']}"
            )
        elif kind == 'revenue_growth_slowing':
            items.append(
                'Consider reviewing why revenue growth slowed from '
                f"{flag['from_value']} in {flag['from_period']} "
                f"to {flag['to_value']} in {flag['to_period']}"
            )
    return items


def build_report(
    query: str,
    claims: list[Claim],
    risk: dict,
    narrative: str,
    metrics_df,
) -> dict:
    """Assemble a reviewable report. Never includes trading language."""
    return {
        'banner': BANNER,
        'summary': _summary(query, claims, risk, narrative),
        'metrics': _metrics_table(metrics_df),
        'findings': _findings(claims),
        'risks': risk,
        'limitations': _limitations(claims),
        'unresolved_questions': _unresolved_questions(claims),
        'considerations': {
            'title': CONSIDERATIONS_TITLE,
            'items': _consideration_items(claims, risk),
        },
    }


def render_markdown(report: dict) -> str:
    """Render the report as markdown with the review banner first."""
    lines = [report['banner'], '']
    lines.extend(['## Summary', report['summary'], ''])
    lines.append('## Metrics')
    for row in report['metrics']:
        lines.append(f"- {row['metric']} {row['period']}: {row['value']}")
    lines.extend(['', '## Findings'])
    for item in report['findings']:
        lines.append(
            f"- [{item['status']}] {item['claim']} "
            f"(source: {item['source']}, date: {item['date']}; {item['reason']})"
        )
    lines.extend(['', '## Risks'])
    confidence = report['risks']['overall_confidence']
    lines.append(
        f"Overall confidence: {confidence['level']}. {confidence['reason']}"
    )
    lines.append('### Data quality')
    for item in report['risks']['data_quality']:
        lines.append(f"- [{item['status']}] {item['claim']} ({item['reason']})")
    lines.append('### Financial')
    for item in report['risks']['financial']:
        lines.append(f"- {item['note']}")
    lines.append('### Business')
    for item in report['risks']['business']:
        lines.append(
            f"- {item['title']} ({item['source']}, {item['date']}): {item['text']}"
        )
    lines.extend(['', '## Limitations'])
    for item in report['limitations']:
        lines.append(f'- {item}')
    lines.extend(['', '## Unresolved questions'])
    for item in report['unresolved_questions']:
        lines.append(f'- {item}')
    lines.extend(['', f"## {report['considerations']['title']}"])
    for item in report['considerations']['items']:
        lines.append(f'- {item}')
    return '\n'.join(lines)
