"""Claim schema: the contract between modules. Percentages are fractions (0.18 = 18%)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


SUPPORTED = 'supported'
CONTRADICTED = 'contradicted'
CONFLICT = 'conflict'
UNSUPPORTED = 'unsupported'
OUTDATED = 'outdated'
UNVERIFIED = 'unverified'


@dataclass
class Claim:
    claim_id: str
    claim: str
    metric: str
    period: str
    value: Optional[float] = None
    source_id: Optional[str] = None
    page_section: str = ''
    date: Optional[str] = None  # ISO string
    status: str = UNVERIFIED
    evidence: list[dict] = field(default_factory=list)
    reason: str = ''
    trail: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Claim:
        return cls(
            claim_id=data['claim_id'],
            claim=data['claim'],
            metric=data['metric'],
            period=data['period'],
            value=data.get('value'),
            source_id=data.get('source_id'),
            page_section=data.get('page_section', ''),
            date=data.get('date'),
            status=data.get('status', UNVERIFIED),
            evidence=list(data.get('evidence') or []),
            reason=data.get('reason', ''),
            trail=list(data.get('trail') or []),
        )

    def add_trail(self, agent: str, action: str) -> None:
        self.trail.append({
            'agent': agent,
            'action': action,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
