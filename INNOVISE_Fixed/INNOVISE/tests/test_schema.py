from src.schema import Claim, UNVERIFIED


def test_claim_to_dict_from_dict_round_trip():
    original = Claim(
        claim_id='c1',
        claim='Revenue grew 18% in FY2024',
        metric='revenue_growth',
        period='FY2024',
        value=0.18,
        source_id='press_release_2024',
        page_section='p.1',
        date='2025-02-10',
        status=UNVERIFIED,
        evidence=[{'source_id': 'press_release_2024', 'passage': 'Revenue grew 18%.'}],
        reason='',
        trail=[{
            'agent': 'analyst',
            'action': 'created',
            'timestamp': '2026-01-01T00:00:00+00:00',
        }],
    )
    restored = Claim.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
