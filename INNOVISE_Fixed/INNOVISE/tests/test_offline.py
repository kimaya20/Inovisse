"""Offline cache-mode pipeline should match a saved ollama run."""

from __future__ import annotations

import json
import os
import re

import pytest

from run_pipeline import DEFAULT_QUERY, run
from src import config

CACHE_PATH = os.path.join(config.CACHE_DIR, 'llm_cache.json')
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'ollama_run.json'
)
TIMESTAMP_RE = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?'
)


def _strip_timestamps(text: str) -> str:
    return TIMESTAMP_RE.sub('<TIMESTAMP>', text)


@pytest.mark.skipif(
    not os.path.isfile(CACHE_PATH),
    reason='LLM cache missing; run python -m scripts.warm_cache',
)
def test_cache_pipeline_matches_ollama_fixture(monkeypatch):
    if not os.path.isfile(FIXTURE_PATH):
        pytest.skip('ollama fixture missing; run python -m scripts.warm_cache')
    monkeypatch.setattr(config, 'LLM_BACKEND', 'cache')
    with open(FIXTURE_PATH, encoding='utf-8') as handle:
        fixture = json.load(handle)
    state = run(DEFAULT_QUERY)
    statuses = {item['claim_id']: item['status'] for item in state['claims']}
    assert statuses == fixture['statuses']
    assert _strip_timestamps(state['markdown']) == _strip_timestamps(fixture['markdown'])
