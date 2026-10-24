"""Warm the LLM cache for the Acme Corp demo.

Run: python -m scripts.warm_cache
"""

from __future__ import annotations

import json
import os

from run_pipeline import DEFAULT_QUERY, rerun_verification, run
from src import config

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tests',
    'fixtures',
    'ollama_run.json',
)


def _save_offline_fixture(state: dict) -> None:
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    payload = {
        'statuses': {
            item['claim_id']: item['status'] for item in state['claims']
        },
        'markdown': state['markdown'],
    }
    with open(FIXTURE_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _apply_demo_correction(state: dict) -> dict:
    for item in state['claims']:
        if item['claim_id'] != 'n1':
            continue
        item['value'] = 0.11
        item['source_id'] = 'acme_10k_fy2024'
        item['date'] = '2025-02-28'
        item['claim'] = 'Revenue grew 11.0% in FY2024'
    return state


def main() -> int:
    os.environ['LLM_BACKEND'] = 'ollama'
    config.LLM_BACKEND = 'ollama'
    print('Running pipeline with LLM_BACKEND=ollama...')
    state = run(DEFAULT_QUERY)
    _save_offline_fixture(state)
    print(f'Wrote fixture {FIXTURE_PATH}')
    print('Replaying demo correction on n1 (value 0.11, source acme_10k_fy2024)...')
    state = _apply_demo_correction(state)
    rerun_verification(state, state['retriever'], state['metrics_df'])
    cache_path = os.path.join(config.CACHE_DIR, 'llm_cache.json')
    print(f'Cache warmed at {cache_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
