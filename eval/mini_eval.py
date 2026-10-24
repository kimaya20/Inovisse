"""Compare local models on claim-support cases and sweep the similarity threshold."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import requests

from src import config
from src.llm import judge_support

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(EVAL_DIR, 'cases.json')
RESULTS_PATH = os.path.join(EVAL_DIR, 'results.json')
OLLAMA_TAGS_URL = 'http://localhost:11434/api/tags'
DEFAULT_MODELS = ['qwen3:4b', 'qwen3:8b', 'gemma4:e2b']
THRESHOLDS = [i / 100 for i in range(10, 75, 5)]
CATEGORIES = ['normal', 'near_miss', 'planted']


def load_cases() -> list[dict]:
    with open(CASES_PATH, encoding='utf-8') as handle:
        payload = json.load(handle)
    return list(payload['cases'])


def installed_models() -> set[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return set()
    names: set[str] = set()
    for item in response.json().get('models', []):
        name = item.get('name') or item.get('model') or ''
        if name:
            names.add(name)
            names.add(name.split(':')[0])
    return names


def model_is_installed(model: str, installed: set[str]) -> bool:
    if model in installed:
        return True
    return any(name == model or name.startswith(model + ':') for name in installed)


def gold_supported(label: str) -> bool:
    return label == 'supported'


def accuracy(rows: list[tuple[bool, bool]]) -> float:
    if not rows:
        return 0.0
    return sum(pred == gold for pred, gold in rows) / len(rows)


def eval_model(model: str, cases: list[dict]) -> dict:
    by_category: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    overall: list[tuple[bool, bool]] = []
    elapsed = 0.0
    for case in cases:
        started = time.perf_counter()
        result = judge_support(
            case['claim'],
            case['passage'],
            model=model,
            use_cache=False,
        )
        elapsed += time.perf_counter() - started
        pair = (bool(result['supported']), gold_supported(case['label']))
        overall.append(pair)
        by_category[case['category']].append(pair)
    n = len(cases)
    return {
        'skipped': False,
        'n': n,
        'accuracy_overall': accuracy(overall),
        'accuracy_by_category': {
            category: accuracy(by_category.get(category, []))
            for category in CATEGORIES
        },
        'avg_seconds': elapsed / n if n else 0.0,
    }


def print_model_table(results: dict[str, dict]) -> None:
    header = f"{'model':<16} {'overall':>8} {'normal':>8} {'near_miss':>10} {'planted':>8} {'sec/call':>8}"
    print(header)
    print('-' * len(header))
    for model, row in results.items():
        if row.get('skipped'):
            print(f'{model:<16} skipped ({row.get("reason", "not installed")})')
            continue
        cats = row['accuracy_by_category']
        print(
            f'{model:<16} {row["accuracy_overall"]:>8.2%} {cats["normal"]:>8.2%} '
            f'{cats["near_miss"]:>10.2%} {cats["planted"]:>8.2%} {row["avg_seconds"]:>8.1f}'
        )


def precision_recall_f1(preds: list[bool], golds: list[bool]) -> dict:
    tp = sum(pred and gold for pred, gold in zip(preds, golds))
    fp = sum(pred and not gold for pred, gold in zip(preds, golds))
    fn = sum(not pred and gold for pred, gold in zip(preds, golds))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def threshold_sweep(cases: list[dict]) -> tuple[list[dict], float]:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    claim_emb = embedder.encode(
        [case['claim'] for case in cases],
        normalize_embeddings=True,
    )
    passage_emb = embedder.encode(
        [case['passage'] for case in cases],
        normalize_embeddings=True,
    )
    scores = np.sum(claim_emb * passage_emb, axis=1)
    golds = [gold_supported(case['label']) for case in cases]
    rows = []
    for threshold in THRESHOLDS:
        preds = [float(score) >= threshold for score in scores]
        metrics = precision_recall_f1(preds, golds)
        rows.append({'threshold': threshold, **metrics})
    best = max(rows, key=lambda row: (row['f1'], -abs(row['threshold'] - config.SIM_THRESHOLD)))
    return rows, best['threshold']


def print_threshold_table(rows: list[dict], recommended: float) -> None:
    header = f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}"
    print(header)
    print('-' * len(header))
    for row in rows:
        mark = ' *' if row['threshold'] == recommended else ''
        print(
            f"{row['threshold']:>10.2f} {row['precision']:>10.3f} "
            f"{row['recall']:>10.3f} {row['f1']:>10.3f}{mark}"
        )
    print(f'Recommended SIM_THRESHOLD (best F1, rough with 20 cases): {recommended:.2f}')
    print('Do not edit config.py automatically; set SIM_THRESHOLD by hand after review.')


def retrieval_scores_for_supported(cases: list[dict]) -> list[dict] | None:
    try:
        from src import retriever as retriever_mod
    except ImportError:
        return None
    search = getattr(retriever_mod, 'search', None) or getattr(retriever_mod, 'retrieve', None)
    if not callable(search):
        print('Retriever not implemented; skipping retrieval scores.')
        return None
    rows = []
    for case in cases:
        if case['label'] != 'supported':
            continue
        hits = search(case['claim'])
        top = []
        for hit in list(hits)[: config.TOP_K]:
            if isinstance(hit, dict):
                top.append(hit.get('score', hit))
            else:
                top.append(hit)
        rows.append({'id': case['id'], 'top_scores': top})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Mini eval for claim-support models.')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases()
    config.LLM_BACKEND = 'ollama'
    installed = installed_models()
    model_results: dict[str, dict] = {}
    if not installed:
        print('Warning: Ollama is not reachable; skipping all models.', file=sys.stderr)
    for model in args.models:
        if not model_is_installed(model, installed):
            print(f'Warning: skipping {model}; not installed.', file=sys.stderr)
            model_results[model] = {
                'skipped': True,
                'reason': 'not installed',
            }
            continue
        print(f'Evaluating {model}...')
        model_results[model] = eval_model(model, cases)
    print()
    print_model_table(model_results)

    sweep_rows: list[dict] = []
    recommended = config.SIM_THRESHOLD
    try:
        sweep_rows, recommended = threshold_sweep(cases)
        print()
        print_threshold_table(sweep_rows, recommended)
    except Exception as exc:
        print(f'Warning: similarity threshold sweep skipped ({exc}).', file=sys.stderr)

    retrieval = retrieval_scores_for_supported(cases)
    results = {
        'models': model_results,
        'threshold_sweep': sweep_rows,
        'recommended_sim_threshold': recommended,
        'retrieval_scores': retrieval,
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print(f'\nWrote {RESULTS_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
