"""One LLM wrapper with ollama and cache backends."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import requests

from src import config

OLLAMA_CHAT_URL = 'http://localhost:11434/api/chat'
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


class CacheMissError(Exception):
    """Raised when the cache backend has no saved reply for this prompt."""


def _cache_path() -> str:
    return os.path.join(config.CACHE_DIR, 'llm_cache.json')


def _cache_key(model: str, prompt: str, json_mode: bool, max_tokens: int) -> str:
    blob = json.dumps(
        {
            'model': model,
            'prompt': prompt,
            'json_mode': json_mode,
            'max_tokens': max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _load_cache() -> dict:
    path = _cache_path()
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _save_reply(key: str, reply: str) -> None:
    cache = _load_cache()
    cache[key] = reply
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_cache_path(), 'w', encoding='utf-8') as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def _strip_think(text: str) -> str:
    return THINK_RE.sub('', text).strip()


def _ask_ollama(prompt: str, json_mode: bool, max_tokens: int, model: str) -> str:
    payload: dict = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': False,
        'options': {
            'temperature': 0,
            'num_predict': max_tokens,
        },
    }
    if json_mode:
        payload['format'] = 'json'
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError('Ollama is unreachable. Start Ollama and retry.') from exc
    content = response.json().get('message', {}).get('content', '')
    return _strip_think(content or '')


def ask(
    prompt: str,
    json_mode: bool = False,
    max_tokens: int = 150,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    """Return a model reply, using the configured ollama or cache backend."""
    model_name = model or config.LLM_MODEL
    backend = config.LLM_BACKEND
    key = _cache_key(model_name, prompt, json_mode, max_tokens)

    if backend == 'cache':
        cache = _load_cache()
        if key in cache:
            return cache[key]
        raise CacheMissError(
            'No cached reply for this prompt. Run with LLM_BACKEND=ollama once to warm the cache.'
        )

    if backend != 'ollama':
        raise ValueError(f"Unknown LLM_BACKEND {backend!r}; use 'ollama' or 'cache'.")

    if use_cache:
        cache = _load_cache()
        if key in cache:
            return cache[key]

    reply = _ask_ollama(prompt, json_mode, max_tokens, model_name)
    _save_reply(key, reply)
    return reply


def _parse_support(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(raw)
        if not match:
            return {'supported': False, 'reason': 'unparseable model output'}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {'supported': False, 'reason': 'unparseable model output'}
    if not isinstance(data, dict) or not isinstance(data.get('supported'), bool):
        return {'supported': False, 'reason': 'unparseable model output'}
    return {'supported': data['supported'], 'reason': str(data.get('reason', ''))}


def judge_support(
    claim_text: str,
    passage_text: str,
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> dict:
    """Return whether a passage supports a claim: {supported: bool, reason: str}."""
    prompt = (
        'You check financial claims. The claim is supported only if the passage '
        'states or directly implies it, with the same metric, period and number. '
        'A passage about a different period, metric or number is NOT support. '
        'Answer JSON with keys supported (boolean) and reason (one sentence).\n\n'
        f'Claim: {claim_text}\n'
        f'Passage: {passage_text}'
    )
    raw = ask(prompt, json_mode=True, max_tokens=150, model=model, use_cache=use_cache)
    return _parse_support(raw)


def write_narrative(metrics_text: str) -> str:
    """Explain already-computed numbers in 5 to 7 sentences; do not calculate."""
    prompt = (
        'Explain the following financial numbers in 5 to 7 sentences. '
        'Use ONLY these numbers. Do not calculate, infer, or add any figures.\n\n'
        f'{metrics_text}'
    )
    return ask(prompt, json_mode=False, max_tokens=300)
