"""LLM wrapper tests; requests.post is monkeypatched so Ollama is not required."""

from __future__ import annotations

import json

import pytest
import requests

from src import config
from src import llm


class FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f'status {self.status_code}')

    def json(self) -> dict:
        return {'message': {'content': self._content}}


def test_ollama_path_posts_payload_strips_think_and_saves_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(config, 'LLM_BACKEND', 'ollama')
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured['url'] = url
        captured['payload'] = json
        captured['timeout'] = timeout
        return FakeResponse(' {"ok": true} <think>hidden</think> leftover ')

    monkeypatch.setattr(llm.requests, 'post', fake_post)
    reply = llm.ask(
        'hello',
        json_mode=True,
        max_tokens=20,
        model='qwen3:4b',
        use_cache=False,
    )
    assert captured['url'] == llm.OLLAMA_CHAT_URL
    payload = captured['payload']
    assert payload['model'] == 'qwen3:4b'
    assert payload['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert payload['stream'] is False
    assert payload['think'] is False
    assert payload['format'] == 'json'
    assert payload['options']['temperature'] == 0
    assert payload['options']['num_predict'] == 20
    assert '<think>' not in reply
    assert 'hidden' not in reply
    cache_file = tmp_path / 'llm_cache.json'
    assert cache_file.is_file()
    saved = json.loads(cache_file.read_text(encoding='utf-8'))
    assert reply in saved.values()


def test_ollama_cache_hit_skips_network(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(config, 'LLM_BACKEND', 'ollama')
    calls = {'n': 0}

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls['n'] += 1
        return FakeResponse('from-network')

    monkeypatch.setattr(llm.requests, 'post', fake_post)
    first = llm.ask('same prompt', model='qwen3:4b', use_cache=True)
    second = llm.ask('same prompt', model='qwen3:4b', use_cache=True)
    assert first == 'from-network'
    assert second == 'from-network'
    assert calls['n'] == 1


def test_cache_backend_raises_cache_miss_error(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(config, 'LLM_BACKEND', 'cache')

    def fail_post(*args, **kwargs):
        raise AssertionError('cache backend must not call Ollama')

    monkeypatch.setattr(llm.requests, 'post', fail_post)
    with pytest.raises(llm.CacheMissError, match='LLM_BACKEND=ollama'):
        llm.ask('uncached prompt')
