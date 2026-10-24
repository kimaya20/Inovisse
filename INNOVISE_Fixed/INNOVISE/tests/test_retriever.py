"""Retriever tests: ranking, source filter, dates, and on-the-fly indexing."""

from __future__ import annotations

import pytest

from src.config import DOCS_DIR, TOP_K
from src.retriever import Retriever


@pytest.fixture(scope='module')
def retriever() -> Retriever:
    loaded = Retriever()
    loaded.load_from_metadata(DOCS_DIR)
    return loaded


def test_revenue_growth_query_returns_10k_in_top_3(retriever: Retriever):
    hits = retriever.search('revenue growth fiscal 2024', k=TOP_K)
    assert any(hit['source_id'] == 'acme_10k_fy2024' for hit in hits)


def test_source_id_filter_returns_only_press_release(retriever: Retriever):
    hits = retriever.search(
        'revenue growth fiscal 2024',
        source_id='press_release_fy2024',
    )
    assert hits
    assert all(hit['source_id'] == 'press_release_fy2024' for hit in hits)


def test_every_result_has_a_non_empty_date(retriever: Retriever):
    hits = retriever.search('revenue growth fiscal 2024')
    assert hits
    assert all(hit.get('date') for hit in hits)


def test_add_document_makes_new_text_searchable(retriever: Retriever):
    retriever.add_document(
        'The tungsten sprocket backlog reached 42 units in lunar fiscal 2099.',
        source_id='upload_sprocket',
        title='Uploaded note',
        date='2026-10-24',
        doc_type='filing',
    )
    hits = retriever.search(
        'tungsten sprocket backlog lunar 2099',
        source_id='upload_sprocket',
    )
    assert any('tungsten sprocket' in hit['text'].lower() for hit in hits)
    assert all(hit['date'] == '2026-10-24' for hit in hits)
