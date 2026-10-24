# How to run:
#   $env:LLM_BACKEND="cache"; streamlit run app.py
#   $env:LLM_BACKEND="ollama"; streamlit run app.py
#   After the first MiniLM download: $env:HF_HUB_OFFLINE="1"; streamlit run app.py

"""Streamlit review UI. No trading or execution functions."""

from __future__ import annotations

import json
import os
import re

import pandas as pd
import streamlit as st

from run_pipeline import DEFAULT_QUERY, rerun_verification, run, save_state
from src import config
from src.audit import get_log, log_action
from src.llm import CacheMissError
from src.report import BANNER
from src.schema import SUPPORTED, Claim

STATUS_COLORS = {
    'supported': '#0a7a3e',
    'conflict': '#c45c00',
    'contradicted': '#b00020',
    'unsupported': '#7b1fa2',
    'outdated': '#5f6368',
    'unverified': '#80868b',
}


def _init_session() -> None:
    defaults = {
        'analyst': '',
        'query': DEFAULT_QUERY,
        'state': None,
        'retriever': None,
        'metrics_df': None,
        'decisions': {},
        'uploads': [],
        'source_dates': {},
        'approved': False,
        'selected_claim_id': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_source_dates() -> dict[str, str]:
    dates = dict(st.session_state.source_dates)
    path = os.path.join(config.DOCS_DIR, 'metadata.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            for entry in json.load(handle):
                dates[entry['source_id']] = entry['date']
    retriever = st.session_state.retriever
    if retriever is not None:
        for chunk in retriever.chunks:
            dates.setdefault(chunk['source_id'], chunk['date'])
    return dates


def _source_ids() -> list[str]:
    ids = list(_load_source_dates().keys())
    return sorted(set(ids))


def _claims() -> list[Claim]:
    state = st.session_state.state
    if not state:
        return []
    return [Claim.from_dict(item) for item in state['claims']]


def _flagged_claims() -> list[Claim]:
    return [claim for claim in _claims() if claim.status != SUPPORTED]


def _unresolved_flagged() -> list[Claim]:
    decisions = st.session_state.decisions
    return [
        claim for claim in _flagged_claims()
        if claim.claim_id not in decisions
    ]


def _status_label(status: str) -> str:
    color = STATUS_COLORS.get(status, '#444')
    return (
        f"<span style='background:{color};color:white;padding:0.15rem 0.5rem;"
        f"border-radius:0.25rem;font-size:0.85rem'>{status}</span>"
    )


def _show_pipeline_error(exc: Exception) -> None:
    message = str(exc)
    if isinstance(exc, CacheMissError):
        st.warning('Not in the offline cache: run once with LLM_BACKEND=ollama')
        return
    if 'Ollama is unreachable' in message:
        st.error('Ollama is unreachable. Start Ollama, then run again.')
        return
    st.error(message)


def _store_runtime(state: dict) -> None:
    st.session_state.retriever = state.pop('retriever', st.session_state.retriever)
    st.session_state.metrics_df = state.pop('metrics_df', st.session_state.metrics_df)
    st.session_state.state = state
    st.session_state.approved = False
    st.session_state.decisions = {}


def _apply_uploads(retriever) -> None:
    for item in st.session_state.uploads:
        if item['applied']:
            continue
        if item['kind'] == 'pdf':
            retriever.add_pdf(
                item['path'],
                item['source_id'],
                item['title'],
                item['date'],
                item['doc_type'],
            )
        else:
            retriever.add_document(
                item['text'],
                item['source_id'],
                item['title'],
                item['date'],
                item['doc_type'],
            )
        item['applied'] = True
        st.session_state.source_dates[item['source_id']] = item['date']


def _run_query(query: str) -> None:
    try:
        with st.spinner('Running pipeline...'):
            state = run(query)
            retriever = state['retriever']
            for item in st.session_state.uploads:
                item['applied'] = False
            had_pending = bool(st.session_state.uploads)
            _apply_uploads(retriever)
            if had_pending:
                state = rerun_verification(state, retriever, state['metrics_df'])
            _store_runtime(state)
            st.session_state.decisions = {}
            st.session_state.selected_claim_id = None
    except Exception as exc:
        _show_pipeline_error(exc)


def _slug(name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', name.strip().lower()).strip('_')
    return slug or 'uploaded_source'


def _handle_upload(uploaded, source_name: str, pub_date: str) -> None:
    if uploaded is None or not source_name.strip() or not pub_date:
        st.warning('Source name, publication date and a file are required to index an upload.')
        return
    try:
        __import__('datetime').date.fromisoformat(pub_date)
    except ValueError:
        st.error('Enter a valid date in YYYY-MM-DD format.')
        return
    source_id = _slug(source_name) + '_' + __import__('uuid').uuid4().hex[:10]
    name = uploaded.name.lower()
    record = {
        'source_id': source_id,
        'title': source_name.strip(),
        'date': pub_date,
        'doc_type': 'filing',
        'applied': False,
        'kind': 'txt',
        'text': '',
        'path': '',
    }
    if name.endswith('.pdf'):
        folder = os.path.join(config.CACHE_DIR, 'uploads')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, source_id + '.pdf')
        with open(path, 'wb') as handle:
            handle.write(uploaded.getvalue())
        record['kind'] = 'pdf'
        record['path'] = path
    else:
        record['text'] = uploaded.getvalue().decode('utf-8', errors='replace')
    st.session_state.uploads.append(record)
    retriever = st.session_state.retriever
    if retriever is not None and st.session_state.state is not None:
        try:
            _apply_uploads(retriever)
            state = rerun_verification(
                st.session_state.state, retriever, st.session_state.metrics_df
            )
            _store_runtime(state)
            st.success(f'Indexed {source_id} and re-verified all claims.')
        except Exception as exc:
            _show_pipeline_error(exc)
    else:
        st.info('Upload saved. It will be indexed on the next Run.')


def _log(claim_id, action, old_value, new_value, note: str) -> None:
    state = st.session_state.state
    log_action(
        report_id=state['report_id'],
        claim_id=claim_id,
        action=action,
        old_value=old_value,
        new_value=new_value,
        note=note,
        analyst=st.session_state.analyst.strip(),
    )


def _sidebar() -> None:
    st.sidebar.header('Reviewer')
    st.sidebar.text_input(
        'Analyst name (required for review actions)',
        key='analyst',
    )
    st.sidebar.text_area('Query', key='query', height=80)
    if st.sidebar.button('Run', type='primary'):
        _run_query(st.session_state.query.strip() or DEFAULT_QUERY)
    st.sidebar.markdown('---')
    st.sidebar.subheader('Upload a document')
    source_name = st.sidebar.text_input('Source name')
    pub_date = st.sidebar.text_input('Publication date (YYYY-MM-DD)')
    uploaded = st.sidebar.file_uploader('PDF or text filing', type=['pdf', 'txt'])
    if st.sidebar.button('Index upload'):
        _handle_upload(uploaded, source_name, pub_date)


def _evidence_panel(claim: Claim) -> None:
    st.subheader('Evidence')
    if not claim.evidence:
        st.info('no evidence found')
    else:
        for item in claim.evidence:
            st.markdown(
                f"**Source:** {item.get('source_id') or '—'}  \n"
                f"**Page / section:** {item.get('page_section') or '—'}  \n"
                f"**Date:** {item.get('date') or '—'}  \n"
                f"**Score:** {item.get('score', '—')}"
            )
            st.text(item.get('text') or 'no evidence found')
    st.subheader('Handoff trail')
    if not claim.trail:
        st.caption('No trail yet.')
        return
    for event in claim.trail:
        st.markdown(
            f"- **{event.get('agent')}** — {event.get('action')}  \n"
            f"  `{event.get('timestamp')}`"
        )


def _review_actions(claim: Claim) -> None:
    st.subheader('Review actions')
    analyst = st.session_state.analyst.strip()
    if not analyst:
        st.warning('Enter an analyst name in the sidebar before Accept, Correct or Dismiss.')
        return
    none_label = '(none)'
    sources = [none_label] + _source_ids()
    current_source = claim.source_id or none_label
    if current_source not in sources:
        sources.append(current_source)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button('Accept as is', key=f'accept_{claim.claim_id}'):
            st.session_state.approved = False
            st.session_state.decisions[claim.claim_id] = 'accept'
            claim.add_trail('Analyst', 'accepted as is')
            _replace_claim(claim)
            save_state(st.session_state.state)
            _log(claim.claim_id, 'accept', claim.status, claim.status, 'accepted as is')
            st.success('Accepted.')
    with col_c:
        dismiss_note = st.text_input('Dismiss note', key=f'dismiss_note_{claim.claim_id}')
        if st.button('Dismiss', key=f'dismiss_{claim.claim_id}'):
            if not dismiss_note.strip():
                st.warning('A note is required to dismiss.')
            else:
                st.session_state.approved = False
                st.session_state.decisions[claim.claim_id] = 'dismiss: ' + dismiss_note.strip()
                claim.add_trail('Analyst', f'dismissed: {dismiss_note.strip()}')
                _replace_claim(claim)
                save_state(st.session_state.state)
                _log(claim.claim_id, 'dismiss', claim.status, claim.status, dismiss_note.strip())
                st.success('Dismissed.')

    st.markdown('**Correct**')
    new_text = st.text_area('Claim text', value=claim.claim, key=f'text_{claim.claim_id}')
    value_default = '' if claim.value is None else str(claim.value)
    new_value_raw = st.text_input('Value (fraction; blank for none)', value=value_default, key=f'val_{claim.claim_id}')
    new_source = st.selectbox(
        'source_id',
        options=sources,
        index=sources.index(current_source) if current_source in sources else 0,
        key=f'src_{claim.claim_id}',
    )
    if st.button('Apply correction and re-verify all claims', key=f'correct_{claim.claim_id}'):
        try:
            new_value = None if new_value_raw.strip() == '' else float(new_value_raw)
        except ValueError:
            st.error('Value must be a number or blank.')
            return
        if new_value is not None and not __import__('math').isfinite(new_value):
            st.error('Value must be finite.')
            return
        old = {'claim': claim.claim, 'value': claim.value, 'source_id': claim.source_id}
        claim.claim = new_text
        claim.value = new_value
        claim.source_id = None if new_source == none_label else new_source
        dates = _load_source_dates()
        if claim.source_id and claim.source_id in dates:
            claim.date = dates[claim.source_id]
        claim.add_trail('Analyst', 'corrected claim fields')
        _replace_claim(claim)
        st.session_state.decisions[claim.claim_id] = 'correct'
        _log(claim.claim_id, 'correct', old, {'claim': claim.claim, 'value': claim.value, 'source_id': claim.source_id}, 'corrected')
        try:
            with st.spinner('Re-verifying all claims...'):
                state = rerun_verification(
                    st.session_state.state,
                    st.session_state.retriever,
                    st.session_state.metrics_df,
                )
                _store_runtime(state)
            st.success('Re-verification complete. Conflicts and confidence were refreshed.')
            st.rerun()
        except Exception as exc:
            _show_pipeline_error(exc)


def _replace_claim(updated: Claim) -> None:
    claims = st.session_state.state['claims']
    for index, item in enumerate(claims):
        if item['claim_id'] == updated.claim_id:
            claims[index] = updated.to_dict()
            break


def _flagged_tab() -> None:
    flagged = _flagged_claims()
    if not flagged:
        st.info('No flagged claims.')
        return
    for claim in flagged:
        st.markdown(
            f"**{claim.claim_id}** {_status_label(claim.status)} — {claim.reason}",
            unsafe_allow_html=True,
        )
        st.caption(claim.claim)
    ids = [claim.claim_id for claim in flagged]
    selected = st.selectbox('Select a claim', options=ids, key='flagged_select')
    st.session_state.selected_claim_id = selected
    chosen = next(claim for claim in flagged if claim.claim_id == selected)
    _evidence_panel(chosen)
    _review_actions(chosen)


def _report_tab() -> None:
    state = st.session_state.state
    st.markdown(state.get('markdown') or '')
    if st.session_state.approved:
        analyst = st.session_state.analyst.strip() or 'analyst'
        labelled = f"Approved by analyst: {st.session_state.get('approved_by', analyst)}\n\n{state.get('markdown') or ''}\n\n## Review decisions\n" + '\n'.join(f'- {k}: {v}' for k, v in st.session_state.decisions.items())
        st.download_button(
            'Download report markdown (Approved by analyst)',
            data=labelled,
            file_name='approved_report.md',
            mime='text/markdown',
        )


def _audit_tab() -> None:
    state = st.session_state.state
    rows = get_log(report_id=state['report_id'] if state else None)
    if not rows:
        st.caption('No audit entries yet.')
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _decision_bar() -> None:
    state = st.session_state.state
    if not state:
        return
    st.markdown('---')
    st.subheader('Report decision')
    feedback = st.text_area('Feedback', key='report_feedback')
    unresolved = _unresolved_flagged()
    analyst_ok = bool(st.session_state.analyst.strip())
    if unresolved:
        st.caption(
            'Approve is disabled until every remaining flagged claim has a decision '
            f"({', '.join(c.claim_id for c in unresolved)})."
        )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            'Approve report',
            disabled=(not analyst_ok) or bool(unresolved),
        ):
            _log(None, 'approve_report', None, 'approved', feedback)
            st.session_state.approved_by = st.session_state.analyst.strip()
            st.session_state.approved = True
            st.success('Report approved.')
    with col_b:
        if st.button('Reject report', disabled=not analyst_ok):
            _log(None, 'reject_report', None, 'rejected', feedback)
            st.session_state.approved = False
            st.warning('Report rejected.')


def main() -> None:
    st.set_page_config(page_title='Acme research review', layout='wide')
    _init_session()
    st.markdown(f'## {BANNER}')
    mode = st.sidebar.radio('Workspace', ['Uploaded document analysis', 'Acme synthetic demonstration'])
    if mode == 'Uploaded document analysis':
        from src.document_ui import document_workspace
        document_workspace()
        return
    st.caption('Fixed Acme FY2024 demonstration. The query does not select another company.')
    _sidebar()
    if st.session_state.state is None:
        st.info('Enter a query and click Run to generate a report for review.')
        return
    tabs = st.tabs(['Report', 'Flagged claims', 'Audit log'])
    with tabs[0]:
        _report_tab()
    with tabs[1]:
        _flagged_tab()
    with tabs[2]:
        _audit_tab()
    _decision_bar()


if __name__ == '__main__':
    main()
