"""Immediate, source-grounded PDF results independent of Ollama or embeddings."""
import hashlib
import json
import streamlit as st
from src.documents import extract_pages, analyze
from src.report import BANNER

def document_workspace():
    st.header('Analyze your document')
    st.caption('Extracted evidence with page citations. Statements are reported by the document, not independently verified. No automatic financial-table normalization or OCR.')
    uploaded = st.file_uploader('Upload a PDF or text document', type=['pdf', 'txt'], key='analysis_upload')
    query = st.text_input('Find evidence about a topic (optional)', placeholder='Revenue, debt, liquidity, risks…')
    if uploaded is None:
        st.info('Upload a document to see its text, numeric passages, and matching evidence immediately.')
        return
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    try:
        if st.session_state.get('document_digest') != digest:
            pages = extract_pages(data, uploaded.name)
            st.session_state.document_pages = pages
            st.session_state.document_digest = digest
        pages = st.session_state.document_pages
        result = analyze(pages, query)
    except Exception as exc:
        st.error(str(exc))
        return
    st.success(f"Read {result['pages']} pages and {result['characters']:,} characters from {uploaded.name}.")
    if result['empty_pages']:
        st.warning(f"Pages with no extracted text: {result['empty_pages']}. Images and scanned pages may require OCR.")
    evidence, numeric, preview = st.tabs(['Relevant evidence', 'Numeric passages', 'Full extracted text'])
    with evidence:
        st.caption(f"Showing {len(result['matches'])} of {result['match_count']} matching passages. Matching uses keywords; try synonyms if needed.")
        if not result['matches']:
            st.info('No passages matched this query. Clear the query to browse the document.')
        for row in result['matches']:
            st.markdown(f"**Page {row['page']} — source excerpt**")
            st.text(row['text'])
    with numeric:
        st.caption('Numbers are quoted in context. These are not normalized financial metrics or independently checked calculations.')
        for row in result['numeric_passages']:
            st.markdown(f"**Page {row['page']}**")
            st.text(row['text'])
        if not result['numeric_passages']:
            st.info('No numeric passages found.')
    with preview:
        for page in pages:
            with st.expander(f"Page {page['page']}"):
                st.text(page['text'] or '[No text extracted]')
    report = BANNER + '\n\n# Document evidence report\n' + uploaded.name + '\n\n'
    report += 'Source excerpts only; not independently verified. Query: ' + query + '\n\n'
    report += '\n\n'.join(f"Page {r['page']}\n{r['text']}" for r in result['matches'])
    st.download_button('Download evidence report', report, 'document_evidence.md', 'text/markdown')
    st.download_button('Download extracted pages', json.dumps(pages, indent=2), 'extracted_pages.json', 'application/json')
