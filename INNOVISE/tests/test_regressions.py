import io
import pandas as pd
import pytest
from src.documents import extract_pages, analyze
from src.ratios import compute_all
from src.schema import Claim
from src.verifier import verify_all

class Retriever:
    def search(self, *a, **kw):
        return [dict(source_id='filing', page_section='Page 1', date='2025-02-28',
                     text='Net margin was 10% in FY2024.', score=1)] if kw.get('source_id') == 'filing' else []

def check(text, source):
    claim = Claim('one', text, 'net_margin', 'FY2024', .1, source, date='2025-02-28')
    metrics = pd.DataFrame([dict(metric='net_margin',period='FY2024',value=.1)])
    verify_all([claim], metrics, Retriever(), judge=lambda *a: {'supported': True})
    return claim

def test_no_source_cannot_pass():
    assert check('Net margin was 10% in FY2024', None).status == 'unsupported'
def test_text_number_mismatch():
    assert check('Net margin was 90% in FY2024', 'filing').status == 'contradicted'
def test_correct_claim():
    assert check('Net margin was 10% in FY2024', 'filing').status == 'supported'
def test_zero_denominators():
    df = pd.DataFrame([dict(period='FY2024',revenue=0, operating_income=1, net_income=1,total_debt=1,total_equity=0)]).set_index('period')
    assert set(compute_all(df).metric) == {'revenue','operating_income','net_income','total_debt','total_equity'}
def test_text_results_and_query():
    pages = extract_pages(b'Revenue was 100. Debt was 20.', 'report.txt')
    assert analyze(pages, 'debt')['matches'][0]['text'] == 'Debt was 20.'
    assert not analyze(pages, 'unrelated')['matches']
def test_empty_text():
    with pytest.raises(ValueError, match='OCR'):
        extract_pages(b'  ', 'report.txt')
def test_blank_pdf():
    from pypdf import PdfWriter
    writer = PdfWriter(); writer.add_blank_page(width=100,height=100)
    stream = io.BytesIO(); writer.write(stream)
    with pytest.raises(ValueError, match='OCR'):
        extract_pages(stream.getvalue(), 'scan.pdf')
def test_pdf_pages():
    from reportlab.pdfgen import canvas
    stream = io.BytesIO(); doc = canvas.Canvas(stream)
    doc.drawString(50,700,'Revenue 100'); doc.showPage()
    doc.drawString(50,700,'Debt 20'); doc.save()
    pages = extract_pages(stream.getvalue(), 'report.pdf')
    assert len(pages) == 2
    assert analyze(pages,'Debt')['matches'][0]['page'] == 2

def test_ui_starts_without_model():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(__import__('pathlib').Path(__file__).resolve().parents[1] / 'app.py').run()
    assert not app.exception
    assert app.sidebar.radio[0].value == 'Uploaded document analysis'
