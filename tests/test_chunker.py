from legal_analyzer.chunker import chunk_document, detect_heading, detect_inline_title, split_sentences
from legal_analyzer.pdf_loader import clean_pages, load_document

from .conftest import SAMPLE_PDF


def test_heading_detection():
    assert detect_heading("12. LIMITATION OF LIABILITY") == (1, "12. LIMITATION OF LIABILITY")
    assert detect_heading("4.2 Payment Terms") == (2, "4.2 Payment Terms")
    assert detect_heading("ARTICLE V - TERMINATION") == (1, "ARTICLE V - TERMINATION")
    assert detect_heading("GOVERNING LAW") == (1, "GOVERNING LAW")
    # sentences and ALL-CAPS disclaimer lines are not headings
    assert detect_heading("1.1 The Company shall provide the services described below.") is None
    assert detect_heading("INCIDENTAL, SPECIAL, CONSEQUENTIAL DAMAGES", at_boundary=False) is None
    assert detect_heading("7. Fees. You must pay all fees when due.") is None


def test_inline_titles():
    assert detect_inline_title("19.1 Entire Agreement. These Terms are the entire agreement.") == (2, "19.1 Entire Agreement")
    assert detect_inline_title("7. Limitation of Liability. In no event will we be liable.") == (1, "7. Limitation of Liability")
    assert detect_inline_title("1.1 The Company shall provide the Services. You agree to pay.") is None


def test_sentence_split_respects_abbreviations():
    text = "Acme Cloud Storage, Inc. is a U.S. company. It may change fees, e.g. monthly. You agree."
    assert split_sentences(text) == [
        "Acme Cloud Storage, Inc. is a U.S. company.",
        "It may change fees, e.g. monthly.",
        "You agree.",
    ]


def test_headers_footers_and_hyphenation_are_cleaned():
    bodies = [
        "The provider may change these terms. Termi-\nnation rights apply and non-\nrefundable fees are due.",
        "Your content is licensed to us.\nWe may share data with partners.",
        "Disputes go to arbitration.\nYou waive class actions.",
    ]
    pages = [f"ACME TERMS OF SERVICE\n{body}\nConfidential - Acme Inc.\nPage {n} of 3" for n, body in enumerate(bodies, 1)]
    cleaned = clean_pages(pages)
    for body_start, text in zip(("The provider", "Your content", "Disputes"), cleaned, strict=True):
        assert text.startswith(body_start)
        assert "ACME TERMS" not in text and "Confidential" not in text and "Page" not in text
    flat = " ".join(cleaned[0].split())
    assert "Termination rights apply and non-refundable fees" in flat


def test_page_tracking_across_page_break():
    pages = [
        "1. TERM\nThis agreement starts today and continues until",
        "either party ends it with thirty days notice.\n2. FEES\nYou pay monthly.",
    ]
    chunks = chunk_document(pages)
    first = chunks[0]
    assert first.section == "1. TERM"
    assert "continues until either party ends it" in first.text
    assert (first.page_start, first.page_end) == (1, 2)
    assert chunks[1].section == "2. FEES" and chunks[1].page_start == 2


def test_long_clause_is_split_with_overlap():
    sentence = "The Supplier shall deliver the goods in accordance with the agreed schedule and specifications. "
    pages = ["5. DELIVERY\n" + sentence * 40]
    chunks = chunk_document(pages, max_chars=600)
    assert len(chunks) > 3
    assert all(len(c.text) <= 600 + 20 for c in chunks)
    assert all(c.section == "5. DELIVERY" for c in chunks)


def test_sample_pdf_chunks_align_with_sections():
    doc = load_document(SAMPLE_PDF)
    assert len(doc.pages) == 4 and not doc.warnings
    chunks = chunk_document(doc.pages)
    sections = [c.section for c in chunks]
    for title in ("11. LIMITATION OF LIABILITY", "13. DISPUTE RESOLUTION AND ARBITRATION", "15. TIME LIMIT ON CLAIMS"):
        assert title in sections
    liability = next(c for c in chunks if c.section == "11. LIMITATION OF LIABILITY")
    assert "TEN U.S. DOLLARS (US $10)" in liability.text
    assert [c.id for c in chunks] == list(range(len(chunks)))
