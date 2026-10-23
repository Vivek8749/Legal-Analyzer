from legal_analyzer.grounding import match_quote
from legal_analyzer.models import Chunk

CHUNKS = {
    0: Chunk(id=0, text="9.1 Acme may suspend or terminate your account at any time, for any reason or no reason.",
             page_start=2, page_end=2),
    1: Chunk(id=1, text="11.2 ACME’S TOTAL LIABILITY WILL NOT EXCEED TEN U.S. DOLLARS (US $10).", page_start=3, page_end=3),
}


def test_exact_quote_is_verified():
    match = match_quote("Acme may suspend or terminate your account at any time", CHUNKS, [0])
    assert match.verified and match.score == 1.0 and match.chunk_id == 0


def test_case_punctuation_and_curly_quotes_are_ignored():
    match = match_quote("acme's total liability will not exceed ten u.s. dollars", CHUNKS, [1])
    assert match.verified and match.chunk_id == 1


def test_wrong_chunk_id_is_corrected():
    match = match_quote("for any reason or no reason", CHUNKS, [1])
    assert match.verified and match.chunk_id == 0


def test_ellipsis_quote_verified_part_by_part():
    match = match_quote("Acme may suspend or terminate ... for any reason or no reason", CHUNKS, [0])
    assert match.verified


def test_fabricated_quote_is_not_verified():
    match = match_quote("Acme will pay every user one million dollars in damages", CHUNKS, [0, 1])
    assert not match.verified
