from legal_analyzer.models import Chunk
from legal_analyzer.retrieval import BM25, HybridIndex, reciprocal_rank_fusion, tokenize
from legal_analyzer.taxonomy import CATEGORY_BY_ID

from .conftest import HashingEmbedder


def test_tokenize_stems_and_keeps_legal_modals():
    tokens = tokenize("The Company may terminate; you shall not be notified of terminations.")
    assert "may" in tokens and "shall" in tokens and "not" in tokens
    assert tokens.count("termin") == 2  # "terminate" and "terminations" share a stem
    assert "the" not in tokens


def test_bm25_ranks_matching_document_first():
    docs = [tokenize(t) for t in ("fees are non-refundable", "binding arbitration and class action waiver", "governing law")]
    scores = BM25(docs).scores(tokenize("arbitration class action"))
    assert scores.argmax() == 1
    assert scores[0] == 0


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]])
    assert fused[1] == fused[2] > fused[3]


def test_hybrid_search_on_sample(sample_record):
    index = sample_record.index
    top = index.search("mandatory arbitration and class action waiver", k=3)
    assert top[0].chunk.section == "13. DISPUTE RESOLUTION AND ARBITRATION"
    queries = CATEGORY_BY_ID["limitation_of_liability"].queries
    assert index.search(queries, k=3)[0].chunk.section == "11. LIMITATION OF LIABILITY"


def test_semantic_rankings_are_fused_in():
    chunks = [Chunk(id=i, text=t, page_start=1, page_end=1) for i, t in enumerate(
        ["We may end your account whenever we like.", "Prices are listed on the website.", "Notices must be in writing."])]
    index = HybridIndex(chunks, HashingEmbedder().embed_documents([c.text for c in chunks]), HashingEmbedder())
    assert index.uses_embeddings
    hit = index.search("end account", k=1)[0]
    assert hit.chunk.id == 0 and hit.keyword_rank == 1 and hit.semantic_rank == 1
