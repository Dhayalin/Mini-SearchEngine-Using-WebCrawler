"""
tests/test_index.py
-------------------
Unit tests for the InvertedIndex: tokenization, build, search, scoring.
"""

import json
import os
import tempfile
import time
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.index import InvertedIndex, tokenize


# ── tokenize ──────────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = tokenize("Hello, World!")
    assert "hello" in tokens
    assert "world" in tokens

def test_tokenize_removes_stopwords():
    tokens = tokenize("this is a test")
    assert "this" not in tokens
    assert "is" not in tokens
    assert "test" in tokens

def test_tokenize_removes_short():
    tokens = tokenize("a I x python")
    assert "a" not in tokens
    assert "python" in tokens


# ── build & search ────────────────────────────────────────────────────────

SAMPLE_PAGES = [
    {
        "url": "https://example.com/python",
        "title": "Python Programming",
        "text": "Python is a high-level programming language used for web development and data science.",
        "outbound_links": ["https://example.com/scrapy"],
        "crawled_at": time.time() - 100,
    },
    {
        "url": "https://example.com/scrapy",
        "title": "Scrapy Web Scraping",
        "text": "Scrapy is a Python framework for large-scale web scraping and crawling of websites.",
        "outbound_links": ["https://example.com/python"],
        "crawled_at": time.time() - 50,
    },
    {
        "url": "https://example.com/fastapi",
        "title": "FastAPI REST API",
        "text": "FastAPI is a modern Python web framework for building APIs quickly.",
        "outbound_links": ["https://example.com/python"],
        "crawled_at": time.time(),
    },
]


@pytest.fixture
def built_index():
    idx = InvertedIndex()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_PAGES, f)
        tmp = f.name
    idx.build(tmp)
    os.unlink(tmp)
    return idx


def test_build_doc_count(built_index):
    assert built_index._num_docs == 3

def test_build_terms_exist(built_index):
    assert "python" in built_index.index
    assert "scrapy" in built_index.index

def test_search_returns_results(built_index):
    results = built_index.search("python")
    assert len(results) > 0

def test_search_relevance_order(built_index):
    results = built_index.search("scrapy crawling")
    urls = [r["url"] for r in results]
    # Scrapy page should rank highest for "scrapy crawling"
    assert "https://example.com/scrapy" in urls[:2]

def test_search_empty_query(built_index):
    results = built_index.search("")
    assert results == []

def test_search_unknown_term(built_index):
    results = built_index.search("xyznotexist")
    assert results == []

def test_scores_normalised(built_index):
    results = built_index.search("python web")
    for r in results:
        assert 0.0 <= r["score"] <= 1.0
        assert 0.0 <= r["tfidf"] <= 1.0
        assert 0.0 <= r["pagerank"] <= 1.0

def test_result_has_required_fields(built_index):
    results = built_index.search("python")
    for r in results:
        for field in ["url", "title", "snippet", "score", "tfidf", "pagerank", "keyword_freq", "recency"]:
            assert field in r, f"Missing field: {field}"

def test_top_k_respected(built_index):
    results = built_index.search("python", top_k=1)
    assert len(results) <= 1


# ── persistence ───────────────────────────────────────────────────────────

def test_save_and_load(built_index):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_index.pkl")
        built_index.save(path)
        loaded = InvertedIndex.load(path)
        assert loaded._num_docs == built_index._num_docs
        results = loaded.search("python")
        assert len(results) > 0


# ── stats ─────────────────────────────────────────────────────────────────

def test_stats(built_index):
    s = built_index.stats()
    assert s["total_documents"] == 3
    assert s["unique_terms"] > 0
    assert s["avg_postings_per_term"] > 0
