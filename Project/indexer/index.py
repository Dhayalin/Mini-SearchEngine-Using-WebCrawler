"""
indexer/index.py
----------------
Builds and queries an inverted index with multi-signal ranking:
  1. TF-IDF   – term relevance within a document
  2. PageRank – structural importance via inbound links
  3. Keyword frequency – raw term count boost
  4. Recency  – newer crawled pages rank slightly higher
"""

import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── Stopwords ──────────────────────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "not", "no", "as", "if", "so",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PageMeta:
    url: str
    title: str
    snippet: str          # first 200 chars of cleaned text
    crawled_at: float
    token_count: int
    inbound_count: int = 0
    pagerank: float = 1.0


@dataclass
class Posting:
    """One entry in a term's postings list."""
    doc_id: int
    tf: float             # term frequency (log-normalised)
    positions: list[int] = field(default_factory=list)


class InvertedIndex:
    """
    In-memory inverted index backed by pickle for persistence.

    index      : term -> {doc_id: Posting}
    docs       : doc_id -> PageMeta
    idf_cache  : term -> IDF score (computed lazily, invalidated on build)
    """

    def __init__(self):
        self.index: dict[str, dict[int, Posting]] = defaultdict(dict)
        self.docs: dict[int, PageMeta] = {}
        self.idf_cache: dict[str, float] = {}
        self._num_docs: int = 0

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, crawled_path: str) -> None:
        """Parse crawled JSON and populate the index."""
        with open(crawled_path, encoding="utf-8") as f:
            pages: list[dict] = json.load(f)

        print(f"🔨 Indexing {len(pages)} pages…")

        # First pass: collect all URLs to compute inbound link counts
        url_to_id: dict[str, int] = {}
        inbound: dict[str, int] = defaultdict(int)

        for i, page in enumerate(pages):
            url_to_id[page["url"]] = i
            for link in page.get("outbound_links", []):
                inbound[link] += 1

        # Second pass: build index
        self.index.clear()
        self.docs.clear()
        self.idf_cache.clear()

        for doc_id, page in enumerate(pages):
            tokens = tokenize(page.get("text", "") + " " + page.get("title", ""))
            if not tokens:
                continue

            # Term frequency (log-normalised)
            freq: dict[str, int] = defaultdict(int)
            positions: dict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokens):
                freq[token] += 1
                positions[token].append(pos)

            # Store metadata
            snippet = page.get("text", "")[:200].strip()
            self.docs[doc_id] = PageMeta(
                url=page["url"],
                title=page.get("title", page["url"]),
                snippet=snippet,
                crawled_at=page.get("crawled_at", 0.0),
                token_count=len(tokens),
                inbound_count=inbound.get(page["url"], 0),
            )

            # Add postings
            for term, count in freq.items():
                tf = 1 + math.log(count)   # log-normalised TF
                self.index[term][doc_id] = Posting(
                    doc_id=doc_id,
                    tf=tf,
                    positions=positions[term],
                )

        self._num_docs = len(self.docs)
        self._compute_pagerank(pages, url_to_id)
        print(f"✅ Index built — {self._num_docs} docs, {len(self.index)} unique terms")

    def _compute_pagerank(
        self,
        pages: list[dict],
        url_to_id: dict[str, int],
        iterations: int = 20,
        damping: float = 0.85,
    ) -> None:
        """Simplified iterative PageRank."""
        n = self._num_docs
        if n == 0:
            return

        rank = {doc_id: 1.0 / n for doc_id in self.docs}

        for _ in range(iterations):
            new_rank: dict[int, float] = {doc_id: (1 - damping) / n for doc_id in self.docs}
            for page in pages:
                src_id = url_to_id.get(page["url"])
                if src_id is None or src_id not in self.docs:
                    continue
                out_links = [
                    url_to_id[u]
                    for u in page.get("outbound_links", [])
                    if u in url_to_id and url_to_id[u] in self.docs
                ]
                if not out_links:
                    # dangling node — distribute evenly
                    share = rank[src_id] / n
                    for doc_id in self.docs:
                        new_rank[doc_id] += damping * share
                else:
                    share = rank[src_id] / len(out_links)
                    for dest_id in out_links:
                        new_rank[dest_id] += damping * share
            rank = new_rank

        for doc_id, pr in rank.items():
            self.docs[doc_id].pagerank = pr

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Multi-signal ranked search.

        Final score = α·tfidf + β·pagerank_norm + γ·freq_norm + δ·recency_norm
        """
        tokens = tokenize(query)
        if not tokens or self._num_docs == 0:
            return []

        # Accumulate TF-IDF scores
        scores: dict[int, float] = defaultdict(float)
        raw_freq: dict[int, int] = defaultdict(int)

        for token in tokens:
            if token not in self.index:
                continue
            idf = self._idf(token)
            for doc_id, posting in self.index[token].items():
                scores[doc_id] += posting.tf * idf
                raw_freq[doc_id] += len(posting.positions)

        if not scores:
            return []

        # Normalise each signal to [0, 1]
        max_tfidf = max(scores.values()) or 1.0
        max_freq = max(raw_freq.values()) or 1.0
        max_pr = max(m.pagerank for m in self.docs.values()) or 1.0
        now = time.time()
        oldest = min(m.crawled_at for m in self.docs.values())
        time_range = (now - oldest) or 1.0

        results = []
        for doc_id, tfidf_score in scores.items():
            meta = self.docs[doc_id]
            tfidf_norm   = tfidf_score / max_tfidf
            freq_norm    = raw_freq[doc_id] / max_freq
            pr_norm      = meta.pagerank / max_pr
            recency_norm = (meta.crawled_at - oldest) / time_range

            # Weighted combination
            final = (
                0.45 * tfidf_norm
                + 0.25 * pr_norm
                + 0.20 * freq_norm
                + 0.10 * recency_norm
            )

            results.append({
                "url": meta.url,
                "title": meta.title,
                "snippet": meta.snippet,
                "score": round(final, 6),
                "tfidf": round(tfidf_norm, 4),
                "pagerank": round(pr_norm, 4),
                "keyword_freq": round(freq_norm, 4),
                "recency": round(recency_norm, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── IDF ────────────────────────────────────────────────────────────────

    def _idf(self, term: str) -> float:
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = len(self.index.get(term, {}))
        idf = math.log((1 + self._num_docs) / (1 + df)) + 1
        self.idf_cache[term] = idf
        return idf

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = "data/index.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 Index saved to {path}")

    @staticmethod
    def load(path: str = "data/index.pkl") -> "InvertedIndex":
        with open(path, "rb") as f:
            idx: InvertedIndex = pickle.load(f)
        print(f"📂 Index loaded — {idx._num_docs} docs, {len(idx.index)} terms")
        return idx

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_documents": self._num_docs,
            "unique_terms": len(self.index),
            "avg_postings_per_term": (
                round(
                    sum(len(v) for v in self.index.values()) / len(self.index), 2
                )
                if self.index
                else 0
            ),
        }


if __name__ == "__main__":
    idx = InvertedIndex()
    idx.build("data/crawled.json")
    idx.save("data/index.pkl")

    results = idx.search("python web scraping")
    for r in results:
        print(f"{r['score']:.4f}  {r['title'][:60]}  {r['url']}")
