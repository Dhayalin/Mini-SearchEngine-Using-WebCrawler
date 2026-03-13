# 🔍 Mini Search Engine

A fully functional, Python-based mini search engine that **crawls**, **indexes**, and **retrieves** web pages with multi-signal ranking.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Scrapy](https://img.shields.io/badge/Scrapy-2.11-60A839?logo=scrapy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Component | Technology | What it does |
|-----------|-----------|--------------|
| **Web Crawler** | Scrapy + BeautifulSoup | Crawls a single domain, extracts text & hyperlinks |
| **Inverted Index** | Pure Python | Maps terms → documents for fast keyword lookup |
| **Ranking Engine** | TF-IDF + PageRank + Frequency + Recency | Multi-signal scoring for relevant results |
| **Search API** | FastAPI | REST endpoints for querying and crawl management |
| **Frontend UI** | HTML / JS | Google-like search interface served by FastAPI |

---

## 🏗️ Architecture

```
mini-search-engine/
├── crawler/
│   └── spider.py          # Scrapy spider + BeautifulSoup extraction
├── indexer/
│   └── index.py           # Inverted index, TF-IDF, PageRank, persistence
├── api/
│   └── main.py            # FastAPI app (search, crawl, stats endpoints)
├── static/
│   └── index.html         # Frontend search UI
├── tests/
│   └── test_index.py      # Unit tests (pytest)
├── build_index.py         # CLI: crawl a URL and build index
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/yourusername/mini-search-engine.git
cd mini-search-engine
pip install -r requirements.txt
```

### 2. Crawl a website and build the index

```bash
# Crawl https://books.toscrape.com (safe demo site, no auth needed)
python build_index.py https://books.toscrape.com --max-pages 50
```

This will:
- 🕷️  Crawl up to 50 pages within the domain
- 💾  Save raw data to `data/crawled.json`
- 📑  Build the inverted index to `data/index.pkl`

### 3. Start the API + UI

```bash
uvicorn api.main:app --reload
```

Open **http://localhost:8000** to use the search UI, or visit **http://localhost:8000/docs** for the interactive API docs.

---

## 🔌 API Reference

### `GET /search`

Search the index and return ranked results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | required | Search query |
| `limit` | int | 10 | Max results (1–50) |

**Example:**
```bash
curl "http://localhost:8000/search?q=python+web+scraping&limit=5"
```

**Response:**
```json
{
  "query": "python web scraping",
  "total_results": 5,
  "search_time_ms": 1.24,
  "results": [
    {
      "url": "https://example.com/page",
      "title": "Web Scraping with Python",
      "snippet": "Python is widely used for web scraping...",
      "score": 0.823,
      "tfidf": 0.91,
      "pagerank": 0.67,
      "keyword_freq": 0.80,
      "recency": 0.55
    }
  ]
}
```

---

### `POST /crawl`

Trigger a new crawl and rebuild the index in the background.

```bash
curl -X POST http://localhost:8000/crawl \
     -H "Content-Type: application/json" \
     -d '{"url": "https://books.toscrape.com", "max_pages": 50}'
```

### `GET /crawl/status`

Poll the progress of an ongoing crawl.

### `GET /stats`

Return index statistics (document count, unique terms, avg postings).

---

## 📐 Ranking Algorithm

Each result is scored using four signals, weighted and summed:

```
score = 0.45 × TF-IDF
      + 0.25 × PageRank
      + 0.20 × Keyword Frequency
      + 0.10 × Recency
```

| Signal | Description |
|--------|-------------|
| **TF-IDF** | Log-normalised term frequency × inverse document frequency |
| **PageRank** | Iterative link-graph authority score (20 iterations, d=0.85) |
| **Keyword Frequency** | Raw count of query terms in the document |
| **Recency** | Higher score for more recently crawled pages |

All signals are normalised to `[0, 1]` before combining.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover: tokenization, index building, search ranking, score normalisation, field validation, and index persistence.

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `DOWNLOAD_DELAY` | `1.0s` | Polite crawl delay between requests |
| `CONCURRENT_REQUESTS` | `4` | Scrapy concurrent requests |
| `DEPTH_LIMIT` | `3` | Max link depth from seed URL |
| `CLOSESPIDER_PAGECOUNT` | `200` | Hard cap on pages per crawl |
| `ROBOTSTXT_OBEY` | `True` | Respect robots.txt |

These are set in `crawler/spider.py` → `custom_settings`.

---

## 🌐 Recommended Demo Sites

These sites are safe to crawl for testing:

| URL | Description |
|-----|-------------|
| `https://books.toscrape.com` | Fake bookstore — crawl-friendly |
| `https://quotes.toscrape.com` | Famous quotes — lightweight |
| `https://crawler-test.com` | Designed for crawler testing |

---

## 📦 Tech Stack

- **[Scrapy](https://scrapy.org/)** — async web crawling framework
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** — HTML parsing and content extraction
- **[FastAPI](https://fastapi.tiangolo.com/)** — high-performance REST API
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server
- **[NumPy](https://numpy.org/)** — numerical operations

---

## 📝 License

MIT — see [LICENSE](LICENSE) for details.
