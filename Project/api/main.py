"""
api/main.py
-----------
FastAPI application exposing:
  GET  /search?q=...&limit=...   — query the index
  POST /crawl                    — trigger a new crawl + re-index
  GET  /stats                    — index statistics
  GET  /                         — serve the frontend UI
"""

import os
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

# Adjust import path when running from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.index import InvertedIndex

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
INDEX_PATH  = DATA_DIR / "index.pkl"
CRAWL_PATH  = DATA_DIR / "crawled.json"
STATIC_DIR  = Path("static")

# ── Global state ───────────────────────────────────────────────────────────
_index: Optional[InvertedIndex] = None
_crawl_status: dict = {"running": False, "message": "idle", "progress": 0}


def load_index() -> None:
    global _index
    if INDEX_PATH.exists():
        try:
            _index = InvertedIndex.load(str(INDEX_PATH))
            print(f"✅ Loaded existing index ({_index._num_docs} docs)")
        except Exception as e:
            print(f"⚠️  Could not load index: {e}")
            _index = None
    else:
        print("ℹ️  No index found. Use POST /crawl to build one.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    yield


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mini Search Engine",
    description="A Python-based search engine with web crawling and TF-IDF ranking.",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Models ─────────────────────────────────────────────────────────────────

class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 100


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float
    tfidf: float
    pagerank: float
    keyword_freq: float
    recency: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]
    search_time_ms: float


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend search UI."""
    ui_path = STATIC_DIR / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Mini Search Engine API</h1><p>Visit <code>/docs</code> for the API.</p>")


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
):
    """Search the index and return ranked results."""
    if _index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Trigger a crawl first via POST /crawl.",
        )

    t0 = time.perf_counter()
    raw_results = _index.search(q, top_k=limit)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = [SearchResult(**r) for r in raw_results]
    return SearchResponse(
        query=q,
        total_results=len(results),
        results=results,
        search_time_ms=round(elapsed_ms, 2),
    )


@app.post("/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Trigger a crawl of the given URL, then rebuild the index."""
    if _crawl_status["running"]:
        return JSONResponse(
            {"message": "A crawl is already in progress.", "status": _crawl_status},
            status_code=409,
        )
    background_tasks.add_task(_crawl_and_index, request.url, request.max_pages)
    return {"message": f"Crawl started for {request.url}", "status": "running"}


@app.get("/crawl/status")
async def crawl_status():
    """Poll the status of the current or last crawl."""
    return _crawl_status


@app.get("/stats")
async def stats():
    """Return index statistics."""
    if _index is None:
        return {"status": "no_index", "message": "Run POST /crawl to build an index."}
    return {"status": "ready", **_index.stats()}


# ── Background crawl task ──────────────────────────────────────────────────

def _crawl_and_index(url: str, max_pages: int) -> None:
    """Run in a background thread: crawl → save → build index → reload."""
    global _index, _crawl_status

    _crawl_status = {"running": True, "message": f"Crawling {url}…", "progress": 0}
    DATA_DIR.mkdir(exist_ok=True)

    try:
        # Scrapy must run in its own thread with a fresh reactor
        from scrapy.crawler import CrawlerProcess
        from crawler.spider import DomainSpider

        _crawl_status["message"] = "Spider running…"
        process = CrawlerProcess()
        process.crawl(
            DomainSpider,
            start_url=url,
            output_path=str(CRAWL_PATH),
        )
        process.start()   # blocks until done

        _crawl_status["progress"] = 60
        _crawl_status["message"] = "Building index…"

        idx = InvertedIndex()
        idx.build(str(CRAWL_PATH))
        idx.save(str(INDEX_PATH))
        _index = idx

        _crawl_status = {
            "running": False,
            "message": f"Done — {idx._num_docs} pages indexed.",
            "progress": 100,
        }

    except Exception as exc:
        _crawl_status = {"running": False, "message": f"Error: {exc}", "progress": 0}
        raise
