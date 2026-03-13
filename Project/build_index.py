"""
build_index.py
--------------
CLI utility to crawl a URL and build the inverted index.

Usage:
    python build_index.py https://books.toscrape.com
    python build_index.py https://books.toscrape.com --max-pages 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crawler.spider import run_crawler
from indexer.index import InvertedIndex

CRAWL_OUT = "data/crawled.json"
INDEX_OUT = "data/index.pkl"


def main():
    parser = argparse.ArgumentParser(description="Mini Search Engine — build index from a URL")
    parser.add_argument("url", help="Seed URL to crawl (single domain)")
    parser.add_argument("--max-pages", type=int, default=50, help="Page cap (default: 50)")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawl, rebuild index from existing crawled.json")
    args = parser.parse_args()

    if not args.skip_crawl:
        print(f"\n🕷️  Starting crawl: {args.url} (max {args.max_pages} pages)\n")
        run_crawler(args.url, output_path=CRAWL_OUT)

    print("\n📑 Building inverted index…")
    idx = InvertedIndex()
    idx.build(CRAWL_OUT)
    idx.save(INDEX_OUT)

    print("\n📊 Index stats:")
    for k, v in idx.stats().items():
        print(f"   {k}: {v}")

    print(f"\n✅ Done! Start the API with:\n   uvicorn api.main:app --reload\n")


if __name__ == "__main__":
    main()
