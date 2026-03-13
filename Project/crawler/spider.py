"""
crawler/spider.py
-----------------
Scrapy spider that crawls a single domain, extracts page content
and hyperlinks, and saves structured data for indexing.
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class DomainSpider(scrapy.Spider):
    name = "domain_spider"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,          # polite crawl delay (seconds)
        "CONCURRENT_REQUESTS": 4,
        "DEPTH_LIMIT": 3,               # max link depth from seed
        "CLOSESPIDER_PAGECOUNT": 200,   # safety cap
        "LOG_LEVEL": "WARNING",
        "USER_AGENT": (
            "MiniSearchEngine/1.0 "
            "(+https://github.com/yourusername/mini-search-engine)"
        ),
    }

    def __init__(self, start_url: str, output_path: str = "data/crawled.json", *args, **kwargs):
        super().__init__(*args, **kwargs)
        parsed = urlparse(start_url)
        self.start_urls = [start_url]
        self.allowed_domains = [parsed.netloc]
        self.output_path = output_path
        self.crawled_pages: list[dict] = []
        self.seen_urls: set[str] = set()

    def parse(self, response):
        url = response.url
        if url in self.seen_urls:
            return
        self.seen_urls.add(url)

        # ── Parse with BeautifulSoup for richer extraction ──────────────────
        soup = BeautifulSoup(response.text, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract visible body text
        text = " ".join(soup.get_text(separator=" ").split())

        # Extract all same-domain links
        outbound_links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            abs_url = urljoin(url, anchor["href"])
            parsed = urlparse(abs_url)
            if parsed.scheme in ("http", "https") and parsed.netloc in self.allowed_domains:
                clean = parsed._replace(fragment="").geturl()
                outbound_links.append(clean)

        page_data = {
            "url": url,
            "title": title,
            "text": text[:50_000],          # cap per-page storage
            "outbound_links": list(set(outbound_links)),
            "crawled_at": time.time(),
            "depth": response.meta.get("depth", 0),
        }

        self.crawled_pages.append(page_data)
        logger.info(f"Crawled [{len(self.crawled_pages)}]: {url}")

        # Follow links
        for link in outbound_links:
            if link not in self.seen_urls:
                yield scrapy.Request(link, callback=self.parse)

    def closed(self, reason):
        """Persist crawled data when spider finishes."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.crawled_pages, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Crawl complete — {len(self.crawled_pages)} pages saved to {self.output_path}")


def run_crawler(start_url: str, output_path: str = "data/crawled.json") -> None:
    """Entry point: launch Scrapy crawler programmatically."""
    process = CrawlerProcess()
    process.crawl(DomainSpider, start_url=start_url, output_path=output_path)
    process.start()


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://books.toscrape.com"
    run_crawler(url)
