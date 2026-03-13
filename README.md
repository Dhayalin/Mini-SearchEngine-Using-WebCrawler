# Mini-SearchEngine-Using-WebCrawler
A Python-based search engine built from scratch that crawls, indexes, and retrieves web pages with multi-signal ranking.
The engine has three core components working in a pipeline: a web crawler built with Scrapy and BeautifulSoup that stays within a single domain, respects robots.txt, and extracts clean page text and hyperlinks; an inverted index that maps terms to documents using log-normalised TF-IDF scoring; and a search API built with FastAPI that returns ranked results in milliseconds.
Results are ranked using four signals combined — TF-IDF relevance (45%), PageRank computed from the crawled link graph (25%), raw keyword frequency (20%), and crawl recency (10%). The project includes a dark-themed web UI, a CLI tool to trigger crawls, and a full pytest suite covering tokenization, indexing, scoring, and persistence.
Tech: Python · Scrapy · BeautifulSoup4 · FastAPI · Uvicorn
