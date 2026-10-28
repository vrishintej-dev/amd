# Match 4: Intelligent Web Information Retrieval

The official Match 4 harness had not been published when this was written. The agent follows the container contract of Matches 2–3 and accepts every likely argument spelling (see `app/app.py`).

- **Live site:** `--url` or `--start-url` plus `--query` or `--question`. A best-first crawler (`app/rag/web.py`) stays on the start site, follows the links whose anchor text and URL best match the request, and converts pages to text with dependency-free HTML parsing (tables kept as `a | b` rows). Its budget is about 12 s and 25 pages, with an https→http fallback.
- **Saved pages:** `--corpus DIR` (HTML files supported), with an optional `--index DIR` first.
- **Answer:** the Mini Challenge 3 engine (see `../mc3-solution/app/rag/`) does BM25 retrieval, then Qwen3-VL-4B on the AMD GPU, then citation verification. It returns **the specific value only**, with the page URL(s) it came from. If the pages don't contain the value, it returns an empty answer rather than a guess.
- **Output:** `/app/output/<id>_output.json` → `{"answer", "citations", "sources", "source_url", "confidence"}`
- **Changes to the shared `rag/` package for Match 4:** `corpus.py` converts .html/.htm files with `web.html_to_text`, and `engine.py` tags pages as "web page" and takes a per-request soft deadline, so the crawl and the answer together stay under 30 s.

The image is built FROM our Mini Challenge 3 image, whose lower layers are the mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` base. The model is already inside it, so the push is a few MB.
