import json
import urllib.request
import urllib.parse
import os

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

def tavily_search(query: str, max_results: int = 5) -> dict:
    """Search the web using Tavily API — built for AI agents"""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not set", "results": []}
    try:
        payload = json.dumps({
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False
        }).encode()

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:400]
            })
        return {"results": results, "query": query}

    except Exception as e:
        return {"error": str(e), "results": []}


def fetch_page(url: str) -> dict:
    """Fetch and clean a web page"""
    try:
        import re
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "content": text[:4000], "length": len(text)}
    except Exception as e:
        return {"error": str(e), "url": url}