"""Product catalog: real products when available, the 2 demo samples otherwise.

Lookup order for the data file:
  1. $RETURNKILLER_CATALOG (explicit path)
  2. data/products.json   (built by scripts/ingest_amazon_reviews.py; not committed)
  3. sample_products.json (two hand-written demo products)

The file is read once per process and indexed by ASIN.
"""
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

BASE = os.path.dirname(os.path.abspath(__file__))

_STOP = {
    "a", "an", "the", "for", "and", "or", "of", "to", "in", "on", "with", "my", "me", "is", "it",
    "this", "that", "will", "can", "do", "does", "how", "what", "any", "some", "find", "search",
}


def _candidate_paths() -> List[str]:
    paths = []
    env = os.environ.get("RETURNKILLER_CATALOG")
    if env:
        paths.append(env)
    paths += [
        os.path.join(BASE, "data", "products.json"),
        os.path.join(BASE, "sample_products.json"),
    ]
    return paths


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


def short_title(title: str, max_len: int = 60) -> str:
    """A speakable name: drop the marketing tail, cut at a word boundary."""
    t = (title or "").split(" - ")[0].split(" | ")[0].strip()
    if len(t) <= max_len:
        return t
    cut = t[:max_len].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut or t[:max_len]


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for path in _candidate_paths():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:  # unreadable or corrupt: try the next candidate
            last_err = e
            continue
        products = data["products"] if isinstance(data, dict) else data
        products = [p for p in products if isinstance(p, dict) and p.get("asin") and p.get("title")]
        if not products:
            continue
        index = {}
        tokens = {}  # kept apart from the product dicts so they stay JSON-serializable
        for p in products:
            tokens[p["asin"]] = set(_tokens(p["title"])) | set(_tokens(p.get("category") or ""))
            index[p["asin"]] = p
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        return {
            "products": products,
            "by_asin": index,
            "tokens": tokens,
            "path": path,
            "source": meta.get("source") or ("sample" if path.endswith("sample_products.json") else "custom"),
            "meta": meta,
        }
    raise RuntimeError(f"No usable product catalog found (last error: {last_err})")


def reload_catalog() -> None:
    load_catalog.cache_clear()


def get_product(asin: str) -> Optional[Dict[str, Any]]:
    return load_catalog()["by_asin"].get(asin)


def _popularity(p: Dict[str, Any]) -> float:
    return float(p.get("rating_number") or len(p.get("reviews") or []))


def summarize(p: Dict[str, Any]) -> Dict[str, Any]:
    dims = p.get("dimensions") or {}
    return {
        "asin": p["asin"],
        "title": p["title"],
        "short_title": short_title(p["title"]),
        "category": p.get("category"),
        "price": p.get("price"),
        "review_count": len(p.get("reviews") or []),
        "has_height": dims.get("height_cm") is not None,
    }


def search_products(query: str = "", limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """Token search over titles/categories. Empty query lists the most popular products."""
    cat = load_catalog()
    q = set(_tokens(query))
    if not q:
        ranked = sorted(cat["products"], key=_popularity, reverse=True)
    else:
        scored = []
        for p in cat["products"]:
            hits = len(q & cat["tokens"][p["asin"]])
            if hits:
                scored.append((hits, _popularity(p), p))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # Require every query token when any product matches all of them; otherwise best partial.
        best = scored[0][0] if scored else 0
        ranked = [p for hits, _, p in scored if hits == best]
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    return {
        "total": len(ranked),
        "products": [summarize(p) for p in ranked[offset : offset + limit]],
    }


def catalog_info() -> Dict[str, Any]:
    cat = load_catalog()
    return {
        "source": cat["source"],
        "products": len(cat["products"]),
        "category": cat["meta"].get("category"),
        "generated_at": cat["meta"].get("generated_at"),
    }
