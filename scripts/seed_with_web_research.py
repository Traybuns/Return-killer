#!/usr/bin/env python3
"""Fill data/products.json with ~230 household products using web research instead of the dataset.

No multi-gigabyte download: each item is researched once with Nova web grounding (cited sources,
dimensions, common fit complaints) and saved, so searching the catalog is instant afterwards.
Entries are the web tier (no review text) and are labelled that way in the app.

  AWS_REGION=us-east-1 python scripts/seed_with_web_research.py            # all terms
  python scripts/seed_with_web_research.py --limit 20                      # quick trial

Needs your AWS credentials locally with bedrock:InvokeModel and bedrock:InvokeTool. Safe to re-run:
products already in the file are skipped, and results are saved as it goes. Each lookup is a billed
web grounding call.
"""
import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import research  # noqa: E402
from analyzer import ReturnKillerAnalyzer  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "..", "data", "products.json")


def household_terms():
    spec = importlib.util.spec_from_file_location("ingest", os.path.join(HERE, "ingest_amazon_reviews.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.HOUSEHOLD)


def load(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("products", []) if isinstance(data, dict) else data, (data.get("meta", {}) if isinstance(data, dict) else {})
    return [], {}


def save(path, products, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "products": products}, f, ensure_ascii=False)
    os.replace(tmp, path)


def main(argv=None, analyzer=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="research only the first N terms")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--term", action="append", default=[], help="extra term to research (repeatable)")
    a = ap.parse_args(argv)

    analyzer = analyzer or ReturnKillerAnalyzer(use_bedrock=True)
    if not (analyzer.use_bedrock and analyzer.bedrock_client):
        sys.exit("Bedrock is not available: check your AWS credentials and region (AWS_REGION=us-east-1).")

    products, meta = load(a.out)
    have = {p["asin"] for p in products}
    terms = list(dict.fromkeys(a.term + household_terms()))
    if a.limit:
        terms = terms[: a.limit]
    todo = [t for t in terms if research.research_id(t) not in have]
    print(f"{len(have)} already saved, researching {len(todo)} terms", file=sys.stderr)

    added = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(research.research_product, analyzer, t): t for t in todo}
        for fut in as_completed(futs):
            term = futs[fut]
            try:
                p = fut.result()
            except Exception as e:  # keep going; re-run to retry
                failed += 1
                print(f"  ! {term}: {str(e)[:120]}", file=sys.stderr)
                continue
            if not p:
                failed += 1
                print(f"  ? {term}: no product identified", file=sys.stderr)
                continue
            p["asin"] = research.research_id(term)  # keyed by the term so re-runs skip it
            products.append(p)
            added += 1
            meta.setdefault("source", "web-research")
            meta["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save(a.out, products, meta)
            print(f"  + {p['title'][:70]}", file=sys.stderr)
    print(f"done: {added} added, {failed} failed, {len(products)} total -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
