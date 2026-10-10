import gzip
import importlib.util
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_ingest():
    spec = importlib.util.spec_from_file_location("ingest", os.path.join(ROOT, "scripts", "ingest_amazon_reviews.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write('{"truncated": ')  # simulates an interrupted download


@pytest.fixture()
def built_catalog(tmp_path, monkeypatch):
    ing = _load_ingest()
    meta = [
        {"parent_asin": "A1", "title": "Acme Air Fryer 5 Qt - Black", "rating_number": 900,
         "features": ["<b>Fast</b>"], "description": ["Cooks."], "categories": ["Kitchen", "Air Fryers"],
         "price": "$59.99", "details": {"Product Dimensions": "10.5\"D x 12\"W x 13\"H"}},
        {"parent_asin": "A2", "title": "Bamboo Cutting Board", "rating_number": 500, "price": None,
         "details": {"Item Dimensions LxWxH": "18 x 12 x 0.75 inches"}},
        {"parent_asin": "A3", "title": "Mystery Gadget", "rating_number": 900,
         "details": {"Product Dimensions": "5 x 8 x 3 inches"}},  # unlabeled axes: no height
        {"parent_asin": "A4", "title": "Rare Air Fryer", "rating_number": 3},
    ]
    revs = [{"parent_asin": "A1", "rating": 1, "title": "x", "text": "<script>alert(1)</script> too big, doesn't fit"}] + \
           [{"parent_asin": "A1", "rating": 5, "title": "ok", "text": "great"} for _ in range(30)]
    mp, rp, out = tmp_path / "m.jsonl.gz", tmp_path / "r.jsonl.gz", tmp_path / "products.json"
    _write_gz(mp, meta)
    _write_gz(rp, revs)
    monkeypatch.setattr(sys, "argv", ["x", "--meta-file", str(mp), "--reviews-file", str(rp), "--out", str(out)])
    ing.main()
    return out


def test_ingest_parses_dimensions_and_stats(built_catalog):
    data = json.loads(built_catalog.read_text())
    by = {p["asin"]: p for p in data["products"]}
    assert set(by) == {"A1"}  # A2/A3 have no reviews; A4 under min ratings
    assert by["A1"]["dimensions"]["height_cm"] == pytest.approx(33.0, abs=0.1)
    assert by["A1"]["price"] == 59.99
    assert by["A1"]["review_stats"] == {"seen": 31, "mismatch_low": 1}
    assert "<b>" not in by["A1"]["bullets"][0]


def test_parse_dimensions_variants():
    ing = _load_ingest()
    h = ing.parse_dimensions({"Item Dimensions LxWxH": "18 x 12 x 0.75 inches"})
    assert h["height_cm"] == pytest.approx(1.9, abs=0.1)
    assert ing.parse_dimensions({"Product Dimensions": "5 x 8 x 3 inches"}) == {}


def test_app_serves_real_catalog_safely(built_catalog, monkeypatch):
    monkeypatch.setenv("RETURNKILLER_CATALOG", str(built_catalog))
    monkeypatch.setenv("RETURNKILLER_USE_BEDROCK", "0")
    import catalog
    catalog.reload_catalog()
    import app
    client = TestClient(app.app)
    try:
        r = client.get("/products?q=air fryer").json()
        assert r["count"] == 1 and r["catalog"]["source"] == "amazon-reviews-2023"
        a = client.get("/analyze/A1").json()
        assert a["risk_basis"] == "review_text_proxy" and a["reviews_analyzed"] >= 1
        assert client.get("/products?q=zzzz").json()["count"] == 0
    finally:
        monkeypatch.delenv("RETURNKILLER_CATALOG")
        catalog.reload_catalog()


def test_short_title_and_search_fallback():
    import catalog
    assert catalog.short_title("Acme Air Fryer - 5 Qt, Black, Digital") == "Acme Air Fryer"
    assert catalog.search_products("air fryer")["total"] >= 1


def test_per_query_cap_and_preset(tmp_path, monkeypatch):
    ing = _load_ingest()
    meta = [{"parent_asin": f"F{i}", "title": f"Brand{i} Air Fryer", "rating_number": 5000} for i in range(6)]
    meta += [{"parent_asin": f"T{i}", "title": f"Brand{i} Toaster Oven", "rating_number": 5000} for i in range(6)]
    revs = [{"parent_asin": p["parent_asin"], "rating": 5, "title": "t", "text": "fine"} for p in meta]
    mp, rp, out = tmp_path / "m.jsonl.gz", tmp_path / "r.jsonl.gz", tmp_path / "p.json"
    _write_gz(mp, meta)
    _write_gz(rp, revs)
    monkeypatch.setattr(sys, "argv", ["x", "--preset", "household", "--per-query", "2",
                                      "--meta-file", str(mp), "--reviews-file", str(rp), "--out", str(out)])
    ing.main()
    titles = [p["title"] for p in json.loads(out.read_text())["products"]]
    assert sum("Air Fryer" in t for t in titles) == 2 and sum("Toaster Oven" in t for t in titles) == 2
