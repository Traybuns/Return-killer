import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research  # noqa: E402

RAW = {
    "title": "Flexispot E7 Standing Desk <img src=x onerror=1>",
    "category": "Office",
    "price_usd": 499,
    "dimensions_cm": {"length": 140, "width": 70, "height": "bad"},
    "common_complaints": [
        {"theme": "Wobble at full height", "severity": "high", "detail": "Shakes when typing.", "fix": "State max stable height."},
        {"theme": "Bigger than photos", "severity": "bogus", "detail": "x", "fix": "Add scale photo."},
    ],
    "listing_tips": ["Show desk next to a chair"],
}


class FakeBedrock:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def converse(self, **kw):
        self.calls.append(kw)
        content = [
            {"text": json.dumps(self.payload)},
            {"citationsContent": {"citations": [{"location": {"web": {"url": "https://example.com/a", "domain": "example.com"}}}]}},
            {"citationsContent": [{"location": {"web": {"url": "javascript:alert(1)"}}}]},
        ]
        return {"output": {"message": {"content": content}}}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("RETURNKILLER_USE_BEDROCK", "0")
    import app
    fake = FakeBedrock(RAW)
    monkeypatch.setattr(app.analyzer, "use_bedrock", True)
    monkeypatch.setattr(app.analyzer, "bedrock_client", fake)
    monkeypatch.setattr(app, "_use_bedrock", True)
    app.analysis_cache._mem.clear()
    return TestClient(app.app), fake


def test_validate_drops_bad_values():
    p = research.validate(RAW, "standing desk", ["https://example.com/a"], "web-standing-desk")
    assert p["dimensions"] == {"length_cm": 140.0, "width_cm": 70.0}
    assert [c["severity"] for c in p["web_research"]["complaints"]] == ["high", "medium"]
    assert research.validate({"title": None}, "x", [], "web-x") is None


def test_ids_roundtrip():
    assert research.research_id("Standing  Desk!") == "web-standing-desk"
    assert research.query_from_id("web-standing-desk") == "standing desk"
    assert not research.is_research_id("B0EXAMPLE01")


def test_research_endpoint_and_analysis(client):
    c, fake = client
    r = c.get("/research", params={"q": "standing desk"})
    assert r.status_code == 200
    pid = r.json()["product"]["asin"]
    assert pid == "web-standing-desk"
    assert fake.calls[0]["toolConfig"] == {"tools": [{"systemTool": {"name": "nova_grounding"}}]}
    a = c.get(f"/analyze/{pid}").json()
    assert a["risk_basis"] == "web_research" and a["engine"] == "bedrock+web"
    assert a["sources"] == ["https://example.com/a"]  # javascript: URL dropped
    assert a["top_complaints"][0]["frequency"] == 0 and a["top_complaints"][0]["example_quotes"] == []
    assert len(fake.calls) == 1  # product cached for the analysis
    assert c.post("/fit", json={"asin": pid, "question": "fit under 40 cm?"}).status_code == 200


def test_unknown_product_404_and_disabled_503(client, monkeypatch):
    c, fake = client
    fake.payload = {"title": None}
    assert c.get("/research", params={"q": "zzzz qqqq"}).status_code == 404
    import app
    monkeypatch.setattr(app, "_use_bedrock", False)
    assert c.get("/research", params={"q": "standing desk"}).status_code == 503


def test_catalog_partial_match_requires_most_tokens():
    import catalog
    assert catalog.search_products("cutting board")["total"] >= 1
    assert catalog.search_products("standing desk cutting")["total"] == 0
