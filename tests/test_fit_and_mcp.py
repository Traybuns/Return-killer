"""Fit-check arithmetic and the /mcp endpoint, all offline (Bedrock disabled)."""
import os
import sys

os.environ["RETURNKILLER_USE_BEDROCK"] = "0"
os.environ.pop("ORIGIN_VERIFY", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from analyzer import fit_check, parse_clearance_cm  # noqa: E402

AIR_FRYER = {"asin": "X", "dimensions": {"length_cm": 35, "width_cm": 28, "height_cm": 32}}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("40 cm", 40.0),
        ("40cm", 40.0),
        ("15 inches", 38.1),
        ('15"', 38.1),
        ("2 feet", 61.0),
        ("under 400 mm", 40.0),
        ("it is 3 in a row, 40 cm", 40.0),  # bare "in" must not be read as inches
        ("no number here", None),
        ("", None),
    ],
)
def test_parse_clearance(text, expected):
    assert parse_clearance_cm(text) == expected


def test_fit_verdict_boundaries():
    assert fit_check(AIR_FRYER, "40 cm")["verdict"] == "likely_fits"   # +8
    assert fit_check(AIR_FRYER, "35 cm")["verdict"] == "likely_fits"   # +3 exactly
    assert fit_check(AIR_FRYER, "34 cm")["verdict"] == "tight"         # +2
    assert fit_check(AIR_FRYER, "32 cm")["verdict"] == "tight"         # 0
    assert fit_check(AIR_FRYER, "30 cm")["verdict"] == "unlikely"      # -2
    assert fit_check(AIR_FRYER, "")["verdict"] == "need_measurement"


def test_fit_without_height_is_unknown_not_a_guess():
    assert fit_check({"asin": "Y", "dimensions": {}}, "40 cm")["verdict"] == "unknown"


@pytest.fixture(scope="module")
def client():
    import app as a

    with TestClient(a.app) as c:  # context manager runs the lifespan that starts the MCP session manager
        yield c


HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def rpc(client, method, params=None, id_=1):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if id_ is not None:
        body["id"] = id_
    return client.post("/mcp", json=body, headers=HEADERS)


def call(client, name, args):
    r = rpc(client, "tools/call", {"name": name, "arguments": args}).json()["result"]
    assert r["isError"] is False
    sc = r["structuredContent"]
    # The server wraps dict results as {"result": {...}}; the simulator page unwraps this.
    assert list(sc.keys()) == ["result"]
    return sc["result"]


def test_mcp_handshake_and_tools(client):
    init = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "t", "version": "1"}}).json()
    assert init["result"]["serverInfo"]["name"] == "ReturnKiller"
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"list_products", "check_fit", "analyze_listing"}
    assert all(t["annotations"]["readOnlyHint"] for t in tools)


def test_mcp_check_fit_is_arithmetic(client):
    r = call(client, "check_fit", {"asin": "B0EXAMPLE02", "space_description": "under my 15 inches cabinet"})
    assert r["verdict"] == "likely_fits" and r["clearance_cm"] == 38.1
    assert "centimeters" in r["spoken"]


def test_mcp_unknown_asin_is_graceful(client):
    r = call(client, "check_fit", {"asin": "NOPE", "space_description": "40 cm"})
    assert r["verdict"] == "unknown" and "couldn't find" in r["spoken"]
    assert "couldn't find" in call(client, "analyze_listing", {"asin": "NOPE"})["spoken"]


def test_mcp_analyze_reports_engine(client):
    r = call(client, "analyze_listing", {"asin": "B0EXAMPLE01"})
    assert r["engine"] == "simulated" and 0 <= r["return_risk_score"] <= 100 and r["spoken"]


def test_existing_routes_still_win_over_the_mount(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/alexa").status_code == 200
    assert client.get("/demo").status_code == 200
    assert client.get("/products").json()["count"] >= 2
