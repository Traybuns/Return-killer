"""
ReturnKiller API
Simple FastAPI server that exposes the analysis engine.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import hmac
import json
import os

from contextlib import asynccontextmanager

from analyzer import ReturnKillerAnalyzer, fit_check
from catalog import get_product, search_products, catalog_info, short_title as catalog_short_title
import research as web_research
from mcp_server import build_mcp


@asynccontextmanager
async def lifespan(_app):
    # The mounted MCP app's own lifespan does not run, so start its session manager here.
    async with _mcp.session_manager.run():
        yield


app = FastAPI(
    lifespan=lifespan,
    title="ReturnKiller API",
    description="AI agent that reduces Amazon returns caused by size & description mismatches",
    version="0.1.0"
)

# CORS_ORIGINS is a comma-separated list (set by Terraform to the CloudFront URL).
# Default "*" keeps local dev working; credentials are only allowed with explicit origins.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set RETURNKILLER_USE_BEDROCK=1 (+ AWS credentials) to enable Nova vision scans
_use_bedrock = os.environ.get("RETURNKILLER_USE_BEDROCK", "").lower() in ("1", "true", "yes")
analyzer = ReturnKillerAnalyzer(use_bedrock=_use_bedrock)

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Lambda request payloads cap at 6 MB

# When deployed, CloudFront (and its WAF) must be the only way in. Terraform sets
# ORIGIN_VERIFY and CloudFront sends it as a header; direct API Gateway hits are rejected.
_origin_verify = os.environ.get("ORIGIN_VERIFY")


@app.middleware("http")
async def require_cloudfront(request: Request, call_next):
    if _origin_verify:
        # Shallow /health is exempt so the Lambda Web Adapter readiness check can reach it.
        is_shallow_health = request.url.path == "/health" and request.query_params.get("deep") != "true"
        if not is_shallow_health and not hmac.compare_digest(
            request.headers.get("x-origin-verify", ""), _origin_verify
        ):
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


class AnalysisCache:
    """DynamoDB-backed cache when CACHE_TABLE is set, in-memory dict otherwise.

    Lambda containers are ephemeral, so an in-memory dict alone loses data between
    invocations. Dict-style access keeps the rest of the code unchanged.
    """

    def __init__(self):
        self._mem: Dict[str, Any] = {}
        self._table = None
        table_name = os.environ.get("CACHE_TABLE")
        if table_name:
            try:
                import boto3
                self._table = boto3.resource("dynamodb").Table(table_name)
            except Exception as e:
                print(f"⚠ DynamoDB cache unavailable, using memory: {e}")

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._mem[key] = value
        if self._table:
            try:
                import time
                self._table.put_item(Item={
                    "pk": f"analysis#{key}",
                    "payload": json.dumps(value),
                    "expires_at": int(time.time()) + 7 * 24 * 3600,
                })
            except Exception as e:
                print(f"⚠ cache write failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        if key in self._mem:
            return self._mem[key]
        if self._table:
            try:
                item = self._table.get_item(Key={"pk": f"analysis#{key}"}).get("Item")
                if item:
                    value = json.loads(item["payload"])
                    self._mem[key] = value
                    return value
            except Exception as e:
                print(f"⚠ cache read failed: {e}")
        return None


analysis_cache = AnalysisCache()


def _research(query: str) -> Optional[Dict[str, Any]]:
    """Web-research a product that is not in the catalog; cached so repeat asks are instant."""
    query = web_research.clean_query(query)
    if len(query) < 2:
        return None
    pid = web_research.research_id(query)
    cached = analysis_cache.get(f"product:{pid}")
    if cached:
        return cached
    product = web_research.research_product(analyzer, query)
    if product:
        analysis_cache[f"product:{product['asin']}"] = product
    return product


def _find_product(asin: str) -> Optional[Dict[str, Any]]:
    found = get_product(asin)
    if found and found.get("lazy"):
        # A name-only catalog entry: fill in dimensions and complaints by researching it now.
        try:
            return _research(found["title"]) or None
        except RuntimeError as e:
            print(f"⚠ research for {asin} failed: {e}")
            return None
    if found or not web_research.is_research_id(asin):
        return found
    try:
        return _research(web_research.query_from_id(asin))
    except RuntimeError as e:
        print(f"⚠ research for {asin} failed: {e}")
        return None


def _get_analysis(asin: str) -> Optional[Dict[str, Any]]:
    cached = analysis_cache.get(asin)
    if cached:
        return cached
    product = _find_product(asin)
    if not product:
        return None
    analysis = analyzer.to_dict(analyzer.analyze_product(product))
    # Don't pin a degraded answer: if Bedrock failed, the offline result must not be served
    # for the next 7 days after Bedrock recovers.
    if not analysis.get("fallback_reason"):
        analysis_cache[asin] = analysis
    return analysis


class AnalyzeRequest(BaseModel):
    asin: Optional[str] = None
    product: Optional[Dict[str, Any]] = None


class FitQuestion(BaseModel):
    asin: str
    question: str  # e.g. "will this fit under my cabinet?"
    space_description: Optional[str] = None


@app.get("/")
def root():
    return {
        "name": "ReturnKiller",
        "version": "0.2.0",
        "build": os.environ.get("BUILD_SHA", "unknown"),
        "status": "running",
        "bedrock_vision": _use_bedrock,
        "catalog": catalog_info(),
        "endpoints": {
            "GET /products": "Search or list products (?q=, ?limit=, ?offset=)",
            "GET /analyze/{asin}": "Analyze a catalog product by ASIN",
            "POST /analyze": "Analyze a custom product payload",
            "POST /fit": "Ask a fit question (Alexa-style)",
            "POST /scan": "Scan a space photo (Bedrock Nova vision)",
            "GET /demo": "Interactive demo UI"
        }
    }


@app.get("/health")
def health(deep: bool = False):
    """Liveness by default; ?deep=true makes a real Bedrock call to prove permissions work."""
    result: Dict[str, Any] = {"status": "ok", "bedrock_enabled": _use_bedrock}
    if deep:
        result.update(analyzer.health_check())
        result.update(web_research.check_grounding(analyzer))
    return result


@app.get("/products")
def list_products(q: str = "", limit: int = 50, offset: int = 0):
    """Search the catalog (token match on title/category) or list the most popular products."""
    result = search_products(q, limit=limit, offset=offset)
    return {
        "count": result["total"], "products": result["products"], "catalog": catalog_info(),
        "research_available": bool(_use_bedrock and analyzer.bedrock_client),
    }


def _research_summary(p: Dict[str, Any]) -> Dict[str, Any]:
    dims = p.get("dimensions") or {}
    return {
        "asin": p["asin"], "title": p["title"], "short_title": catalog_short_title(p["title"]),
        "category": p.get("category"), "price": p.get("price"), "review_count": 0,
        "has_height": dims.get("height_cm") is not None, "source": "web", "lazy": False,
        "sources": (p.get("web_research") or {}).get("sources", []),
    }


@app.get("/research")
def research_endpoint(q: str):
    """Look up any product on the web (Nova web grounding). Slower and costlier than /products."""
    if not (_use_bedrock and analyzer.bedrock_client):
        raise HTTPException(status_code=503, detail="Web research is not enabled on this server")
    try:
        product = _research(q)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not product:
        raise HTTPException(status_code=404, detail="I could not identify a product from that search")
    return {"product": _research_summary(product)}


@app.get("/analyze/{asin}")
def analyze_by_asin(asin: str):
    product = _find_product(asin)

    if not product:
        raise HTTPException(status_code=404, detail=f"Product {asin} not found in the catalog")

    result = analyzer.analyze_product(product)
    result_dict = analyzer.to_dict(result)
    analysis_cache[asin] = result_dict
    return {**result_dict, "reviews": _sample_reviews(product)}


def _sample_reviews(product: Dict[str, Any], limit: int = 12) -> list:
    """Reviews to show beside the analysis: lowest stars first, so the problems lead."""
    out = []
    for r in sorted(product.get("reviews") or [], key=lambda x: x.get("rating") or 0)[:limit]:
        try:
            rating = max(1, min(5, int(r.get("rating") or 0)))
        except (TypeError, ValueError):
            continue
        out.append({"rating": rating, "title": str(r.get("title") or "")[:120], "text": str(r.get("text") or "")[:500]})
    return out


@app.post("/analyze")
def analyze_custom(req: AnalyzeRequest):
    if req.product:
        product = req.product
    elif req.asin:
        product = _find_product(req.asin)
        if not product:
            raise HTTPException(status_code=404, detail="ASIN not found")
    else:
        raise HTTPException(status_code=400, detail="Provide either 'asin' or 'product'")
    
    result = analyzer.analyze_product(product)
    result_dict = analyzer.to_dict(result)
    analysis_cache[result.asin] = result_dict
    return result_dict


@app.post("/fit")
def fit_question(req: FitQuestion):
    """Simulate an Alexa 'will it fit?' interaction."""
    
    product = _find_product(req.asin)
    analysis = _get_analysis(req.asin)
    if not product or not analysis:
        raise HTTPException(status_code=404, detail="Product not found")

    spoken = analysis["alexa_fit_response"]

    # If the shopper described a space, answer with real arithmetic on the listed height.
    fit = fit_check(product, req.space_description or req.question)
    if fit["clearance_cm"] is not None:
        spoken = fit["spoken"]

    return {
        "asin": req.asin,
        "question": req.question,
        "fit_verdict": fit["verdict"],
        "spoken_response": spoken,
        "size_chart": analysis["size_chart_text"],
        "visual_suggestions": analysis["visual_suggestions"],
        "risk_score": analysis["return_risk_score"],
        "screen_title": f"Size Check: {analysis['title'][:40]}",
        "dimensions_summary": analysis.get("size_chart_text", "").split("\n")[0] if analysis.get("size_chart_text") else "Dimensions unavailable"
    }






@app.post("/scan")
async def scan_space(
    file: UploadFile = File(...),
    asin: Optional[str] = Form(None),
    hint: Optional[str] = Form(None),
):
    """
    Upload a photo of a cabinet/shelf/space.
    Uses Bedrock Nova vision when RETURNKILLER_USE_BEDROCK=1.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")
    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Image file is empty")

    product = None
    if asin:
        product = _find_product(asin)

    result = analyzer.scan_space(
        image_bytes=image_bytes,
        media_type=file.content_type or "image/jpeg",
        product=product,
        user_hint=hint,
    )
    return result


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    """Glass / bubble polished demo UI with Alexa-first interaction."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ReturnKiller — Will it fit?</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #070a0f;
      --glass: rgba(255,255,255,0.045);
      --glass-strong: rgba(255,255,255,0.08);
      --glass-border: rgba(255,255,255,0.12);
      --glass-highlight: rgba(255,255,255,0.18);
      --accent: #ff9900;
      --accent-2: #ff6b00;
      --accent-soft: rgba(255,153,0,0.15);
      --accent-glow: rgba(255,153,0,0.35);
      --text: #f4f6f8;
      --text-secondary: #b8c0cc;
      --muted: #7d8996;
      --danger: #ff5c6c;
      --danger-soft: rgba(255,92,108,0.14);
      --success: #00e5a0;
      --success-soft: rgba(0,229,160,0.12);
      --warning: #ffb020;
      --radius: 24px;
      --radius-sm: 16px;
      --radius-pill: 999px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Soft ambient orbs */
    body::before, body::after {
      content: "";
      position: fixed;
      border-radius: 50%;
      filter: blur(80px);
      z-index: 0;
      pointer-events: none;
    }
    body::before {
      width: 420px;
      height: 420px;
      top: -120px;
      left: -80px;
      background: rgba(255,153,0,0.14);
    }
    body::after {
      width: 380px;
      height: 380px;
      bottom: -100px;
      right: -60px;
      background: rgba(0,229,160,0.08);
    }

    .wrap {
      position: relative;
      z-index: 1;
      max-width: 1040px;
      margin: 0 auto;
      padding: 14px 20px 80px;
    }

    /* Header */
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      font-size: 1.2rem;
      letter-spacing: -0.02em;
    }
    .logo-mark {
      width: 38px;
      height: 38px;
      background: linear-gradient(145deg, var(--accent), var(--accent-2));
      border-radius: 14px;
      display: grid;
      place-items: center;
      font-size: 18px;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.15) inset,
        0 8px 24px var(--accent-glow);
    }

    /* Glass card base */
    .glass {
      background: var(--glass);
      backdrop-filter: blur(24px) saturate(140%);
      -webkit-backdrop-filter: blur(24px) saturate(140%);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius);
      box-shadow:
        0 8px 32px rgba(0,0,0,0.25),
        0 1px 0 var(--glass-highlight) inset;
    }

    /* Hero */
    .hero {
      text-align: center;
      margin-bottom: 14px;
    }
    .hero h1 {
      font-size: clamp(1.4rem, 3.6vw, 1.9rem);
      font-weight: 800;
      letter-spacing: -0.035em;
      line-height: 1.15;
      margin-bottom: 4px;
    }
    .hero h1 span {
      background: linear-gradient(135deg, var(--accent), #ffc14d);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .hero p {
      color: var(--muted);
      font-size: 0.95rem;
      max-width: 520px;
      margin: 0 auto;
    }

    /* Product picker */
    .picker {
      padding: 22px;
      margin-bottom: 20px;
    }
    .review { padding: 12px 0; border-top: 1px solid var(--glass-border); }
    .review:first-child { border-top: 0; padding-top: 0; }
    .review-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 4px; font-weight: 600; font-size: 0.92rem; }
    .stars { color: var(--accent); letter-spacing: 1px; font-size: 0.85rem; }
    .review p { color: var(--text-secondary); font-size: 0.9rem; line-height: 1.55; }
    .alexa-main { margin-bottom: 18px; }
    .alexa-frame { display: block; width: 100%; height: clamp(440px, calc(100vh - 190px), 600px); border: 1px solid var(--glass-border); border-radius: 20px; background: #070a0f; }
    .alexa-actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
    .alexa-actions .btn { text-decoration: none; }
    .search-results { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; max-height: 320px; overflow-y: auto; }
    .result-item {
      text-align: left; background: rgba(0,0,0,0.25); border: 1px solid var(--glass-border); color: var(--text);
      border-radius: 14px; padding: 12px 16px; font-family: inherit; font-size: 0.95rem; cursor: pointer;
    }
    .result-item:hover { border-color: rgba(255,153,0,0.5); }
    .result-item[aria-selected="true"] { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .result-item small { display: block; color: var(--muted); margin-top: 2px; }
    .search-empty { color: var(--muted); font-size: 0.9rem; padding: 8px 4px; }
    #productSearch {
      flex: 1; min-width: 220px; max-width: 420px;
      background: rgba(0,0,0,0.25); border: 1px solid var(--glass-border); color: var(--text);
      padding: 12px 18px; border-radius: var(--radius-pill); font-size: 0.95rem; font-family: inherit;
    }
    #productSearch:focus { outline: none; border-color: rgba(255,153,0,0.5); box-shadow: 0 0 0 4px var(--accent-soft); }
    .picker-label {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      margin-bottom: 12px;
      text-align: center;
    }
    .picker-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: center;
    }
    select {
      flex: 1;
      min-width: 220px;
      max-width: 420px;
      appearance: none;
      background: rgba(0,0,0,0.25) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%237d8996' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E") no-repeat right 16px center;
      border: 1px solid var(--glass-border);
      color: var(--text);
      padding: 14px 42px 14px 18px;
      border-radius: var(--radius-pill);
      font-size: 0.95rem;
      font-family: inherit;
      cursor: pointer;
      transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    }
    select:hover, select:focus {
      border-color: rgba(255,153,0,0.5);
      outline: none;
      box-shadow: 0 0 0 4px var(--accent-soft);
      background-color: rgba(0,0,0,0.35);
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #111;
      border: none;
      padding: 14px 26px;
      border-radius: var(--radius-pill);
      font-weight: 700;
      font-size: 0.95rem;
      font-family: inherit;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.2s, filter 0.15s;
      white-space: nowrap;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.2) inset,
        0 6px 24px var(--accent-glow);
    }
    .btn:hover {
      filter: brightness(1.07);
      transform: translateY(-2px);
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.25) inset,
        0 10px 32px var(--accent-glow);
    }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    /* Results */
    #results {
      display: none;
      animation: rise 0.45s cubic-bezier(0.22,1,0.36,1);
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(18px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Alexa hero card — main interaction */
    .alexa-hero {
      padding: 28px 24px 24px;
      margin-bottom: 18px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }
    .alexa-hero::before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(255,153,0,0.12), transparent 70%);
      pointer-events: none;
    }
    .alexa-hero > * { position: relative; }
    .alexa-kicker {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--accent);
      margin-bottom: 10px;
    }
    .alexa-spoken {
      color: var(--text-secondary);
      font-size: 1.05rem;
      line-height: 1.65;
      max-width: 560px;
      margin: 0 auto 22px;
    }
    .ask-alexa-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      background: linear-gradient(135deg, #00caff, #00a3ff 40%, #0066ff);
      color: #fff;
      border: none;
      padding: 18px 36px;
      border-radius: var(--radius-pill);
      font-weight: 750;
      font-size: 1.1rem;
      font-family: inherit;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.2s, filter 0.15s;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.25) inset,
        0 8px 32px rgba(0,140,255,0.4);
      margin-bottom: 16px;
    }
    .ask-alexa-btn:hover {
      filter: brightness(1.08);
      transform: translateY(-2px) scale(1.02);
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.3) inset,
        0 12px 40px rgba(0,140,255,0.5);
    }
    .ask-alexa-btn:active { transform: translateY(0) scale(1); }
    .ask-alexa-btn .mic {
      width: 22px;
      height: 22px;
      display: grid;
      place-items: center;
    }
    .alexa-hint {
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 18px;
    }
    .fit-input-wrap {
      display: flex;
      gap: 10px;
      max-width: 520px;
      margin: 0 auto;
      flex-wrap: wrap;
      justify-content: center;
    }
    .fit-input-wrap input {
      flex: 1;
      min-width: 200px;
      background: rgba(0,0,0,0.3);
      border: 1px solid var(--glass-border);
      color: var(--text);
      padding: 14px 18px;
      border-radius: var(--radius-pill);
      font-size: 0.92rem;
      font-family: inherit;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .fit-input-wrap input:focus {
      outline: none;
      border-color: rgba(0,163,255,0.6);
      box-shadow: 0 0 0 4px rgba(0,163,255,0.12);
    }
    .fit-input-wrap input::placeholder { color: var(--muted); }
    .btn-soft {
      background: var(--glass-strong);
      color: var(--text);
      border: 1px solid var(--glass-border);
      box-shadow: none;
      padding: 14px 20px;
    }
    .btn-soft:hover {
      background: rgba(255,255,255,0.1);
      filter: none;
      box-shadow: none;
    }
    #fitResult {
      display: none;
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: var(--radius-sm);
      background: rgba(0,163,255,0.08);
      border: 1px solid rgba(0,163,255,0.2);
      text-align: left;
      max-width: 560px;
      margin-left: auto;
      margin-right: auto;
      animation: rise 0.35s ease;
    }
    #fitResult .label {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #5cc8ff;
      margin-bottom: 6px;
    }
    #fitResult .text {
      color: var(--text-secondary);
      font-size: 0.95rem;
      line-height: 1.6;
    }

    /* Grid of insight cards */
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 700px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      padding: 20px;
      transition: border-color 0.2s, transform 0.2s;
    }
    .card:hover {
      border-color: var(--glass-highlight);
    }
    .card.full { grid-column: 1 / -1; }
    .card-title {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 14px;
    }

    /* Risk */
    .risk-row {
      display: flex;
      align-items: center;
      gap: 20px;
      flex-wrap: wrap;
    }
    .risk-number {
      font-size: 3rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      line-height: 1;
    }
    .risk-number.high { color: var(--danger); }
    .risk-number.medium { color: var(--warning); }
    .risk-number.low { color: var(--success); }
    .risk-level {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 5px 12px;
      border-radius: var(--radius-pill);
      margin-bottom: 8px;
    }
    .risk-level.high { background: var(--danger-soft); color: var(--danger); }
    .risk-level.medium { background: rgba(255,176,32,0.15); color: var(--warning); }
    .risk-level.low { background: var(--success-soft); color: var(--success); }
    .risk-summary {
      color: var(--text-secondary);
      font-size: 0.88rem;
      white-space: pre-line;
      line-height: 1.6;
    }

    /* Complaints */
    .complaint {
      background: rgba(0,0,0,0.22);
      border-radius: 14px;
      padding: 12px 14px;
      margin-bottom: 8px;
      border-left: 3px solid var(--accent);
    }
    .complaint:last-child { margin-bottom: 0; }
    .complaint-top {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 5px;
    }
    .complaint-theme { font-weight: 600; font-size: 0.92rem; }
    .tag {
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 2px 8px;
      border-radius: var(--radius-pill);
    }
    .tag.high { background: var(--danger-soft); color: var(--danger); }
    .tag.medium { background: rgba(255,176,32,0.15); color: var(--warning); }
    .tag.neutral { background: rgba(255,255,255,0.06); color: var(--muted); }
    .complaint-fix {
      color: var(--text-secondary);
      font-size: 0.85rem;
      margin-bottom: 4px;
    }
    .complaint-quote {
      color: var(--muted);
      font-size: 0.8rem;
      font-style: italic;
    }

    .bullet-list, .visual-list { list-style: none; }
    .bullet-list li, .visual-list li {
      position: relative;
      padding: 7px 0 7px 20px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.9rem;
      color: var(--text-secondary);
    }
    .bullet-list li:last-child, .visual-list li:last-child { border-bottom: none; }
    .bullet-list li::before {
      content: "";
      position: absolute;
      left: 0;
      top: 13px;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 10px var(--accent-glow);
    }
    .visual-list li::before {
      content: "→";
      position: absolute;
      left: 0;
      top: 7px;
      color: var(--accent);
      font-weight: 600;
    }

    .size-box {
      background: rgba(0,0,0,0.25);
      border-radius: 14px;
      padding: 14px;
      font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace;
      font-size: 0.82rem;
      line-height: 1.7;
      color: var(--text-secondary);
      white-space: pre-wrap;
    }

    .empty-state { color: var(--muted); font-size: 0.9rem; }

    .btn.loading { pointer-events: none; opacity: 0.7; }
    .btn.loading::after {
      content: "";
      width: 14px;
      height: 14px;
      border: 2px solid transparent;
      border-top-color: #111;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      margin-left: 4px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Modal for Alexa connect */
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.55);
      backdrop-filter: blur(8px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .modal-backdrop.open { display: flex; }
    .modal {
      width: 100%;
      max-width: 400px;
      padding: 28px 24px;
      text-align: center;
      animation: rise 0.3s ease;
    }
    .modal h3 {
      font-size: 1.25rem;
      font-weight: 750;
      margin-bottom: 8px;
    }
    .modal p {
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 20px;
      line-height: 1.55;
    }
    .modal .phrase {
      background: rgba(0,0,0,0.35);
      border-radius: 14px;
      padding: 14px 16px;
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 18px;
      border: 1px solid var(--glass-border);
    }
    .modal-actions {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .modal-actions .btn { width: 100%; }

    .scan-row {
      display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;
      margin: 14px 0 6px; align-items: center;
    }
    .scan-btn {
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(255,255,255,0.06);
      border: 1px dashed var(--glass-border);
      color: var(--text-secondary);
      padding: 12px 18px; border-radius: var(--radius-pill);
      font-weight: 600; font-size: 0.9rem; font-family: inherit;
      cursor: pointer; transition: border-color 0.15s, background 0.15s, color 0.15s;
    }
    .scan-btn:hover {
      border-color: rgba(255,153,0,0.5); background: var(--accent-soft); color: var(--text);
    }
    .scan-preview {
      display: none; margin: 12px auto 0; max-width: 280px;
      border-radius: 16px; overflow: hidden;
      border: 1px solid var(--glass-border);
    }
    .scan-preview img { width: 100%; display: block; max-height: 180px; object-fit: cover; }
    .scan-status {
      display: none; margin-top: 10px; font-size: 0.85rem; color: var(--muted);
    }
    .scan-status.active { display: block; }
    #scanResultBox {
      display: none; margin-top: 16px; text-align: left; max-width: 560px;
      margin-left: auto; margin-right: auto; padding: 16px 18px;
      border-radius: var(--radius-sm);
      background: rgba(255,153,0,0.08);
      border: 1px solid rgba(255,153,0,0.22);
      animation: rise 0.35s ease;
    }
    #scanResultBox .label {
      font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--accent); margin-bottom: 6px;
    }
    #scanResultBox .metrics {
      display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;
    }
    #scanResultBox .metric {
      background: rgba(0,0,0,0.25); border-radius: 999px;
      padding: 4px 12px; font-size: 0.8rem; color: var(--text-secondary);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="header">
      <div class="logo">
        <div class="logo-mark">🛡️</div>
        ReturnKiller
      </div>
    </header>

    <section class="hero">
      <h1>Ask Alexa before you buy. <span>Skip the return.</span></h1>
      <p>Name a product and your space. Alexa checks whether it fits.</p>
    </section>

    <section class="alexa-main" id="alexaMain">
      <iframe src="/alexa" title="Ask Alexa" class="alexa-frame" allow="microphone; camera"></iframe>
    </section>

    <section class="glass picker">
      <div class="picker-label">Or look up a product yourself</div>
      <form class="picker-row" id="searchForm">
        <input id="productSearch" type="search" placeholder="Search any product, e.g. Ninja air fryer" autocomplete="off" />
        <button class="btn" type="submit" id="searchBtn">Search</button>
      </form>
      <div id="searchResults" class="search-results" role="listbox" aria-label="Search results"></div>
      <div class="picker-row" style="margin-top:14px">
        <button class="btn" id="analyzeBtn" type="button" onclick="runAnalysis()" disabled>Analyze</button>
      </div>
    </section>

    <div id="results">
      <!-- Alexa first -->
      <section class="glass alexa-hero">
        <div class="alexa-kicker">Check your space</div>
        <p id="alexaResponse" class="alexa-spoken">Analyze a product to hear how Alexa answers “will it fit?”</p>

        <div class="fit-input-wrap">
          <input id="fitQuestion" placeholder="Or type a space… e.g. under a 40cm cabinet" />
          <button class="btn btn-soft" onclick="askFit()">Check fit</button>
        </div>

        <div id="fitResult">
          <div class="label">Alexa says</div>
          <div id="fitResultText" class="text"></div>
        </div>
      </section>

      <div class="grid" style="margin-bottom:14px">
        <div class="glass card full" id="riskCard">
          <div class="card-title">Return risk</div>
          <div class="risk-row">
            <div id="riskScore" class="risk-number">—</div>
            <div>
              <div id="riskLevel" class="risk-level">—</div>
              <div id="summary" class="risk-summary"></div>
              <div id="sources" class="risk-summary" style="margin-top:8px"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="glass card" id="reviewsCard" style="display:none;margin-bottom:14px">
        <div class="card-title">What buyers said</div>
        <div id="reviewsList"></div>
        <button class="btn btn-soft" id="reviewsMore" type="button" style="display:none;margin-top:10px">Show more reviews</button>
      </div>

      <div class="grid">
        <div class="glass card">
          <div class="card-title">Customer complaints</div>
          <div id="complaints"></div>
        </div>
        <div class="glass card">
          <div class="card-title">Suggested bullets</div>
          <ul id="bullets" class="bullet-list"></ul>
        </div>
      </div>

      <div class="grid" style="margin-top:14px">
        <div class="glass card">
          <div class="card-title">Size chart</div>
          <div id="sizeChart" class="size-box"></div>
        </div>
        <div class="glass card">
          <div class="card-title">Visual fixes</div>
          <ul id="visuals" class="visual-list"></ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Alexa connect modal -->
  <div class="modal-backdrop" id="alexaModal" onclick="closeAlexaBackdrop(event)">
    <div class="glass modal" onclick="event.stopPropagation()">
      <h3>Ask Alexa</h3>
      <p>Say this on any Echo device, or open the Alexa app and try it there.</p>
      <div class="phrase" id="alexaPhrase">“Alexa, ask Return Killer if this will fit under my cabinet”</div>
      <div class="modal-actions">
        <button class="btn" onclick="copyPhrase()">Copy phrase</button>
        <button class="btn btn-soft" onclick="tryAlexaApp()">Open Alexa app</button>
        <button class="btn btn-soft" onclick="closeAlexa()">Close</button>
      </div>
    </div>
  </div>

  <script>
    let currentAsin = null;
    let currentTitle = "";
    let lastSpoken = "";

    function esc(v) {
      return String(v == null ? '' : v).replace(/[&<>"']/g, ch => (
        {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
    }

    let searchSeq = 0;
    async function searchProducts(q) {
      const seq = ++searchSeq;
      const box = document.getElementById('searchResults');
      box.replaceChildren();
      if (!q) return;
      let data = { products: [], research_available: false };
      try {
        const res = await fetch('/products?limit=5&q=' + encodeURIComponent(q));
        data = await res.json();
      } catch (e) { /* fall through to research */ }
      if (seq !== searchSeq) return;  // a newer search superseded this one
      data.products.forEach(p => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'result-item'; b.setAttribute('role', 'option');
        b.setAttribute('aria-selected', 'false');
        b.dataset.asin = p.asin; b.dataset.title = p.short_title || p.title;
        b.textContent = p.short_title || p.title;
        const sm = document.createElement('small');
        sm.textContent = [p.category, p.price != null ? '$' + p.price : null].filter(Boolean).join(' · ');
        b.append(sm);
        b.addEventListener('click', () => selectProduct(b));
        box.append(b);
      });
      if (data.research_available) {
        // Every search does a real web lookup, then analyzes what it found.
        const rb = researchButton(q);
        box.prepend(rb);
        researchProduct(q, rb, true);
      } else if (!data.products.length) {
        const d = document.createElement('div');
        d.className = 'search-empty';
        d.textContent = 'No match. Live web search is not enabled on this server.';
        box.append(d);
      } else if (data.products.length === 1) {
        selectProduct(box.firstChild);
      }
    }

    function researchButton(q) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'result-item research-item';
      b.textContent = 'Searching the web for "' + q + '"...';
      const sm = document.createElement('small');
      sm.textContent = 'Reading current sources. Takes 10-20 seconds.';
      b.append(sm);
      b.addEventListener('click', () => researchProduct(q, b));
      return b;
    }

    async function researchProduct(q, btn, thenAnalyze) {
      btn.disabled = true; btn.firstChild.textContent = 'Searching the web for "' + q + '"...';
      try {
        const res = await fetch('/research?q=' + encodeURIComponent(q));
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Research failed');
        const p = data.product;
        const item = document.createElement('button');
        item.type = 'button'; item.className = 'result-item'; item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', 'false');
        item.dataset.asin = p.asin; item.dataset.title = p.short_title || p.title;
        item.textContent = p.short_title || p.title;
        const sm = document.createElement('small');
        sm.textContent = 'Found on the web' + (p.category ? ' · ' + p.category : '');
        item.append(sm);
        item.addEventListener('click', () => selectProduct(item));
        btn.replaceWith(item);
        selectProduct(item);
        if (thenAnalyze) runAnalysis();
      } catch (e) {
        btn.disabled = false; btn.firstChild.textContent = (e.message || 'Search failed') + '. Click to try again.';
      }
    }

    function selectProduct(btn) {
      document.querySelectorAll('.result-item').forEach(x => x.setAttribute('aria-selected', String(x === btn)));
      currentAsin = btn.dataset.asin;
      document.getElementById('analyzeBtn').disabled = false;
      document.getElementById('analyzeBtn').textContent = 'Analyze ' + btn.dataset.title.slice(0, 32);
    }

    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('searchForm').addEventListener('submit', (ev) => {
        ev.preventDefault();
        searchProducts(document.getElementById('productSearch').value.trim());
      });
    });

    function renderReviews(reviews) {
      const card = document.getElementById('reviewsCard');
      const list = document.getElementById('reviewsList');
      const more = document.getElementById('reviewsMore');
      list.replaceChildren();
      card.style.display = reviews.length ? 'block' : 'none';
      const draw = (r) => {
        const d = document.createElement('div'); d.className = 'review';
        const h = document.createElement('div'); h.className = 'review-head';
        const st = document.createElement('span'); st.className = 'stars';
        st.textContent = '★'.repeat(r.rating) + '☆'.repeat(5 - r.rating);
        st.setAttribute('aria-label', r.rating + ' out of 5 stars');
        const t = document.createElement('span'); t.textContent = r.title || '';
        h.append(st, t);
        const p = document.createElement('p'); p.textContent = r.text || '';
        d.append(h, p); list.append(d);
      };
      reviews.slice(0, 4).forEach(draw);
      more.style.display = reviews.length > 4 ? 'inline-flex' : 'none';
      more.onclick = () => { reviews.slice(4).forEach(draw); more.style.display = 'none'; };
    }

    async function runAnalysis() {
      const asin = currentAsin;
      if (!asin) return;

      const btn = document.getElementById('analyzeBtn');
      btn.classList.add('loading');
      btn.disabled = true;

      try {
        const res = await fetch(`/analyze/${asin}`);
        const data = await res.json();

        document.getElementById('results').style.display = 'block';
        currentTitle = data.title || "";
        lastSpoken = data.alexa_fit_response || "";

        const score = data.return_risk_score;
        const level = score >= 65 ? 'high' : score >= 40 ? 'medium' : 'low';
        const levelLabel = level === 'high' ? 'High Risk' : level === 'medium' ? 'Medium Risk' : 'Low Risk';

        const scoreEl = document.getElementById('riskScore');
        scoreEl.textContent = score;
        scoreEl.className = 'risk-number ' + level;

        const levelEl = document.getElementById('riskLevel');
        levelEl.textContent = levelLabel;
        levelEl.className = 'risk-level ' + level;

        let summary = data.summary || '';
        summary = summary.replace(/^Return Risk:.*\\n\\n?/i, '');
        document.getElementById('summary').textContent = summary.trim() +
          (data.risk_basis === 'review_text_proxy'
            ? ' (Estimated from ' + (data.reviews_analyzed || 0) + ' review texts; Amazon does not publish return reasons.)'
            : '');
        const srcEl = document.getElementById('sources');
        srcEl.replaceChildren();
        (data.sources || []).forEach(u => {
          if (!/^https?:\\/\\//.test(u)) return;
          const a = document.createElement('a');
          a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
          try { a.textContent = new URL(u).hostname.replace(/^www\\./, ''); } catch (e) { a.textContent = u; }
          a.style.marginRight = '12px';
          srcEl.append(a);
        });
        if (srcEl.children.length) srcEl.prepend(document.createTextNode('Sources: '));

        const complaintsEl = document.getElementById('complaints');
        if (!data.top_complaints || data.top_complaints.length === 0) {
          complaintsEl.innerHTML = '<p class="empty-state">No major complaint themes detected.</p>';
        } else {
          complaintsEl.innerHTML = data.top_complaints.map(c => `
            <div class="complaint">
              <div class="complaint-top">
                <span class="complaint-theme">${esc(c.theme)}</span>
                <span class="tag ${esc(c.severity)}">${esc(c.severity)}</span>
                <span class="tag neutral">${c.frequency ? c.frequency + ' mention' + (c.frequency !== 1 ? 's' : '') : 'reported online'}</span>
              </div>
              ${c.detail ? `<p class="complaint-fix">${esc(c.detail)}</p>` : ''}
              <p class="complaint-fix">${esc(c.suggested_fix)}</p>
              ${c.example_quotes && c.example_quotes[0] ? `<p class="complaint-quote">“${esc(c.example_quotes[0])}”</p>` : ''}
            </div>
          `).join('');
        }

        document.getElementById('bullets').innerHTML =
          (data.improved_bullets || []).map(b => `<li>${esc(b)}</li>`).join('') ||
          '<li class="empty-state">No suggestions</li>';

        renderReviews(data.reviews || []);
        const frame = document.querySelector('.alexa-frame');
        if (frame && frame.contentWindow) {
          frame.contentWindow.postMessage({ type: 'rk-product', asin: asin, title: data.title || '',
            short_title: (data.title || '').split(' - ')[0].slice(0, 60) }, location.origin);
        }
        document.getElementById('sizeChart').textContent = data.size_chart_text || '—';
        document.getElementById('alexaResponse').textContent = data.alexa_fit_response || '—';
        document.getElementById('fitResult').style.display = 'none';

        document.getElementById('visuals').innerHTML =
          (data.visual_suggestions || []).map(v => `<li>${esc(v)}</li>`).join('') ||
          '<li class="empty-state">No suggestions</li>';

        // Update Alexa phrase with product context
        const short = (currentTitle.split(' - ')[0] || 'this product').slice(0, 40);
        document.getElementById('alexaPhrase').textContent =
          `“Alexa, ask Return Killer if the ${short} will fit under my cabinet”`;

        document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (e) {
        alert('Analysis failed. Is the server running?');
      } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
      }
    }

    async function askFit() {
      const question = document.getElementById('fitQuestion').value.trim() || 'Will this fit?';
      if (!currentAsin) return;

      try {
        const res = await fetch('/fit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            asin: currentAsin,
            question: question,
            space_description: question
          })
        });
        const data = await res.json();
        document.getElementById('fitResultText').textContent = data.spoken_response;
        document.getElementById('fitResult').style.display = 'block';
        lastSpoken = data.spoken_response;
      } catch (e) {
        alert('Fit request failed.');
      }
    }

    function openAlexa() {
      document.getElementById('alexaModal').classList.add('open');
    }
    function closeAlexa() {
      document.getElementById('alexaModal').classList.remove('open');
    }
    function closeAlexaBackdrop(e) {
      if (e.target === document.getElementById('alexaModal')) closeAlexa();
    }
    function copyPhrase() {
      const text = document.getElementById('alexaPhrase').textContent.replace(/[“”]/g, '');
      navigator.clipboard.writeText(text).then(() => {
        const btn = event.target;
        const prev = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = prev; }, 1500);
      });
    }
    function tryAlexaApp() {
      // Deep link attempts — works when Alexa app is installed
      // Skill invocation isn't universally linkable pre-publish; this opens Alexa
      const phrase = encodeURIComponent(
        document.getElementById('alexaPhrase').textContent.replace(/[“”]/g, '')
      );
      // Try Alexa app schemes (best-effort)
      window.location.href = 'https://alexa.amazon.com/';
      // Fallback note is already in the modal
    }

    async function onSpacePhoto(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      if (!currentAsin) {
        alert('Analyze a product first, then scan your space.');
        event.target.value = '';
        return;
      }

      const preview = document.getElementById('scanPreview');
      const img = document.getElementById('scanPreviewImg');
      img.src = URL.createObjectURL(file);
      preview.style.display = 'block';

      const status = document.getElementById('scanStatus');
      status.textContent = 'Analyzing your space with vision…';
      status.classList.add('active');
      document.getElementById('scanResultBox').style.display = 'none';

      const form = new FormData();
      form.append('file', file);
      form.append('asin', currentAsin);
      const hint = document.getElementById('fitQuestion').value.trim();
      if (hint) form.append('hint', hint);

      try {
        const res = await fetch('/scan', { method: 'POST', body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Scan failed');
        }
        const data = await res.json();
        status.classList.remove('active');

        const spoken = data.spoken_summary || 'Scan complete.';
        document.getElementById('scanResultText').textContent = spoken;
        const metrics = [];
        if (data.space_type) metrics.push('Type: ' + data.space_type);
        if (data.estimated_clearance_height_cm != null) metrics.push('Height ~ ' + data.estimated_clearance_height_cm + ' cm');
        if (data.estimated_clearance_width_cm != null) metrics.push('Width ~ ' + data.estimated_clearance_width_cm + ' cm');
        if (data.estimated_depth_cm != null) metrics.push('Depth ~ ' + data.estimated_depth_cm + ' cm');
        if (data.fit_verdict) metrics.push('Fit: ' + String(data.fit_verdict).replace(/_/g, ' '));
        if (data.confidence) metrics.push('Confidence: ' + data.confidence);
        if (data.engine) metrics.push('Engine: ' + data.engine);
        document.getElementById('scanMetrics').innerHTML = metrics.map(m => '<span class="metric">' + esc(m) + '</span>').join('');
        document.getElementById('scanResultBox').style.display = 'block';

        document.getElementById('fitResultText').textContent = spoken + (data.advice ? ' ' + data.advice : '');
        document.getElementById('fitResult').style.display = 'block';
        lastSpoken = spoken;
      } catch (e) {
        status.textContent = e.message || 'Scan failed';
        status.classList.add('active');
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      const input = document.getElementById('fitQuestion');
      if (input) {
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') askFit();
        });
      }
    });
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@app.get("/alexa", response_class=HTMLResponse)
def alexa_simulator():
    """Simulated Alexa+ experience: a browser MCP client talking to /mcp."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "alexa.html")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ---- MCP server (Alexa+) --------------------------------------------------
# Mounted last at "/" so every route above wins; only /mcp falls through to it.
def _research_summary_for(query: str) -> Optional[Dict[str, Any]]:
    product = _research(query)
    return _research_summary(product) if product else None


_mcp = build_mcp(_find_product, _get_analysis, search_products, _research_summary_for)
app.mount("/", _mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)