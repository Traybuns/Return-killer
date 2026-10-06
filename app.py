"""
ReturnKiller API
Simple FastAPI server that exposes the analysis engine.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os

from analyzer import ReturnKillerAnalyzer, load_sample_products

app = FastAPI(
    title="ReturnKiller API",
    description="AI agent that reduces Amazon returns caused by size & description mismatches",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (set use_bedrock=True when AWS credentials are available)
analyzer = ReturnKillerAnalyzer(use_bedrock=False)

# In-memory cache of analyses
analysis_cache: Dict[str, Any] = {}


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
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "GET /products": "List sample products",
            "GET /analyze/{asin}": "Analyze a sample product by ASIN",
            "POST /analyze": "Analyze a custom product payload",
            "POST /fit": "Ask a fit question (Alexa-style)",
            "GET /demo": "Simple HTML demo UI"
        }
    }


@app.get("/products")
def list_products():
    products = load_sample_products()
    return {
        "count": len(products),
        "products": [
            {
                "asin": p["asin"],
                "title": p["title"],
                "category": p.get("category"),
                "review_count": len(p.get("reviews", [])),
            }
            for p in products
        ]
    }


@app.get("/analyze/{asin}")
def analyze_by_asin(asin: str):
    products = load_sample_products()
    product = next((p for p in products if p["asin"] == asin), None)
    
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {asin} not found in sample data")
    
    result = analyzer.analyze_product(product)
    result_dict = analyzer.to_dict(result)
    analysis_cache[asin] = result_dict
    return result_dict


@app.post("/analyze")
def analyze_custom(req: AnalyzeRequest):
    if req.product:
        product = req.product
    elif req.asin:
        products = load_sample_products()
        product = next((p for p in products if p["asin"] == req.asin), None)
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
    
    # Get or create analysis
    if req.asin in analysis_cache:
        analysis = analysis_cache[req.asin]
    else:
        products = load_sample_products()
        product = next((p for p in products if p["asin"] == req.asin), None)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        result = analyzer.analyze_product(product)
        analysis = analyzer.to_dict(result)
        analysis_cache[req.asin] = analysis
    
    # Build response
    spoken = analysis["alexa_fit_response"]
    
    # If user provided space context, enhance the answer
    if req.space_description:
        spoken += f" Based on your description of '{req.space_description}', "
        spoken += "I recommend measuring the exact opening and comparing it to the dimensions on screen."
    
    return {
        "asin": req.asin,
        "question": req.question,
        "spoken_response": spoken,
        "size_chart": analysis["size_chart_text"],
        "visual_suggestions": analysis["visual_suggestions"],
        "risk_score": analysis["return_risk_score"],
        "screen_title": f"Size Check: {analysis['title'][:40]}",
        "dimensions_summary": analysis.get("size_chart_text", "").split("\n")[0] if analysis.get("size_chart_text") else "Dimensions unavailable"
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    """Minimal HTML demo so we can click around immediately."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ReturnKiller Demo</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --accent: #ff9900;
      --text: #e7e9ea;
      --muted: #8b98a5;
      --danger: #f4212e;
      --success: #00ba7c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
      max-width: 960px;
      margin: 0 auto;
    }
    h1 { font-size: 1.8rem; margin-bottom: 4px; }
    .subtitle { color: var(--muted); margin-bottom: 32px; }
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
      border: 1px solid #2f3336;
    }
    .card h2 { font-size: 1.1rem; margin-bottom: 12px; color: var(--accent); }
    button {
      background: var(--accent);
      color: #000;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.95rem;
    }
    button:hover { filter: brightness(1.1); }
    button.secondary { background: #2f3336; color: var(--text); }
    select, input {
      background: #0f1419;
      border: 1px solid #2f3336;
      color: var(--text);
      padding: 10px;
      border-radius: 8px;
      width: 100%;
      margin-bottom: 12px;
      font-size: 0.95rem;
    }
    .risk {
      font-size: 2rem;
      font-weight: 700;
    }
    .risk.high { color: var(--danger); }
    .risk.medium { color: var(--accent); }
    .risk.low { color: var(--success); }
    pre {
      background: #0f1419;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 0.85rem;
      white-space: pre-wrap;
    }
    .complaint {
      border-left: 3px solid var(--accent);
      padding-left: 12px;
      margin-bottom: 12px;
    }
    .tag {
      display: inline-block;
      background: #2f3336;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      margin-right: 6px;
    }
    .tag.high { background: #3d1214; color: #ff7a7a; }
    .tag.medium { background: #3d2e00; color: #ffcc66; }
    #results { display: none; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .row > * { flex: 1; min-width: 200px; }
  </style>
</head>
<body>
  <h1>🛡️ ReturnKiller</h1>
  <p class="subtitle">Stop size & description returns before they happen</p>

  <div class="card">
    <h2>1. Choose a product</h2>
    <select id="productSelect">
      <option value="">Loading products...</option>
    </select>
    <button onclick="runAnalysis()">Analyze Listing</button>
  </div>

  <div id="results">
    <div class="card">
      <h2>Return Risk Score</h2>
      <div id="riskScore" class="risk">—</div>
      <pre id="summary"></pre>
    </div>

    <div class="card">
      <h2>Top Customer Complaints</h2>
      <div id="complaints"></div>
    </div>

    <div class="card">
      <h2>Improved Bullets (suggested)</h2>
      <ul id="bullets" style="padding-left: 20px;"></ul>
    </div>

    <div class="card">
      <h2>Size Chart</h2>
      <pre id="sizeChart"></pre>
    </div>

    <div class="card">
      <h2>Alexa “Will it fit?” Response</h2>
      <p style="margin-bottom: 12px; color: var(--muted);">Simulated Echo Show spoken response:</p>
      <pre id="alexaResponse"></pre>
      <div style="margin-top: 12px;">
        <input id="fitQuestion" placeholder="Ask a follow-up, e.g. will it fit under a 40cm cabinet?" />
        <button class="secondary" onclick="askFit()">Ask Alexa</button>
      </div>
      <pre id="fitResult" style="margin-top: 12px; display:none;"></pre>
    </div>

    <div class="card">
      <h2>Visual Suggestions</h2>
      <ul id="visuals" style="padding-left: 20px;"></ul>
    </div>
  </div>

  <script>
    let currentAsin = null;

    async function loadProducts() {
      const res = await fetch('/products');
      const data = await res.json();
      const select = document.getElementById('productSelect');
      select.innerHTML = data.products.map(p =>
        `<option value="${p.asin}">${p.title} (${p.asin})</option>`
      ).join('');
    }

    async function runAnalysis() {
      const asin = document.getElementById('productSelect').value;
      if (!asin) return;
      currentAsin = asin;

      const res = await fetch(`/analyze/${asin}`);
      const data = await res.json();

      document.getElementById('results').style.display = 'block';

      const scoreEl = document.getElementById('riskScore');
      scoreEl.textContent = data.return_risk_score + ' / 100';
      scoreEl.className = 'risk ' + (
        data.return_risk_score >= 65 ? 'high' :
        data.return_risk_score >= 40 ? 'medium' : 'low'
      );

      document.getElementById('summary').textContent = data.summary;

      const complaintsEl = document.getElementById('complaints');
      if (data.top_complaints.length === 0) {
        complaintsEl.innerHTML = '<p style="color:var(--muted)">No major complaint themes detected.</p>';
      } else {
        complaintsEl.innerHTML = data.top_complaints.map(c => `
          <div class="complaint">
            <strong>${c.theme}</strong>
            <span class="tag ${c.severity}">${c.severity}</span>
            <span class="tag">${c.frequency} mentions</span>
            <p style="margin-top:6px;color:var(--muted);font-size:0.9rem">${c.suggested_fix}</p>
            <p style="margin-top:4px;font-size:0.85rem;font-style:italic">"${c.example_quotes[0] || ''}"</p>
          </div>
        `).join('');
      }

      document.getElementById('bullets').innerHTML =
        data.improved_bullets.map(b => `<li style="margin-bottom:6px">${b}</li>`).join('');

      document.getElementById('sizeChart').textContent = data.size_chart_text;
      document.getElementById('alexaResponse').textContent = data.alexa_fit_response;

      document.getElementById('visuals').innerHTML =
        data.visual_suggestions.map(v => `<li style="margin-bottom:6px">${v}</li>`).join('');
    }

    async function askFit() {
      const question = document.getElementById('fitQuestion').value || 'Will this fit?';
      if (!currentAsin) return;

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
      const el = document.getElementById('fitResult');
      el.style.display = 'block';
      el.textContent = data.spoken_response;
    }

    loadProducts();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
