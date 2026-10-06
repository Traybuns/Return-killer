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
    """Polished interactive demo UI."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ReturnKiller — Stop preventable returns</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f14;
      --bg-elevated: #111820;
      --card: #151d28;
      --card-hover: #1a2433;
      --border: rgba(255,255,255,0.06);
      --border-strong: rgba(255,255,255,0.1);
      --accent: #ff9900;
      --accent-soft: rgba(255,153,0,0.12);
      --accent-glow: rgba(255,153,0,0.25);
      --text: #f0f2f5;
      --text-secondary: #c5cdd6;
      --muted: #7a8694;
      --danger: #ff4757;
      --danger-soft: rgba(255,71,87,0.12);
      --success: #00d68f;
      --success-soft: rgba(0,214,143,0.12);
      --warning: #ffb020;
      --radius: 16px;
      --radius-sm: 10px;
      --shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255,153,0,0.08), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(0,214,143,0.04), transparent);
    }

    .header {
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(16px);
      background: rgba(11,15,20,0.85);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
    }
    .header-inner {
      max-width: 1080px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .logo-mark {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--accent), #ff6b00);
      border-radius: 8px;
      display: grid;
      place-items: center;
      font-size: 16px;
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .badge {
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 3px 8px;
      border-radius: 20px;
      letter-spacing: 0.03em;
    }

    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 24px 80px;
    }

    .hero { margin-bottom: 36px; }
    .hero h1 {
      font-size: clamp(1.8rem, 4vw, 2.4rem);
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.2;
      margin-bottom: 10px;
    }
    .hero h1 span { color: var(--accent); }
    .hero p {
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 520px;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 28px;
      box-shadow: var(--shadow);
    }
    .panel-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: stretch;
    }
    select {
      flex: 1;
      min-width: 240px;
      appearance: none;
      background: var(--bg-elevated) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%237a8694' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E") no-repeat right 14px center;
      border: 1px solid var(--border-strong);
      color: var(--text);
      padding: 12px 40px 12px 16px;
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      font-family: inherit;
      cursor: pointer;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    select:hover, select:focus {
      border-color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: linear-gradient(135deg, var(--accent), #ff6b00);
      color: #111;
      border: none;
      padding: 12px 22px;
      border-radius: var(--radius-sm);
      font-weight: 650;
      font-size: 0.95rem;
      font-family: inherit;
      cursor: pointer;
      transition: transform 0.12s, box-shadow 0.15s, filter 0.15s;
      white-space: nowrap;
      box-shadow: 0 4px 16px var(--accent-glow);
    }
    .btn:hover {
      filter: brightness(1.08);
      transform: translateY(-1px);
      box-shadow: 0 6px 24px var(--accent-glow);
    }
    .btn:active { transform: translateY(0); }
    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }
    .btn-ghost {
      background: transparent;
      color: var(--text-secondary);
      border: 1px solid var(--border-strong);
      box-shadow: none;
    }
    .btn-ghost:hover {
      background: var(--card-hover);
      border-color: var(--muted);
      filter: none;
      box-shadow: none;
    }

    #results {
      display: none;
      animation: fadeUp 0.4s ease;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    @media (max-width: 720px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px;
      transition: border-color 0.2s, background 0.2s;
    }
    .card:hover { border-color: var(--border-strong); }
    .card.full { grid-column: 1 / -1; }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
      gap: 12px;
    }
    .card-title {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
    }

    .risk-hero {
      display: flex;
      align-items: center;
      gap: 24px;
      flex-wrap: wrap;
    }
    .risk-number {
      font-size: 3.2rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      line-height: 1;
    }
    .risk-number.high { color: var(--danger); }
    .risk-number.medium { color: var(--warning); }
    .risk-number.low { color: var(--success); }
    .risk-meta { flex: 1; min-width: 180px; }
    .risk-level {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 4px 10px;
      border-radius: 20px;
      margin-bottom: 8px;
    }
    .risk-level.high { background: var(--danger-soft); color: var(--danger); }
    .risk-level.medium { background: rgba(255,176,32,0.15); color: var(--warning); }
    .risk-level.low { background: var(--success-soft); color: var(--success); }
    .risk-summary {
      color: var(--text-secondary);
      font-size: 0.9rem;
      white-space: pre-line;
      line-height: 1.6;
    }

    .complaint {
      background: var(--bg-elevated);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      margin-bottom: 10px;
      border-left: 3px solid var(--accent);
    }
    .complaint:last-child { margin-bottom: 0; }
    .complaint-top {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }
    .complaint-theme { font-weight: 600; font-size: 0.95rem; }
    .tag {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 2px 8px;
      border-radius: 20px;
    }
    .tag.high { background: var(--danger-soft); color: var(--danger); }
    .tag.medium { background: rgba(255,176,32,0.15); color: var(--warning); }
    .tag.neutral { background: rgba(255,255,255,0.06); color: var(--muted); }
    .complaint-fix {
      color: var(--text-secondary);
      font-size: 0.88rem;
      margin-bottom: 6px;
    }
    .complaint-quote {
      color: var(--muted);
      font-size: 0.82rem;
      font-style: italic;
      border-left: 2px solid var(--border-strong);
      padding-left: 10px;
    }

    .bullet-list, .visual-list { list-style: none; }
    .bullet-list li, .visual-list li {
      position: relative;
      padding: 8px 0 8px 22px;
      border-bottom: 1px solid var(--border);
      font-size: 0.92rem;
      color: var(--text-secondary);
    }
    .bullet-list li:last-child, .visual-list li:last-child { border-bottom: none; }
    .bullet-list li::before {
      content: "";
      position: absolute;
      left: 0;
      top: 14px;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }
    .visual-list li::before {
      content: "→";
      position: absolute;
      left: 0;
      top: 8px;
      color: var(--accent);
      font-weight: 600;
    }

    .size-box {
      background: var(--bg-elevated);
      border-radius: var(--radius-sm);
      padding: 16px;
      font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace;
      font-size: 0.85rem;
      line-height: 1.7;
      color: var(--text-secondary);
      white-space: pre-wrap;
    }

    .alexa-block {
      background: linear-gradient(135deg, rgba(255,153,0,0.06), rgba(0,214,143,0.04));
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 16px;
      margin-bottom: 14px;
    }
    .alexa-label {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      margin-bottom: 8px;
    }
    .alexa-text {
      color: var(--text-secondary);
      font-size: 0.95rem;
      line-height: 1.6;
    }
    .fit-row {
      display: flex;
      gap: 10px;
      margin-top: 4px;
    }
    .fit-row input {
      flex: 1;
      background: var(--bg-elevated);
      border: 1px solid var(--border-strong);
      color: var(--text);
      padding: 12px 14px;
      border-radius: var(--radius-sm);
      font-size: 0.92rem;
      font-family: inherit;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .fit-row input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .fit-row input::placeholder { color: var(--muted); }
    #fitResult {
      display: none;
      margin-top: 12px;
      animation: fadeUp 0.3s ease;
    }

    .empty-state {
      color: var(--muted);
      font-size: 0.9rem;
      padding: 8px 0;
    }

    .btn.loading {
      pointer-events: none;
      opacity: 0.7;
    }
    .btn.loading::after {
      content: "";
      width: 14px;
      height: 14px;
      border: 2px solid transparent;
      border-top-color: #111;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      margin-left: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header class="header">
    <div class="header-inner">
      <div class="logo">
        <div class="logo-mark">🛡️</div>
        ReturnKiller
      </div>
      <span class="badge">Hackathon Demo</span>
    </div>
  </header>

  <main>
    <section class="hero">
      <h1>Stop <span>size & description</span> returns before they happen</h1>
      <p>Analyze any listing for return risk, get concrete fixes, and preview the Alexa “will it fit?” experience.</p>
    </section>

    <section class="panel">
      <div class="panel-label">Select a product to analyze</div>
      <div class="controls">
        <select id="productSelect">
          <option value="">Loading products…</option>
        </select>
        <button class="btn" id="analyzeBtn" onclick="runAnalysis()">
          Analyze Listing
        </button>
      </div>
    </section>

    <div id="results">
      <div class="grid">
        <div class="card full">
          <div class="card-header">
            <span class="card-title">Return Risk Score</span>
          </div>
          <div class="risk-hero">
            <div id="riskScore" class="risk-number">—</div>
            <div class="risk-meta">
              <div id="riskLevel" class="risk-level">—</div>
              <div id="summary" class="risk-summary"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Top Customer Complaints</span>
          </div>
          <div id="complaints"></div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Suggested Bullets</span>
          </div>
          <ul id="bullets" class="bullet-list"></ul>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Size Chart & Fit Guide</span>
          </div>
          <div id="sizeChart" class="size-box"></div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Visual Fixes</span>
          </div>
          <ul id="visuals" class="visual-list"></ul>
        </div>
      </div>

      <div class="card full" style="margin-top:16px">
        <div class="card-header">
          <span class="card-title">Alexa · “Will it fit?” Experience</span>
        </div>
        <div class="alexa-block">
          <div class="alexa-label">Spoken response on Echo Show</div>
          <div id="alexaResponse" class="alexa-text"></div>
        </div>
        <div class="fit-row">
          <input id="fitQuestion" placeholder="Ask a follow-up… e.g. will it fit under a 40cm cabinet?" />
          <button class="btn btn-ghost" onclick="askFit()">Ask Alexa</button>
        </div>
        <div id="fitResult" class="alexa-block">
          <div class="alexa-label">Alexa replies</div>
          <div id="fitResultText" class="alexa-text"></div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentAsin = null;

    async function loadProducts() {
      try {
        const res = await fetch('/products');
        const data = await res.json();
        const select = document.getElementById('productSelect');
        select.innerHTML = data.products.map(p =>
          `<option value="${p.asin}">${p.title}</option>`
        ).join('');
      } catch (e) {
        document.getElementById('productSelect').innerHTML =
          '<option value="">Could not load products</option>';
      }
    }

    async function runAnalysis() {
      const asin = document.getElementById('productSelect').value;
      if (!asin) return;
      currentAsin = asin;

      const btn = document.getElementById('analyzeBtn');
      btn.classList.add('loading');
      btn.disabled = true;

      try {
        const res = await fetch(`/analyze/${asin}`);
        const data = await res.json();

        document.getElementById('results').style.display = 'block';

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
        document.getElementById('summary').textContent = summary.trim();

        const complaintsEl = document.getElementById('complaints');
        if (!data.top_complaints || data.top_complaints.length === 0) {
          complaintsEl.innerHTML = '<p class="empty-state">No major complaint themes detected.</p>';
        } else {
          complaintsEl.innerHTML = data.top_complaints.map(c => `
            <div class="complaint">
              <div class="complaint-top">
                <span class="complaint-theme">${c.theme}</span>
                <span class="tag ${c.severity}">${c.severity}</span>
                <span class="tag neutral">${c.frequency} mention${c.frequency !== 1 ? 's' : ''}</span>
              </div>
              <p class="complaint-fix">${c.suggested_fix}</p>
              ${c.example_quotes && c.example_quotes[0] ? `<p class="complaint-quote">“${c.example_quotes[0]}”</p>` : ''}
            </div>
          `).join('');
        }

        document.getElementById('bullets').innerHTML =
          (data.improved_bullets || []).map(b => `<li>${b}</li>`).join('') ||
          '<li class="empty-state">No suggestions</li>';

        document.getElementById('sizeChart').textContent = data.size_chart_text || '—';
        document.getElementById('alexaResponse').textContent = data.alexa_fit_response || '—';
        document.getElementById('fitResult').style.display = 'none';

        document.getElementById('visuals').innerHTML =
          (data.visual_suggestions || []).map(v => `<li>${v}</li>`).join('') ||
          '<li class="empty-state">No suggestions</li>';

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
        const el = document.getElementById('fitResult');
        document.getElementById('fitResultText').textContent = data.spoken_response;
        el.style.display = 'block';
      } catch (e) {
        alert('Fit request failed.');
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      loadProducts();
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
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
