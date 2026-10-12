"""ReturnKiller MCP server (Streamable HTTP, stateless).

Exposes the analysis engine to Alexa+ (or any MCP client) as three read-only tools.
Mounted into the FastAPI app so one Lambda serves the REST API, the demo and /mcp.
"""
from typing import Any, Callable, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from analyzer import fit_check

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def _level(score: float) -> str:
    return "high" if score >= 65 else "medium" if score >= 40 else "low"


def build_mcp(
    get_product: Callable[[str], Optional[Dict[str, Any]]],
    get_analysis: Callable[[str], Dict[str, Any]],
    research: Callable[[str], Optional[Dict[str, Any]]],
) -> FastMCP:
    mcp = FastMCP(
        "ReturnKiller",
        instructions=(
            "ReturnKiller tells shoppers whether an Amazon product will fit their space or match "
            "the listing before they order, so they don't have to return it. Every product is "
            "looked up live on the web. Call research_product with the product the shopper named, "
            "then check_fit when they describe a space, or analyze_listing for return risk. "
            "Answer only what the shopper asked. Read the 'spoken' field aloud."
        ),
        stateless_http=True,
        json_response=True,
        # Runs behind CloudFront with an origin secret, not on localhost, so the
        # localhost-oriented DNS-rebinding host check does not apply.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @mcp.tool(name="research_product", annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
    def research_product_tool(query: str) -> Dict[str, Any]:
        """Look up a product on Amazon and the web (can take up to ~20 s). Returns the product
        with an id usable by check_fit and analyze_listing."""
        try:
            product = research(query)
        except RuntimeError as e:
            if "quota" in str(e).lower():
                return {"product": None, "spoken": str(e)}
            return {"product": None, "spoken": "I couldn't reach my web research right now. Please try again."}
        if not product:
            return {"product": None, "spoken": f"I couldn't find a product called {query}. Try the full product name."}
        return {"product": product, "spoken": f"I found {product['short_title']}."}

    @mcp.tool(annotations=READ_ONLY)
    def check_fit(asin: str, space_description: str = "") -> Dict[str, Any]:
        """Will this product fit? Pass the ASIN and the space the shopper described,
        e.g. 'under my 40 cm cabinet' or '15 inches of clearance'. Returns a verdict
        computed from the listed dimensions, not estimated by a model."""
        product = get_product(asin)
        if not product:
            return {"verdict": "unknown", "spoken": f"I couldn't find a product with ASIN {asin}."}
        result = fit_check(product, space_description)
        result["asin"] = asin
        result["dimensions_cm"] = product.get("dimensions")
        return result

    @mcp.tool(annotations=READ_ONLY)
    def analyze_listing(asin: str) -> Dict[str, Any]:
        """Return risk analysis for a product: score, top complaint themes grounded in
        real reviews, and the listing changes that would prevent returns."""
        analysis = get_analysis(asin)
        if not analysis:
            return {"spoken": f"I couldn't find a product with ASIN {asin}."}
        score = analysis["return_risk_score"]
        complaints = analysis.get("top_complaints") or []
        top = complaints[0] if complaints else None
        seen = analysis.get("reviews_analyzed") or 0
        proxy = analysis.get("risk_basis") == "review_text_proxy"
        spoken = f"The return risk is {_level(score)}, {score:.0f} out of 100"
        web = analysis.get("risk_basis") == "web_research"
        spoken += (", estimated from what reviewers wrote." if proxy
                   else ", from web research rather than store reviews." if web else ".")
        if top and web:
            spoken += f" Buyers commonly report {top['theme'].lower()}."
        elif top:
            if seen:
                spoken += (f" The biggest issue is {top['theme'].lower()}, mentioned in "
                           f"{top['frequency']} of the {seen} reviews I looked at.")
            else:
                spoken += f" The biggest issue is {top['theme'].lower()}, mentioned by {top['frequency']} customers."
        return {
            "asin": asin,
            "title": analysis["title"],
            "return_risk_score": score,
            "risk_level": _level(score),
            "risk_basis": analysis.get("risk_basis"),
            "reviews_analyzed": seen,
            "sources": (analysis.get("sources") or [])[:5],
            "top_complaints": [
                {"theme": c["theme"], "mentions": c["frequency"], "detail": c.get("detail", ""), "severity": c["severity"],
                 "fix": c["suggested_fix"]} for c in complaints[:3]
            ],
            "size_chart": analysis.get("size_chart_text"),
            "improved_bullets": (analysis.get("improved_bullets") or [])[:3],
            "engine": analysis.get("engine"),
            "spoken": spoken,
        }

    return mcp
