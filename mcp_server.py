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
    search: Callable[..., Dict[str, Any]],
    research: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> FastMCP:
    mcp = FastMCP(
        "ReturnKiller",
        instructions=(
            "ReturnKiller tells shoppers whether an Amazon product will actually fit or match "
            "their expectations before they order, so they don't have to return it. Call "
            "search_products to find a product by name (if it returns nothing, call research_product, which looks it up on the web), check_fit when the shopper describes a "
            "space, and analyze_listing for return risk. Read the 'spoken' field aloud."
        ),
        stateless_http=True,
        json_response=True,
        # Runs behind CloudFront with an origin secret, not on localhost, so the
        # localhost-oriented DNS-rebinding host check does not apply.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    def _find(query: str, limit: int) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 5))
        res = search(query, limit=limit)
        items = [
            {k: p[k] for k in ("asin", "short_title", "category", "price", "has_height")}
            for p in res["products"]
        ]
        total = res["total"]
        if not items:
            spoken = f"I couldn't find anything matching {query}." if query else "I have no products loaded."
        else:
            names = "; ".join(p["short_title"] for p in items)
            more = f" There are {total} matches in all." if total > len(items) else ""
            spoken = f"I found {total} matching product{'s' if total != 1 else ''}. Top: {names}.{more}"
        return {"total": total, "products": items, "spoken": spoken}

    @mcp.tool(name="search_products", annotations=READ_ONLY)
    def search_products_tool(query: str = "", limit: int = 5) -> Dict[str, Any]:
        """Search the catalog by product name or category, e.g. 'air fryer'.
        Returns at most 5 products with their ASINs."""
        return _find(query, limit)

    @mcp.tool(name="research_product", annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
    def research_product_tool(query: str) -> Dict[str, Any]:
        """Look up ANY product on the web when it is not in the catalog. Slower (up to ~20 s).
        Returns the product with an ASIN-like id usable by check_fit and analyze_listing."""
        if research is None:
            return {"product": None, "spoken": "Web research is not available right now."}
        try:
            product = research(query)
        except RuntimeError as e:
            if "quota" in str(e).lower():
                return {"product": None, "spoken": str(e)}
            return {"product": None, "spoken": "I couldn't reach my web research right now. Please try again."}
        if not product:
            return {"product": None, "spoken": f"I couldn't identify a product for {query}."}
        return {"product": product,
                "spoken": f"I found {product['short_title']} online. Ask me whether it will fit, or how risky it is."}

    @mcp.tool(name="list_products", annotations=READ_ONLY)
    def list_products_tool(limit: int = 5) -> Dict[str, Any]:
        """List the most-reviewed products ReturnKiller can check (max 5)."""
        return _find("", limit)

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
