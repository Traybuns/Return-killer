"""
ReturnKiller - Core Analysis Engine
Uses Amazon Bedrock Nova (or falls back to simulated analysis) to detect
size/description mismatches and generate listing improvements.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class ComplaintInsight:
    theme: str
    frequency: int
    severity: str  # high, medium, low
    example_quotes: List[str]
    suggested_fix: str


@dataclass
class SizeIssue:
    issue_type: str  # scale_mismatch, missing_reference, dimension_unclear, storage_fit
    description: str
    severity: str
    recommendation: str


@dataclass
class AnalysisResult:
    asin: str
    title: str
    return_risk_score: float  # 0-100
    top_complaints: List[ComplaintInsight]
    size_issues: List[SizeIssue]
    improved_bullets: List[str]
    improved_title_suggestion: str
    size_chart_text: str
    alexa_fit_response: str
    visual_suggestions: List[str]
    summary: str
    analyzed_at: str


class ReturnKillerAnalyzer:
    """
    Analyzes product listings for return risks related to 
    description accuracy and size/fit issues.
    """

    def __init__(self, use_bedrock: bool = False, region: str = "us-east-1"):
        self.use_bedrock = use_bedrock
        self.region = region
        self.bedrock_client = None
        
        if use_bedrock:
            try:
                import boto3
                self.bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=region
                )
                print("✓ Bedrock client initialized")
            except Exception as e:
                print(f"⚠ Could not initialize Bedrock: {e}")
                print("  Falling back to simulated analysis")
                self.use_bedrock = False

    def analyze_product(self, product: Dict[str, Any]) -> AnalysisResult:
        """Main entry point - analyze a product and return insights."""
        
        if self.use_bedrock and self.bedrock_client:
            return self._analyze_with_bedrock(product)
        else:
            return self._analyze_simulated(product)

    def _analyze_simulated(self, product: Dict[str, Any]) -> AnalysisResult:
        """
        High-quality simulated analysis that mimics what Nova would produce.
        This lets us demo and iterate without requiring AWS credentials.
        """
        reviews = product.get("reviews", [])
        return_reasons = product.get("return_reasons", {})
        dimensions = product.get("dimensions", {})
        title = product.get("title", "")
        bullets = product.get("bullets", [])
        
        # Extract complaint themes from reviews
        complaints = self._extract_complaints(reviews)
        
        # Detect size issues
        size_issues = self._detect_size_issues(reviews, dimensions, title)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(return_reasons, complaints, size_issues)
        
        # Generate improved content
        improved_bullets = self._generate_improved_bullets(bullets, complaints, size_issues, dimensions)
        improved_title = self._improve_title(title, size_issues, dimensions)
        size_chart = self._generate_size_chart(dimensions, product.get("category", ""))
        
        # Alexa response
        alexa_response = self._generate_alexa_fit_response(product, dimensions, size_issues)
        
        # Visual suggestions
        visual_suggestions = self._generate_visual_suggestions(size_issues, dimensions)
        
        # Summary
        summary = self._generate_summary(risk_score, complaints, size_issues)
        
        return AnalysisResult(
            asin=product.get("asin", "UNKNOWN"),
            title=title,
            return_risk_score=risk_score,
            top_complaints=complaints,
            size_issues=size_issues,
            improved_bullets=improved_bullets,
            improved_title_suggestion=improved_title,
            size_chart_text=size_chart,
            alexa_fit_response=alexa_response,
            visual_suggestions=visual_suggestions,
            summary=summary,
            analyzed_at=datetime.now(timezone.utc).isoformat()
        )

    def _extract_complaints(self, reviews: List[Dict]) -> List[ComplaintInsight]:
        """Extract and cluster complaint themes from reviews."""
        themes = {
            "size_smaller_than_expected": {
                "keywords": ["smaller", "small", "tiny", "looks huge", "looks big", "misleading size", "size"],
                "quotes": [],
                "fix": "Add clear scale reference images (phone, banana, hand) and explicit dimensions in the first bullet."
            },
            "color_mismatch": {
                "keywords": ["color", "darker", "lighter", "different color", "looks different"],
                "quotes": [],
                "fix": "Update main image to accurately represent true product color under natural lighting. Add color swatch note."
            },
            "storage_fit": {
                "keywords": ["cabinet", "fit under", "storage", "too tall", "won't fit", "clearance"],
                "quotes": [],
                "fix": "Add storage dimensions and a 'Will it fit?' section showing height when stored upright/flat."
            },
            "capacity_misleading": {
                "keywords": ["capacity", "basket", "space", "family", "not enough", "smaller than"],
                "quotes": [],
                "fix": "Clarify usable capacity vs external dimensions. Show real food quantity photos."
            }
        }
        
        for review in reviews:
            if review.get("rating", 5) >= 4:
                continue  # Skip positive reviews for complaint extraction
            text = (review.get("text", "") + " " + review.get("title", "")).lower()
            
            for theme_key, theme_data in themes.items():
                if any(kw in text for kw in theme_data["keywords"]):
                    theme_data["quotes"].append(review.get("text", "")[:120])
        
        insights = []
        for theme_key, data in themes.items():
            if data["quotes"]:
                severity = "high" if len(data["quotes"]) >= 2 else "medium"
                insights.append(ComplaintInsight(
                    theme=theme_key.replace("_", " ").title(),
                    frequency=len(data["quotes"]),
                    severity=severity,
                    example_quotes=data["quotes"][:3],
                    suggested_fix=data["fix"]
                ))
        
        # Sort by frequency
        insights.sort(key=lambda x: x.frequency, reverse=True)
        return insights

    def _detect_size_issues(self, reviews: List[Dict], dimensions: Dict, title: str) -> List[SizeIssue]:
        """Detect specific size-related problems."""
        issues = []
        
        # Check if dimensions exist and are complete
        if not dimensions or not all(k in dimensions for k in ["length_cm", "width_cm", "height_cm"]):
            issues.append(SizeIssue(
                issue_type="dimension_unclear",
                description="Product dimensions are missing or incomplete in the listing data.",
                severity="high",
                recommendation="Add full L x W x H dimensions in centimeters and inches in the product description and bullets."
            ))
        
        # Check review language for scale problems
        size_complaints = 0
        for review in reviews:
            text = (review.get("text", "") + " " + review.get("title", "")).lower()
            if any(w in text for w in ["smaller", "bigger", "size", "fit", "scale", "looks"]):
                size_complaints += 1
        
        if size_complaints >= 2:
            issues.append(SizeIssue(
                issue_type="scale_mismatch",
                description=f"Multiple customers ({size_complaints}) report the product appears different in size than expected from photos.",
                severity="high",
                recommendation="Add at least one image showing the product next to a common object (smartphone, credit card, banana, or hand) for scale."
            ))
        
        # Storage fit issues
        storage_mentions = sum(
            1 for r in reviews 
            if any(w in (r.get("text", "") + r.get("title", "")).lower() 
                   for w in ["cabinet", "under", "storage", "clearance", "shelf"])
        )
        if storage_mentions >= 1:
            issues.append(SizeIssue(
                issue_type="storage_fit",
                description="Customers are returning the product because it does not fit in their intended storage space.",
                severity="high",
                recommendation="Add a 'Storage & Fit' section with height when stored upright and flat, plus common cabinet clearance guidance."
            ))
        
        # Missing reference objects in title/bullets
        if "compact" in title.lower() or "large" in title.lower() or "small" in title.lower():
            issues.append(SizeIssue(
                issue_type="missing_reference",
                description="Title uses relative size words (compact/large/small) without absolute measurements.",
                severity="medium",
                recommendation="Pair relative size words with exact dimensions, e.g., 'Compact 12-inch Air Fryer' instead of just 'Compact'."
            ))
        
        return issues

    def _calculate_risk_score(self, return_reasons: Dict, complaints: List, size_issues: List) -> float:
        """Calculate a 0-100 return risk score focused on description/size issues."""
        score = 20.0  # baseline
        
        # Return reason weighting
        not_as_desc = return_reasons.get("not_as_described", 0)
        wrong_size = return_reasons.get("wrong_size", 0)
        total_returns = sum(return_reasons.values()) or 1
        
        desc_ratio = (not_as_desc + wrong_size) / total_returns
        score += desc_ratio * 40
        
        # Complaint frequency
        score += min(len(complaints) * 8, 25)
        
        # Size issues
        high_severity = sum(1 for i in size_issues if i.severity == "high")
        score += high_severity * 7
        
        return min(round(score, 1), 100.0)

    def _generate_improved_bullets(
        self, 
        original: List[str], 
        complaints: List[ComplaintInsight],
        size_issues: List[SizeIssue],
        dimensions: Dict
    ) -> List[str]:
        """Generate improved bullet points that address return drivers."""
        improved = []
        
        # Always lead with clear dimensions if available
        if dimensions:
            l, w, h = dimensions.get("length_cm"), dimensions.get("width_cm"), dimensions.get("height_cm")
            if l and w and h:
                inches = f"{l/2.54:.1f}\" x {w/2.54:.1f}\" x {h/2.54:.1f}\""
                improved.append(
                    f"Exact Dimensions: {l} x {w} x {h} cm ({inches}) — see scale photos for real-life size"
                )
        
        # Address top complaints
        for complaint in complaints[:2]:
            if "size" in complaint.theme.lower():
                improved.append(
                    "True-to-photo size — includes scale reference images so you know exactly what to expect"
                )
            elif "color" in complaint.theme.lower():
                improved.append(
                    "Color shown is accurate under natural light — see customer photos for real-world appearance"
                )
            elif "storage" in complaint.theme.lower() or "fit" in complaint.theme.lower():
                improved.append(
                    "Storage-friendly design — check the dimension chart to confirm it fits your cabinet or shelf"
                )
        
        # Keep strongest original bullets (filtered)
        for bullet in original:
            lower = bullet.lower()
            if not any(skip in lower for skip in ["premium", "perfect", "easy to clean", "eco-friendly"]):
                if bullet not in improved:
                    improved.append(bullet)
        
        # Ensure we have 5
        while len(improved) < 5 and original:
            candidate = original[len(improved) % len(original)]
            if candidate not in improved:
                improved.append(candidate)
            else:
                break
                
        return improved[:5]

    def _improve_title(self, title: str, size_issues: List[SizeIssue], dimensions: Dict) -> str:
        """Suggest a more accurate title."""
        if not dimensions:
            return title + " — See Exact Dimensions in Description"
        
        l = dimensions.get("length_cm")
        if l and "x" not in title.lower() and not any(c.isdigit() for c in title[:20]):
            # Add dimension hint
            return f"{title} ({l:.0f}cm)"
        return title

    def _generate_size_chart(self, dimensions: Dict, category: str) -> str:
        """Generate human-readable size guidance."""
        if not dimensions:
            return "Dimensions not available. Please measure your space before ordering."
        
        l = dimensions.get("length_cm", 0)
        w = dimensions.get("width_cm", 0)
        h = dimensions.get("height_cm", 0)
        
        lines = [
            f"Product Size: {l} cm (L) × {w} cm (W) × {h} cm (H)",
            f"In inches: {l/2.54:.1f}\" × {w/2.54:.1f}\" × {h/2.54:.1f}\"",
            "",
            "Will it fit?",
            f"• Counter / shelf depth needed: at least {w + 5:.0f} cm recommended",
            f"• Height clearance (upright): {h + 3:.0f} cm",
            f"• Comparable to: a large laptop or medium serving tray" if l > 40 else f"• Comparable to: a standard dinner plate or tablet",
        ]
        return "\n".join(lines)

    def _generate_alexa_fit_response(
        self, 
        product: Dict, 
        dimensions: Dict, 
        size_issues: List[SizeIssue]
    ) -> str:
        """Generate a natural Alexa spoken + visual response for fit questions."""
        title = product.get("title", "this product")
        short_name = title.split(" - ")[0].split(",")[0]
        
        if not dimensions:
            return (
                f"I don't have exact dimensions for the {short_name} yet. "
                "I recommend checking the product detail page for the size chart, "
                "or measuring the space where you plan to put it."
            )
        
        l = dimensions.get("length_cm", 0)
        w = dimensions.get("width_cm", 0)
        h = dimensions.get("height_cm", 0)
        
        response = (
            f"The {short_name} measures {l:.0f} by {w:.0f} by {h:.0f} centimeters. "
            f"That's about {l/2.54:.1f} inches long. "
        )
        
        # Add practical guidance
        if h > 30:
            response += (
                f"It is {h:.0f} cm tall, so check that you have enough clearance under your cabinets. "
                "Most standard cabinets have about 45 to 50 cm of space. "
            )
        
        response += (
            "I've put a size comparison on the screen so you can see how it relates to everyday objects. "
            "Would you like me to compare it to a specific space in your home?"
        )
        
        return response

    def _generate_visual_suggestions(self, size_issues: List[SizeIssue], dimensions: Dict) -> List[str]:
        """Suggest visual assets that would reduce returns."""
        suggestions = [
            "Add a scale reference photo: product next to a standard smartphone or banana",
            "Include a top-down photo with a ruler or measuring tape visible",
        ]
        
        if any(i.issue_type == "storage_fit" for i in size_issues):
            suggestions.append(
                "Show the product stored under a cabinet or on a shelf with dimensions overlaid"
            )
        
        if dimensions and dimensions.get("height_cm", 0) > 25:
            suggestions.append(
                "Add a lifestyle image showing real clearance height in a typical kitchen"
            )
        
        suggestions.append(
            "Generate an Alexa Show visual: side-by-side of product vs common kitchen objects"
        )
        
        return suggestions

    def _generate_summary(
        self, 
        risk_score: float, 
        complaints: List[ComplaintInsight], 
        size_issues: List[SizeIssue]
    ) -> str:
        """Human-readable executive summary."""
        level = "LOW" if risk_score < 40 else "MEDIUM" if risk_score < 65 else "HIGH"
        
        summary = f"Return Risk: {level} ({risk_score}/100)\n\n"
        
        if complaints:
            summary += "Top customer friction points:\n"
            for c in complaints[:3]:
                summary += f"  • {c.theme} ({c.frequency} mentions) — {c.severity} severity\n"
        
        if size_issues:
            summary += "\nSize & description issues found:\n"
            for i in size_issues:
                summary += f"  • [{i.severity.upper()}] {i.description}\n"
        
        summary += "\nRecommended immediate actions:\n"
        summary += "  1. Add scale-reference images\n"
        summary += "  2. Lead bullets with exact dimensions\n"
        summary += "  3. Add a 'Will it fit?' section for storage contexts\n"
        
        return summary

    def to_dict(self, result: AnalysisResult) -> Dict:
        """Convert AnalysisResult to JSON-serializable dict."""
        return {
            "asin": result.asin,
            "title": result.title,
            "return_risk_score": result.return_risk_score,
            "top_complaints": [asdict(c) for c in result.top_complaints],
            "size_issues": [asdict(s) for s in result.size_issues],
            "improved_bullets": result.improved_bullets,
            "improved_title_suggestion": result.improved_title_suggestion,
            "size_chart_text": result.size_chart_text,
            "alexa_fit_response": result.alexa_fit_response,
            "visual_suggestions": result.visual_suggestions,
            "summary": result.summary,
            "analyzed_at": result.analyzed_at
        }


def load_sample_products(path: str = None) -> List[Dict]:
    """Load sample products from JSON."""
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), 
            "..", "data", "sample_products.json"
        )
    with open(path) as f:
        data = json.load(f)
    return data["products"]


if __name__ == "__main__":
    # Quick self-test
    analyzer = ReturnKillerAnalyzer(use_bedrock=False)
    products = load_sample_products()
    
    print("=" * 60)
    print("ReturnKiller Analysis Engine - Self Test")
    print("=" * 60)
    
    for product in products:
        print(f"\n📦 Analyzing: {product['title'][:50]}...")
        result = analyzer.analyze_product(product)
        print(f"   Risk Score: {result.return_risk_score}/100")
        print(f"   Complaints found: {len(result.top_complaints)}")
        print(f"   Size issues: {len(result.size_issues)}")
        print(f"\n{result.summary}")
        print("-" * 40)
