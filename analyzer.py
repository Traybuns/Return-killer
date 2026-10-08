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

    def __init__(self, use_bedrock: bool = False, region: Optional[str] = None):
        self.use_bedrock = use_bedrock
        region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
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

    @staticmethod
    def _model_ids() -> List[str]:
        """Nova 2 Lite first (override with RETURNKILLER_MODEL_ID), then fall back."""
        ids = []
        override = os.environ.get("RETURNKILLER_MODEL_ID")
        if override:
            ids.append(override)
        ids += [
            "us.amazon.nova-2-lite-v1:0",
            "global.amazon.nova-2-lite-v1:0",
            "amazon.nova-2-lite-v1:0",
        ]
        return list(dict.fromkeys(ids))

    def health_check(self) -> Dict[str, Any]:
        """Make a tiny real Bedrock call so a missing permission shows up immediately."""
        if not self.use_bedrock or not self.bedrock_client:
            return {"bedrock": "disabled", "ok": True}
        last_err = None
        for model_id in self._model_ids():
            try:
                self.bedrock_client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": "Reply with the word ok."}]}],
                    inferenceConfig={"maxTokens": 10, "temperature": 0},
                )
                return {"bedrock": "ok", "model_id": model_id, "region": self.region, "ok": True}
            except Exception as e:
                last_err = e
        return {"bedrock": "error", "error": str(last_err)[:300], "region": self.region, "ok": False}

    def analyze_product(self, product: Dict[str, Any]) -> AnalysisResult:
        """Main entry point - analyze a product and return insights."""
        # Product listing analysis currently uses the high-quality simulated path.
        # Space photo scanning uses Bedrock Nova when enabled (see scan_space).
        return self._analyze_simulated(product)

    def scan_space(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
        product: Optional[Dict[str, Any]] = None,
        user_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a photo of a real-world space (cabinet, shelf, etc.) using
        Amazon Nova vision on Bedrock. Returns estimated clearances and fit advice.
        """
        if self.use_bedrock and self.bedrock_client:
            try:
                return self._scan_space_bedrock(image_bytes, media_type, product, user_hint)
            except Exception as e:
                print(f"⚠ Bedrock space scan failed: {e}")
                return self._scan_space_simulated(product, user_hint, error=str(e))
        return self._scan_space_simulated(product, user_hint)

    def _scan_space_bedrock(
        self,
        image_bytes: bytes,
        media_type: str,
        product: Optional[Dict[str, Any]],
        user_hint: Optional[str],
    ) -> Dict[str, Any]:
        import base64

        fmt = "jpeg"
        if "png" in (media_type or "").lower():
            fmt = "png"
        elif "webp" in (media_type or "").lower():
            fmt = "webp"
        elif "gif" in (media_type or "").lower():
            fmt = "gif"

        product_ctx = "No product selected."
        dims = {}
        if product:
            dims = product.get("dimensions") or {}
            product_ctx = (
                f"Product: {product.get('title', 'Unknown')}\n"
                f"Dimensions (cm): L={dims.get('length_cm')} W={dims.get('width_cm')} H={dims.get('height_cm')}"
            )

        hint = user_hint or "No extra hint from user."

        prompt = f"""You are a spatial fit assistant for Amazon shoppers.
Analyze this photo of a real-world storage or placement space (cabinet, shelf, counter, floor corner, etc.).

{product_ctx}
User hint: {hint}

Estimate what you can see. Be honest about uncertainty.
Return ONLY valid JSON (no markdown) with this shape:
{{
  "space_type": "cabinet|shelf|counter|closet|floor|other",
  "estimated_clearance_height_cm": number or null,
  "estimated_clearance_width_cm": number or null,
  "estimated_depth_cm": number or null,
  "confidence": "high|medium|low",
  "observations": ["short bullet", "..."],
  "fit_verdict": "likely_fits|tight|unlikely|unknown",
  "spoken_summary": "2-3 sentences a voice assistant would say to the shopper",
  "advice": "one practical next step"
}}

Rules:
- Prefer centimeters.
- If you cannot measure reliably, use null and confidence low — do not invent precise numbers.
- If product dimensions are known, compare them in fit_verdict and spoken_summary.
- spoken_summary must be natural speech, no JSON.
"""

        model_ids = self._model_ids()

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        last_err = None

        for model_id in model_ids:
            try:
                body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": fmt,
                                        "source": {"bytes": b64},
                                    }
                                },
                                {"text": prompt},
                            ],
                        }
                    ],
                    "inferenceConfig": {
                        "maxTokens": 800,
                        "temperature": 0.2,
                    },
                }
                # Converse API
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": fmt,
                                        "source": {"bytes": image_bytes},
                                    }
                                },
                                {"text": prompt},
                            ],
                        }
                    ],
                    inferenceConfig={"maxTokens": 800, "temperature": 0.2},
                )
                text = response["output"]["message"]["content"][0]["text"]
                parsed = self._parse_json_loose(text)
                parsed["engine"] = "bedrock"
                parsed["model_id"] = model_id
                parsed["product_asin"] = (product or {}).get("asin")
                if dims:
                    parsed["product_dimensions_cm"] = dims
                return parsed
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"All Bedrock vision models failed: {last_err}")

    def _parse_json_loose(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            return {
                "space_type": "other",
                "estimated_clearance_height_cm": None,
                "estimated_clearance_width_cm": None,
                "estimated_depth_cm": None,
                "confidence": "low",
                "observations": ["Could not parse model response"],
                "fit_verdict": "unknown",
                "spoken_summary": text[:400] if text else "I could not analyze that photo.",
                "advice": "Try a clearer photo with a known object for scale.",
            }

    def _scan_space_simulated(
        self,
        product: Optional[Dict[str, Any]] = None,
        user_hint: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback when Bedrock is off or fails — still demoable."""
        dims = (product or {}).get("dimensions") or {}
        h = dims.get("height_cm")
        w = dims.get("width_cm")
        hint = (user_hint or "").lower()

        # Toy heuristic from hint text if present
        est_h = 45.0
        if "40" in hint:
            est_h = 40.0
        elif "30" in hint:
            est_h = 30.0
        elif "50" in hint:
            est_h = 50.0

        verdict = "unknown"
        if h:
            if est_h >= h + 3:
                verdict = "likely_fits"
            elif est_h >= h:
                verdict = "tight"
            else:
                verdict = "unlikely"

        title = (product or {}).get("title") or "this product"
        short = title.split(" - ")[0]

        spoken = (
            f"From your photo I estimated about {est_h:.0f} centimeters of clearance. "
        )
        if h:
            spoken += f"The {short} is {h:.0f} cm tall. "
            if verdict == "likely_fits":
                spoken += "It should fit with a little room to spare."
            elif verdict == "tight":
                spoken += "It may fit, but clearance looks tight — measure once to be sure."
            else:
                spoken += "It probably will not fit in that opening."
        else:
            spoken += "Select a product so I can compare exact dimensions."

        if error:
            spoken = (
                "I could not reach the vision model, so this is a demo estimate only. "
                + spoken
            )

        return {
            "space_type": "cabinet" if "cabinet" in hint else "shelf",
            "estimated_clearance_height_cm": est_h,
            "estimated_clearance_width_cm": 60.0,
            "estimated_depth_cm": 35.0,
            "confidence": "low",
            "observations": [
                "Simulated scan (enable Bedrock Nova for real vision)",
                f"Hint used: {user_hint or 'none'}",
            ],
            "fit_verdict": verdict,
            "spoken_summary": spoken,
            "advice": "Enable Bedrock and retake the photo in good light with the opening fully visible.",
            "engine": "simulated",
            "product_asin": (product or {}).get("asin"),
            "product_dimensions_cm": dims or None,
            "error": error,
        }

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
    """Load sample products from JSON (supports flat repo or backend/ layout)."""
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "data", "sample_products.json"),
            os.path.join(base, "..", "data", "sample_products.json"),
            os.path.join(base, "sample_products.json"),
        ]
        path = next((c for c in candidates if os.path.isfile(c)), candidates[0])
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