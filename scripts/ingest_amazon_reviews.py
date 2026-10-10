#!/usr/bin/env python3
"""Build data/products.json from the Amazon Reviews 2023 dataset (McAuley Lab).

Streams the gzipped JSONL files, so nothing big is stored and you can stop early.
Run it on your own machine (the files are large), then redeploy.

  python scripts/ingest_amazon_reviews.py --category Home_and_Kitchen \
      --query "air fryer" --query "cutting board" --max-products 150

Offline / already downloaded:  --meta-file meta_X.jsonl.gz --reviews-file X.jsonl.gz

Citation (required by the dataset's authors): Hou et al., "Bridging Language and Items for
Retrieval and Recommendation" / Amazon Reviews 2023, arXiv:2403.03952. The page states no
explicit license; check https://amazon-reviews-2023.github.io before redistributing the data.

Amazon does not publish return reasons. The output carries `review_stats`
(reviews seen / low-star reviews that cite a size or description mismatch) so the analyzer
can produce an honest, labelled proxy score.
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "data", "products.json")

MISMATCH = re.compile(
    r"too (small|big|large|tall|short|wide|narrow|tight|loose)|smaller than|bigger than|larger than|"
    r"runs (small|large)|doesn'?t fit|does not fit|didn'?t fit|did not fit|won'?t fit|"
    r"not as (described|pictured|shown|advertised)|misleading|different (from|than) (the )?(photo|picture|image|description|listing)|"
    r"looks? (nothing )?like the (photo|picture|image)|color (is )?(different|off)|"
    r"not (the )?(size|color)|wrong size|false advertising|smaller in person|bigger in person",
    re.I,
)


# ~200 household and appliance terms. With --per-query 1 this yields up to ~200 varied products.
HOUSEHOLD = [
    "air fryer", "toaster oven", "microwave", "blender", "coffee maker", "espresso machine",
    "electric kettle", "rice cooker", "slow cooker", "pressure cooker", "stand mixer", "food processor",
    "toaster", "waffle maker", "dish rack", "cutting board", "knife block", "cookware set", "baking sheet",
    "food storage container", "trash can", "robot vacuum", "vacuum cleaner", "air purifier", "humidifier",
    "dehumidifier", "space heater", "tower fan", "floor lamp", "desk lamp", "bookshelf", "shoe rack",
    "storage bin", "drawer organizer", "closet organizer", "laundry hamper", "ironing board",
    "shower curtain", "bath mat", "mattress topper", "curtain rod", "wall shelf", "step stool", "spice rack",
    "paper towel holder", "water filter pitcher", "ice maker", "mini fridge", "wine cooler", "juicer",
    "sandwich maker", "popcorn maker", "bread machine", "electric grill", "hand mixer", "immersion blender",
    "can opener", "salad spinner", "nightstand", "coat rack", "ottoman", "bar stool", "tv stand",
    "coffee table", "side table", "console table", "bedside lamp", "table lamp", "pendant light",
    "wall clock", "alarm clock", "picture frame", "mirror", "throw pillow", "throw blanket", "comforter",
    "duvet cover", "bed sheets", "pillow", "weighted blanket", "memory foam pillow", "bath towel",
    "hand towel", "shower head", "toilet brush", "soap dispenser", "toothbrush holder", "bathroom scale",
    "towel rack", "laundry basket", "clothes drying rack", "garment rack", "hangers", "steam iron",
    "garment steamer", "fabric shaver", "lint remover", "mop", "broom", "dustpan", "spray bottle", "bucket",
    "sponge", "scrub brush", "squeegee", "window cleaner", "steam mop", "carpet cleaner", "handheld vacuum",
    "cordless vacuum", "stick vacuum", "paper shredder", "desk organizer", "filing cabinet", "bookends",
    "magazine rack", "umbrella stand", "doormat", "key holder", "wall hooks", "over the door hook",
    "under bed storage", "vacuum storage bag", "garment bag", "shoe organizer", "jewelry organizer",
    "cosmetic organizer", "lazy susan", "turntable organizer", "fridge organizer", "pantry organizer",
    "bread box", "fruit bowl", "cake stand", "serving tray", "serving bowl", "salad bowl", "mixing bowls",
    "measuring cups", "measuring spoons", "kitchen scale", "kitchen timer", "meat thermometer", "oven mitts",
    "apron", "dish towels", "dish soap dispenser", "sink caddy", "dish drying mat", "colander", "strainer",
    "grater", "peeler", "kitchen shears", "knife sharpener", "chef knife", "paring knife", "bread knife",
    "steak knives", "cast iron skillet", "nonstick pan", "frying pan", "saucepan", "stockpot", "dutch oven",
    "roasting pan", "casserole dish", "muffin pan", "cake pan", "pie dish", "cookie sheet", "cooling rack",
    "rolling pin", "pizza stone", "pizza cutter", "tea kettle", "teapot", "french press", "pour over coffee",
    "coffee grinder", "milk frother", "travel mug", "water bottle", "thermos", "lunch box", "bento box",
    "ice cube tray", "popsicle mold", "wine glasses", "coffee mugs", "drinking glasses", "plate set",
    "dinnerware set", "flatware set", "chopsticks", "utensil holder", "candle", "diffuser", "night light",
    "extension cord", "power strip", "surge protector", "smoke detector", "fire extinguisher", "flashlight",
    "tool box", "screwdriver set", "step ladder", "stud finder", "tape measure", "doorstop", "door lock",
    "window blinds", "blackout curtains", "area rug", "runner rug", "floor mat", "ceiling fan",
    "portable fan", "heating pad", "electric blanket", "sewing machine", "sewing kit", "clothes steamer",
    "laundry detergent dispenser", "trash bags", "compost bin", "recycling bin", "pet bowl", "cat litter box",
    "dog bed",
]


def open_stream(src):
    """Return a text stream of decompressed lines from a URL or local path."""
    if re.match(r"https?://", src):
        raw = urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "returnkiller-ingest"}), timeout=60)
    else:
        raw = open(src, "rb")
    if src.endswith(".gz"):
        raw = gzip.GzipFile(fileobj=raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def jsonl(src, max_lines):
    with open_stream(src) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                return
            try:
                yield json.loads(line)
            except ValueError:
                continue  # truncated final line of an interrupted/partial file


def _to_cm(value, unit):
    unit = (unit or "").lower()
    if unit.startswith(("in", '"')):
        return round(value * 2.54, 1)
    if unit.startswith("mm"):
        return round(value / 10, 1)
    if unit.startswith("cm"):
        return round(value, 1)
    if unit.startswith(("ft", "feet")):
        return round(value * 30.48, 1)
    return None


NUM = r"(\d+(?:\.\d+)?)"


def parse_dimensions(details):
    """Height only from an unambiguous source: an 'H' label, or the third value of 'LxWxH'.

    Anything else (e.g. 'Product Dimensions: 5 x 8 x 3 inches' with no axis names) is
    left out rather than guessed: a wrong height produces a confidently wrong fit verdict.
    """
    out = {}
    for key, raw in (details or {}).items():
        if not isinstance(raw, str):
            continue
        k = key.lower()
        if "dimension" not in k:
            continue
        # e.g. 10.5"D x 12"W x 13"H
        m = re.search(NUM + r"\s*(?:\"|in(?:ches)?|cm|mm)?\s*h\b", raw, re.I)
        unit = re.search(r"(inches|inch|in\b|cm|mm|\")", raw, re.I)
        if m and unit:
            h = _to_cm(float(m.group(1)), unit.group(1))
            if h:
                out["height_cm"] = h
                break
        # e.g. 'Item Dimensions LxWxH': '9.84 x 9.84 x 5.51 inches'
        if re.search(r"l\s*x\s*w\s*x\s*h", k):
            m = re.search(NUM + r"\s*x\s*" + NUM + r"\s*x\s*" + NUM + r"\s*(inches|inch|in|cm|mm)\b", raw, re.I)
            if m:
                l, w, h = (_to_cm(float(m.group(i)), m.group(4)) for i in (1, 2, 3))
                out.update({"length_cm": l, "width_cm": w, "height_cm": h})
                break
    return out


def clean(text, limit=600):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def matching_query(title, queries):
    """First query contained in the title ('' when there are no queries), else None."""
    if not queries:
        return ""
    low = title.lower()
    return next((q for q in queries if q in low), None)


def build_product(meta):
    title = clean(meta.get("title"), 300)
    if not title:
        return None
    details = meta.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = {}
    images = []
    for im in meta.get("images") or []:
        u = im.get("hi_res") or im.get("large")
        if u:
            images.append(u)
    cats = meta.get("categories") or []
    price = meta.get("price")
    try:
        price = float(str(price).replace("$", "").replace(",", "")) if price not in (None, "") else None
    except ValueError:
        price = None
    return {
        "asin": meta.get("parent_asin"),
        "title": title,
        "bullets": [clean(b, 300) for b in (meta.get("features") or [])][:8],
        "description": clean(" ".join(meta.get("description") or []), 800),
        "images": images[:3],
        "dimensions": parse_dimensions(details),
        "category": (cats[-1] if cats else meta.get("main_category")),
        "price": price,
        "rating_number": meta.get("rating_number") or 0,
        "average_rating": meta.get("average_rating"),
        "reviews": [],
        "review_stats": {"seen": 0, "mismatch_low": 0},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="Home_and_Kitchen", help="dataset category file name")
    ap.add_argument("--meta-file", help="local or URL path to meta jsonl(.gz); default: dataset URL")
    ap.add_argument("--reviews-file", help="local or URL path to reviews jsonl(.gz); default: dataset URL")
    ap.add_argument("--query", action="append", default=[], help="keep products whose title contains this (repeatable)")
    ap.add_argument("--preset", choices=["household"], help="household: ~200 kitchen, bath, bedroom, cleaning and storage items")
    ap.add_argument("--per-query", type=int, default=0, help="keep at most N products per --query term (variety)")
    ap.add_argument("--min-ratings", type=int, default=None, help="skip products with fewer ratings overall")
    ap.add_argument("--max-products", type=int, default=None)
    ap.add_argument("--max-meta-lines", type=int, default=0, help="stop scanning metadata after N lines (0 = all)")
    ap.add_argument("--max-review-lines", type=int, default=3_000_000, help="stop scanning reviews after N lines")
    ap.add_argument("--reviews-per-product", type=int, default=40)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    meta_src = a.meta_file or f"{BASE}/meta_categories/meta_{a.category}.jsonl.gz"
    rev_src = a.reviews_file or f"{BASE}/review_categories/{a.category}.jsonl.gz"
    queries = [q.lower() for q in a.query]
    if a.preset == "household":
        queries = list(dict.fromkeys(queries + HOUSEHOLD))
        a.per_query = a.per_query or 1
        a.max_products = a.max_products or 250
        a.max_meta_lines = a.max_meta_lines or 2_000_000  # rare terms never fill; bound the scan
        # Review lines are sampled, so only well-reviewed products collect enough for a score.
        a.min_ratings = a.min_ratings if a.min_ratings is not None else 1000
    a.max_products = a.max_products or 150
    a.min_ratings = a.min_ratings if a.min_ratings is not None else 200
    taken = {}

    print(f"[1/2] metadata: {meta_src}", file=sys.stderr)
    products = {}
    for i, meta in enumerate(jsonl(meta_src, a.max_meta_lines), 1):
        if (meta.get("rating_number") or 0) >= a.min_ratings and meta.get("parent_asin"):
            q = matching_query(clean(meta.get("title"), 300), queries)
            if q is not None and not (a.per_query and taken.get(q, 0) >= a.per_query):
                p = build_product(meta)
                if p:
                    products[p["asin"]] = p
                    taken[q] = taken.get(q, 0) + 1
        if i % 200000 == 0:
            print(f"  scanned {i:,} items, kept {len(products)}", file=sys.stderr)
        if len(products) >= a.max_products:
            break
        if a.per_query and queries and all(taken.get(q, 0) >= a.per_query for q in queries):
            break
    print(f"  kept {len(products)} products", file=sys.stderr)
    if not products:
        sys.exit("No products matched; relax --query/--min-ratings.")

    print(f"[2/2] reviews: {rev_src}", file=sys.stderr)
    for i, r in enumerate(jsonl(rev_src, a.max_review_lines), 1):
        p = products.get(r.get("parent_asin"))
        if p:
            text = clean(r.get("text"), 500)
            rating = r.get("rating") or 0
            hit = bool(MISMATCH.search(f"{r.get('title') or ''} {text}"))
            st = p["review_stats"]
            st["seen"] += 1
            if rating <= 3 and hit:
                st["mismatch_low"] += 1
            # keep a bounded, useful sample: mismatch/low-star first, then a few others
            keep = hit or rating <= 2 or len(p["reviews"]) < a.reviews_per_product // 4
            if keep and len(p["reviews"]) < a.reviews_per_product and text:
                p["reviews"].append({"rating": int(rating), "title": clean(r.get("title"), 120), "text": text})
        if i % 500000 == 0:
            print(f"  scanned {i:,} reviews", file=sys.stderr)

    out = [p for p in products.values() if p["review_stats"]["seen"] > 0]
    for p in out:
        p["reviews"].sort(key=lambda x: x["rating"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "source": "amazon-reviews-2023",
                "category": a.category,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "citation": "Hou et al., Amazon Reviews 2023, arXiv:2403.03952",
                "review_lines_scanned": a.max_review_lines,
            },
            "products": out,
        }, f, ensure_ascii=False)
    with_h = sum(1 for p in out if p["dimensions"].get("height_cm"))
    print(f"wrote {len(out)} products ({with_h} with height) -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
