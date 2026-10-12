import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
# Tests exercise the optional catalog path with a fixture; production ships none.
os.environ.setdefault("RETURNKILLER_CATALOG", os.path.join(HERE, "fixtures", "sample_products.json"))
