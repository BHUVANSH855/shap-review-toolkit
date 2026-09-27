import sys

from shap_review.engine import ReviewEngine

print(ReviewEngine().full_scan(sys.argv[1] if len(sys.argv) > 1 else "."))
