import time

from shap_review.fuzzing.engine import TreeExplainerFuzzer

s = time.monotonic()
r = TreeExplainerFuzzer(0).run(1000)
print({"iterations": r["iterations"], "seconds": round(time.monotonic() - s, 3)})
