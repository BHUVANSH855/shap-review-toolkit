import argparse
import json

from shap_review.fuzzing.engine import TreeExplainerFuzzer

p = argparse.ArgumentParser()
p.add_argument("--iterations", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
print(json.dumps(TreeExplainerFuzzer(a.seed).run(a.iterations), indent=2))
