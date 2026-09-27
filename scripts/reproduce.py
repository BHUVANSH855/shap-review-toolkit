import json
import sys

from shap_review.reproduction.runner import run_script

print(json.dumps(run_script(sys.argv[1]), indent=2))
