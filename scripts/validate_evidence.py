import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path

from shap_review.evidence import EvidenceCorpus

if __name__ == "__main__":
    corpus = EvidenceCorpus.from_directory(
        Path(__file__).resolve().parents[1] / "data/evidence/issues"
    )
    errors = corpus.validate()
    print(f"records={len(corpus.records)} invalid={len(errors)}")
    for k, v in errors.items():
        print(k, v)
    raise SystemExit(1 if errors else 0)
