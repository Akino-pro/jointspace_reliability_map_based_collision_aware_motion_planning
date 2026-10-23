"""Independent hit-and-run validation of seven-dimensional spatial IRIS."""
import argparse
import json
from pathlib import Path
from scripts import spatial7R_model as model
from scripts.spatial4R_validate_iris import validate as shared_validate


def validate(checkpoint="data/iris/spatial7R_30cm20deg_iris_regions.npz", samples=5000, seed=2026):
    return shared_validate(checkpoint, samples, seed, model_module=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-regions", default="data/iris/spatial7R_30cm20deg_iris_regions.npz")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", default="results/spatial7r/spatial7R_30cm20deg_iris_validation.json")
    args = parser.parse_args()
    report = validate(args.iris_regions, args.samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["collisions"]:
        raise SystemExit("Collision found; reject this IRIS cache")
