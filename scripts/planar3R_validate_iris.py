"""Independent hit-and-run sampling of cached IRIS region interiors."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.planar3R_collision import is_collision_free
from scripts.planar3R_iris_regions import interior_point, load_iris_checkpoint


def validate_regions(path, samples=5000, seed=2026):
    if samples < 1:
        raise ValueError("samples must be positive")
    regions, metadata = load_iris_checkpoint(path)
    rng = np.random.default_rng(seed)
    collisions = 0
    model = metadata["model"]
    for region in regions:
        q = interior_point(region.A, region.b)
        if q is None:
            raise ValueError("IRIS region has no full-dimensional interior")
        for k in range(samples + 50):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            projection = region.A @ direction
            slack = region.b - region.A @ q
            negative, positive = projection < -1e-12, projection > 1e-12
            lower = np.max(slack[negative] / projection[negative])
            upper = np.min(slack[positive] / projection[positive])
            q = q + rng.uniform(lower, upper) * direction
            if k >= 50:
                collisions += not is_collision_free(
                    q, model["obstacles"], model["link_lengths"], model["link_radius"]
                )
    return {"checkpoint": str(path), "regions": len(regions), "seed": seed,
            "samples_per_region": samples, "sampled_configurations": samples * len(regions),
            "collisions": collisions, "iris_options": metadata.get("options"),
            "generation_seconds": metadata["generation_seconds"],
            "method": "hit-and-run sampling; not a continuous collision certificate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-regions", default='data/iris/planar3R_iris_regions.npz')
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default='results/planar3r/iris_region_validation.json')
    args = parser.parse_args()
    report = validate_regions(args.iris_regions, args.samples, args.seed)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)
    if report["collisions"]:
        raise SystemExit("Collision samples found; do not use this checkpoint for planning")


if __name__ == "__main__":
    main()
