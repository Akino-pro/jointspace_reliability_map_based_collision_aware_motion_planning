"""Independent capsule-sphere checks of hit-and-run samples in spatial IRIS."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import spatial4R_model as model
from scripts.planar3R_iris_regions import interior_point, load_iris_checkpoint


def validate(checkpoint=None, samples=5000, seed=2026, model_module=None):
    arm = model_module or model
    dimension = len(arm.A)
    checkpoint = checkpoint or f"data/iris/spatial4R_{model.ROBOT_PARAMETER_SET}_iris_midpoint_regions.npz"
    if samples < 1:
        raise ValueError("Sample count must be positive")
    regions, metadata = load_iris_checkpoint(checkpoint, expected_dimension=dimension)
    if metadata["model"]["dh"] != arm.PARAMETERS:
        raise ValueError("Checkpoint DH parameters differ from the validation model")
    rng = np.random.default_rng(seed)
    collisions = 0
    for region in regions:
        q = interior_point(region.A, region.b)
        if q is None:
            raise ValueError("Spatial IRIS region lacks a full-dimensional interior")
        for index in range(samples+50):
            direction = rng.normal(size=dimension)
            direction /= np.linalg.norm(direction)
            projection, slack = region.A @ direction, region.b-region.A @ q
            lower = np.max(slack[projection < -1e-12]/projection[projection < -1e-12])
            upper = np.min(slack[projection > 1e-12]/projection[projection > 1e-12])
            q += rng.uniform(lower, upper)*direction
            if not region.contains(q, tol=1e-7):
                raise RuntimeError("Validation sampler left the IRIS halfspaces")
            if index >= 50:
                collisions += not arm.collision_free(q, metadata["model"]["obstacles"],
                                                       metadata["model"]["link_radius"])
    return dict(checkpoint=checkpoint, regions=len(regions), samples_per_region=samples,
                sampled_configurations=samples*len(regions), collisions=int(collisions), seed=seed,
                method=f"{dimension}D hit-and-run sampling; not a continuous collision certificate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["ftwmax", "rcmax"], default="ftwmax")
    parser.add_argument("--iris-regions", default=None)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    model.select_robot(args.robot)
    args.output = args.output or f"results/spatial4r/spatial4R_{args.robot}_iris_midpoint_validation.json"
    report = validate(args.iris_regions, args.samples)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["collisions"]:
        raise SystemExit("Collision found; reject this spatial IRIS cache")
