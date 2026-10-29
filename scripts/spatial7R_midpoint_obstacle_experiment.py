"""Build new-scene IRIS coverage and run the 60 cm / base-Z 90-degree task."""
import json
from pathlib import Path
import numpy as np
from scripts import spatial7R_model as model
from scripts.spatial4R_iris import generate_regions
from scripts.spatial7R_two_stage_gcs_planner import plan_before_failure

CHECKPOINT = "data/iris/spatial7R_60cm90deg_iris_regions.npz"
OUTPUT = Path("results/spatial7r/spatial7R_60cm90deg_two_stage_path.json")


def main():
    scene = json.loads(Path("results/spatial7r/spatial7R_60cm90deg_scene.json").read_text())
    seed_path = np.array(scene["seed_path"])
    if not model.validate_path(seed_path, model.OBSTACLES, spacing=.005)[0]:
        raise ValueError("IRIS seed corridor must be collision free")
    # This auxiliary corridor supplies IRIS seeds only, never the returned GCS path.
    seeds = [seed_path[1], seed_path[0], seed_path[-1]]
    generate_regions(CHECKPOINT, model.OBSTACLES, max_regions=3, seeds=seeds,
                     coverage_radius=.65, model_module=model)
    report = plan_before_failure(checkpoint=CHECKPOINT, max_paths=200, max_trials=300)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
          ("success", "status", "message", "original_region_count", "region_count", "edge_count",
           "target_set_count", "cost", "endpoint_error", "orientation_error_radians", "total_seconds")}), flush=True)
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
