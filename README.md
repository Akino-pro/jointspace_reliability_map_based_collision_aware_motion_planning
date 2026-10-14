# Planar 3R reliability-aware motion planning

An experimental reliability-aware motion planner for a planar three-joint robot.
The current pre-failure workflow is:

```text
Collision-free IRIS polytopes -> reliability intersections -> connectivity filtering -> GCS path -> collision and endpoint checks
```

The default link lengths are `[1,1,1]`, with start position `(0.5,0.5)` and goal
position `(1.5,1.5)`. The obstacle is a circle centered at `(0.75,0.75)` with
radius `0.15`. The planner uses joint reliabilities `0.7`, `0.5`, and `0.3`.

## Main files

| File | Purpose |
| --- | --- |
| `planar3R_drake_iris.py` | IRIS region generation and checkpoint resume |
| `planar3R_iris_regions.py` | Checkpoint I/O, reliability intersections, and polytope adjacency |
| `planar3R_gcs_planner.py` | Main entry point, scene and reliability definitions, and JSON output |
| `planar3R_iris_gcs_planner.py` | GCS solving, candidate selection, and path validation |
| `planar3R_validate_iris.py` | Independent sampling of region interiors |
| `planar3R_post_failure_planner.py` | 2D GCS replanning after one planar 3R joint locks |
| `planar3R_collision.py` | Forward-kinematic link geometry and analytic collision checks |
| `planar3R_two_point_ssm_views.py` | IK, SSM tracing, endpoint refinement, and visualization |
| `planar3R_bbox_reliability_partition.py` | SSM bounding-box reliability partitions and visualization |
| `gcs/` | Drake GCS wrappers |
| `Dockerfile.drake` | Reproducible Docker runtime |
| `tests/` | Pre-failure and post-failure geometry, termination, and Drake integration tests |

Other reliability-map and animation scripts remain available for independent
research and historical visualization. Grid-based planning and the fallback
planner have been removed.

Single-joint locked-failure replanning uses 2D slices of the original IRIS regions
and a pure joint-space path-length objective. It stops and reports immediately
if the goal is unreachable, all goal configurations collide, or the current
coverage has no connecting path. See [POST_FAILURE_WORKFLOW.md](POST_FAILURE_WORKFLOW.md).

Revolute joints are assumed to have no mechanical angle limits. Both workflows
use the original IRIS regions without additional joint-angle clipping. The
cached IRIS model uses `[-pi,pi]^3` as a coordinate chart for periodic angles;
connections across opposite chart boundaries are not yet implemented. Thus the
current graph does not cover every possible periodic route.

## Current status and execution

See [IRIS_WORKFLOW.md](IRIS_WORKFLOW.md) for details. The current checkpoint
contains six IRIS regions generated with stronger numerical verification
settings. No collisions were detected in 30,000 sampled interior configurations.
The full planning run completed on 2026-10-13: reliability intersections produced
82 regions, connectivity filtering retained 43 regions, and total planning time
was approximately 30.3 seconds. The latest result is in `iris_path.json`.
Collision checks passed for 668 sampled path configurations, with endpoint error
approximately `2.94e-13`.

Using the configured container:

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_gcs_planner.py
docker stop planar3r-iris-validation
```

IRIS collision verification and independent checks are numerical. Detecting no
collisions does not establish a strict continuous collision-free certificate.
The current model does not check robot self-collision.
