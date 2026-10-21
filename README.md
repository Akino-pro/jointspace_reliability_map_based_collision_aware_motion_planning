# Reliability-aware motion planning for planar 3R and spatial 4R

Experimental reliability-aware motion planning for planar three-joint and
standard-DH spatial four-joint robots. The original planar pre-failure workflow is:

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
| `spatial4R_model.py` | Active reference DH parameters, spatial FK/Jacobian, SSMs, and capsule collisions |
| `spatial4R_iris.py` | Spatial Drake model and 4D IRIS cache generation |
| `spatial4R_two_stage_gcs_planner.py` | Spatial position-task pre-failure two-stage planning |
| `SPATIAL4R_WORKFLOW.md` | Spatial configuration, validation, measured results, and coverage limits |
| `gcs/` | Drake GCS wrappers |
| `Dockerfile.drake` | Reproducible Docker runtime |
| `tests/` | Pre-failure and post-failure geometry, termination, and Drake integration tests |

Other reliability-map and animation scripts remain available for independent
research and historical visualization. Grid-based planning and the fallback
planner have been removed.

An additional fixed-start workflow implements adaptive local SSM hulls, one GCS
solve per effective target intersection, continuous IK projection, and a second full GCS solve to
the selected fixed goal. See [TWO_STAGE_SSM_WORKFLOW.md](TWO_STAGE_SSM_WORKFLOW.md)
for geometric epsilon, assumptions, validation, and comparison options. Run
`planar3R_two_stage_gcs_planner.py --geometric-epsilon 0.025 --ssm-arc-spacing 0.044`
in the Drake container; its default fixed start is waypoint 0 of `iris_path.json`
and its output is `two_stage_path.json`.

Single-joint locked-failure replanning uses 2D slices of the original IRIS regions
and a pure joint-space path-length objective. It stops and reports immediately
if the goal is unreachable, all goal configurations collide, or the current
coverage has no connecting path. See [POST_FAILURE_WORKFLOW.md](POST_FAILURE_WORKFLOW.md).

Revolute joints are assumed to have no mechanical angle limits. Both workflows
use the original IRIS regions without additional joint-angle clipping. The
cached IRIS model uses `[-pi,pi]^3` as a coordinate chart for periodic angles;
connections across opposite chart boundaries are not yet implemented. Thus the
current graph does not cover every possible periodic route.

## Spatial 4R prototype

The spatial pre-failure planner selects `--robot ftwmax|rcmax` (default: the first
robot, ftwmax), with joint reliabilities `[0.9,0.7,0.5,0.3]` from
`spatial_4r_reliability_maps_proposed_method.py` without running its grid code.
It traces all found position-task SSM branches, uses fixed arc spacing 0.044,
epsilon 0.025, and both-stage rounding budgets 200/300. The tested local-coverage
example completed with MOSEK: cost 0.0660305054, cached runtime 6.83 s, and
fixed-goal graph gap 14.53%. It is a feasible prototype, not a global optimum
certificate or full configuration-space coverage. A larger-domain attempt
exceeded the Docker memory budget; the completed benchmark uses explicitly
local IRIS inflation domains, not physical joint limits. See
[SPATIAL4R_WORKFLOW.md](SPATIAL4R_WORKFLOW.md) for the scene, assumptions and commands.

## Historical planar status and execution

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
