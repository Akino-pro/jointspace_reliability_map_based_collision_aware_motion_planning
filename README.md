# Reliability-aware motion planning for planar 3R and spatial 4R/7R

Experimental reliability-aware motion planning for planar three-joint and
standard-DH spatial four- and seven-joint robots. The original planar pre-failure workflow is:

```text
Collision-free IRIS polytopes -> reliability intersections -> connectivity filtering -> GCS path -> collision and endpoint checks
```

## Project layout and commands

Run commands from the repository root. Python entry points are now modules under
`scripts/`; use `python -m scripts.MODULE`, rather than the previous root filenames.

| Directory | Contents |
| --- | --- |
| `scripts/` | Robot models, geometry, planning, diagnostics and visualization code |
| `gcs/` | Shared Drake GCS implementation |
| `tests/` | Automated geometry, solver and replay tests |
| `docs/` | Workflow guides and `PROJECT_CONVERSATION_HANDOFF.txt` |
| `data/iris/` | Original and historical IRIS checkpoints |
| `data/ssm/` | Saved SSM traces |
| `data/seeds/` | IRIS seed inputs |
| `results/planar3r/` | Planar planning results and comparisons |
| `results/spatial4r/` | Spatial planning results and comparisons |
| `results/spatial7r/` | Seven-joint full-pose planning results and comparisons |
| `results/diagnostics/` | Solver and environment diagnosis reports |
| `visualizations/` | Scene models, GIFs, PNGs and replay artifacts |
| `logs/` | Solver diagnostic logs |
| `config/` | Dockerfile, optional dependencies and old-to-new file map |
| `licenses/` | Local MOSEK license, ignored by Git |

```powershell
# Native MuJoCo playback of the saved pre-failure paths
python -m scripts.planar3R_mujoco_viewer
python -m scripts.spatial4R_mujoco_viewer

# Native playback of the successful spatial locked-joint example
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_post_failure_path.json

# Drake workflow in the configured container
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial4R_post_failure_planner --failed-joint 4
```

See [the layout guide](docs/FILE_ORGANIZATION.md) and the machine-readable
[file map](config/file_locations.json). Detailed workflow commands below use the
same module convention. Historical results have been retained with updated path
references; computed configurations, costs and checkpoint geometry are preserved.

The default link lengths are `[1,1,1]`, with start position `(0.5,0.5)` and goal
position `(1.5,1.5)`. The obstacle is a circle centered at `(0.75,0.75)` with
radius `0.15`. The planner uses joint reliabilities `0.7`, `0.5`, and `0.3`.

## Main files

| File | Purpose |
| --- | --- |
| `scripts/planar3R_drake_iris.py` | IRIS region generation and checkpoint resume |
| `scripts/planar3R_iris_regions.py` | Checkpoint I/O, reliability intersections, and polytope adjacency |
| `scripts/planar3R_gcs_planner.py` | Main entry point, scene and reliability definitions, and JSON output |
| `scripts/planar3R_iris_gcs_planner.py` | GCS solving, candidate selection, and path validation |
| `scripts/planar3R_validate_iris.py` | Independent sampling of region interiors |
| `scripts/planar3R_post_failure_planner.py` | 2D GCS replanning after one planar 3R joint locks |
| `scripts/planar3R_collision.py` | Forward-kinematic link geometry and analytic collision checks |
| `scripts/planar3R_mujoco_viewer.py` | Native MuJoCo arm/obstacle visualization and saved-path playback |
| `scripts/planar3R_two_point_ssm_views.py` | IK, SSM tracing, endpoint refinement, and visualization |
| `scripts/planar3R_bbox_reliability_partition.py` | SSM bounding-box reliability partitions and visualization |
| `scripts/spatial4R_model.py` | Active reference DH parameters, spatial FK/Jacobian, SSMs, and capsule collisions |
| `scripts/spatial4R_iris.py` | Spatial Drake model and 4D IRIS cache generation |
| `scripts/spatial4R_two_stage_gcs_planner.py` | Spatial position-task pre-failure two-stage planning |
| `docs/SPATIAL4R_WORKFLOW.md` | Spatial configuration, validation, measured results, and coverage limits |
| `gcs/` | Drake GCS wrappers |
| `config/Dockerfile.drake` | Reproducible Docker runtime |
| `tests/` | Pre-failure and post-failure geometry, termination, and Drake integration tests |

Other reliability-map and animation scripts remain available for independent
research and historical visualization. Grid-based planning and the fallback
planner have been removed.

An additional fixed-start workflow implements adaptive local SSM hulls, one GCS
solve per effective target intersection, continuous IK projection, and a second full GCS solve to
the selected fixed goal. See [docs/TWO_STAGE_SSM_WORKFLOW.md](docs/TWO_STAGE_SSM_WORKFLOW.md)
for geometric epsilon, assumptions, validation, and comparison options. Run
`python -m scripts.planar3R_two_stage_gcs_planner --geometric-epsilon 0.025 --ssm-arc-spacing 0.044`
in the Drake container; its default fixed start is waypoint 0 of `results/planar3r/iris_path.json`
and its output is `results/planar3r/two_stage_path.json`.

Single-joint locked-failure replanning uses 2D slices of the original IRIS regions
and a pure joint-space path-length objective. It stops and reports immediately
if the goal is unreachable, all goal configurations collide, or the current
coverage has no connecting path. See [docs/POST_FAILURE_WORKFLOW.md](docs/POST_FAILURE_WORKFLOW.md).

Revolute joints are assumed to have no mechanical angle limits. Both workflows
use the original IRIS regions without additional joint-angle clipping. The
cached IRIS model uses `[-pi,pi]^3` as a coordinate chart for periodic angles;
connections across opposite chart boundaries are disabled by default. The
two-stage and post-failure entry points accept optional `--periodic` connections;
this restricted graph still does not cover every possible periodic route. See
[the comparison guide](docs/LEGACY_METHOD_COMPARISON.md) for scope and examples.

## Planar 3R MuJoCo viewer

Run `python -m scripts.planar3R_mujoco_viewer` on the native host to display the arm,
obstacles, goal and saved trajectory. Use `--path FILE.json` for another
pre/post-failure result, or `--render` to export MuJoCo PNG/GIF previews.
Space pauses, R restarts, L toggles looping, and T toggles the tip trace.
The viewer preserves the original unwrapped GCS segments and replays positions;
it does not simulate a tracking controller or rerun planning. See
[docs/MUJOCO_VISUALIZATION.md](docs/MUJOCO_VISUALIZATION.md) for commands and assumptions.

## Spatial 4R prototype

Single locked-joint replanning is available in `scripts/spatial4R_post_failure_planner.py`.
It searches up to four physical IK goals, slices original IRIS regions into 3D,
and runs one unweighted-length GCS with budgets 200/300. The failed angle stays
fixed in every four-angle waypoint. Numerical IK completeness is not certified;
unsupported singular IK and missing coverage stop before GCS. See
[docs/SPATIAL4R_POST_FAILURE_WORKFLOW.md](docs/SPATIAL4R_POST_FAILURE_WORKFLOW.md) for commands,
status meanings and the completed obstacle benchmark.

Up to four detected SSM branches share one sequentially refined reliability
map, using the maximum per-branch reliability, exact convex box merging and
common arc sampling density. All branches supply first-stage GCS targets.
The default numerical discovery budget is now 80 IK attempts (`--ssm-ik-attempts`);
branch completeness is not certified. Reports include per-branch statistics.

Run `python -m scripts.spatial4R_mujoco_viewer` on the native host to replay the first
robot's saved 200/300 result with its DH geometry and spherical obstacles.
Use `--render` for PNG/GIF exports or `--path FILE.json` for another spatial
result, including the second robot. See [docs/MUJOCO_VISUALIZATION.md](docs/MUJOCO_VISUALIZATION.md).

The current spatial example places the radius-0.10 sphere at the start/goal
Cartesian midpoint and uses separate `midpoint` IRIS and result files. To
regenerate its connected cover, pass `--generate-iris --iris-seeds
data/seeds/spatial4R_ftwmax_midpoint_iris_seeds.json` to the planner. Historical results
below used the previous obstacle location; see the spatial workflow for details.

The spatial pre-failure planner selects `--robot ftwmax|rcmax` (default: the first
robot, ftwmax), with joint reliabilities `[0.9,0.7,0.5,0.3]` from
`scripts/spatial_4r_reliability_maps_proposed_method.py` without running its grid code.
It traces all found position-task SSM branches, uses fixed arc spacing 0.044,
epsilon 0.025, and both-stage rounding budgets 200/300. The tested local-coverage
example completed with MOSEK: cost 0.0660305054, cached runtime 6.83 s, and
fixed-goal graph gap 14.53%. It is a feasible prototype, not a global optimum
certificate or full configuration-space coverage. A larger-domain attempt
exceeded the Docker memory budget; the completed benchmark uses explicitly
local IRIS inflation domains, not physical joint limits. See
[docs/SPATIAL4R_WORKFLOW.md](docs/SPATIAL4R_WORKFLOW.md) for the scene, assumptions and commands.

## Spatial 7R pre-failure prototype

Shared adjacency screening and the optional direct sampled-SSM baseline are
documented in [the legacy-method comparison](docs/LEGACY_METHOD_COMPARISON.md).
All three pre-failure planners accept `--compare-sampled`. Planar 3R additionally
accepts optional `--periodic` connections, disabled by default; the two-stage
method and existing reliability model remain the primary workflow.

`python -m scripts.spatial7R_two_stage_gcs_planner` uses the DH parameters and
reliabilities from `scripts/spatial_7r_FTW_estimation_ssm.py`, without importing
or executing its workspace grid. It plans to a full position-and-orientation
goal, traces up to eight regular one-dimensional SSM branches, and uses the
same two-stage GCS workflow with both rounding budgets 200/300. Exact reliability
partitioning is restricted to original IRIS coverage to control seven-dimensional
growth, with explicit resource limits and all-branch maximum reliability.

The current local example has exactly 30 cm translation and 20 degrees of
relative rotation: one detected branch, 92 samples, 24 free regions and four
targets. It completed in 32.16 s including IRIS generation, with final weighted
cost 0.0043086900 and position/orientation errors near machine precision.
Three first-stage targets returned no feasible path, so target-selection/global
optimality is not certified. See [the 7R workflow](docs/SPATIAL7R_WORKFLOW.md)
for commands, source geometry, measured results and limitations.

## Historical planar status and execution

See [docs/IRIS_WORKFLOW.md](docs/IRIS_WORKFLOW.md) for details. The current checkpoint
contains six IRIS regions generated with stronger numerical verification
settings. No collisions were detected in 30,000 sampled interior configurations.
The full planning run completed on 2026-10-13: reliability intersections produced
82 regions, connectivity filtering retained 43 regions, and total planning time
was approximately 30.3 seconds. The latest result is in `results/planar3r/iris_path.json`.
Collision checks passed for 668 sampled path configurations, with endpoint error
approximately `2.94e-13`.

Using the configured container:

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_gcs_planner
docker stop planar3r-iris-validation
```

IRIS collision verification and independent checks are numerical. Detecting no
collisions does not establish a strict continuous collision-free certificate.
The current model does not check robot self-collision.
