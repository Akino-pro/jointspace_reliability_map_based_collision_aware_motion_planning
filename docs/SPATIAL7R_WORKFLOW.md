# Spatial 7R pre-failure planning

## Affine-dimension adjacency attempt

The requested d-minus-one filter is implemented and tested. Applied to all
2,856 regions, it keeps 38,094 of 869,670 directed edges (95.62% fewer).
All regions remain in one connected component and all 23 target sets retain
the exact original target hash. The 19,047 candidate unordered pairs passed
affine tests: 18,204 dimension-six intersections and 843 dimension-seven
overlaps. Tolerance is 1e-7 after normalizing halfspace normals. Shared
filtering/preprocessing took 238.99 s, reusing the original adjacency cache.

Both 200/300 methods now pass graph construction but still fail before the
first native solve result within this VM's memory. At the 6 GiB worker limit,
union/independent attempts lasted 34.08/37.29 s. Removing redundant regional
and edge constraints preserves the same geometry, graph, targets and cost;
it reduced regional rows 58,848->44,186 and edge membership rows
591,087->93,552. At 7 GiB the compact union attempt lasted64.90s before SIGKILL,
and the first independent target lasted48.45s before supervisor termination.
These are aborted attempt times, not complete solve timings. No final path
or new feasible cost is available. No arbitrary nearest-neighbor pruning,
corridor cropping, reliability merging, or replacement seed trajectory was used.

See [the full report and commands](../results/spatial7r/facet_graph_60cm90deg_comparison/README.md).
Optional API inputs: `min_intersection_dimension`,
`candidate_adjacency_checkpoint`, `intersection_dimension_tolerance`, and
`compact_constraints`. Defaults preserve the existing planner behavior.
178 regression tests:170 passed,eight optional MuJoCo skips.

## Complete-graph attempts at 200/300

The full new-scene graph was explicitly prepared without cropping after
raising experiment-only guards: 2,856 regions, 869,670 directed edges and
23 effective targets. Shared preprocessing took 264.93 s. Independent-target
and union-hull workers were attempted sequentially at 200/300 in both stages.
They stopped at the supervisor's 6 GiB RSS threshold during GCS construction:
62.29 s for the first independent target and 65.29 s for the union graph.
Neither reached relaxation, sampling or stage two; no new path/cost exists.
See [complete measurements and logs](../results/spatial7r/full_graph_60cm90deg_comparison/README.md).

An axis-bound contact diagnostic found that at least 831,576 edges (95.62%)
have intersection-dimension upper bound below six. The original authors'
optional d-minus-one intersection rule would leave at most 38,094 candidates
for exact rank checks. That rule was not applied to these full-graph attempts.
It is a possible next experiment, requiring connectivity and path-cost checks,
not a guarantee of unchanged optima for arbitrary convex-region graphs.
The vectorized adjacency broad phase preserves all original nonempty-contact
edges and passes the existing exact all-pairs parity tests. Adjacency caching
is keyed by the exact region geometry/reliability fingerprint.

## Active task: 60 cm / base-Z 90 degrees / exact midpoint obstacle

The current default task preserves the original start configuration and
translation direction, doubles the position displacement to 0.60 m, and
sets `R_goal = Rz(+90 degrees) @ R_start`. This is a rotation about the fixed
base/world Z axis, not the local tool Z axis. The sphere radius remains 0.08 m;
its centre is exactly `(p_start + p_goal)/2`, with no vertical offset.
World positions (m): start (3.0205333342,-1.4946400076,0.5399182592),
goal (3.1878890016,-0.9193011224,0.5711792222), obstacle centre
(3.1042111679,-1.2069705650,0.5555487407).

New artifacts are kept separate from the previous 30 cm / 20-degree task:
`results/spatial7r/spatial7R_60cm90deg_scene.json`,
`data/iris/spatial7R_60cm90deg_iris_regions.npz`,
`results/spatial7r/spatial7R_60cm90deg_two_stage_path.json`, and
`visualizations/spatial7R_60cm90deg_mujoco/`.
Run `python -m scripts.spatial7R_midpoint_obstacle_experiment` to build the
new IRIS coverage and solve the primary method at 200/300 in both stages.
The auxiliary safe corridor supplies IRIS seeds only, not the final GCS path.
The previous five-budget results below describe the historical task and
must not be interpreted as performance for this new scene.

The first attempt generated three original IRIS regions (162.14 s total),
then stopped at `resource_limit`: reliability intersections produced 2,856
regions against the configured limit of 160. No GCS call or new optimized
trajectory was produced. This is a computational coverage/partition limit,
not an infeasibility result. The saved report explicitly records failure.

Run `python -m scripts.spatial7R_mujoco_viewer --scene-only` to inspect the
new task without a connecting trajectory. Press S for the start robot pose
and G for the goal pose; both fixed endpoint coordinate frames remain visible.
Add `--render` for the paired endpoint image and obstacle close-up in
`visualizations/spatial7R_60cm90deg_mujoco/`. Default viewer execution also
falls back to this preview when the active task has no successful saved path.
Both endpoints passed native MuJoCo contact checks. Twelve native MuJoCo
tests passed; the Docker regression ran 172 tests with 164 passes and eight
optional MuJoCo skips.

## Historical 30 cm / 20-degree MuJoCo visualization

Run `python -m scripts.spatial7R_mujoco_viewer --path results/spatial7r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json --output-dir visualizations/spatial7R_mujoco_20261029` on the native host to replay
the matched-budget experiment's 200/300 path with its saved seven-joint DH
parameters and exact capsule/sphere geometry. Use `--render` to export PNG/GIF
and start/goal side-by-side and close-up images into
`visualizations/spatial7R_mujoco_20261029/`. This is kinematic path replay.

The fixed start and goal tool coordinate frames are both shown and labeled
START_X/Y/Z and GOAL_X/Y/Z: local X is red, Y green, Z blue. The blue and green
spheres mark the start and goal tool positions; the larger red sphere is the
actual obstacle. The task constrains full SE(3) pose (three position and three
orientation degrees of freedom), not just the XYZ tool location. The replay
report saves both homogeneous poses and verifies the final tool rotation.

World coordinates in metres: start (3.020533,-1.494640,0.539918),
goal (3.104211,-1.206971,0.555549), obstacle centre
(3.062372,-1.350805,0.747733), radius 0.08. The task translation is 0.30 m
and relative orientation change is 20 degrees. The original path and
obstacle positions are preserved. Eleven native MuJoCo tests passed, including
seven-joint random-pose FK/rotation and contact parity against the planner.
The rendered replay passed 41 sampled collision/contact checks with maximum
MuJoCo/planner tool-position difference 2.24e-15 m.

Window controls: mouse rotate/zoom, Space pause, R restart, L toggle loop,
T toggle path and frame-axis geometry, Esc close.

## Historical 30 cm / 20-degree matched-stage budget sweep (2026-10-29)

`python -m scripts.spatial7R_per_target_budget_sweep` runs only the primary
independent-target method at 50/75, 100/150, 200/300, 300/450 and 400/600.
Each first-stage target receives the full case budget; stage two uses the same
budget, plus the retained first-stage route restriction. Both 7R defaults
remain 200/300. The 4R API and CLI now explicitly default both stages to 200/300.
The runner now defaults to the new 60 cm / 90-degree cache and a separate
output directory. The following completed measurements used the previous scene.

All five fresh runs returned final cost 0.004308689987573309. Total measured
times, including preprocessing, were 15.804, 16.765, 15.139, 14.985 and 14.849 s.
All final paths passed collision, graph, position and orientation checks.
The graph/task and target fingerprints matched across all runs: 24 regions,
424 directed edges, 4 targets, one detected SSM branch and 92 samples.
Each case sampled only two unique first-stage routes in total, with one
successful restriction and one target returning a valid path. Stage two
sampled one unique route. Larger caps did not produce additional unique
routes or improve cost in this scene. Three missing target solutions do not
prove infeasibility. Single-run timing differences do not establish a speed
advantage for larger budgets; no global optimality is certified.

See [full measured results and charts](../results/spatial7r/per_target_matched_stage_budget_sweep_20261029/README.md).

Optional comparisons now include `--compare-union-hulls`,
`--compare-nonlinear-goal` and `--compare-all`, alongside `--compare-sampled`.
See [goal-set comparisons](GOAL_SET_COMPARISONS.md) for the terminal models,
native nonlinear solver settings and the measured four-method 7R experiment.

The entry point is `python -m scripts.spatial7R_two_stage_gcs_planner`, run from
the repository root in the Drake container. The implementation uses the actual
DH constants and reliabilities in `scripts/spatial_7r_FTW_estimation_ssm.py`.
Only numeric AST assignments are read; the reference module is never imported,
so its top-level workspace grid and Robotics Toolbox dependencies do not run.

## Robot and task

| Joint | a | d | alpha (degrees) | Reliability |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.4 | -0.4 | -62 | 0.2 |
| 2 | 0.8 | -0.6 | -79 | 0.3 |
| 3 | 0.2 | 0.2 | 90 | 0.4 |
| 4 | 1.0 | 0.6 | 29 | 0.5 |
| 5 | 0.6 | -0.8 | 81 | 0.6 |
| 6 | 0.4 | 0.2 | -80 | 0.7 |
| 7 | 0.0 | 0.0 | 0 | 0.8 |

The reference uses all six position/orientation constraints. This workflow
therefore plans to a full SE(3) goal pose. A regular 6x7 Jacobian has a
one-dimensional nullspace, so ordered SSM curves and the existing arc-length
hull approximation remain applicable. A position-only 7R task would have a
four-dimensional regular SSM and is not handled by this curve entry point.

The zero seventh DH displacement is intentional and copied exactly. Joint 7
changes tool orientation without changing tool position; the Drake plant still
has seven active revolute joints. The source CA values are retained as reference
metadata but are not enforced, following the current project's unlimited-angle
assumption. The [-pi,pi]^7 chart and local IRIS inflation bounds restrict
coverage, not physical joint motion. Opposite-chart boundary edges remain absent.

## Workflow

1. Validate the fixed start, full goal pose, collision scene and search budgets.
2. Generate or load local seven-dimensional IRIS regions using capsules for both
   DH offset and link segments. Self-collision remains disabled.
3. Use multistart full-pose IK and nullspace continuation to discover up to
   **eight** regular closed SSM branches, per the user's requested limit. Search
   defaults to 80 IK starts and seed 1; discovery is not certified complete.
4. Subdivide each original IRIS bounding domain at every discovered branch's
   coordinate bounds. Each resulting cell unions the individual branch recovery
   sets and receives the conditional reliability of that union, counting each
   failure event and the no-failure event once. Different failures may select
   different branches. This is the planar/4R rule in seven dimensions.
5. Merge equal-reliability cells only if their union is exactly a box, then
   intersect with the full original IRIS halfspaces. Restricting refinement to
   exact LP coordinate bounds avoids constructing unreachable parts of a global
   partition; it does not replace collision polytopes with their bounding boxes.
6. Sample every branch at a common arc spacing, split at chart seams, form
   adaptive local sample hulls and intersect them with connected free regions.
7. Run one independent first-stage GCS for every effective target intersection.
   Validate each returned route against its region sequence and target hull.
8. Project the selected first-stage endpoint onto the selected local continuous
   SSM with **both position and orientation equality constraints**.
9. Run a fresh fixed-start/fixed-goal GCS, retaining the first-stage region route
   as an additional convex restriction. This is not a direct projection stitch.
10. Restore input winding offsets and validate graph membership, collisions,
    start, fixed goal, Cartesian position and orientation errors.

Costs remain `(0.01 + 1 - R) * Euclidean joint-space length`, with all seven
joint coordinates included. Reliability follows the project's conditional
at-most-one-joint-failure model, using the seven reference reliabilities.

The rotation residual is the SO(3) logarithm of actual rotation times goal
rotation transpose, with its analytic local Jacobian. A half-turn orientation
error cannot disappear as it can with an unqualified cross-product residual.
Singular/higher-nullity SSM traces are skipped and reported as trace failures;
no curve is fabricated for them.

## Defaults and commands

- Both GCS stages: 200 unique paths, 300 rounding trials, MOSEK.
- Geometric epsilon: 0.025; common arc spacing: 0.044 rad.
- Start q: [0.20,-0.62,0.30,-0.55,0.44,-0.36,0.12].
- The current task has exactly 0.30 m Cartesian separation and a 20-degree
  relative orientation change. Start XYZ is [3.0205333342,-1.4946400076,0.5399182592],
  goal XYZ is [3.1042111679,-1.2069705650,0.5555487407].
- Goal pose is FK at [0.4430183224,-0.7496097719,0.4620122149,-0.6310061075,
  0.5372073289,-0.4248048860,0.3716597124]; the final
  configuration is estimated on its SSM, not fixed to these reference angles.
- Sphere center: Cartesian endpoint midpoint plus [0,0,0.20], radius 0.08.
- Capsule radius: 0.015; local IRIS inflation half-width: 0.16 rad.
- Default cache: `data/iris/spatial7R_30cm20deg_iris_regions.npz`.
- Default result: `results/spatial7r/spatial7R_30cm20deg_two_stage_path.json`.

```powershell
docker start planar3r-iris-validation
# First run: generate the small local IRIS cover.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_two_stage_gcs_planner --generate-iris
# Reuse the cache, preserving the first run's report.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_two_stage_gcs_planner --output results/spatial7r/spatial7R_30cm20deg_cached_path.json
# Preprocess without GCS.
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial7R_two_stage_gcs_planner --geometry-only --output results/spatial7r/spatial7R_30cm20deg_geometry.json
# Independent collision sampling of the saved regions.
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial7R_validate_iris --samples 5000
```

CLI overrides include `--q` (seven angles), `--goal-position X Y Z` together
with `--goal-quaternion X Y Z W`, `--iris-regions`, `--iris-radius`,
`--ssm-ik-attempts`, `--ssm-seed`, `--geometric-epsilon`, `--ssm-arc-spacing`,
`--max-rounded-paths`, `--max-rounding-trials`, `--solver` and `--output`.
An altered scene or inflation domain needs its own matching cache; it is not
silently substituted into an old checkpoint. Custom goals may require new seed
coverage beyond the supplied reference seed.

Exact partitioning and GCS graphs can grow sharply with dimensionality and
branch count. Module-level resource limits are 20,000 local partition cells,
160 free regions, 1,500 directed edges and 128 effective targets. The API accepts
overrides. Exceeding a limit reports `resource_limit` before GCS, rather than
dropping branches, returning false infeasibility or exhausting Docker memory.
Other preflight statuses include `goal_unreachable` for a conservative radial
reach exclusion, `start_in_collision`, `ssm_sampling_failed` and
`no_path_in_iris_coverage`. No missing path in a limited cover is a proof of
global physical impossibility.

## Current 30 cm / 20-degree benchmark

The default task completed with measured Cartesian separation
0.30000000000000004 m and relative SO(3) rotation 20.000000000000004 degrees.
IRIS seeds are tried in midpoint/start/reference-goal order. A single midpoint
region covered both endpoint configurations, avoiding duplicate overlapping
coverage in this example. The sphere remains above the Cartesian midpoint;
this is a local feasibility example, not an obstacle-detour benchmark.

| Metric | Value |
| --- | ---: |
| Discovered SSM branches / arc samples | 1 / 92 |
| Local sample hulls / effective targets | 23 / 4 |
| Original IRIS / reliability free regions | 1 / 24 |
| Directed graph edges | 424 |
| IRIS generation | 16.82 s |
| First / second GCS stage | 13.18 / 0.35 s |
| Total including IRIS | 32.16 s |
| Final weighted cost | 0.004308689987573309 |
| Final position / orientation error | 5.44e-16 m / 2.37e-16 rad |
| Path collision samples | 45, all passed |
| Independent IRIS collision samples | 5,000, all passed |

The final route is [16], with R=1 and weight 0.01. Its unweighted joint-space
length is approximately 0.430869 rad. The retained first-stage restriction
matches the native second-stage result. The projection-connection comparison
cost is 0.004308778347183775; it is not used as the final path.

Only one of four first-stage targets returned a validated feasible path. Its
individual graph gap is 1.97%; the aggregate first-stage bound is unavailable.
The fixed-goal second-stage bound agrees with its cost to numerical precision.
This does not certify globally optimal target selection on the continuous SSM.

An earlier start/goal-seeded attempt generated two original regions, 72 free
regions and 2,392 directed edges, then exhausted Docker memory during the first
GCS solve. Its checkpoint and failure report are preserved under names containing
`spatial7R_30cm20deg_endpoint_seed`; no final path came from that attempt.
The directed-edge guard was reduced to 1,500 before rerunning successfully with
midpoint seeding. A resource guard reduces risk; it is not a memory guarantee.

## Original 18.6 cm benchmark (historical)

The following results use the previous goal. Their reports and original
`data/iris/spatial7R_iris_local_regions.npz` cache are preserved. The current
defaults above use an independent 30 cm / 20-degree scene and cache.

This is a small, feasible local full-pose demonstration. The sphere is above
the Cartesian midpoint; it does not block the final direct joint-space segment.
It is not a difficult obstacle-detour benchmark.

| Metric | Value |
| --- | ---: |
| Detected SSM branches | 1 after 80 attempts, cap 8 |
| Total traced arc length | 3.672692304 rad |
| Arc samples / local hulls | 84 / 22 |
| Original IRIS regions | 1 |
| Local reliability cells / free regions | 24 / 24 |
| Directed graph edges / effective targets | 424 / 5 |
| First-stage feasible returned targets | 3 of 5 |
| Fresh IRIS generation | 16.31 s |
| First full run, including IRIS | 32.12 s |
| Cached preprocessing | 1.44 s |
| Cached first stage | 13.84 s |
| Cached second stage | 0.33 s |
| Cached total | 15.66 s |
| Final weighted cost | 0.0017474369143094923 |
| Final Cartesian position error | 1.02e-15 |
| Final orientation error | 1.23e-16 rad |
| Collision samples | 19, all passed |

The selected final configuration is approximately
[0.2991141632,-0.6481254145,0.3194587531,-0.5639247640,0.4016982777,-0.4203832141,0.2393105394].
The final graph route is [16], with one distinct rounded route. Its retained
first-stage route yields the same cost. The weighted cost is small because
this segment lies in an R=1 region with weight 0.01; its unweighted joint-space
length is approximately 0.1747436914 rad.

The historical default report also includes the requested projection-connection
comparison: first-stage path plus a direct projection segment costs
0.001966858465907098, versus final reoptimized cost 0.0017474369143094923.
This comparison is never used as the final path. Adding its sampled checks and
rerunning the default workflow gave 17.09 s, with the same selected goal and
final cost. The earlier cached report above took 15.66 s; the original cold
report is retained as `spatial7R_two_stage_cold_path.json`.

Numerical limitation: two first-stage targets returned no feasible rounded
path. The other three returned validated feasible graph paths, but their
relaxation bounds exceed their costs beyond tolerance (`inconsistent_bounds`).
The aggregate first-stage bound is unavailable. Thus no reliable certification
of first-stage target ranking or continuous-SSM global optimality is claimed.
The final **fixed-goal** graph bound agrees with its final cost to numerical
precision, which is a narrower statement. Improving the first-stage numerical
conditioning/coverage remains follow-up work, not a completed optimality proof.

Historical exploratory artifacts are retained: radius 0.35 produced 196 regions
and stopped at the resource limit; see `spatial7R_geometry_radius035_limit.json`
and `data/iris/spatial7R_iris_radius035_regions.npz`. Multiplying all objectives
by 1000 was tested, returned no feasible rounded first-stage paths and was not
retained; its report is `spatial7R_scaled_two_stage_path.json`. The working
implementation keeps the original objective scale.

## Post-failure planning (2026-10-26)

Entry point: `python -m scripts.spatial7R_post_failure_planner`. One joint is
locked at its failure angle; the six remaining coordinates are active. The
original full SE(3) goal is preserved. The cost is the sum of Euclidean segment
lengths in those six coordinates, with every region weight equal to one.
Reliability maps, reliability probabilities and the pre-failure 0.01 offset
are not used after failure.

The planner substitutes the locked angle into **all original IRIS halfspaces**,
forming 6D collision-free slices without reliability subdivision. Periodic
multistart IK finds intersections of the full-pose goal SSMs with the locked
angle plane. Each regular intersection is a discrete goal configuration. All
found collision-free goals in the source-connected slice graph attach to a
single GCS with a zero-cost bookkeeping sink. This compares paths to different
goal branches directly; the sink adds no physical motion or cost. No projection
or second planning stage is needed after failure. All goal attachments in all
containing regions are retained.

Default budgets: 256 IK starts (seed 0), 200 rounded paths / 300 trials, MOSEK.
The IK search runs its full budget with no inherited four-root cap. A root
index identifies an endpoint, not an original pre-failure SSM branch ID: a
branch can intersect the locked-angle plane more than once. IK discovery is
not certified complete, and GCS relaxation/rounding does not certify global
branch selection. Solver bounds are reported for the graph of found goals.
Rank-deficient locked-joint goals return `unsupported_singular_ik`; a continuous
singular goal set is not represented as a fabricated finite set of points.

The output restores input winding offsets and holds the supplied failed angle
exactly. Checks include all six pose residuals, the start and locked angle,
graph/target membership, physical path length and sampled capsule collisions.
The original scene is preserved when loading a pre-failure report; mismatched
robot or checkpoint geometry is rejected. Self-collision remains disabled and
spatial periodic seam connections remain deferred.

```powershell
# Read the default successful pre-failure report and lock joint 7 at waypoint 0.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial7R_post_failure_planner --failed-joint 7
# Supply a failure state explicitly; the default goal and IRIS scene are retained.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial7R_post_failure_planner --failed-joint 7 --q 0.44 -0.75 0.46 -0.63 0.54 -0.42 0.37
```

Other flags: `--path`, `--waypoint-index`, `--iris-regions`, `--ik-starts`,
`--ik-seed`, `--max-rounded-paths`, `--max-rounding-trials`, `--solver` and
`--output`. A custom full goal requires both `--goal-position X Y Z` and
`--goal-quaternion X Y Z W`. The default output is
`results/spatial7r/spatial7R_post_failure_path.json`.

Real-scene experiments reused the 30 cm / 20-degree IRIS checkpoint without
generating new regions. At the saved pre-failure start, all seven lock cases
returned `no_path_in_iris_coverage`. This means that no found safe root is
connected within the limited cache; it is not proof of physical unreachability.
Reports: `results/spatial7r/spatial7R_post_failure_joint{1..7}.json`.

At 95% of the saved straight pre-failure segment,
`q_failure = 0.05*q_start + 0.95*q_final`, **all seven lock cases succeeded**
with the original obstacle scene. Each had one connected goal and one GCS
region. Results are saved as
`results/spatial7r/spatial7R_post_failure_near_goal_joint{1..7}.json`.

| Locked joint | Unweighted path cost (rad) |
| --- | ---: |
| 1 | 0.04244412 |
| 2 | 0.02155192 |
| 3 | 0.02226220 |
| 4 | 0.03581863 |
| 5 | 0.01996857 |
| 6 | 0.03927720 |
| 7 | 0.04771531 |

Eleven new post-failure tests passed in Docker, including actual MOSEK solves
for every locked joint, winding restoration and a controlled multi-goal graph
that selects the shorter endpoint path. Full regression: 153 tests, 147 passed,
six optional MuJoCo tests skipped. These results do not establish complete IK
enumeration, continuous collision certification or performance on difficult
obstacle-detour scenes.

## Verification and scope

All 16 7R tests pass, including the default 30 cm / 20-degree task regression.
They check reference parsing without execution, exact
DH pose/collision agreement with Drake at 100 random configurations, full-pose
residual Jacobians, the zero-offset wrist, half-turn rotations, closed actual
SSM pose accuracy, all 128 reliability subsets, eight-branch aggregation and
target inclusion, exact retained IRIS boundaries, resource stopping, unreachable
preflight and seven-coordinate GCS cost. Eight-branch fixtures are controlled
tests, not a claim that this physical example has eight branches.

The full Docker suite ran 125 tests: 119 passed and six optional MuJoCo tests
were skipped in the headless environment. That pre-failure change added no 7R
MuJoCo viewer. The post-failure planner is documented above. The saved 7D IRIS region also passed 5,000 independent
hit-and-run collision samples; this does not certify continuous collision
freedom. No new packages were needed and no reference grid was run.
