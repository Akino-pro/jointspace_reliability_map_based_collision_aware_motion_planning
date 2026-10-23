# Spatial 4R single locked-joint replanning

`scripts/spatial4R_post_failure_planner.py` supports one of the four joints locking at
its failure-time angle, with the remaining joints healthy. The task constrains
XYZ position only. It follows the planar 3R post-failure workflow, with three
active coordinates and three-dimensional slices of the original 4D IRIS regions.
Joint reliability is not used after the failure; cost is ordinary Euclidean
path length in the three active joint angles, in radians. The fourth angle is
constant in every output configuration. Both available DH robots are supported
through the selected robot parameters; the completed obstacle benchmark uses
the first robot, ftwmax.

## Run

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_post_failure_planner --failed-joint 4 --path results/spatial4r/spatial4R_ftwmax_midpoint_two_stage_path.json --waypoint-index 0
```

The output defaults to `results/spatial4r/spatial4R_post_failure_path.json`. Instead of a saved
waypoint, provide `--q Q1 Q2 Q3 Q4 --goal X Y Z`. `--failed-joint` is required
and accepts 1..4. `--robot ftwmax|rcmax` selects a model; by default it is inferred
from the saved path, or ftwmax for explicit angles. `--iris-regions` overrides
the source report's original IRIS cache. The CLI checks model and collision
scene agreement before solving. A failure does not move the obstacle to a new
failure/goal midpoint: the checkpoint's existing physical scene is preserved.

Success exits with 0; all stopped/failed cases exit with 1 and save a structured
report. Default rounding settings are 200 unique paths and 300 trials, with
MOSEK. `--max-rounded-paths`, `--max-rounding-trials` and `--solver` are available.

## IK and stopping behavior

The three active angles normally produce discrete IK roots. Following the
requested model, the search supports at most four distinct physical goal
configurations (`IK_MAX_SOLUTIONS=4`). Periodic angle equivalents are deduplicated
before counting this limit. Both +/-pi chart endpoint representations can be
retained for the same physical root, so the number of chart representations can
exceed four without adding distinct physical configurations.

Unlike the planar equivalent-2R analytic solver, this implementation uses
numerical multistart IK. It starts with the failure configuration, a deterministic
5x5x5 angular seed cover, then random starts as needed, within a total budget
of 256 starts (`--ik-starts`), seed 0 (`--ik-seed`). Each solve uses the analytic
active-coordinate Jacobian and a maximum of 160 evaluations. Search stops at
four periodically distinct roots or the start budget. FK residual must be at
most 1e-8; periodic root deduplication uses 1e-5. No claim of certified root
completeness is made, even if the four-root limit is reached numerically.

Conservative radial reach exclusions can prove some goals unreachable. For a
locked first joint, a tighter reach sphere is centered on the fixed first DH
endpoint. If no numerical IK root is found without a reach exclusion, report
`ik_search_failed`, not proven unreachability. The report explicitly marks
`ik_completeness_certified=false` and records the numerical search settings.

Rank-deficient goal IK (minimum active-Jacobian singular value below 1e-7) stops
with `unsupported_singular_ik`. This conservatively excludes singular isolated
endpoints as well as possible continuous goal sets; rank deficiency alone does
not prove that the solution set is continuous. The already-at-goal case returns
the supplied configuration and zero cost without requiring an IK solve.

| Status | Meaning |
| --- | --- |
| `success` | Validated locked-joint path, or already at the goal |
| `unreachable` | A conservative geometric reach exclusion proves impossibility |
| `ik_search_failed` | No root found within the numerical budget; inconclusive |
| `unsupported_singular_ik` | A found goal IK is rank deficient; unsupported |
| `start_in_collision` | Failure configuration collides |
| `all_found_goals_in_collision` | Every found root collides; enumeration is not certified |
| `no_path_in_iris_coverage` | No found safe root connects through current slices |
| `solver_failed` | The single GCS solve returned no feasible rounded path |
| `path_validation_failed` | Returned path fails geometry, endpoint, lock or cost checks |
| `invalid_input` | Invalid state, robot, cache or source-scene mismatch |

Failure stops immediately at its workflow check. No extra regions, changed
targets, solve retries or fallback failure planners are generated automatically.

## GCS and returned path

Substitute the locked angle into every original IRIS region:
`A_active u <= b - A_locked * locked_angle`. Reliability subdivisions are not
used. Keep slices with a numerical 3D interior, build their adjacency graph,
and retain the source-connected component. Stop before GCS if none of the found
safe IK goals has coverage there. All covered goals enter one GCS problem;
multiple goal points feed a zero-cost bookkeeping sink. The sink is removed
from the physical path, and every region's length weight is exactly 1.

The output restores every input winding offset and uses the original supplied
failure angle exactly for the locked column. Original DH capsule-sphere checks
validate path samples; endpoint position, initial configuration and locked-angle
errors must be within tolerance. The path's active Euclidean length is also
checked against the reported GCS cost.

Reported GCS bounds apply only to the found goals in the covered slice graph.
Missing IK roots, limited IRIS coverage and missing opposite-chart boundary
connections prevent a global physical-path optimality certificate. Joints have
no physical angle limits; the internal [-pi,pi] chart is not an artificial limit.
Self-collision is disabled, as in pre-failure planning.

## Completed verification

Thirteen dedicated tests pass in Docker/Drake. They cover all four lock choices,
FK-valid roots, the four-root cap, slice substitution, unweighted GCS length,
multiple-IK selection, exact locked columns and input winding restoration.
Failure tests cover reach exclusion, inconclusive IK, singular IK, colliding
goals, source collisions and disconnected slices, including no-GCS stopping.
The full-GCS unit cases use a declared obstacle-free full-chart fixture and do
not substitute for the separate actual-obstacle benchmark.

The actual midpoint-obstacle ftwmax task was tested from its original start:

| Locked joint | Result | GCS calls | Runtime |
| --- | --- | ---: | ---: |
| 1 | No numerical IK root found | 0 | 0.75 s |
| 2 | No numerical IK root found | 0 | 0.62 s |
| 3 | No numerical IK root found | 0 | 0.64 s |
| 4 | Validated path | 1 | 0.94 s |

See `results/spatial4r/spatial4R_post_failure_benchmark.json` and `results/spatial4r/spatial4R_post_failure_joint1.json`
through `results/spatial4r/spatial4R_post_failure_joint4.json`. Joint 4 remains at -0.3 rad.
Two safe IK configurations were found; one is in connected slice coverage.
Three slice regions with four directed edges produce cost/length
0.7353883585602623 rad. All 77 sampled path configurations pass collision checks;
endpoint error is 9.42e-16 and lock error is exactly zero. The rounding budget
is 200/300, but this graph contains only one sampled unique route; 299 duplicate
trials were discarded. The graph-model bound agrees with cost to numerical
tolerance; this is not a certificate over unfound goals or uncovered free space.

## MuJoCo

```powershell
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_post_failure_path.json
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_post_failure_path.json --render --output-dir visualizations/spatial4R_post_failure_mujoco
```

The failed joint marker is red. Loading rejects trajectories whose indicated
locked column varies. No dynamic controller is simulated; these are the actual
saved joint-space segments. The exported joint-4 example has 56 frames at half
speed; 69 replay samples show no analytic collisions or MuJoCo contacts. Tip FK
agreement is within 1.58e-15. Ten host visualization tests pass, including the
new locked-joint loading, interpolation and marker checks. Sampled checks are
not continuous collision-free certificates.
