# Replanning after a single locked-joint failure

For the spatial 4R counterpart (three active joints, XYZ task and up to four
physical IK configurations), see
[docs/SPATIAL4R_POST_FAILURE_WORKFLOW.md](SPATIAL4R_POST_FAILURE_WORKFLOW.md).
The rest of this document describes the planar 3R analytic workflow.

This workflow supports exactly one joint of the current planar 3R robot
(link lengths `[1,1,1]`) locking at its current angle. The other two joints are
assumed healthy, with no subsequent failures. The task constrains only the
2D end-effector position. The entry point is `scripts/planar3R_post_failure_planner.py`.

## Execution

Start the configured Docker container from PowerShell:

```powershell
docker start planar3r-iris-validation
```

Simulate joint 3 locking at a waypoint of the pre-failure path, then attempt to
reach the original goal:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_post_failure_planner --failed-joint 3 --path results/planar3r/iris_path.json --waypoint-index 0
```

Alternatively, provide the three joint angles at failure in radians. The locked
angle is taken from this configuration; there is no separate angle parameter:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_post_failure_planner --failed-joint 3 --q -0.3182651673106314 1.8792519611450273 -0.4722077464279502 --gx 1.5 --gy 1.5 --output results/planar3r/post_failure_success.json
```

The default output is `results/planar3r/post_failure_path.json`; use `--output` to change it.
Success returns exit code 0 and failure returns 1. Failures also produce JSON
containing `status`, the reason, and `gcs_solve_calls`. `--failed-joint` accepts
exactly one of 1, 2, or 3. Joints are assumed to have no mechanical angle stops.
Finite failure angles outside `[-pi,pi]` are mapped to the cached periodic
coordinate chart internally; their winding offsets are restored in reported IK
solutions and output paths. The supplied locked angle remains exactly fixed.
Replanning slices the original IRIS regions with no additional angular clipping.
The current graph does not connect opposite chart boundaries, so periodic route
coverage remains incomplete.

## Workflow

1. Validate the failure configuration and scene.
2. Analytically solve the equivalent 2R IK for the locked joint, then map all solutions back to the original 3R configuration. There are generally zero, one, or two physical solutions; equivalent angle representations at the chart boundary may increase the number of representations.
3. Check goal collisions using the original three links. Equivalent straight links are not substitutes for collision geometry.
4. Substitute the locked angle into each original collision-free IRIS region to obtain 2D halfspaces: `A_active u <= b - A_locked alpha`. Reliability-partitioned regions are not used.
5. Check connectivity between the failure state and valid goals in the current slice graph. Stop immediately if none are connected.
6. Include all connected goals in one GCS solve. All region weights are 1; the cost is Euclidean path length in the two active joints' configuration space. A zero-cost dummy sink selects an IK goal and does not represent physical motion.
7. Reconstruct the 3D joint path with the locked joint fixed throughout, restore supplied angle windings, and check endpoints and path collisions.

## Termination statuses

| Status | Meaning |
| --- | --- |
| `success` | Already at the goal, or a path was found and passed validation |
| `unreachable` | The goal is kinematically unreachable after the joint locks |
| `start_in_collision` | The failure configuration itself is in collision |
| `all_goals_in_collision` | All goal IK solutions are in collision |
| `no_path_in_iris_coverage` | No connecting path exists within the current 2D IRIS slice coverage |
| `solver_failed` | The single GCS solve failed |
| `path_validation_failed` | The returned path failed validation |
| `unsupported_degenerate_ik` | Degenerate kinematics produce a continuous goal set that this discrete-goal entry point does not support |
| `invalid_input` | The input or checkpoint is invalid |

After failure, the planner does not automatically generate regions, change the
goal, relax constraints, or retry a solve. One native Drake GCS call still uses
convex relaxation and rounding internally to produce a discrete executable path.

A lack of connecting coverage does not prove that no path exists in the full
free space. Only slices with a 2D interior are retained; the original IRIS
coverage and boundary contacts remain subject to numerical precision.
IRIS verification and sampled path checks are not continuous collision-free
proofs. Robot self-collision policy is unchanged from the pre-failure workflow.

## Completed validation

- All 24 tests passed in Docker. Integration tests cover all three possible locked joints and verify selection of the shorter goal in one solve, the length objective, preservation of the locked angle, and input/output winding offsets for unrestricted angles.
- At the current pre-failure path's start: locking joint 1 or 2 reports an unreachable goal; locking joint 3 reports no connecting path within current coverage. None of these three cases called GCS. Reports are `results/planar3r/post_failure_joint1.json`, `results/planar3r/post_failure_joint2.json`, and `results/planar3r/post_failure_path.json`.
- The explicit `--q` example above is an artificially constructed near-goal failure state, not a waypoint sampled from the pre-failure path. With joint 3 locked, it reaches the goal using two 2D regions and one GCS call. Path length is approximately 0.05 rad, endpoint error is approximately `7.86e-15`, and sampled collision checks pass. The report is `results/planar3r/post_failure_success.json`.

Test commands:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 -m unittest discover -s tests -v
docker stop planar3r-iris-validation
```
