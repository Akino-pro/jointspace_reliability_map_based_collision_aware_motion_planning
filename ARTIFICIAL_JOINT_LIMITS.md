# Artificial pre-failure joint limits

Artificial limits are optional software constraints on pre-failure motion.
They are independent of the physical angular chart `[-pi,pi]^3` and are released
after a joint locks. The locked joint then remains fixed, while the two healthy
joints recover their full physical ranges.

## Execution

Provide three lower and three upper angles in radians. Every interval must have
positive width and lie inside `[-pi,pi]`. Omitting both options preserves the
full-range behavior. Providing just one option is invalid.

The tested box `[-3,3]^3` is slightly smaller than `[-pi,pi]^3` (approximately
plus/minus 171.9 degrees rather than 180 degrees):

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_gcs_planner.py --joint-lower -3 -3 -3 --joint-upper 3 3 3 --output iris_path_artificial_limits.json
```

An asymmetric box is also supported; specify each joint's bounds separately.
The resulting joint-space box is convex, although its collision-free portion
and Cartesian workspace generally are not.

## Reachability and early termination

A Cartesian goal is outside the limited workspace only if no joint configuration
inside the artificial box can reach it. Cartesian coordinates are not compared
directly with joint-angle bounds.

The planner checks the goal before computing SSMs or constructing GCS. Bounded
IK establishes reachability when a valid solution is found. Conservative
Cartesian interval bounds, with subdivision when needed, can establish
unreachability. An exhausted finite search reports `reachability_unresolved`
rather than falsely asserting that no IK solution exists. Planning stops in
either failure case and writes an English JSON report.

- `goal_unreachable`: the goal is outside the workspace permitted by the limits.
- `start_unreachable`: the start position has no configuration within the limits.
- `reachability_unresolved`: the numerical/interval preflight cannot resolve reachability within its search budget.

These preflight failures have `gcs_solve_calls: 0` and exit code 1.

Test an unreachable goal using the same box:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_gcs_planner.py --joint-lower -3 -3 -3 --joint-upper 3 3 3 --gx -3 --gy 0 --output artificial_limits_unreachable.json
```

The position `(-3,0)` is reachable in the full physical chart only when all links
align to the left: joint 1 must be at `+pi` or `-pi`, with joints 2 and 3 at zero.
It is therefore excluded by `[-3,3]^3`. The tested rejection took approximately
0.0002 seconds within the planner, with no SSM tracing or GCS solve.

## Region and path constraints

The original full-range IRIS checkpoint remains unchanged. Pre-failure planning
intersects each IRIS polytope with the artificial box, then intersects those
regions with the existing reliability partitions. All resulting cells remain
convex. IK candidates must lie inside the box, and bounded-IK witnesses supplement
sparse SSM sampling when necessary.

The final path is checked against the limits. Straight segments between valid
waypoints stay inside the box by convexity. Reliability still uses the full-range
target SSM because artificial limits are released after failure; the hypothetical
post-failure recovery configurations must not be limited to the pre-failure box.

The saved path records `artificial_joint_limits` and `gcs_solve_calls`. Use it as
the source of a locked-joint failure simulation:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_post_failure_planner.py --failed-joint 3 --path iris_path_artificial_limits.json --waypoint-index 0
```

Post-failure planning reads the original full-range IRIS checkpoint, ignores
the pre-failure artificial box, and reports `artificial_limits_released: true`.
It does not release the locked joint or expand beyond physical limits. Lack of
connecting IRIS slice coverage still stops planning without adding regions.

## Validation

- The limited default-scene run completed using the original six IRIS regions, 82 reliability intersections, and 43 solve regions. Its result is `iris_path_artificial_limits.json`; all path angles satisfy `[-3,3]^3` and collision/endpoint checks passed. Total time was approximately 39.4 seconds in this single run.
- The outside-workspace test above produced `goal_unreachable` without calling GCS; its report is `artificial_limits_unreachable.json`.
- The full 30-test suite includes an integration case in which the post-failure goal is outside a smaller artificial box, but becomes reachable after its limits are released.
- Original full-range results in `iris_path.json` and the IRIS checkpoint are preserved.

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 -m unittest discover -s tests -v
docker stop planar3r-iris-validation
```
