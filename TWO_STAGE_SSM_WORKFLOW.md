# Adaptive SSM targets and two-stage GCS

Current user-selected defaults: geometric epsilon 0.025, 300 SSM samples,
and 200 paths / 300 trials in both stages. The stage-one route remains an
additional stage-two candidate. These settings correspond to the full result
in `two_stage_all_paths200_trials300.json`. The epsilon 0.0125 experiment is
retained for comparison. Historical default values below describe earlier runs.

`planar3R_two_stage_gcs_planner.py` implements the latest idea in
`PROJECT_CONVERSATION_HANDOFF.txt`. The original sampled endpoint planner remains
available. Both use the existing IRIS cache and the same robot geometry.

## Run

In the configured Drake container, run:

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_two_stage_gcs_planner.py --start-path iris_path.json --geometric-epsilon 0.025 --ssm-samples 300
docker exec planar3r-iris-validation /usr/bin/python3 -m unittest discover -s tests -v
docker stop planar3r-iris-validation
```

The fixed start defaults to waypoint 0 of `iris_path.json`. Use `--q a b c` to
supply three angles in radians, or `--start-path FILE --waypoint-index N`.
The output defaults to `two_stage_path.json`; the existing result is preserved.
Workspace goal flags are `--gx` and `--gy`. The workflow neither generates new
IRIS regions nor searches across epsilon values.

The default reliability source is `goal`. Existing code constructs a different
cost map for each traced SSM branch. This planner uses one explicit cost map,
selected by zero-based `--reliability-branch` (default 0). For N effective target
intersections, a successful run performs N independent first-stage GCS calls
and one final fixed-configuration GCS call. All targets use the same cost map.
`--reliability-source start|goal|none`,
`--min-reliability`, and `--reliability-epsilon` retain their existing meanings.
The cost is weighted Euclidean joint-space length, with region weight
`reliability_epsilon + 1 - reliability`.

## Solver quality and exact mode

The ordinary solver now follows the official relaxation-and-rounding workflow:
solve the convex relaxation ONCE, save its lower bound, sample distinct paths
from that solution, and solve their convex restrictions. It does not rerun the
full relaxation for rounding. The selected stage-one route remains an extra
candidate in the final solve, and quality is recomputed using the best returned
feasible cost after this comparison.

CLI controls (independent of geometric epsilon):

```text
--max-rounded-paths 200
--max-rounding-trials 300
--stage2-max-rounded-paths 200
--stage2-max-rounding-trials 300
--rounding-seed 0
--relative-gap-tolerance 0.01
```

The first two budgets apply to each first-stage intersection (and the optional
sampled baseline); the stage-two budgets apply only to the final fixed-goal
solve. Current defaults are stage one 200/300 and stage two 200/300. Settings for
both stages are saved separately. The extra known-route restriction remains
outside the random candidate count.

Every solved candidate records `quality`. `stage1_quality` uses the minimum
lower bound across ALL target intersections and the best feasible candidate
cost. If a target lacks a bound, the aggregate bound is unavailable rather
than silently excluding that target. This bound concerns the sampled target
sets in the graph, not the continuous SSM. `stage2_quality` concerns only the
selected fixed endpoint in that graph. Both include lower/upper bounds,
absolute gap, relative gap `(U-L)/L`, and `within_tolerance`. The default quality
test is `U-L <= 1e-6 + 0.01*L`. Missing, nonfinite, inconsistent, and zero bounds
are handled explicitly. These are numerical solver bounds, subject to solver
tolerances, and do not certify full physical-space coverage.

Path validation success and quality acceptance are separate: a collision-checked
feasible path can be returned with `quality.status=gap_exceeds_tolerance`.
The planner does not automatically repeat GCS or sweep epsilon to meet a gap.
Changing budgets or tolerance requires an explicit parameter choice.

`--no-rounding` requests the mixed-integer GCS formulation. A conic MIP solver
capability check runs before expensive preprocessing. Before license activation,
the container had Clarabel for continuous solves, disabled Mosek, and unavailable
Gurobi. That historical check returned `mixed_integer_solver_unavailable` with
zero GCS calls. MOSEK has now been activated as described below. A successful MIP
incumbent is not reused as a lower bound without a reported bound; if that bound
is unavailable, quality remains `bound_unavailable` and no certificate is claimed.

## MOSEK activation and bounded mixed-integer runs

The user supplied `mosek.lic` in the project directory on 2026-10-19. `*.lic` is
ignored by Git. Do not print license contents or commit a license. No new solver
package was needed: the Drake image already includes MOSEK support.

Start the existing container, copy the license, and pass its path to each new
Docker process using MOSEK:

```powershell
docker start planar3r-iris-validation
docker cp mosek.lic planar3r-iris-validation:/tmp/mosek.lic
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 planar3R_verify_mosek.py
```

`MOSEKLM_LICENSE_FILE` on a `docker exec` command applies only to that process;
it does not change the container's saved environment. Repeat `-e` when launching
another solve. Copy the license again if the container or `/tmp` is recreated.

Mixed-integer controls are separate from the geometric and reporting tolerances:

```text
--no-rounding
--mip-time-limit 60
--mip-relative-gap 0.01
```

The time limit applies to EACH mixed-integer optimizer call; setup and path
validation add time. The stopping gap is a solver target, not a claim that it
was met. Inspect the solver termination and `mip_details.solution_status`.
The planner never treats a merely feasible incumbent as a proof of optimality.

The license and a tiny mixed-integer cone problem were verified successfully
in `mosek_environment_check.json` (known optimal cost 0.3, binary value 0).
The saved real fixed-endpoint case can be checked without rerunning all targets:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u planar3R_diagnose_stage2.py --mip-only --output stage2_mosek_mip_review.json
```

The first 60-second mixed-integer test returned an integer-feasible solution
with cost 0.8346454952 in about 62.89 seconds of solve/setup time (optimizer
time 60.88 seconds). MOSEK solution status 2 is primal feasible, not integer
optimal. It passed 653 collision samples and fixed-goal equality. The 1% target
was not certified. This raw MIP test did not seed the known cheaper route and
does not replace the saved feasible 0.3850930626 result. Routine two-stage
planning still compares the preserved stage-one region route with its native
final solve before choosing the final output. 52 tests passed after this setup.

## Algorithm

1. Check scene/cache compatibility and fixed-start collision freedom. Keep all
   reliability-partitioned free regions connected to the fixed start.
2. Trace ordered goal SSM branches with the existing sampler, refine FK drift,
   split failed refinement portions and periodic chart seams, then subsample.
   The sample budget is approximately 300 per traced branch; preserving chart
   piece endpoints can increase the total slightly.
3. Recursively split each sampled portion at its middle index until twice its
   maximum distance to the endpoint chord is at most `geometric_epsilon`.
   Adjacent groups share an endpoint. The first/last group also preserves the
   trace endpoints; no extra edge closes a trace or crosses a periodic seam.
4. Construct each sample convex hull in its affine subspace, without artificial
   thickness. Intersect it with the connected free regions using halfspaces.
   Feasibility LPs preserve point, line, and plane intersections.
5. Solve a separate GCS from the original fixed start to EACH nonempty target
   intersection, sequentially, then select the lowest returned feasible cost.
   Each solve has one direct target vertex and no artificial sink. Its incoming
   motion uses its associated free region's usual weight. Record each target's
   cost, endpoint, solve time, and failure status. A failed target is not retried;
   other prescribed candidate solves still run. If all fail, stop the workflow.
6. Numerically minimize distance from the selected terminal point to continuous
   FK equality, constrained by the selected free region and the local samples'
   coordinate box. Nearby samples seed SLSQP; valid exact samples are also
   candidates. Reject colliding or inaccurate results. This is numerical local
   projection; the box restricts branch association heuristically and does not
   certify it near singularities or closely approaching branches. It can exclude
   unsampled extrema. No nearest-sample-only endpoint claim is made.
7. Create a fresh GCS with the original start, the projected fixed goal, **all**
   retained free regions, and their original adjacencies. The full route is free
   to change. One fixed-goal vertex accepts motion from every region containing
   that configuration. Preserve the selected first-stage region sequence as an
   additional convex-restriction candidate, with all waypoint positions optimized
   again for the fixed goal. Compare it against the native rounded result and
   return the lower-cost solution. This does not freeze the full graph search or
   append the projection segment to the returned path.
8. Check start equality, fixed-goal equality, goal FK, and sampled collisions
   using the original three-link geometry. Recompute the feasible path's cost
   from its actual region segments; also retain the solver's reported cost.

Inputs outside the canonical chart are planned using a consistent winding
offset and restored in the result. Segments are never individually wrapped.
This preserves the fixed physical configuration, but does not add routes across
opposite boundaries of the cached `[-pi,pi]^3` chart.

Failures stop without changing the task, regenerating regions, or retrying GCS.
Reports distinguish missing IRIS coverage, sampling failure, projection failure,
solver failure, unavailable Drake, and failed path validation. A coverage failure
does not establish physical impossibility.

## What epsilon controls

For a sampled group, let `d` be the maximum sample distance to its endpoint
chord. Convexity puts the sample hull within distance `d` of that chord.
The continuous *sampled polyline* has an axial projection spanning the chord,
so every chord point is within `d` of some polyline point. Thus `2*d` bounds
the distance from any point in the sample hull to the sampled polyline.
The implementation controls this quantity, including interior hull points,
rather than using an unspecified minimum boundary distance. This conservative
metric can split backtracking portions unnecessarily.

This is **not** a continuous SSM coverage or distance certificate. A two-sample
chord has bound zero relative to its own polyline even if the actual curve
bends away. Smaller epsilon cannot recover missing samples. Exact sample hulls
can miss the continuous curve, so first-stage costs are not reported as lower
bounds on the full continuous task. Final cost monotonicity, global optimality,
and continuous collision freedom are not asserted. The existing reliability
bounding-box approximation and finite chart coverage still apply.

## Results and comparison

JSON includes envelope and target counts, actual sample count, region/edge
counts, preprocessing time, both solve times, projection time/distance, selected
local target, estimated goal, final FK error, collision samples, and actual cost.
`gcs_solve_calls` counts the two-stage pipeline only; native GCS rounding may
internally solve several convex restrictions.
`stage1_candidates` records the individual target solves; `stage1_gcs_solve_calls`
and `stage2_gcs_solve_calls` distinguish N candidate calls from the final call.
Progress is printed before and after each target solve. The earlier combined
multi-target implementation is superseded; its saved two-call results remain
historical comparisons, not timings of this sequential method.

`stage1_region_route` preserves the selected route for the final solver.
`stage2_native_cost`, `stage2_additional_route_results`, `stage2_solution_source`,
and `stage2_region_route` make final route selection auditable. The extra convex
restriction is counted separately in `stage2_additional_convex_restriction_calls`;
it does not repeat the full GCS relaxation or change the N+1 top-level count.

Every run that reaches a valid projection also reports
`projection_connection_baseline`: the first-stage path followed by the straight
segment to the projected SSM goal. Its cost is `stage1_cost + connector_weight *
connector_length`, using the selected region's actual reliability weight. The
report includes the connector weight/reliability, length, cost, complete path,
same-region containment, FK/start errors, sampled collision checks, and timing.
This comparison adds no GCS calls and never replaces the second-stage output.
If it passes validation, `stage2_minus_projection_connection_cost` compares the
replanned cost against it; a positive value means the second-stage returned
solution is more expensive. This is not a global optimality gap. An invalid
comparison is marked unsuccessful and is not treated as a feasible upper bound.

`--compare-sampled` optionally runs a **separate** baseline GCS after successful
final validation. It uses the same fixed start, scene, cost map, region graph,
and refined samples, but only sampled point targets. It places all those points
in one graph, rather than repeating the legacy planner's per-pair solves, and
reports its own time, cost, validation status, and solve count under
`sampled_baseline`. This can be more expensive than the legacy default of one
candidate pair; it is an endpoint-representation comparison, not an identical
legacy runtime comparison. It does not perform an epsilon sweep.

On 2026-10-18, local Windows preprocessing of the default saved start and scene
produced 27 envelopes, 16 target intersections, and 43 connected free regions
with 824 directed adjacency edges at geometric epsilon 0.05. Geometry took
approximately 4.9 seconds in that run. No robotics Drake or Docker executable
was available, so neither real GCS stage nor a real baseline was run here.
Tests cover geometry, continuous projection, stop conditions, and the two-stage
control flow; a dedicated Drake integration test checks target motion costs and
sink handling when robotics Drake is available.

On 2026-10-19, the corrected per-intersection workflow was run in the Drake
container with 300 samples and geometric epsilon 0.025. It produced 40 hulls,
20 target intersections, and 21 GCS calls. All 20 candidate solves succeeded.
Stage one took 568.18 seconds; the final fixed-goal solve took 27.41 seconds;
the total was 599.07 seconds. The best candidate cost was 0.5813465168.
Its projected-goal connection baseline cost was 0.5813467627 and passed 656
collision samples. The final GCS returned cost 0.8667624205 and passed 683
collision samples, with goal FK error 9.07e-13. The higher final cost requires
separate investigation of route selection/rounding; no optimality is claimed.
Full results: `two_stage_path_per_intersection.json`. All 43 tests passed.

Subsequent diagnosis reproduced the 0.8667624205 native result on the same fixed
endpoints and cost model. Native rounding compares at most 10 random paths and
missed a feasible low-cost region sequence. Recovering the simple sequence
v1 -> v3 -> v11 -> v28 from the saved stage-one waypoints and solving its convex
restriction gave cost 0.3850930626, matching independently recomputed cost and
passing sampled collision checks. The unrounded relaxation cost was 0.3618987263,
a lower bound for this fixed-endpoint graph model, not the entire SSM task or
uncovered physical space. No global optimality is certified. The planner now
retains the selected stage-one route as an additional optimized candidate to
avoid discarding known feasible routes. All 44 tests passed after this repair.
`planar3R_diagnose_stage2.py` reproduces the diagnosis for the saved default case;
its JSON output is `stage2_diagnosis.json`.

With the new routine the fixed-endpoint replay is saved separately as
`stage2_quality_review.json`: lower bound 0.3618987263, feasible upper bound
0.3850930626, relative gap 0.0640906824 (6.4091%), above the requested 1%.
The old native-only result has gap about 139.50%. Exact-mode environment check:
`two_stage_exact_environment_check.json`. The diagnostic utility now uses the
ordinary solve's stored lower bound instead of solving a duplicate relaxation;
its current default output is `stage2_quality_review.json`.
The completed quality-reporting change passed all 52 tests, including real
Drake relaxation/restriction integration, updated-bound candidate selection,
zero/missing/inconsistent bounds, and unavailable-MIP-solver handling.

## Stage-two budget comparison: 10/100 versus 100/100

The saved first-stage 20-intersection result (10 paths, 100 trials per solve),
300 samples, epsilon 0.025, and projected fixed goal were reused unchanged.
The final stage was rerun with Clarabel and seed 0, requesting 100 distinct paths
in at most 100 trials. Drake found 99 distinct routes and one duplicate. The
best random candidate cost improved from 0.8667624205 to 0.5239423200. The extra
known-route restriction still won at 0.3850930626, so final gap remained 6.4091%
against the unchanged lower bound 0.3618987263. Final-stage solve time was
26.53 seconds; 652 collision samples passed. The first stage was not rerun.
Result: `stage2_paths100_trials100.json`. All 52 tests passed, including an
assertion that candidate-stage budgets stay at 10/100 while the final stage
receives 100/100. To replay this comparison:

```powershell
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -u planar3R_diagnose_stage2.py --stage2-only --max-rounded-paths 100 --max-rounding-trials 100 --output stage2_paths100_trials100.json
```

## Complete rerun with both stages at 100/100

At the user's request, all 20 first-stage intersections and the final stage
were rerun with 100 distinct-path budget / 100 sampling trials, using Clarabel,
seed 0, 300 SSM samples, epsilon 0.025, and the same saved start/cost map. This
run did not reuse first-stage results. Command:

```powershell
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -u planar3R_two_stage_gcs_planner.py --start-path iris_path.json --geometric-epsilon 0.025 --ssm-samples 300 --max-rounded-paths 100 --max-rounding-trials 100 --stage2-max-rounded-paths 100 --stage2-max-rounding-trials 100 --output two_stage_all_paths100_trials100.json
```

All 20 solves succeeded (95-100 unique routes each). The lowest returned
first-stage cost was 0.3816510474 at target index 19 (the 20th intersection),
not target index 0 from the earlier 10/100 run. First-stage aggregate lower
bound was 0.3584417133 and gap 6.4751%; the ranking is still not certified.
The projected goal was [0.6087380070, -0.6279264205, 1.9159107976], so the
fixed endpoint differs from previous replay experiments. Projection distance
was 0.0078651106 rad; the validated connection baseline cost was 0.3817296985.

The final stage found 99 unique random routes, best random cost 0.3991851967.
Preserving first-stage route [1,3,28] as an extra optimized candidate reduced
the returned cost to 0.3817177111. This is reoptimization, not direct stitching.
Its fixed-goal lower bound was 0.3585083602, giving gap 6.4739%; 1% was not met.
The path passed 618 collision samples; endpoint FK error was 1.76e-13.
First-stage time 559.70 s, final-stage time 26.44 s, total 589.67 s (9m50s).
The earlier full 10/100 run took 599.07 s; these are single measurements, not
evidence that increasing candidate count makes solving faster. All parameters,
per-target costs/bounds/counts and final candidates are in the output JSON.

### Second-stage replay: 150 paths / 200 trials

`stage2_paths150_trials200.json` reuses the first-stage result from
`two_stage_all_paths100_trials100.json`; no first-stage solve was repeated.
The fixed endpoints and extra candidate route [1,3,28] remain identical.
With seed 0 and Clarabel, sampling found 150 unique routes (5 duplicates
discarded); all 150 restrictions succeeded. The native rounded result now
finds route [1,3,28] itself, with cost 0.3817177111, matching the extra route.
Previously native rounding at 100/100 returned 0.3991851967, while the extra
route already gave the same final cost 0.3817177111. Thus the final cost is
unchanged. Solve time was 27.66 s versus 26.44 s previously. The fixed-goal
lower bound remains 0.3585083602 and gap 6.4739%, exceeding the 1% tolerance.
Validation passed 618 collision samples and endpoint FK error was 1.76e-13.
The replay utility accepts `--input` to select the saved first-stage result.

### Second-stage replay: 200 paths / 300 trials

`stage2_paths200_trials300.json` reuses the same 100/100 first-stage result,
fixed endpoints, extra route, seed 0, and Clarabel solver. Sampling found
200 unique routes and discarded 6 duplicates; all 200 restrictions succeeded.
Native and final costs remain 0.3817177111, with route [1,3,28]. Solve time
was 27.46 s versus 27.66 s at 150/200 and 26.44 s at 100/100. These single
measurements show little runtime difference, not a speedup guarantee.
The lower bound remains 0.3585083602 and gap 6.4739%; 1% is still not met.
The final path passed 618 collision samples; endpoint FK error was 1.76e-13.

### Second-stage replay: 1000 paths / 1100 trials

`stage2_paths1000_trials1100.json` reuses the same first-stage 100/100 result
and fixed endpoints, with seed 0 and Clarabel. Sampling found 1000 unique
routes and discarded 49 duplicates; all 1000 restrictions succeeded.
Solve time was 29.66 s, compared with 27.46 s for 200/300: an increase of
2.19 s (7.99%) in these individual measurements. Native cost was
0.381717711086879; the retained first-stage route gave 0.381717711086575.
This difference is numerical only; the best final cost did not improve.
The lower bound and gap remain 0.3585083602 and 6.4739%. The selected path
passed 618 collision samples, with endpoint FK error 1.76e-13.

### Full rerun: both stages 200 paths / 300 trials

Both public planner defaults and CLI defaults now use 200/300 for each stage.
`two_stage_all_paths200_trials300.json` reruns the complete example, retaining
300 SSM samples, geometric epsilon 0.025, seed 0, and the extra stage-one route.
All 20 first-stage calls succeeded with 200 unique rounded routes each.
Compared with the full 100/100 run, 13 of 20 returned first-stage costs improved
by more than 1e-8, but the winner remained target index 19 (20th intersection).
Its cost 0.381651047429187 and projected goal configuration were unchanged.
The second stage found 200 routes, with native and final cost 0.381717711086575;
the extra first-stage route matched that cost. Fixed-goal gap remains 6.4739%.
The path passed collision and endpoint validation.

First-stage time was 586.10 s, second-stage time 26.89 s, total 616.53 s.
The previous full 100/100 run took 559.70 s, 26.44 s, and 589.67 s respectively:
total increase 26.86 s (4.55%). The earlier 150/200 experiment was a second-stage
replay, not a full two-stage rerun: it reused the old first stage, took 27.66 s
for stage two, and returned the same goal and cost. No directly measured full
150/200 runtime exists. These are single runs, so small timing differences
should not be interpreted as guaranteed speed improvements.
All 22 tests in test_two_stage_gcs.py passed after updating the default budgets.

### Geometry only: epsilon halved to 0.0125

`two_stage_geometry_epsilon_00125.json` keeps the same 300 SSM samples and
scene, but halves geometric epsilon from 0.025 to 0.0125. Local SSM hulls
increase from 40 to 55; effective target intersections increase from 20 to 28.
The 6 cached IRIS regions and 43 reliability-partitioned free regions do not
change (824 directed graph edges). Total preprocessing time was 3.72 s.
Neither planning stage ran: all GCS solve counts are zero. The new CLI flag
`--geometry-only` returns counts immediately after hull/intersection construction.
Use `--geometric-epsilon 0.0125 --ssm-samples 300 --geometry-only` to reproduce.
The geometric epsilon CLI default remains 0.05; 0.0125 is this experiment's
explicit setting. No path-cost comparison has been computed at 0.0125.

### Full run: epsilon 0.0125, both stages 100 paths / 150 trials

The subsequent full run is saved in two_stage_epsilon00125_paths100_trials150.json.
It keeps 300 SSM samples, the same scene and seed 0, and uses Clarabel.
All 28 stage-one targets produced 100 unique candidates; stage two also produced
100. There were 29 full GCS calls plus the additional stage-one route restriction.
Target index 22 (23rd intersection), region 11, won at cost 0.3818634313717322.
Its projected goal was [0.3757867697535929,-0.15490638851605437,1.7679057112502206],
with projection distance 0.0008716871409386708. The stage-one route [1,3,11]
reoptimized for the fixed goal gave final cost 0.3818714856801985, beating
native stage-two cost 0.5215069012896434. The projection-connection comparison
baseline was 0.3818721482431416. Fixed-goal gap is 6.4705%; 1% remains unmet.
The final path passed 619 collision samples, with endpoint FK error 1.88e-13.

Stage one took 832.20 s, stage two 30.92 s, total 866.82 s (14m26.82s).
Compared with epsilon 0.025 and both-stage 200/300, final cost increased by
0.0001537746 (0.0403%) and total runtime increased by 250.29 s (40.60%).
The selected goal changed, and both epsilon and solver budgets changed;
this is not an isolated epsilon experiment or evidence of monotonic quality.
The extra first-stage route again prevented a worse random-rounding result.
This run used explicit flags; code defaults remain 200/300 and CLI geometric
epsilon remains 0.05. No new paths were solved during documentation updates.
