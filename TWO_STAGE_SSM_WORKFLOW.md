# Adaptive SSM targets and two-stage GCS

`planar3R_two_stage_gcs_planner.py` implements the latest idea in
`PROJECT_CONVERSATION_HANDOFF.txt`. The original sampled endpoint planner remains
available. Both use the existing IRIS cache and the same robot geometry.

## Run

In the configured Drake container, run:

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_two_stage_gcs_planner.py --start-path iris_path.json --geometric-epsilon 0.05 --ssm-samples 300
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
selected by zero-based `--reliability-branch` (default 0), so a successful run
performs exactly two top-level GCS calls. All sampled goal branches compete
within that selected map. `--reliability-source start|goal|none`,
`--min-reliability`, and `--reliability-epsilon` retain their existing meanings.
The cost is weighted Euclidean joint-space length, with region weight
`reliability_epsilon + 1 - reliability`.

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
5. Solve one GCS with all nonempty target intersections. Each target accepts
   motion only from its associated free region. Its incoming motion has that
   region's usual weight. A target can only exit to the artificial zero-cost
   sink. The sink waypoint is removed from physical paths.
6. Numerically minimize distance from the selected terminal point to continuous
   FK equality, constrained by the selected free region and the local samples'
   coordinate box. Nearby samples seed SLSQP; valid exact samples are also
   candidates. Reject colliding or inaccurate results. This is numerical local
   projection; the box restricts branch association heuristically and does not
   certify it near singularities or closely approaching branches. It can exclude
   unsampled extrema. No nearest-sample-only endpoint claim is made.
7. Create a fresh GCS with the original start, the projected fixed goal, **all**
   retained free regions, and their original adjacencies. The full route is free
   to change. The returned motion is exclusively this second-stage path.
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
