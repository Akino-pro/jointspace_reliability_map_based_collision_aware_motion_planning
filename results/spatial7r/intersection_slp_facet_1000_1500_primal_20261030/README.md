# 7R intersection + SLP pruning, union-hull 1000/1500

Validated success on the existing 60 cm / base-Z 90-degree scene. Both GCS
stages use 1000 paths / 1500 trial caps, seed 0. Representative path pruning
uses alpha 1.1, 2000 attempts per target, and an offline 80-iteration SLP graph.
Only union-hull ran; the per-target method was not rerun.

## Input adjacency and size limitation

The original nonempty-intersection graph has 2856 regions, 869670 directed
region edges and 23 targets. Its intersection-vertex representation would
have 434835 vertices and 184684472 undirected within-region edges. This exceeds
the bounded-memory implementation's capacity and was not materialized.

This experiment instead uses the existing adjacency policy requiring intersection
dimension >=6: all 2856 input regions and 23 targets remain, with 38094 directed
region edges. The intersection representative graph has 19047 vertices and
277236 undirected within-region edges. Lower-dimensional contacts are excluded;
this can change the admissible paths and optimum, so comparison with the old
full-nonempty-adjacency center runs is not a matched benchmark.

Size counts were computed before allocation and saved in
`results/spatial7r/intersection_graph_size_20261030.json`. A capacity guard now
rejects graphs over 50000 intersection vertices or 2000000 representative edges
before allocating all intersection sets/edges; it never silently discards edges.

## Offline SLP

The collective representative edge cost fell from 105583.43818623 to
29148.90774041. SLP stopped at its cap of 80 iterations, without a convergence
certificate. Maximum original-halfspace violation was 1.08e-8. Initial offline
representative construction/SLP took 127.456 s (SLP itself 127.022 s).
The optimized graph is cached at
`data/representative/spatial7r_facet_intersection_slp_80.npz` with geometry,
weight and optimizer-setting checks.

This reproduces the intersection representative/SLP preprocessing idea for our
piecewise constant reliability density. Final sampled routes are converted
back to the existing region-vertex GCS; it is not the complete paper GCSGC.
See `docs/INTERSECTION_SLP_PREPROCESSING.md` for smoothing, trust-region rules,
path conversion and limitations.

## Completed run (cached offline graph)

| Quantity | Value |
| --- | ---: |
| Retained regions / directed region edges | 198 / 852 |
| Targets retained | 23 |
| Representative graph load + query pruning | 16.705 s |
| Geometry preprocessing (cached IRIS/adjacency) | 17.165 s |
| GCS stage one | 34.143 s |
| GCS stage two | 23.351 s |
| Total worker task time | 91.544 s |
| Sampled peak worker RSS | 2.84 GiB |
| Final weighted feasible cost | 0.597770631497 |
| Joint SE(3) task residual | 0.0 (reported floating-point result) |

Each stage generated 1000 unique paths. Collision, original-region path
membership and recomputed weighted-cost checks all passed. The 91.544 s total
includes preprocessing, representative query pruning, both GCS stages and
validation; it excludes the already cached offline SLP construction and the
failed initial solver attempt. Do not interpret it as a successful cold-run
total. Historical IRIS and adjacency generation are also excluded.

The initial cold run used MOSEK automatic solve form and failed at the first
relaxation with `kSolverSpecificError`, rescode 0, solution_status 0. Its full
report/log and RSS are preserved in
`../intersection_slp_facet_1000_1500_20261030/`. The successful retry used MOSEK
primal form, retaining the same graph, costs, targets, alpha, seed and tolerances.
Redundant constraints were compacted using the existing routine. Full rounding
result history is not retained; every sampled candidate is still optimized and
compared and the best result retained. Both settings are explicit in reports.

Reported costs are feasible upper bounds for this pruned adjacency model, not
certified global continuous-goal optima. Timings are sequential single runs,
not repeated statistical timing evidence. 29 intersection/pruning/7R tests pass.

## Reproduction

From the workspace root, using the configured Drake container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_full_graph_supervisor --methods union_hulls --facet-adjacency --compact-constraints --representative-pruning --pruning-method intersection-slp --pruning-alpha 1.1 --pruning-trials 2000 --slp-iterations 80 --slp-cache data/representative/spatial7r_facet_intersection_slp_80.npz --max-rounded-paths 1000 --max-rounding-trials 1500 --discard-rounding-history --mosek-solve-form primal --output-dir results/spatial7r/intersection_slp_facet_1000_1500_repeat
```

`union_hulls.json` contains the full trajectory, solver diagnostics and pruning
statistics. `comparison_summary.json` contains supervisor timing and peak RSS.
`union_hulls.log` records phase progress. Input graph and target fingerprints
are checked against the existing facet-adjacency preparation report.
