# Intersection-vertex representative graphs and SLP preprocessing

This implements the structure and collective-cost objective of Clark & Xie,
TRO 2025, Section III-C and Eq. (29), specialized to this repository's constant
regional reliability densities. It is opt-in; the region-center method remains.

A completed 7R run with union-hull 1000/1500 is documented in
[the 7R report](../results/spatial7r/intersection_slp_facet_1000_1500_primal_20261030/README.md).
It uses the pre-existing intersection-dimension >=6 adjacency policy to fit
the intersection representative graph in memory; this differs from the full
nonempty-intersection graph and is stated explicitly in the report.

## What is reproduced

For every adjacent pair of regions i,j, create one representative vertex with
feasible set V_ij = Q_i intersect Q_j. Every pair of such vertices sharing a
region r is connected, with edge cost w_r times their Euclidean distance.
Both endpoints lie in convex Q_r, so this segment lies in Q_r. The original
halfspaces are used without geometric inflation, including flat intersections.

One point per vertex is optimized collectively:

    minimize sum_(u,v,r) w_r ||q_u - q_v||, subject to q_v in V_v.

The paper's spatially varying affine densities lead to nonconvex edge costs.
Our densities w_r = 0.01 + 1 - reliability_r are constant, so this particular
collective objective is convex. The implementation uses a smoothed norm
sqrt(||delta||^2 + 1e-12) for gradients and reports the original unsmoothed cost.
SLP solves sparse linear subproblems with the exact intersection constraints
and a box trust region. Initial radius is 0.15 radians; acceptance requires
positive actual smoothed decrease and actual/predicted ratio >=0.1. Radius
halves on rejection and grows by 1.5 (up to 1) for ratios >0.75.

For this first attempt the cap is 80 iterations. It stopped at that cap, not
at a convergence certificate. Collective unsmoothed cost fell from
4015.66137332 to 1723.64301750 (about 57.1%), with maximum halfspace violation
1.11e-16. No claim of an optimal representative graph is made.

The full 4R representative graph has 601 intersection vertices and 13,662
undirected within-region edges (27,324 directed equivalents). This topology
is different from the 56-region / 1,202-directed-edge graph used by the main GCS.
Geometry/weight/optimizer fingerprints protect a reusable offline NPZ cache.

## What remains different from the complete paper

- The final optimization still uses the existing region-vertex GCS. Sampled
  intersection routes are translated into directed region transitions, with
  consecutive duplicates collapsed and region loops shortcut through their
  convex region. The resulting transitions are checked against the input graph.
- The union of region transitions can recombine into additional paths; it is
  not an exact retention of all intersection-to-intersection turn constraints.
- There is no ReLU cost fitting, McCormick relaxation, or reproduction of the
  paper's complete nonconvex GCSGC formulation.
- SLP is our explicit trust-region implementation of the stated optimization
  objective; the paper's mention of SLP alone does not specify these local
  radius, smoothing and stopping parameters.

## 4R experiment

All cases use the same input geometry, 13 targets, seed 0, 2,000 representative
path attempts PER TARGET, and 2,000 paths / 3,000 trial caps in BOTH GCS stages.
Final trajectories use original region constraints and physical costs. Input
graph/task and hull fingerprints match the historical per-target reference.

| Representative method | alpha | Final region vertices / directed edges | Final feasible cost | Total seconds |
| --- | ---: | --- | ---: | ---: |
| Region centers (previous) | 1.1 | 10 / 18 | 0.0910805904 | 5.12 |
| Region centers (previous) | 1.5 | 32 / 223 | 0.0780563844 | 20.47 |
| Intersection centers, no SLP | 1.1 | 46 / 358 | 0.2125063080 | 37.42 |
| Intersection + SLP | 1.02 | 26 / 113 | unavailable: selected path failed validation | 10.64 |
| Intersection + SLP | 1.1 | 44 / 277 | 0.0780462636 | 27.58 |
| Intersection + SLP | 1.5 | 56 / 732 | 0.0910805900 | 49.64 |
| Historical per-target 200/300 | n/a | 56 / 1202 | 0.0780798317 | 115.85 |

The alpha=1.1 SLP run includes initial offline construction/optimization (1.312 s,
of which SLP is 1.283 s); the SLP alpha=1.02 and 1.5 cases reuse that cache.
The uncached alpha=1.1 breakdown is geometry 2.192 s,
representative construction/SLP/path pruning 9.212 s, GCS stage one 10.430 s,
stage two 5.698 s, and remaining validation overhead. Times are sequential
single runs, not repeated statistical timing evidence. The historical reference
was read, not rerun. No defaults were changed to favor the new method.

All successful final reports pass collision, endpoint, graph membership and
cost checks. For alpha=1.02 the relaxation succeeded and 702 convex restrictions
reported success, but the selected result failed the independent original-region
path check. A separate diagnostic rerun confirms this in
`intersection_slp_alpha102_validation_diagnostic.json`; the case is NOT counted
as a valid solution or evidence of physical infeasibility. Tolerances were not
loosened. Greater alpha did not monotonically improve the finite-rounding result.

## Files and reproduction

- Implementation: `scripts/gcs_intersection_slp.py`
- 4R integration: `scripts/spatial4R_two_stage_gcs_planner.py`
- Raw ablation reports: `results/spatial4r/intersection_slp_20261030/`
- Cold alpha=1.1 report: `results/spatial4r/union_hull_intersection_slp_alpha110_20261030.json`
- Optimized offline cache: `data/representative/spatial4r_intersection_slp_80.npz`
- Summary: `results/spatial4r/intersection_slp_20261030/summary.csv`

Run the alpha=1.1 variant in the configured Drake container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --union-hulls-only --representative-pruning --pruning-method intersection-slp --slp-cache data/representative/spatial4r_intersection_slp_80.npz --pruning-alpha 1.1 --pruning-trials 2000 --max-rounded-paths 2000 --max-rounding-trials 3000 --stage2-max-rounded-paths 2000 --stage2-max-rounding-trials 3000 --output results/spatial4r/intersection_slp_repeat.json
```

To reproduce the cold optimization with this command, choose a new cache path.
Existing caches are checked and never silently reused for different inputs.
Five dedicated tests cover intersection membership and objective reduction,
within-region edge construction, path translation, direct same-region goals,
loop erasure, and cache reuse/invalidation. Together with pruning, spatial and
two-stage regression checks, 57 tests pass.
