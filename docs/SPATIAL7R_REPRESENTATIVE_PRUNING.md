# Pre-model representative graph pruning for 7R

The existing 2,856-region scene has 869,670 directed nonempty-intersection
edges. Historical full-graph workers exceeded the 6 GiB RSS limit while
allocating the Drake graph. Filtering intersection dimension and compacting
constraints allowed construction, but the solve still exceeded the limit.

`--representative-pruning` now selects a directed subgraph before constructing
any Drake objects. It is opt-in; historical full-graph modes are unchanged.

Implementation:

1. Find a feasible Chebyshev center in each region using an LP.
2. Weight directed center connections by the source region's reliability cost
   weight times center distance; connect the real start and target representatives.
3. For each target separately, compute reverse Dijkstra distances and retain one
   shortest surrogate path. This avoids suppressing a more expensive target in
   the first-stage target comparison.
4. Sample simple paths subject to accumulated surrogate cost plus remaining
   Dijkstra distance being at most `alpha` times that target's surrogate optimum.
5. Retain the union of successful directed paths, remap regions and targets, and
   use the resulting graph for both planning stages. Original collision
   halfspaces, reliability weights, IK projection and path validation remain.

This follows the graph-pruning idea in Clark and Xie, TRO 2025, Section IV-C,
but uses centers of this project's region vertices, not their SLP-optimized
intersection representatives. It is not an exact reproduction. Center weights
are not lower bounds on continuous GCS costs. The reduced graph can exclude
the original optimal trajectory; reported gaps apply only to the reduced graph.
It currently rejects periodic edges with nonzero chart shifts.

Reproduce with the configured Drake container, from the repository root:

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_full_graph_supervisor --representative-pruning --pruning-alpha 1.1 --pruning-trials 100 --output-dir results/spatial7r/representative_pruned_alpha110
```

The existing `full_graph_60cm90deg_comparison/prepare.json` and `adjacency.npz`
are reused. Output goes to a separate directory, preserving historical runs.
The supervisor runs union-hull and per-target methods sequentially with the
original 200/300 rounding budgets and samples worker RSS every 0.2 seconds.
Peak RSS includes representative-graph construction and GCS solving; it is a
sampled worker peak, not total Docker/host memory.

On the current scene, alpha=1.1, 100 trials per target, seed=0 retains 212
regions and 1,403 directed edges, preserving paths to all 23 targets. See the
saved `comparison_summary.json`, method reports and logs for measured outcomes.

Measured runs (200 rounded paths / 300 trials, cached original adjacency):

| Method | Total seconds | Sampled peak worker RSS | Final weighted cost | Result |
| --- | ---: | ---: | ---: | --- |
| Union of target hulls | 74.92 | 3.84 GiB | 0.61633209 | Validated success |
| Per target, then fixed-goal stage | 494.77 | 4.10 GiB | 0.57668801 | Validated success, 23/23 first-stage targets |

Total time includes geometry reconstruction, representative pruning (about
12.8 seconds), both GCS stages and validation. It excludes the historical IRIS
generation and original adjacency-cache generation. No full-graph optimum was
computed, so these costs cannot establish the pruning loss against that optimum.

Region routes in these reports use reduced indices; original region indices
are recorded in `representative_pruning.retained_original_region_indices`.
The `graph_task_sha256` remains the original input fingerprint for comparing
methods, rather than a fingerprint of the selected subgraph.

Checks: `tests.test_gcs_representative_pruning`, `tests.test_gcs_quality`, and
`tests.test_two_stage_gcs` (38 tests), plus the actual 7R two-stage runs.
