# 4R union-hull representative pruning: alpha comparison

The unchanged ftwmax midpoint scene has 56 free regions, 1202 directed region
edges and 13 target sets. Each target receives 2000 representative path
attempts, seed 0. Both subsequent GCS stages use 2000 paths / 3000 trial caps,
with MOSEK primal solve form and the same original geometric constraints and costs.
No per-target GCS reference was rerun. Input graph/task and target fingerprints
match the saved per-target reference in both runs.

| Method | Retained regions | Directed region edges | Final weighted cost | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| Pruned union-hull, alpha 1.1 | 10 | 18 | 0.091080590359 | 5.118 |
| Pruned union-hull, alpha 1.5 | 32 | 223 | 0.078056384372 | 20.473 |
| Historical per-target 200/300 | 56 | 1202 | 0.078079831671 | 115.854 |

Alpha 1.5 returns a feasible cost about 0.0300% lower than the reference,
with approximately 5.66x lower single-run time. This is numerical experimental
evidence for this scene and seed, not repeated statistical timing evidence or
a global optimality certificate. Alpha 1.1 is about 16.65% higher cost; its
kept-region set excludes original region 7, used by the historical reference route.

Alpha 1.5 timing: geometry 2.287 s, representative pruning 2.887 s, stage one
10.015 s, stage two 5.233 s; total 20.473 s includes remaining validation overhead.
It produced 2000 unique GCS paths in each stage. Alpha 1.1 produced only 8 and
4 unique paths under the same caps (2992 / 2996 duplicate samples discarded).
Both retain paths to all 13 targets in the representative graph. Every final
path passes collision, graph membership, endpoint and weighted-cost checks.
Alpha 1.5 task residual is 1.65e-15.

The pruner is a center-based heuristic, not an admissible cost-bound method;
it can exclude better routes. Original region indices are retained in each
report's `representative_pruning.retained_original_region_indices`. Quality
bounds apply only to the pruned graph. A fallback feasibility LP handles a
thin target whose inscribed-radius LP fails, using the identical halfspaces.

Raw reports:

- `union_hull_pruned_alpha110_trials2000_paths2000_20261030.json`
- `union_hull_pruned_alpha150_trials2000_paths2000_20261030.json`
- Reference: `per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json`

Reproduce alpha 1.5 in the configured Drake container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --union-hulls-only --representative-pruning --pruning-alpha 1.5 --pruning-trials 2000 --max-rounded-paths 2000 --max-rounding-trials 3000 --stage2-max-rounded-paths 2000 --stage2-max-rounding-trials 3000 --output results/spatial4r/union_hull_pruned_alpha150_repeat.json
```
