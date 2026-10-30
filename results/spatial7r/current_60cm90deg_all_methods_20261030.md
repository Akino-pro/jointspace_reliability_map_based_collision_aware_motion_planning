# Current 7R example: all recorded methods

Task: 60 cm translation and +90 degrees about base Z. Existing input partition:
2856 reliable collision-free regions, one discovered SSM branch, 54 samples,
23 effective hull targets. This summary excludes the older 30 cm / 20 degree
example, different scaled/local scenes, and post-failure planning tasks.

## Validated successful runs

| Preprocessing / planner | GCS paths/trials per call, both stages | Final regions / directed edges | Final weighted cost | Task time | Sampled peak worker RSS |
| --- | --- | --- | ---: | ---: | ---: |
| Region-center pruning + union-hull | 200/300 | 212 / 1403 | 0.616332092695 | 74.920 s | 3.84 GiB |
| Same region-center pruning + per-target | 200/300 PER TARGET in stage one | 212 / 1403 | 0.576688012520 | 494.766 s | 4.10 GiB |
| Intersection + SLP pruning + union-hull, MOSEK primal | 1000/1500 | 198 / 852 | 0.597770631497 | 91.544 s with offline cache | 2.84 GiB |
| Center pruning, 1000 attempts/target + union-hull, compact primal/1-thread | 1000/1500 | 314 / 3188 | 0.580011597931 | 621.319 s | 6.84 GiB |

All four final paths pass collision, goal task, original-region membership and
weighted-cost validation. Per-target 200/300 assigns the full first-stage cap
to EACH of the 23 targets; union-hull assigns its cap to one joint call.

Center pruning starts from all 869670 nonempty-intersection directed edges,
alpha 1.1 and 100 representative path attempts per target. Both center-method
runs with 100 attempts share the same selected graph. The two methods were measured sequentially.

The later 1000-attempt center run uses the same full input graph and targets,
but retains 314 regions / 3188 edges. It needed a 7 GiB guard, compact
constraints, MOSEK primal form with one internal thread, and release of unused
preprocessing graph references. Three earlier 6 GiB attempts failed. See
[its detailed report](center_alpha110_search1000_gcs1000_1500_rss7_released_20261030/README.md).
Its 621.319 s excludes those failed attempts; solver settings differ from the
earlier timings and should not be treated as a matched speed comparison.

Intersection/SLP starts from the existing dimension>=6 adjacency policy, with
38094 directed edges over the same input regions/targets; low-dimensional
contacts are excluded. Its representative graph has 19047 intersection vertices
and 277236 undirected edges. Pruning uses alpha 1.1, 2000 attempts per target.
SLP lowers collective cost from 105583.438186 to 29148.907740, stopping at the
80-iteration cap rather than a convergence certificate. It retains original
intersection halfspaces and then converts selected routes to region transitions
for the existing final GCS formulation.

Initial offline representative construction/SLP costs 127.456 s (SLP itself
127.022 s). The successful 91.544 s run REUSES this cache and includes geometry
17.165 s, query pruning 16.705 s, GCS stage one 34.143 s, stage two 23.351 s
and remaining validation overhead. It excludes the offline work and earlier
failed attempt. These are not a measured successful cold-run total.

Input adjacency, representative sampling counts and GCS budgets differ between
center and SLP runs. Their cost/time comparison is observational, not a matched
ablation proving one preprocessing method intrinsically superior. Historical
IRIS/adjacency construction is excluded; timings are single-run measurements,
not repeated statistical estimates. All returned costs are feasible upper
bounds rather than certified global continuous-goal optima.

## Failed and stopped runs

| Configuration | Methods attempted | Outcome |
| --- | --- | --- |
| Initial default region cap of 160 | Primary entry point | Stopped at 2856 regions before constructing the GCS; software budget, not measured OOM |
| No graph pruning: 2856 regions / 869670 directed edges | union-hull and per-target, 200/300 | Both hit 6 GiB while building the GCS model (about 65.3 / 62.3 s to stop) |
| Dimension>=6 adjacency: 2856 / 38094, no path pruning | Both, 200/300 | Both hit 6 GiB during the solve phase (34.1 / 37.3 s to stop) |
| Same adjacency + redundant-constraint compaction | Both, 200/300 | Both still hit 6 GiB during the solve phase (37.5 / 41.6 s to stop) |
| Same compact model, guard increased to 7 GiB | Both, 200/300 | Per-target hit 7 GiB; union-hull exited -9 near the limit, with no recorded definitive cause |
| Intersection + SLP pruning, automatic MOSEK form | union-hull, 1000/1500 | Relaxation returned kSolverSpecificError / UNKNOWN before sampling; preserved cold attempt took 168.999 s including offline SLP |

The last solver failure was followed by the validated primal-form retry above,
with the same graph, targets, costs and tolerances. Its solver status is not an
infeasibility certificate.

A full nonempty-adjacency intersection graph was size-checked, not materialized:
434835 vertices / 184684472 undirected edges. It exceeds the bounded-memory
implementation capacity; this motivated the explicitly stated adjacency policy.

## Scope and source records

The sampled-point-target and native nonlinear-terminal baselines have not been
run on THIS 60 cm / 90 degree example. Their earlier 30 cm / 20 degree results
must not be mixed into this table.

Successful source directories:

- `representative_pruned_alpha110/` (union-hull and per-target)
- `intersection_slp_facet_1000_1500_primal_20261030/`

Failure directories:

- `full_graph_60cm90deg_comparison/`
- `facet_graph_60cm90deg_comparison/`
- `facet_graph_60cm90deg_compact_comparison/`
- `facet_graph_60cm90deg_compact_rss7_comparison/`
- `intersection_slp_facet_1000_1500_20261030/`

The accompanying `current_60cm90deg_all_methods_20261030.csv` and `.json` flatten
the supervisor records without altering original reports. Failed-worker times
are labeled `until_termination`; completed report times are `total_task`.

Among validated runs, the lowest returned cost is the per-target center-pruned
run, the shortest recorded task time is the center-pruned union-hull run, and
the lowest sampled peak worker memory is the intersection/SLP union-hull run.
