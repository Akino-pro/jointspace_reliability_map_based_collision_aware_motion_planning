# 7R center pruning: alpha 1.1, 1000 attempts/target, union-hull 1000/1500

Validated success on the current 60 cm / base-Z 90-degree example. Full input
adjacency: 2856 regions, 869670 nonempty-intersection directed edges, 23 targets.
This does not use the dimension>=6 adjacency of the intersection/SLP experiment.
Only union-hull ran. Input graph and target fingerprints match the previous
center-pruned per-target reference; no per-target GCS was rerun.

Pruning: alpha 1.1, 1000 attempts PER TARGET, seed 0; retains 314 regions and
3188 directed edges. Both GCS stages use 1000 paths / 1500 trial caps.

| Quantity | Measured value |
| --- | ---: |
| Final weighted feasible cost | 0.580011597931 |
| Successful task time | 621.319 s |
| Geometry preprocessing | 19.836 s |
| Representative path pruning | 33.072 s |
| GCS stage one | 299.946 s |
| GCS stage two | 268.120 s |
| Sampled peak worker RSS | 6.84 GiB |
| Stage-one unique / solver-successful paths | 1000 / 999 |
| Stage-two unique / solver-successful paths | 1000 / 1000 |
| Selected target | 22 |
| SE(3) task residual | 0.0 (floating-point reported result) |

Collision, original-region membership, endpoint task and recomputed cost checks
all passed. These costs are feasible upper bounds, not certified global optima.

## Memory and timing conditions

Three attempts hit the 6 GiB guard: original constraints (101.3 s), compact
constraints (97.3 s), and compact/primal/one internal MOSEK thread (96.6 s).
Their preserved directories are respectively:

- `../center_alpha110_search1000_gcs1000_1500_20261030/`
- `../center_alpha110_search1000_gcs1000_1500_compact_20261030/`
- `../center_alpha110_search1000_gcs1000_1500_compact_primal_thread1_20261030/`

The successful retry uses the same selected graph and costs, with redundant
constraint compaction (edge rows 51337 -> 10259; region rows 6186 -> 4945),
MOSEK primal form, one internal MOSEK thread, and a 7 GiB RSS guard. GCS
algorithm parallelism remains at its installed default, separately from MOSEK
thread count. The no-longer-needed original connectivity graph is released and
unreachable preprocessing objects collected before GCS allocation. No rounding
result history is accumulated; every candidate is still solved and compared,
retaining the best result and counts. No feasibility tolerances were loosened.

621.319 s is the successful attempt only, excluding the three failed attempts.
It reuses IRIS/adjacency caches but recomputes center pruning. No SLP runs.

| Historical reference | Cost | Task seconds |
| --- | ---: | ---: |
| Center pruning 100 attempts/target, union-hull 200/300 | 0.616332092695 | 74.920 |
| Same 100-attempt graph, per-target 200/300 | 0.576688012520 | 494.766 |
| This 1000-attempt graph, union-hull 1000/1500 | 0.580011597931 | 621.319 |

The new cost is 0.5763% above the saved per-target reference. Its time is worse
in this memory-constrained configuration. Graph sizes, rounding budgets, solver
form/thread settings and memory management differ; these single runs are not
a matched timing ablation or repeated statistical benchmark. 49 regression
checks passed after the changes.

## Reproduce

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_full_graph_supervisor --methods union_hulls --representative-pruning --pruning-method centers --pruning-alpha 1.1 --pruning-trials 1000 --max-rounded-paths 1000 --max-rounding-trials 1500 --discard-rounding-history --compact-constraints --mosek-solve-form primal --mosek-threads 1 --rss-limit-gib 7 --output-dir results/spatial7r/center_alpha110_search1000_repeat
```

`union_hulls.json` stores the trajectory and validation, `comparison_summary.json`
stores RSS and timing, and `union_hulls.log` records both GCS stages.
