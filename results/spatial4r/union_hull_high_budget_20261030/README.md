# 4R union-hull higher sampling budgets

Only union-hull ran. Each stage uses the listed paths/trial caps, seed 0, MOSEK primal form.
No graph, targets, collision geometry, objective weights or solver tolerances changed.
Input graph and target fingerprints match the saved per-target reference.

Reference: per-target 200/300, cost 0.078079831671, total 115.854 s. It was not rerun.

| Paths/trials in both stages | Final cost | Gap to reference | Total seconds | Actual stage 1 / stage 2 paths | Success |
| --- | ---: | ---: | ---: | ---: | --- |
| 800/1200 | 0.089816724048 | 15.032% | 75.005 | 800 / 800 | True |
| 1600/2400 | 0.089816724048 | 15.032% | 88.875 | 1600 / 1600 | True |
| 3200/4800 | 0.089816724048 | 15.032% | 106.757 | 3200 / 3200 | True |
| 6400/9600 | 0.089816724048 | 15.032% | 143.670 | 6400 / 6400 | True |
| 12800/19200 | 0.078045911348 | -0.043% | 227.401 | 12800 / 12800 | True |
| 25600/38400 | 0.078045940249 | -0.043% | 403.263 | 25600 / 25600 | True |

Every successful final path is checked for goal, collision, graph membership and cost.
The gap is against a saved feasible reference, not a certified global optimum.
Trials are caps; actual attempt counts are not exposed by Drake.
This combines sequential single runs, not repeated statistical measurements.
Full high-budget worker RSS and outcomes are in summary.json; raw reports and logs are preserved.

The initial 25600-path attempt hit the 6 GiB RSS guard after 255.7 s. Its failure remains in summary.json.
The completed retry is in ../union_hull_high_budget_bounded_history_20261030, with full-result history disabled.
Every candidate is still solved in the original order; only the best result and success counters are retained.
The retry's successful-run time excludes the failed initial attempt; it is not a cumulative experiment time.
The default history-retaining behavior is unchanged for other runs.

Reproduce the high-budget runs in the configured Drake container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.union_hull_large_budget_sweep --robots 4r --budgets 6400 12800 25600 --output-dir results/spatial4r/union_hull_high_budget_repeat
```

Retry the 25600-path case with bounded result history:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.union_hull_large_budget_sweep --robots 4r --budgets 25600 --discard-rounding-history --output-dir results/spatial4r/union_hull_high_budget_bounded_repeat
```
