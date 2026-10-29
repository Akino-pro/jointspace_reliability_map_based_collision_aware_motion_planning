# Four-method comparison: completed experiment matrix

Updated 2026-10-28. All twelve method/example combinations have been attempted:
ten returned validated paths, two returned no feasible path. No unrun entries
remain in this matrix. A failed solver return does not prove infeasibility.

Costs are region-weighted joint-space lengths; lower is better. Each time
includes shared preprocessing once and excludes fresh IRIS generation and
other methods' computation. Compare costs within each example, not across
different robots. These are single runs, not repeated timing statistics.

| Example | Method | Final weighted cost | Time (s) | Outcome |
| --- | --- | ---: | ---: | --- |
| Planar 3R | Main: independent hull targets, two stages | 0.3817177694 | 115.80 | Validated |
| Planar 3R | Sampled point targets | 0.3817192602 | 21.95 | Validated |
| Planar 3R | Joint hull targets, two stages | 0.3850137377 | 21.11 | Validated |
| Planar 3R | Native nonlinear terminal | — | 92.42 | No feasible rounded path returned |
| Spatial 4R | Main: independent hull targets, two stages | 0.0780798317 | 110.10 | Validated |
| Spatial 4R | Sampled point targets | 0.0920764848 | 14.16 | Validated |
| Spatial 4R | Joint hull targets, two stages | — | 9.80 | No feasible rounded first-stage path returned |
| Spatial 4R | Native nonlinear terminal | 0.2813945056 | 129.63 | Validated |
| Spatial 7R | Main: independent hull targets, two stages | 0.0043086900 | 14.06 | Validated |
| Spatial 7R | Sampled point targets | 0.0040315218 | 3.87 | Validated |
| Spatial 7R | Joint hull targets, two stages | 0.0043086900 | 5.32 | Validated |
| Spatial 7R | Native nonlinear terminal | 0.0039795184 | 120.78 | Validated |

All examples use `union_of_branch_recovery_sets`, existing IRIS caches and no
periodic seam connections. All methods retain the same graph and weights
within each example. Per-call rounding budgets are 200 paths / 300 trials.
Nonlinear restrictions use SNOPT with 500 major iterations. All feasible
results passed their endpoint task and collision checks; this is numerical
validation, not continuous collision certification or global optimality.

## New planar run

The 2026-10-28 run uses the original **forward** planar task:

- Start q = [-2.972349782093813, -2.26926174311021, -1.2375903668686903].
- Cartesian goal = [1.5, 1.5], epsilon 0.025, SSM arc spacing 0.044, seed 0.
- One discovered goal branch, 301 samples, 43 free regions, 824 edges,
  20 hull/region targets; shared preprocessing 3.33814328 s.
- Main: 20 first-stage solves plus one fixed-goal solve; task error 2.22e-16.
- Sampled: 91 covered point targets; task error 1.10e-11.
- Joint hulls: 20 targets, two GCS stages; task error 3.33e-15.
- Nonlinear: 43 regional terminals, 152 unique sampled restrictions from
  300 trials; no feasible result returned. It was allowed to finish, not
  stopped because of a resource guard.
- Entire sequential process completed in 241.262282376 s. Observed Docker
  memory remained around 3 GiB, below its 7.57 GiB limit.

The previous planar memory problem concerned a **different reverse two-branch
task** to [0.5, 0.5] with 110 regions / 2,776 edges. It is not the planar example
in this comparison. The old max-across-branches forward report is preserved
as historical data; all rows above use the current union rule. This forward
example has one branch, so its weighted costs match the historical values.

## Sources and interpretation

- New planar report: `results/planar3r/all_goal_comparisons_20261028.json`.
- Planar log: `logs/planar3R_all_goal_comparisons_20261028.log`.
- Spatial 4R report (2026-10-26): `results/spatial4r/spatial4R_all_goal_comparisons.json`.
- Spatial 7R report (2026-10-26): `results/spatial7r/spatial7R_30cm20deg_all_goal_comparisons.json`.
- Machine-readable combined summary: `results/diagnostics/final_four_method_comparisons_20261028.json`.

The forward 3R main and sampled costs are almost identical; joint hulls return
a slightly higher cost. Main has the lowest returned cost in 4R. The nonlinear
method has the lowest returned cost in the local 7R case but takes much longer;
its task-free relaxation is weak. These observations do not establish that
one method is generally superior or certify the globally best continuous SSM
endpoint. The completed 4R and 7R experiments were reused, not rerun or averaged.
