# 4R union-hull relaxation diagnosis and successful runs

The ftwmax midpoint example uses the unchanged 56-region, 1202-directed-edge
graph and 13 target sets. The default MOSEK relaxation selected the dual form,
stalled and returned UNKNOWN (`kSolverSpecificError`, rescode 0, solution_status
0), before any path sampling. This was not an infeasibility certificate.

Solving the identical relaxation with `MSK_IPAR_INTPNT_SOLVE_FORM=1` (primal)
returned OPTIMAL, lower-bound cost 0.0687300216393. Tolerances, geometry,
targets, objective and seed were unchanged. Diagnostic reports are in
`results/diagnostics/spatial4r_union_hull_solver_20261030`; `primal.log` records
the solver iterations and solution summary.

The 4R union-hull-only entry point and union-hull comparisons now explicitly
select primal form for MOSEK. This option is recorded in their solver settings.
The primary per-target 4R method and its 200/300 defaults remain unchanged.

Full two-stage union-hull runs (seed 0; matching stage budgets):

| Paths/trials per stage | Final weighted cost | Total seconds | Result |
| --- | ---: | ---: | --- |
| 800/1200 | 0.089816724048 | 75.005 | Validated |
| 1600/2400 | 0.089816724048 | 88.875 | Validated |
| 3200/4800 | 0.089816724048 | 106.757 | Validated |

All runs generated the requested number of unique paths in each stage and
passed collision, graph-membership, cost and endpoint checks (task error
9.16e-16). None reran the primary per-target method. Against the saved
per-target 200/300 result, cost 0.0780798316713 and total 115.854 s, these
union-hull costs are about 15.03% higher. Larger budgets did not improve them.
The primal relaxation fix makes the union method work, but does not establish
equivalent solution quality to per-target planning in this 4R example.

Reports and the historical-reference fingerprint checks are saved under
`results/spatial4r/union_hull_primal_20261030`. Timing is from sequential single
runs with cached IRIS, not repeated statistical measurements. Costs are
feasible upper bounds, not certified global continuous-goal optima.

Reproduce only union-hull (in the configured Drake container):

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --union-hulls-only --max-rounded-paths 800 --max-rounding-trials 1200 --stage2-max-rounded-paths 800 --stage2-max-rounding-trials 1200 --output results/spatial4r/union_hull_primal_repeat.json
```
