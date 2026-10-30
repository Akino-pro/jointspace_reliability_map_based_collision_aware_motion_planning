# Union-hull: matching larger budgets in both stages

Follow-up: the 4R relaxation issue was resolved by selecting MOSEK primal
form, without changing geometry or tolerances. Successful 800/1600/3200 runs
are saved in `results/spatial4r/union_hull_primal_20261030`; see
[the diagnosis and results](../../../docs/SPATIAL4R_UNION_HULL_SOLVER.md).
The failed automatic-form measurements below are preserved as historical data.

Only method two (union-hull) ran. Both stages used 800/1200, 1600/2400,
or 3200/4800 paths/trial caps. Seed 0. No primary-method solve ran.
The original graphs and hull targets match the historical references by SHA256.
No representative pruning was used on these 3R/4R graphs.

| Robot | Paths/trials in each stage | Final weighted cost | Total seconds | Outcome |
| --- | --- | ---: | ---: | --- |
| 3R | 800/1200 | 0.381717767939 | 23.496 | Validated |
| 3R | 1600/2400 | 0.381717767939 | 36.495 | Validated |
| 3R | 3200/4800 | 0.381717767939 | 64.790 | Validated |
| 4R | 800/1200 | unavailable | 10.860 | First stage failed; 0 candidate paths |
| 4R | 1600/2400 | unavailable | 10.950 | First stage failed; 0 candidate paths |
| 4R | 3200/4800 | unavailable | 11.049 | First stage failed; 0 candidate paths |

Historical first-method references (read only, not rerun):

- 3R: `results/planar3r/per_target_budget_sweep_20261029/paths200_trials300.json`,
  cost 0.381717769444, total 104.294970 s. Each of 20 targets received 200/300;
  stage two used 200/300. A lower-budget historical 100/150 case also achieved
  the same cost in 86.691367 s (stage two fixed at 200/300).
- 4R: `results/spatial4r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json`,
  cost 0.0780798316713, total 115.854341 s. Both stages used per-call 200/300.

3R generated the requested 800, 1600 and 3200 unique candidate paths in each
stage. Every final path passed collision, goal, graph-membership and cost checks;
the task residual was 2.22e-16. All three runs selected target 19. The cost
difference from the historical reference is about -1.51e-9, numerical agreement.
800/1200 is the fastest of the tested larger budgets, with about 4.44x speedup
against the saved 200/300 per-target reference. This is a comparison of single
runs at different times, not repeated statistical timing evidence.

For 4R, a separate diagnostic 800/1200 rerun (also union-hull only) recorded
`relaxation_success=false`, `SolutionResult.kSolverSpecificError`, MOSEK
`rescode=0`, `solution_status=0`. It failed before `SamplePaths`, so the large
sampling budgets did not get exercised. The main three runs all had the same
pre-rounding log pattern and zero candidates; the diagnostic confirms this
failure mode for the 800/1200 case. This is not evidence of physical infeasibility
or proof that larger sampling cannot work after resolving the relaxation issue.
See `failure_diagnosis/4r_paths800.json`.

Peak sampled worker RSS was 1.92-2.03 GiB for 3R and 2.84-2.87 GiB for 4R;
no worker hit the 6 GiB guard. Times include fresh geometry preprocessing and
both executed planning stages, reuse cached IRIS, and exclude plotting. Trial
counts are caps; Drake does not expose actual sampling-trial counts.

`summary.csv`, `summary.json`, individual reports and logs retain measurements.
To reproduce in the configured container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.union_hull_large_budget_sweep --output-dir results/diagnostics/union_hull_large_budget_repeat
```
