# Spatial 7R independent-target budget sweep

Robot: spatial7R. Existing midpoint scene and IRIS cache; no scene or graph modification.
Both stages use each case's path/trial budget; each first-stage target receives the full budget.
Only the primary per-target two-stage method runs. Rounding seed 0; SSM search seed 1.
Times include shared preprocessing once and exclude plotting/file generation and other methods.
Each budget is one sequential run, not a statistical timing benchmark.

| Per-target paths/trials | Targets with returned paths | Actual stage-one routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50/75 | 1/4 | 2 | 0.0043086444 | 0.0043086900 | 13.892 | 0.389 | 15.804 | success |
| 100/150 | 1/4 | 2 | 0.0043086444 | 0.0043086900 | 14.284 | 0.368 | 16.765 | success |
| 200/300 | 1/4 | 2 | 0.0043086444 | 0.0043086900 | 13.105 | 0.353 | 15.139 | success |
| 300/450 | 1/4 | 2 | 0.0043086444 | 0.0043086900 | 12.977 | 0.355 | 14.985 | success |
| 400/600 | 1/4 | 2 | 0.0043086444 | 0.0043086900 | 12.840 | 0.365 | 14.849 | success |

Graph/task and target fingerprints are checked across budgets. Failed targets remain in the report.
No returned path for a target is not proof of physical infeasibility.
Actual unique routes and successful restrictions are recorded separately from requested caps.
Actual native sampling-trial attempt counts are not exposed and are not inferred from caps.
Second-stage random caps match the case budget,plus one retained first-stage route restriction.
Returned costs are feasible upper bounds, not globally certified continuous-goal optima.
No joint-hull, sampled-goal or nonlinear-goal comparison was executed.
The 7R defaults are not changed automatically by this experiment.

Full precision: summary.json, summary.csv and each case JSON. Plot: cost_time.png and cost_time.svg.
