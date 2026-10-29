# Spatial 4R independent-target budget sweep

Robot: ftwmax. Existing midpoint scene and IRIS cache; no scene or graph modification.
Each target receives its full budget, without total-budget division. Stage two is fixed at 200/300.
Only the primary per-target two-stage method runs. Rounding seed 0; SSM search seed 1.
Times include shared preprocessing once and exclude plotting/file generation and other methods.
Each budget is one sequential run, not a statistical timing benchmark.

| Per-target paths/trials | Targets with returned paths | Actual stage-one routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50/75 | 11/13 | 550 | 0.1306891148 | 0.1306891535 | 92.209 | 6.294 | 100.823 | success |

Graph/task and target fingerprints are checked across budgets. Failed targets remain in the report.
No returned path for a target is not proof of physical infeasibility.
Actual unique routes and successful restrictions are recorded separately from requested caps.
Actual native sampling-trial attempt counts are not exposed and are not inferred from caps.
Second-stage random cap is 200/300 plus one retained first-stage route restriction.
Returned costs are feasible upper bounds, not globally certified continuous-goal optima.
No joint-hull, sampled-goal or nonlinear-goal comparison was executed.
The 4R defaults are not changed automatically by this experiment.

Full precision: summary.json, summary.csv and each case JSON. Plot: cost_time.png and cost_time.svg.
