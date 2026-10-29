# Spatial 4R independent-target budget sweep

Robot: ftwmax. Existing midpoint scene and IRIS cache; no scene or graph modification.
Both stages use each case's path/trial budget; each first-stage target receives the full budget.
Only the primary per-target two-stage method runs. Rounding seed 0; SSM search seed 1.
Times include shared preprocessing once and exclude plotting/file generation and other methods.
Each budget is one sequential run, not a statistical timing benchmark.

| Per-target paths/trials | Targets with returned paths | Actual stage-one routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50/75 | 11/13 | 550 | 0.1306891148 | 0.1306891535 | 93.627 | 5.702 | 101.527 | success |
| 100/150 | 11/13 | 1100 | 0.1306891148 | 0.1306891535 | 98.502 | 6.014 | 106.981 | success |
| 200/300 | 11/13 | 2200 | 0.0911111046 | 0.0780798317 | 105.618 | 7.832 | 115.854 | success |
| 300/450 | 11/13 | 3300 | 0.0911111046 | 0.0780798317 | 113.541 | 8.398 | 124.368 | success |
| 400/600 | 11/13 | 4400 | 0.0911111046 | 0.0780798317 | 120.563 | 9.280 | 132.316 | success |

Graph/task and target fingerprints are checked across budgets. Failed targets remain in the report.
No returned path for a target is not proof of physical infeasibility.
Actual unique routes and successful restrictions are recorded separately from requested caps.
Actual native sampling-trial attempt counts are not exposed and are not inferred from caps.
Second-stage random caps match the case budget,plus one retained first-stage route restriction.
Returned costs are feasible upper bounds, not globally certified continuous-goal optima.
No joint-hull, sampled-goal or nonlinear-goal comparison was executed.
The 4R defaults are not changed automatically by this experiment.

Full precision: summary.json, summary.csv and each case JSON. Plot: cost_time.png and cost_time.svg.

## Completed outcome (2026-10-29)

All five runs completed with matching budgets in both stages and full fresh
end-to-end timing. No saved first-stage or second-stage solver result was
reused. Graph/task and hull-target fingerprints match in all five cases.
The graph has56free regions,1202directed region edges and13target sets.

Final costs for50/75and100/150are0.13068915352676208;for200/300,300/450and
400/600they are0.07807983167132608. The improvement is about40.3%. This is
the smallest tested budget reaching the lowest returned cost for this case
and seed,not a global optimality or convergence certificate.

Every run returned validated final paths.11of13first-stage targets returned
paths in every case;two did not return feasible candidates,without proving
physical infeasibility. Actual first-stage routes and successful restrictions
were550,1100,2200,3300,4400. Stage two generated50,100,200,300,400routes,
respectively,plus one retained-route restriction per run. Collision samples
all passed and endpoint errors were below2.69e-10.

Full measured times were101.5272,106.9806,115.8543,124.3676,132.3159s.
These are fresh sequential single-run measurements,not stitched replay
estimates or repeated statistical measurements. The old paused fixed-stage-two
worker was terminated before this sweep;its completed report was preserved
and its pause record marked superseded. No paused duration enters these times.

The planar3R primary defaults are now100/150in BOTH stages following explicit
user authorization. Spatial4R defaults remain200/300in both stages;this
experiment does not change spatial7R settings.
