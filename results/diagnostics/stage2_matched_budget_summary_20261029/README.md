# Final matched-budget stage-two replay results

The separate 4R first-stage sweep is paused at100/150; only50/75 has a completed first-stage report.
No first-stage GCS was rerun. The saved projected endpoint is reused in every case.
Both stages now use that case's own path/trial cap:50/75,100/150,200/300,300/450,400/600.

| Budget in both stages | 3R independent cost | 3R joint cost | Independent new stage2(s) | Joint new stage2(s) | Independent stitched total(s) | Joint stitched total(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50/75 | 0.3818392430 | 0.3861941211 | 4.254 | 9.555 | 77.401 | 17.124 |
| 100/150 | 0.3817177694 | 0.3850137378 | 2.911 | 8.706 | 85.301 | 15.714 |
| 200/300 | 0.3817177694 | 0.3850137377 | 3.410 | 8.897 | 103.543 | 16.788 |
| 300/450 | 0.3817177694 | 0.3850137377 | 3.426 | 9.201 | 120.665 | 18.078 |
| 400/600 | 0.3817177679 | 0.3850137377 | 3.468 | 9.595 | 139.512 | 19.448 |

## Completed spatial4R case

50/75 in both stages:cost=0.1306891535;new stage2=6.220s;stitched total=100.749s. Path validated.
The100/150,200/300,300/450,400/600 new 4R sweep cases are not reported as completed.

## Interpretation and limitations

All11replayed paths passed graph membership,collision sampling,true endpoint task and recomputed weighted-cost checks.
Independent3R and4R reports include the original stage-one route; it is retained as one additional restriction.
The old joint-hull reports saved only the projected endpoint,not the stage-one route. Joint replays therefore
run the full fixed-goal graph with no extra retained route. No old stage-two route is used as a substitute.
This asymmetry is explicit;the joint50/75 cost change cannot be attributed solely to reducing the rounding budget.
Actual joint and independent path/trial settings match the stage-one case;seeds remain0 and graphs are fingerprint checked.
The tiny independent400vs100cost difference(about1.5e-9)is numerically negligible,not evidence of a meaningful improvement.
There is no global continuous-SSM optimum certificate and no equal-total-candidate-budget claim.

Stitched total=old measured method total-old measured stage-two time+new measured stage-two time.
It is an estimate combining different executions,not a freshly measured end-to-end duration.
The 4R paused process was not resumed. Its pause record is in results/spatial4r/per_target_budget_sweep_20261029/pause.json.
The previous sweeps and all source reports remain unchanged;source SHA256 digests are recorded per replay.

Sources:
- results/diagnostics/stage2_matched_budget_independent_20261029/summary.json
- results/diagnostics/stage2_matched_budget_joint_20261029/summary.json

A supplementary attempt to change the global3R stage-two default was rejected by automatic review as out of scope.
No global default change was applied;all replay budgets were passed explicitly.
