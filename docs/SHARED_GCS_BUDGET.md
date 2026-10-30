# Shared candidate budget for planar 3R

The optional total-budget mode compares independent hull-target solves and
joint hull-target selection under the same stage-one maximum candidate-path
and sampling-trial budgets. It does not change the task, region graph,
reliability weights, endpoint projection or stage-two algorithm. Historical
per-call defaults and saved reports are preserved.

Enable with `--method per-target`, `--stage1-total-paths B` and `--stage1-total-trials T`.
Each of N targets receives floor(B/N) paths and floor(T/N) trials, with one
additional unit assigned to the first remainder targets in the existing
deterministic target order. Totals are exactly conserved, no targets are
dropped, and unused quotas are not reassigned. B and T must be positive
integers, B >= N and T >= B. Unsupported insufficient budgets fail explicitly.
This mode applies to rounding, not mixed-integer exact solves.

The joint-hull comparison receives B paths and T trials for its one stage-one
call. The independent method shares them across N calls. Stage two has its
own independent, identical settings on both methods. The existing retained
stage-one-route convex restriction is an additional stage-two candidate on
both methods and is not included in the random candidate cap.

For the forward planar example, N=20, B=200 and T=300 means 10 paths / 15 trials
per independent target versus 200 paths / 300 trials for the joint graph.
Stage two remains 200 paths / 300 trials. This experiment enables only the
joint-hull comparison: sampled and nonlinear comparisons are not run.

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.planar3R_two_stage_gcs_planner --compare-union-hulls --stage1-total-paths 200 --stage1-total-trials 300 --output results/planar3r/shared_budget_200paths_300trials_20261029.json
```

Reports distinguish requested quotas, actual generated unique routes and
successful restrictions. Per-target diagnostics include failures as well as
successful returned paths. Drake's `SamplePaths` API does not expose the
actual number of trial attempts; `rounding_trials_attempted` is therefore
null, rather than incorrectly reporting the trial cap as a measured count.

Equal candidate caps are not equal computation time. The independent method
still solves N separate relaxations; the joint graph has multiple terminal
sets and different relaxed flows. A restriction can succeed yet the returned
path later fail membership validation. Timing and final physical-task checks
remain necessary. These experiments do not introduce a region-count ratio
heuristic or impose a wall-clock limit.

## Forward planar experiment (2026-10-29)

Only the independent method and joint-hull method ran. The sampled and
nonlinear comparisons are absent from the report. The unchanged forward
task has one goal branch, 301 SSM samples, 43 regions, 824 directed edges and
20 hull targets. Both use the current recovery-union reliability map,
epsilon 0.025, arc spacing 0.044, rounding seed 0 and MOSEK. Existing IRIS was
loaded; shared preprocessing took 2.989995291 s.

| Metric | Independent target solves | Joint target solve |
| --- | ---: | ---: |
| Stage-one total path cap | 200 (10 per target) | 200 |
| Stage-one total trial cap | 300 (15 per target) | 300 |
| Actual stage-one unique routes | 200 | 200 |
| Successful stage-one restrictions | 200 | 200 |
| Stage-one returned path cost | 0.7073967172 | 0.3850137203 |
| Selected hull target (one-based) | 20 | 1 |
| Stage-one time | 63.2856 s | 5.8612 s |
| Stage-two random path cap / trial cap | 200 / 300 | 200 / 300 |
| Actual stage-two unique routes | 200 | 200 |
| Additional retained-route restrictions | 1 | 1 |
| Stage-two time | 4.2176 s | 10.2969 s |
| Final weighted path cost | 0.3817168047 | 0.3850137377 |
| Total including shared preprocessing | 70.5811 s | 19.1756 s |
| Endpoint Cartesian error | 4.97e-16 | 3.33e-15 |
| Collision samples, all passed | 618 | 650 |

Both methods returned validated paths. The independent method's final cost
was about 0.86% lower, while the joint method was faster. The independent
first-stage path was substantially worse; full fixed-goal replanning recovered
a much better route. Thus this run does not show that independent solves
necessarily find a better first-stage path. The two methods selected different
task endpoints, and final cost includes that choice and the second replan.

The historical 2026-10-28 independent run used 200 paths per target (4,000
generated in total) and cost 0.3817177694 in 115.7992 s. The new 200-total-path
run has almost the same final cost with fewer candidate restrictions. Timing
differences are single measurements across sessions, not repeated statistics.
Equal path and trial caps do not equalize the 20 independent relaxations
against one joint relaxation, so there is still no equal-time comparison.

Report: `results/planar3r/shared_budget_200paths_300trials_20261029.json`.
Log: `logs/planar3R_shared_budget_20261029.log`.
Regression log: `logs/shared_budget_tests_20261029.log` (168 tests: 162 passed,
six optional MuJoCo tests skipped). Tests include exact quota conservation,
remainder allocation, insufficient budgets, comparison forwarding, independent
stage-two settings and a real GCS case where actual routes differ from the cap.
