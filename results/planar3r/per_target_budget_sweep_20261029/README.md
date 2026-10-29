# Planar 3R independent-target budget sweep

Each target receives its full configured path/trial cap. There is no total-budget division.
Only the original per-target two-stage method ran. Seed 0; second stage fixed at 200/300.
The same 43 free regions, 824 directed edges and 20 target intersections are used throughout.
Costs are feasible returned costs, not certified continuous-SSM optima. Times are single measurements.

| Per-target cap | Total requested paths | Actual stage-one paths | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50/75 | 1000 | 1000 | 0.3817856289 | 0.3818392430 | 70.042 | 4.418 | 77.565 | success |
| 100/150 | 2000 | 2000 | 0.3816459432 | 0.3817177694 | 79.076 | 4.300 | 86.691 | success |
| 200/300 | 4000 | 4000 | 0.3816459432 | 0.3817177694 | 96.864 | 4.161 | 104.295 | success |
| 300/450 | 6000 | 6000 | 0.3816459432 | 0.3817177694 | 113.950 | 4.197 | 121.436 | success |
| 400/600 | 8000 | 8000 | 0.3816459432 | 0.3817177694 | 132.711 | 4.237 | 140.282 | success |

## Comparison with the completed joint-hull sweep

Joint-hull measurements are reused from its separate sweep, not rerun or included in primary timings.
Both methods have the same per-call stage-one cap and fixed second-stage 200/300 budget.
The independent method performs 20 stage-one relaxations and gets 20 times the stage-one path cap.

| Paths/trials per call | Independent final cost | Joint final cost | Independent total (s) | Joint total (s) |
| --- | ---: | ---: | ---: | ---: |
| 50/75 | 0.3818392430 | 0.3850137377 | 77.565 | 17.415 |
| 100/150 | 0.3817177694 | 0.3850137377 | 86.691 | 17.075 |
| 200/300 | 0.3817177694 | 0.3850137377 | 104.295 | 17.845 |
| 300/450 | 0.3817177694 | 0.3850137377 | 121.436 | 19.004 |
| 400/600 | 0.3817177694 | 0.3850137377 | 140.282 | 20.019 |

Total times include preprocessing once and exclude other methods or file/chart generation.
Each stage-two solve additionally considers the retained first-stage route as a convex restriction.
Path/trial settings are caps; actual SamplePaths trial counts are unavailable and not inferred.
Graph/task and target fingerprints must match both across this sweep and the earlier joint sweep.
All per-target allocations and diagnostics are preserved in the individual JSON reports.
No IRIS regeneration, reference grid, sampled-goal or nonlinear-goal experiment was performed.

Charts: cost_time.png and cost_time.svg. Full precision data: summary.json and summary.csv.

## Observed outcome

All five cases succeeded. Actual stage-one route counts were exactly1000,
2000,4000,6000,8000;all corresponding convex restrictions succeeded and all
20targets returned a feasible candidate in every case. Both task and graph
fingerprints match across cases and match the earlier joint-hull sweep.

The50/75 case selected target17(zero-based),while100/150 and larger selected
target19. Final cost decreased from0.3818392430276426 to0.3817177694444778,
then remained identical through400/600. All final paths passed collision and
task checks;endpoint errors were at most2.23e-16. Each stage-two solve generated
200routes and considered one additional retained-route restriction.

For this case and seed,100/150 is the smallest tested per-target budget to
reach the best returned final cost among these five cases. This is not a
global optimality or stochastic convergence certificate. The previous
shared-total-budget ablation used different per-target settings and remains
a separate historical experiment, not overwritten by this sweep.

An independent first-stage call contains43free-region vertices,one source and
one target(45vertices). The20target problems are solved separately. The joint
first-stage graph contains43free regions,20targets,one source and one sink
(65vertices). Equal per-call caps therefore give the independent method20times
the stage-one path budget and20separate relaxations;this is not an equal-total
budget or equal-time comparison.
