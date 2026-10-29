# Planar 3R joint-hull first-stage budget sweep

Only the joint-hull two-stage method ran. Seed 0; second-stage budget fixed at 200/300.
The first-stage graph has 43 free regions, 20 convex targets, one source and one sink (65 vertices).
Times include graph construction and optimization; total adds shared preprocessing once.
Each budget was measured once, in ascending order. This is not a statistical timing comparison.

| Paths / trial cap | Stage-one actual routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total incl. preprocessing (s) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50/75 | 50 | 0.3850137203 | 0.3850137377 | 4.526 | 9.846 | 17.415 | success |
| 100/150 | 100 | 0.3850137203 | 0.3850137377 | 3.895 | 10.066 | 17.075 | success |
| 200/300 | 200 | 0.3850137203 | 0.3850137377 | 4.790 | 9.955 | 17.845 | success |
| 300/450 | 300 | 0.3850137203 | 0.3850137377 | 5.795 | 10.128 | 19.004 | success |
| 400/600 | 400 | 0.3850137203 | 0.3850137377 | 6.611 | 10.165 | 20.019 | success |

Both graph/task and target-polytope fingerprints are checked across all runs.
Each successful path passes task, collision, graph membership and weighted-cost checks.
Second-stage candidate counts are reported separately; one retained-route restriction is additional to the random caps.
The trial counts are upper bounds: Drake does not expose actual SamplePaths attempt counts.
No primary, sampled-goal or nonlinear-goal method was run.
No IRIS regions, graph edges, target hulls, objective weights or rounding seeds were changed.
Final returned costs are feasible upper bounds, not certified continuous-SSM optima.

Per-case JSON files, summary.json and summary.csv retain the unrounded measurements.
The cost/time figure is saved as cost_time.png and cost_time.svg when matplotlib is available.
