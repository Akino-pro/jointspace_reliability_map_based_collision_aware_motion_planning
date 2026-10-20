# Reverse benchmark: Clarabel and MOSEK diagnosis

Controlled checks use the first target of the same merged reverse benchmark:
93 regions, 1960 region edges, 1962 edges including endpoints. Both use
continuous convex relaxation; no route sampling or mixed-integer search runs.
`planar3R_diagnose_relaxation.py` captures inputs before the first planning
solve and reconstructs that graph with explicitly selected solvers.

| Relaxation solver | GCS preprocessing | Solve wall time | Result |
| --- | --- | ---: | --- |
| Clarabel | Off | 79.19 s | MaxTime, 7 iterations |
| MOSEK (20 threads reported) | Off | 10.32 s | Success |
| Clarabel | MOSEK | 82.73 s including preprocessing | MaxTime, 8 iterations; Clarabel internal time 72.43 s |
| MOSEK (1 thread) | Off | 18.99 s | Success |

All cases requested a 60-second solver limit. It is not a hard wall-clock
deadline: Clarabel checked termination after exceeding it. A separate
100-second worker watchdog was enabled; none of these cases hit the watchdog.
Times exclude the approximately 6-second graph/geometry reconstruction.

Confirmed observations:

- The slowdown occurs without GCS preprocessing and before random rounding.
  Equal-weight route sampling cannot explain these diagnostic failures.
- Clarabel did advance through iterations. Its recorded termination is MaxTime,
  not evidence of an infinite loop, infeasibility, or an implementation deadlock.
- Clarabel's formulation has 15695 variables, 213040 constraint rows,
  4525222 nonzeros in A, and 1961 second-order cones. Its log reports direct
  QDLDL linear algebra with one thread. The initial iteration is expensive and
  feasibility residuals are still large at the recorded time limit.
- MOSEK solves the same graph relaxation without GCS preprocessing. Its log
  reports its own presolve, solution of the dual formulation, and different
  factorization setup. Even forcing one thread succeeds in about 19 seconds,
  so the difference is not solely caused by MOSEK using more threads.
- MOSEK no-preprocessing objectives were 1.1238846645 (default threads) and
  1.1238832555 (one thread); they are relaxation values for FIRST target index
  0, not the final selected target or the final feasible path cost 0.465795.

Interpretation: this installation's Clarabel configuration is substantially
less efficient on this large sparse conic model. Iteration cost and slower
numerical convergence explain the observed waiting. A unique deeper cause
(factorization fill, scaling, ill-conditioning, or a specific solver option)
has not been isolated by profiling or numerical ablation. In particular,
equal reliability values have not been established as the cause.

Evidence files (each JSON has a corresponding full `.log`):

- `relaxation_clarabel_no_preprocessing.json`
- `relaxation_mosek_no_preprocessing.json`
- `relaxation_clarabel_mosek_preprocessing.json`
- `relaxation_mosek_single_thread.json`

Example reproduction in the existing container:

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u planar3R_diagnose_relaxation.py --solver clarabel --preprocessing-solver mosek --time-limit 60 --watchdog-seconds 100 --output NEW_DIAGNOSIS.json
```

The license environment enables MOSEK for the explicitly selected preprocessing
solver; it does not override the explicit Clarabel relaxation solver.
No production solver selection, tolerances, costs, or sampling defaults were
changed by these diagnostics. The complete 28-target planner was not rerun.
