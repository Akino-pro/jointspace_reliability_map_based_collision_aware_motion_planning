# Goal-set planning comparisons

Matched-stage-two replays now reuse the saved stage-one endpoint for each
completed sweep case and set stage two to that case's own budget. See
[the final matched-budget summary](../results/diagnostics/stage2_matched_budget_summary_20261029/README.md).
The unfinished4R first-stage sweep remains paused. Older joint reports omitted
the first-stage route,so their endpoint-only replays have no extra retained
route;this caveat is explicit in the summary. Future joint reports now save
the first-stage route,route shifts and waypoints for complete replayability.

The matching [per-target five-budget sweep](../results/planar3r/per_target_budget_sweep_20261029/README.md)
uses each configured budget in full for every target. It is separate from
the earlier shared-total-budget ablation and reuses the joint sweep only for
comparison, without rerunning other methods.

A five-budget experiment for the joint-hull method alone is available in
[the budget-sweep report](../results/planar3r/union_hull_budget_sweep_20261029/README.md).
It varies first-stage budgets from50/75 through400/600 while fixing stage two
at200/300. All five runs returned the same validated final cost; timings and
the standalone cost/time figure are saved with the per-case reports.

An optional equal-total-candidate-budget ablation for planar 3R is now
documented in [shared GCS budgets](SHARED_GCS_BUDGET.md). It runs only the
independent and joint-hull methods and preserves the four-method records below.

The experiment matrix is now complete as of 2026-10-28: the original forward
3R example was run with all four methods under the current union-recovery
rule. See [the final 12-entry comparison](../results/diagnostics/final_four_method_comparisons_20261028.md).
Ten combinations returned validated paths; the 3R nonlinear and 4R joint-hull
methods returned no feasible paths in their recorded runs. The earlier
"planar not rerun" notes below describe the 2026-10-26 session only.

All three pre-failure planners (planar 3R, spatial 4R and full-pose spatial 7R)
support three independent comparisons. The main per-target two-stage algorithm
is unchanged. Enable individual methods with `--compare-sampled`,
`--compare-union-hulls`, `--compare-nonlinear-goal`, or all three with
`--compare-all`. Existing outputs are not overwritten when a separate `--output`
is supplied.

| Method | First solve | Actual task endpoint | Top-level GCS calls |
| --- | --- | --- | ---: |
| Main method | One independent solve per hull/region target | Local task projection followed by fixed-goal replan | N+1 |
| Sampled baseline (existing) | All safe covered samples in one graph | Selected sample | 1 |
| Union-hull baseline (new) | All existing hull/region targets in one graph | Same local projection and fixed-goal replan as main | 2 |
| Nonlinear-goal baseline (new) | Convex relaxation followed by native nonlinear restrictions | Real task equality enforced while optimizing the entire chosen route | 1 |

## Union-hull comparison

The targets remain separate convex nodes, connected to a common bookkeeping
sink. Their union need not be convex; no large convex hull is constructed.
The selected terminal's physical motion is charged using its containing
region's weight, and the artificial sink is excluded from the returned path.
Projection uses the same selected local SSM envelope and free region as the
main method. The second GCS uses the same fixed endpoint and retains the first
route as an additional convex restriction; it is not a direct connector stitch.

Thus this is an ablation of per-target solves versus one joint target solve.
Exact first-stage optima would agree, but relaxation and finite rounding can
produce different returned candidates. One solve is a larger graph; it need
not be N times faster. If projection fails, the comparison reports failure
without silently changing targets or borrowing the main result.

## Native nonlinear-goal comparison

The implementation uses Drake's `GraphOfConvexSets.Transcription.kRestriction`
to place the real endpoint task equality in every nonlinear restriction:

- 3R: both Cartesian position coordinates.
- 4R: all three Cartesian position coordinates.
- 7R: three position coordinates and three SO(3) logarithm residual components,
  with the same full goal pose and analytic local Jacobian as the main method.

Convex terminals cover **all connected free regions**, not just those hit by
sample hulls. The relaxation drops the task equality and solves the convex
graph. This is a valid outer surrogate for the task inside the same graph,
but often a very weak one: relaxed flows can favor terminals near the source
and fail to explore useful routes. There is no claim of reproducing the
strongest surrogate design in the literature. Sample hulls are deliberately
not imposed on nonlinear endpoints: they can miss the actual SSM between
samples. The nonlinear endpoint can move throughout its selected free region.

Drake samples routes from that relaxation and solves each restriction with
SNOPT (default) or IPOPT (`--nonlinear-solver ipopt`). The route's intermediate
points and endpoint are optimized jointly. There is no endpoint projection,
fixed-goal second GCS, or main-method solution used as a seed. The conic solver
for the relaxation is unchanged. The restriction iteration budget is 500;
SNOPT feasibility tolerance is 1e-8 and optimality tolerance is 1e-7. IPOPT
uses tolerance 1e-8 and a per-restriction CPU cap of 60 seconds. Unsupported
solvers and failed restrictions are explicitly reported. This method requires
relaxation/rounding; with `--no-rounding` it reports `unsupported_solve_mode`.

Native API references:

- [GCS shortest paths and nonconvex restriction support](https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1optimization_1_1_graph_of_convex_sets.html)
- [Target subgraphs containing multiple convex sets](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_gcs_trajectory_optimization.html)
- [Wrangel and Tedrake, IROS 2024](https://doi.org/10.1109/IROS58592.2024.10802426)

## Fairness and saved results

All methods share the scene, connected IRIS/reliability regions, physical edge
costs, starting chart and per-call rounding settings. Before failure, costs
remain `(0.01 + 1 - R) * joint-space length`. Main and comparisons never replace
or seed each other's results. All comparisons run before the main stage one,
so an ordinary main-method failure still leaves their reports available.

`union_hull_baseline` and `nonlinear_goal_baseline` record success/failure,
cost, physical waypoints, graph membership, sampled collisions, task residual,
recomputed weighted cost, solver/budget information, timing and a common
graph/task fingerprint. Paths restore input winding offsets. Primary solve
counts exclude comparisons; each comparison records its own GCS call count.
Bounds are graph-specific; neither finite sampling nor local nonlinear solves
certify the globally best continuous task endpoint.

`two_stage_total_seconds` excludes **all** enabled comparison times. Summaries
add shared preprocessing to each method's incremental time once. This charges
the nonlinear baseline for the shared SSM/map construction even though its
terminal constraints do not need the sampled hulls; report that convention
when comparing performance. Per-call budgets are the same, but total numbers
of restrictions need not be equal. Single sequential runs are not timing
statistics. No extra IRIS generation or reference workspace grid is required.

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_two_stage_gcs_planner --compare-all --output results/spatial7r/spatial7R_30cm20deg_all_goal_comparisons.json
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --compare-all --output results/spatial4r/spatial4R_all_goal_comparisons.json
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.planar3R_two_stage_gcs_planner --compare-all --output results/planar3r/all_goal_comparisons.json
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.summarize_planner_comparisons results/spatial7r/spatial7R_30cm20deg_all_goal_comparisons.json --output results/diagnostics/all_goal_comparisons_summary.json
```

The planar union-recovery graph has previously reached Docker's memory budget;
the commands above document available entry points, not completed experiments.
See measured reports for what actually ran. Existing 3R periodic behavior is
preserved; new comparisons retain route shifts and validate continuous lifts.

## Measured 7R comparison (2026-10-26)

The unchanged 30 cm / 20-degree task was rerun with its existing IRIS cache,
union-of-branch-recovery reliability rule, epsilon 0.025, arc spacing 0.044,
MOSEK relaxation and 200 paths / 300 trials. The nonlinear restrictions used
SNOPT. Shared preprocessing took 1.5622 seconds. All four final paths passed
their task and collision checks.

| Method | Final weighted cost | Time including shared preprocessing | GCS calls |
| --- | ---: | ---: | ---: |
| Main, per-target two-stage | 0.0043086899876 | 14.06 s | 5 |
| Sampled baseline | 0.0040315217986 | 3.87 s | 1 |
| Union-hull two-stage | 0.0043086899876 | 5.32 s | 2 |
| Nonlinear-goal | 0.0039795183786 | 120.78 s | 1 |

The union-hull solve considered four targets and returned the same final goal
and cost as main. Its first stage generated eight unique rounded paths; final
stage generated one. The nonlinear graph had 24 terminal regions and 200
sampled restrictions. Its final full-pose residual norm was 3.39e-10. Its lower
bound was only 3.35e-10, so the cheaper feasible path is not a global optimality
certificate. Main's fixed-goal bound likewise does not certify its continuous
endpoint selection. Timings are a single sequential measurement and exclude
fresh IRIS generation. The task remains a local example without a forced
obstacle detour.

Report: `results/spatial7r/spatial7R_30cm20deg_all_goal_comparisons.json`.
Log: `logs/spatial7R_all_goal_comparisons.log`.

## Measured spatial 4R comparison (2026-10-26)

The ftwmax midpoint task was rerun with the existing local IRIS cache,
the same union-recovery map and MOSEK 200/300. Shared preprocessing took
2.3517 seconds; the graph had 56 regions and 13 hull targets.

| Method | Final weighted cost | Time including shared preprocessing | Status |
| --- | ---: | ---: | --- |
| Main, per-target two-stage | 0.0780798316713 | 110.10 s | Validated |
| Sampled baseline | 0.0920764847952 | 14.16 s | Validated |
| Union-hull two-stage | — | 9.80 s | No feasible rounded first-stage path returned |
| Nonlinear-goal | 0.2813945055571 | 129.63 s | Validated |

The union-hull failure is not proof of target infeasibility. No alternate
target selection, solver retry or primary-result substitution was used.
The nonlinear method's true-task residual norm was 7.96e-14, but its task-free
relaxation bound was only 1.12e-10. It found a worse feasible route than main;
the weak surrogate and finite route sampling remain limitations. These results
must be retained alongside the favorable local 7R result.

Report: `results/spatial4r/spatial4R_all_goal_comparisons.json`.
Combined summary: `results/diagnostics/all_goal_comparisons_summary.json`.

Eight new comparison tests passed, including native nonlinear tasks on real
3R/4R/7R models, union-target selection, periodic routes, task AutoDiff, explicit
failure handling and winding restoration. Full Docker regression: 161 tests,
155 passed and six optional MuJoCo tests skipped. Full real planar benchmark
was not rerun in this session.
