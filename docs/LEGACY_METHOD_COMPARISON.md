# Legacy-inspired graph improvements and sampled-goal baseline

The two-stage method remains the primary planner. These changes borrow ideas
from `obs_avoid_ft_mp_Landon` without copying its reliability model, random
probabilities, center-distance graph pruning or custom collision inflation.

## Adjacency screening

`scripts/planar3R_iris_regions.py::polytope_adjacency` now first checks
conservative axis-aligned bounds. Single-coordinate halfspaces supply inexpensive
outer bounds; missing coordinate bounds are obtained with LPs. Bounds expand
outward slightly for numerical tolerance. Only clearly separated pairs are
discarded. Remaining pairs still undergo the original full-halfspace feasibility
LP, retaining face, edge and point contacts. Solver failures are reported, not
silently classified as disjoint geometry.

This shared helper applies to planar 3R, spatial 4R/7R and post-failure slices.
It uses no new packages. This is an AABB broad phase, not an R-tree dependency;
pair screening remains quadratic, while expensive intersection LPs decrease.
It preserves actual graph edges and does not reduce a dense GCS graph's memory
requirements. `broad_phase=False` is available for exhaustive verification.
Reports include candidate/LP counts before source-component filtering; these
counts may refer to more regions than the final connected graph.

## Optional planar periodic connections

Historical and benchmark defaults keep periodic connections **off**. Enable
`--periodic` on either `scripts.planar3R_two_stage_gcs_planner` or
`scripts.planar3R_post_failure_planner`. There is no spatial 4R/7R periodic CLI.
Periodicity is an optional coverage extension, not a premise of the method.

For an edge from region U to region V with translation s, the physical segment
ends at xV+s in U's chart. It costs wU*norm(xV+s-xU) and must satisfy U's original
halfspaces there. The original V point stays in V's chart. All combinations of
single-period coordinate shifts are tested when their AABBs overlap, including
multi-axis seams. Parallel edges retain their separate shifts. No collision
polytope is replaced by its AABB or expanded for convenience.

Path extraction accumulates chart offsets to return continuous, unwrapped
angles. Segments are checked in their source charts; projections use the original
target chart. The retained first-stage route records both region IDs and shifts
so the second-stage convex restriction preserves its seam choice. The final
fixed-goal check compares physical angles modulo 2pi. After failure, only active
coordinates have periodic edges; the failed coordinate stays exactly constant.

This is not a complete universal-cover search: it connects distinct existing
regions and retains GCS simple-region paths. It does not represent all routes
that revisit a region in another winding, or an internal wrap within one sole
full-chart region. Missing routes remain a coverage limitation, not physical
infeasibility. This option makes no new global-optimality claim.

## Fair direct sampled-goal baseline

All three pre-failure entry points accept `--compare-sampled`. The common
implementation is `scripts/sampled_goal_baseline.py`. It receives the same fixed
start, already refined SSM envelope samples, connected free regions, reliability
weights and graph used by the primary planner. Epsilon, arc spacing, scene and
robot remain unchanged. It does not resample, change reliability aggregation,
prune routes heuristically or add IRIS coverage.

Every distinct collision-free sample covered by the connected graph becomes one
point terminal, attached to **all** containing regions. Duplicate samples are
removed, not valid alternatives. A zero-cost bookkeeping sink selects a terminal;
it is excluded from the returned physical trajectory. The baseline runs one
GCS with the same per-call stage-one solver, path/trial budgets and rounding seed.
Both methods therefore have the same per-call settings, not an equal total
number of convex restrictions: the primary method independently solves each
hull target and then runs its fixed-goal second stage.

The baseline validates source, goal task, region membership and collision samples,
and saves its path, cost, graph bound/gap, selected sample, solver settings, timing
and an input fingerprint. It runs after shared preprocessing and before stage
one, so a later primary failure does not erase its result. It never replaces or
seeds the primary solution. It is an adaptation of the legacy terminal-selection
approach onto our common graph, not an exact reproduction of the entire legacy
GGCS implementation.

`sampled_baseline.seconds` measures its incremental work, including point-terminal
preparation and validation. `two_stage_total_seconds` excludes that baseline work.
The summary adds common `geometry_seconds` to baseline time for a fair cached
end-to-end comparison. Fresh IRIS generation, when requested, is part of shared
preprocessing. Original cache-generation time is metadata and is not charged
again during a cached run. Runs are sequential single measurements, not a
statistical speed benchmark. Per-graph solver gaps do not certify the unknown
continuous-SSM optimum, and cost comparisons must account for failed targets.

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.planar3R_two_stage_gcs_planner --compare-sampled --output results/planar3r/two_stage_sampled_comparison.json
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --compare-sampled --output results/spatial4r/spatial4R_ftwmax_midpoint_sampled_comparison.json
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial7R_two_stage_gcs_planner --compare-sampled --output results/spatial7r/spatial7R_30cm20deg_sampled_comparison.json
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.summarize_planner_comparisons results/planar3r/two_stage_sampled_comparison.json results/spatial4r/spatial4R_ftwmax_midpoint_sampled_comparison.json results/spatial7r/spatial7R_30cm20deg_sampled_comparison.json
```

Historical result files are preserved; these commands create separate reports.

## Measured cached comparisons (2026-10-23)

Both methods used MOSEK, 200 paths / 300 trials per GCS call, and periodic
connections were disabled. Times include shared preprocessing for each method.

| Robot/task | Samples / covered point targets | Hull targets | Two-stage cost | Sampled cost | Two-stage time | Sampled time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Planar 3R | 301 / 91 | 20 | 0.3817177694 | 0.3817192602 | 110.15 s | 17.89 s |
| Spatial 4R, ftwmax midpoint | 153 / 53 | 13 | 0.0780798317 | 0.0920764848 | 119.72 s | 14.68 s |
| Spatial 7R, 30 cm / 20 degrees | 92 / 6 | 4 | 0.0043086900 | 0.0040315218 | 14.89 s | 5.40 s |

All six final paths passed task and collision checks. The 3R costs are very
close; 4R benefits from the primary method in this run. The 7R sampled baseline
is cheaper, exposing the primary first-stage target-selection limitation:
three of four hull targets returned no feasible rounded solution. In 4R two of
13 targets returned no solution. These failures do not prove target infeasibility.
Keep both reported methods independent rather than silently replacing the
primary result with the cheaper baseline. This experiment does not establish
which method is generally superior.

Adjacency LP counts before source-component filtering decreased from 3,321 to
1,883 in 3R, 1,540 to 724 in 4R, and 276 to 212 in 7R. No bounding LPs were
needed for these axis-bounded reliability cells. Final connected graphs retained
43/56/24 regions and 824/1,202/424 directed edges respectively. These counts
describe construction work, not a measured solver-speed improvement.

Machine-readable results and quality bounds are collected in
`results/diagnostics/planner_sampled_comparisons.json`. Baseline paths are stored
in each paired report's `sampled_baseline` field. They use the planner's internal
start chart with continuous lifts; a nonprincipal input winding can be restored
by adding the primary report's input-to-chart offset to every baseline waypoint.

## Verification

`tests/test_graph_improvements.py` checks screened/exhaustive adjacency equality,
lower-dimensional contacts, disjoint polytopes with overlapping AABBs, bounding
LP fallback, multi-axis periodic shifts, retained-route shifts, continuous path
length, all-containing-region terminal attachments, sample filtering and actual
MOSEK solves. End-to-end planar tests cross a seam through both primary stages
and through the locked-joint replanner. The full Docker suite ran 138 tests:
132 passed and six optional MuJoCo tests were skipped.
