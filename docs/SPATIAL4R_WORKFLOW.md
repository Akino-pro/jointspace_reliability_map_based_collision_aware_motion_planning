# Spatial 4R pre-failure planning

## Up to four SSM branches (2026-10-22)

The spatial planner applies the same multi-branch workflow as planar 3R to one
through four detected regular closed SSM branches. `scripts/spatial4R_model.py` exposes
`MAX_SSM_BRANCHES=4`, `MAX_SSM_IK_ATTEMPTS=80`, `SSM_TRACE_STEP=0.025` and
`SSM_DUPLICATE_TOLERANCE=0.005`. The previous IK budget was 20. Use
`--ssm-ik-attempts N --ssm-seed S` to control the search without changing GCS
rounding settings. Search stops at four distinct traces or the attempt budget.
Period-aware point-to-trace-segment comparison replaces the old 0.06 vertex
distance rule, so sample spacing does not require a broad deduplication radius.
This is still numerical deduplication and multistart search, not proof of branch
completeness; reports retain `all_branches_certified=false` and record the stop
reason, failed IK/traces and duplicate starts.

Each branch yields a 4D bounding-box partition of up to 3^4=81 cells. Successive
branches refine the existing partition; they do not create independent cost
maps. Each common cell unions the recoverable joint IDs from all branches and
evaluates its conditional reliability once on that union. Different single-joint
failures may recover to different branches; this does not create a new physical
SSM branch or join target hulls across branches. Equal-reliability neighbors are merged only when their
union is exactly a box, then intersected with the original IRIS regions.

All detected branches use the same arc spacing (0.044 by default), so the total
sample count grows with their total length. Hulls remain branch-local and split
at periodic seams. Every effective target intersection from every branch enters
the existing first-stage independent GCS loop. Projection uses only the selected
local branch; stage two uses the estimated fixed configuration, with the
retained first-stage route as an additional candidate. Both budgets remain
200/300. Branches outside the connected IRIS cover still contribute to the
reliability map, but naturally have no effective target intersections there.

Reports include `ssm_branch_summaries` with each branch's bounding box, arc
length, sample count, hull count and effective target count. Each first-stage
candidate records its `ssm_branch` index (zero-based).

Validation: all 13 existing spatial tests and six new multi-branch tests passed.
Controlled four-branch fixtures verify discovery/capping, periodic deduplication,
unioned recovery sets, branch-order invariance, convex merging, common arc density
and all-branch target inclusion. These fixtures are not physical four-branch
robot examples. A historical max-rule benchmark at the real ftwmax target
[0.9487804194,-1.2546894825,-0.5848754942] produced two distinct closed branches,
625 raw reliability boxes merged to 205, arc lengths 15.85076 and 19.05339 rad,
and 364+436=800 samples at spacing 0.044. Its hull counts are 42+52=94 and maximum
traced FK residual is 9.97e-10. See `results/spatial4r/spatial4R_ftwmax_multibranch_validation.json`
and the raw traces in `data/ssm/spatial4R_ftwmax_multibranch_ssm_example.npz`. This is an
SSM/reliability preprocessing check, not a new GCS planning experiment.

The current midpoint-obstacle task was also reprocessed with the new defaults:
one detected branch after 80 attempts, 153 samples, 56 connected free regions,
1202 edges and 13 targets, unchanged geometry. See
`results/spatial4r/spatial4R_ftwmax_midpoint_multibranch_geometry.json` (2.39 s, zero GCS calls).
Its existing successful trajectory and visualization remain applicable.

## Current obstacle placement (2026-10-22)

The default example now places the radius-0.10 sphere at the Cartesian midpoint
of the actual start and goal positions. For ftwmax the center is
[1.4594625111524193, 2.1170965052121957, -0.5076625057796179]. The start and
reference goal configurations remain collision-free; the old saved path collides
with this sphere and must not be reused with the new scene. Custom task endpoints
and the selected robot determine the midpoint at planning time.

New scene files use `midpoint` in their names. Existing `local` caches and saved
paths retain their historical geometry. Use `--legacy-obstacle` with an explicit
historical cache to reproduce the old sphere at [0.5,0,0.3]. The tables below
describe those historical experiments, not the midpoint scene.

Straight interpolation seeds left the first two IRIS regions disconnected.
`data/seeds/spatial4R_ftwmax_midpoint_iris_seeds.json` supplies a sampled collision-free
detour corridor solely for IRIS inflation (not as a replacement final path).
Its via configuration is [0.6751336532,-0.0439006786,0.7859599663,-0.3347099340];
13 seeds were placed along the two segments after a deterministic seed-7 search.
Three new IRIS regions cover the corridor. Their 15,000 independent hit-and-run
samples had no collisions. Reliability subdivision gives 56 connected free
regions, 1202 directed edges and 13 effective SSM targets, with 153 SSM samples.

The successful fresh run took 179.09 s: IRIS 70.70 s, first stage 99.38 s and
second stage 7.28 s. Both stages use 200/300. Eleven of thirteen first-stage
targets returned feasible paths; targets 11 and 13 returned no rounded solution,
so the aggregate first-stage bound is unavailable. Target 8 won with cost
0.0911111046. Full fixed-goal replanning reduced the final cost to
0.07807983167132608; the retained first-stage route costs 0.0911111508.
The final graph lower bound is 0.0688190746, relative gap 13.4567%. This is a
validated feasible result, not a global optimality certificate. Path checks:
102 configurations, no collisions, endpoint error 9.16e-16. The initial
disconnected attempt took another 47.21 s and is not included in 179.09 s.
MuJoCo export: 71 frames, 88 replay checks, no analytic collisions or contacts.

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --generate-iris --iris-seeds data/seeds/spatial4R_ftwmax_midpoint_iris_seeds.json
python -m scripts.spatial4R_mujoco_viewer
```

Default cache: `data/iris/spatial4R_ftwmax_iris_midpoint_regions.npz`.
Default report: `results/spatial4r/spatial4R_ftwmax_midpoint_two_stage_path.json`.
IRIS seeds are only needed to regenerate the cache, not to reuse it. The initial
disconnected attempt is preserved in `results/spatial4r/spatial4R_ftwmax_midpoint_initial_disconnected.json`
and `data/iris/spatial4R_ftwmax_iris_midpoint_disconnected_regions.npz`. Missing paths in a
limited IRIS cover do not prove that no collision-free physical path exists.

## Current first-robot experiment: ftwmax

The default robot is now the FIRST reference model, ftwmax. `--robot rcmax`
selects the second model for reproducing the historical results below. Each
model has separate default cache/output names. The source is parsed without
execution; only numeric/list arithmetic and np.pi are evaluated. The default
rounding budgets remain 200/300; 1000/1500 was explicitly tested.

The first robot uses a=[0.5,0.48,0.76,0.95], d=[-0.29,0,0.05,1], and
alpha=[85,-53,-89,68] degrees. Reliabilities [0.9,0.7,0.5,0.3], no CA limits.
With the same start/reference angles and sphere, its start XYZ is
[2.0780462171,1.6396001579,-0.3902624283] and goal XYZ is
[0.8408788052,2.5945928525,-0.6250625833]. Found 1 SSM branch, 153 samples,
32 local hulls, 11 targets, 16 free regions and 240 directed edges. Both runs
use the same local IRIS cache (radius 0.9), epsilon 0.025, arc spacing 0.044,
MOSEK, seed 0, and retained first-stage route.

| Metric | Both stages 200/300 | Both stages 1000/1500 |
| --- | ---: | ---: |
| First-stage winning target (1-based) | 7 | 6 |
| First-stage best cost | 0.07676953521773118 | 0.07675044859275859 |
| Final fixed-goal cost | 0.07676821560717718 | 0.07685758311854818 |
| First-stage time | 17.60 s | 40.56 s |
| Second-stage time | 0.83 s | 2.28 s |
| GCS stage time sum | 18.43 s | 42.83 s |
| Final graph gap | 11.6968% | 11.6822% |

Every first-stage solve and stage two collected the requested 200 or 1000
distinct routes. Larger sampling improved one first-stage target enough to
change the winner; its best cost dropped 0.0249%. Projection then selected
different fixed goal configurations. Final cost rose 0.1164%; this does NOT
mean more sampled routes worsen the SAME fixed-endpoint optimization problem.
Native and retained-route final costs tie in both experiments. The algorithm's
estimated endpoint selection is not guaranteed to improve monotonically.

The 1000/1500 run took 61.15s including 17.41s of IRIS generation; the 200/300
run reused that cache and took 19.23s. Compare GCS times above rather than
these unequal cold/cache totals. Both paths passed collision/FK checks. The
1000 result passed 95 collision samples with FK error 9.16e-16; the 200 result
passed 93 with FK error 1.11e-16. The ftwmax IRIS cache passed 5000 independent
collision samples, and all 80 tests passed with ftwmax as the active model.

Files: results/spatial4r/spatial4R_ftwmax_paths1000_trials1500.json,
results/spatial4r/spatial4R_ftwmax_paths200_trials300.json, data/iris/spatial4R_ftwmax_iris_local_regions.npz,
spatial4R_ftwmax_iris_validation.json.

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --robot ftwmax --generate-iris --max-rounded-paths 1000 --max-rounding-trials 1500 --output NEW_FTWMAX_RESULT.json
```

The sections below retain the historical second-robot (rcmax) results. To
reproduce them, explicitly pass `--robot rcmax --legacy-obstacle --iris-regions data/iris/spatial4R_iris_local_regions.npz`
and a NEW output filename. The old unprefixed rcmax cache is preserved.

The implemented workflow is fixed-start, position-only planning through 4D
collision-free IRIS regions, joint-reliability subdivision, adaptive SSM target
selection, and a final fixed-goal GCS solve. It retains the selected stage-one
route as an additional convex restriction in stage two. Single locked-joint
post-failure planning is now available separately; see
[docs/SPATIAL4R_POST_FAILURE_WORKFLOW.md](SPATIAL4R_POST_FAILURE_WORKFLOW.md).

## Reference robot

`scripts/spatial4R_model.py` reads the selected top-level constant assignments from
`scripts/spatial_4r_reliability_maps_proposed_method.py` using Python AST. It does not
import or execute that script, start its grid computation, or require Robotics
Toolbox/SpatialMath. This historical table describes rcmax, which overrides
ftwmax in the original script but is not the current planner default:

| Joint | Standard-DH a | d | alpha (rad) | Reliability |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.04732540449946272 | -0.0020589855582313892 | 1.1599121046989387 | 0.9 |
| 2 | 0.8501471809443848 | -0.12359280272943952 | -0.5154988273227061 | 0.7 |
| 3 | 0.0018628495298751034 | 0.0025723721804752 | 0.03775362985949342 | 0.5 |
| 4 | 0.8495872435309159 | 0.04538990560835715 | 0.5365523262566805 | 0.3 |

The original CA array is recorded as reference metadata but is NOT applied as
artificial or mechanical joint limits, following the current planar workflow.
Periodic angles use the finite chart [-pi,pi]^4; opposite chart boundaries are
not connected. The task constrains XYZ position, leaving a generic 1D SSM.

Both the standard-DH axial offsets d and link lengths a are represented as
capsule segments of radius 0.015. Collision checks consider spherical obstacles.
Self-collision is explicitly disabled, consistently in Drake and the analytic
checker. This is a model assumption, not a full hardware collision model.

## Planning and configuration

- IK with the analytic 3x4 position Jacobian, numerical nullspace continuation,
  periodic branch duplicate detection, and goal equality refinement.
- Up to four traced branches, with 20 IK attempts. This search is not complete;
  a failed trace is reported and is not proof of physical unreachability.
- All found branches are sampled with shared arc interval 0.044 radians. Seam
  pieces are sampled independently and interpolated configurations refined.
- Local hull epsilon 0.025. The hull/projection machinery supports four joint
  coordinates; sampled hulls are not certified continuous SSM outer envelopes.
- Each branch induces 3^4=81 reliability boxes. Branch maps are overlaid and
  their joint-recovery sets are unioned before computing conditional reliability.
  All 15 nonempty-condition
  values match the reference script's formulas exactly. The no-condition value
  is zero. Equal-reliability cells merge only when their union is exactly a box.
- Intersect these boxes with IRIS. Drop an original IRIS halfspace only when
  its maximum over the entire box is strictly below its bound; this eliminates
  box-implied rows without expanding the feasible intersection.
- For N target intersections, stage one performs N independent GCS solves,
  projects only the lowest-cost returned endpoint onto the corresponding local
  SSM, and stage two runs full fixed-start-to-fixed-goal GCS with the retained
  region route as an extra candidate. No projection connector is used as the
  final solution; its cost is saved only as a comparison.
- Both stages request 200 distinct candidate routes / 300 sampling trials,
  seed 0. Fewer than 200 routes can be returned when duplicates consume trials.
- Region cost is `(0.01 + 1 - R) * Euclidean joint-space length` in four
  dimensions. MOSEK is explicitly selected for the spatial example; this is
  continuous convex relaxation plus rounding, NOT mixed-integer search.
- Reports retain graph relaxation bounds and gaps. Missing coverage, numerical
  solver failure, and physical radial unreachability have different statuses.

Main files: `scripts/spatial4R_model.py`, `scripts/spatial4R_iris.py`,
`scripts/spatial4R_two_stage_gcs_planner.py`, `scripts/spatial4R_validate_iris.py`.
Shared geometry and GCS helpers in the planar-named modules were generalized
to joint-space dimension without changing the planar robot model.

## Tested scene and coverage limits

The source script is a reliability-map experiment, not a point-to-point collision
scene. This new deterministic test uses:

- Start configuration `[0.2,-0.8,0.6,-0.3]`, with position
  `[1.3921760958,0.0630204588,-1.0303980729]`.
- Goal position `[1.0048114470,1.3268089240,-0.3027628522]`, generated by reference
  configuration `[0.8,-0.4,0.2,0.4]`. That configuration seeds IK/IRIS but is NOT
  imposed as the final goal configuration.
- Sphere centered at `[0.5,0,0.3]`, radius 0.10; capsule radius 0.015.

An initial larger-domain IRIS cache `data/iris/spatial4R_iris_regions.npz` generated one
region in 90.41 s. Its reliability graph had 207 regions and 3890 directed edges,
with 20 targets. Full optimization did not complete: a run was killed by the
container OOM mechanism (about 7.57 GiB RAM available). A preceding numerical
attempt returned no path for its first target; this is not an infeasibility
proof. The larger cache is preserved; no completed full-chart result is claimed.

The successful first prototype instead uses local IRIS inflation domains of
seed +/-0.9 radians, clipped to the coordinate chart. The cache is
`data/iris/spatial4R_iris_local_regions.npz`. These are FINITE COVERAGE domains for region
generation, not physical limits. No success/failure or optimality conclusion
extends to configurations outside the represented regions. In particular,
tracing both SSM branches does not mean both are fully covered by this cache.

## Completed results

- 2 SSM branches, lengths 8.8865763 and 8.8867140 radians; 205 and 204 samples,
  409 total, zero sample-refinement failures, 10 local hulls.
- 625 overlay boxes -> 247 merged boxes. Local IRIS intersection gives
  21 connected free regions, 104 directed edges, and 3 effective target sets.
- Three first-stage GCS calls plus one final call and one extra convex
  restriction. First-stage route counts 113, 105, 200; final count 122.
- Selected target index 1, SSM branch index 0. First-stage cost
  0.06601626039626814; projection distance 0.002982211080047504.
- Final goal configuration
  `[0.7882687205502752,-0.39373734534478555,0.7531786903199453,-0.1474218192119493]`.
- Final cost 0.0660305054225731. Native result and retained route [1,9,19]
  tie at this cost. Goal FK error 7.04e-16, start/fixed-goal configuration errors
  zero; collision validation passes 99 path samples.
- Fixed-goal lower bound 0.05765536717912147; gap (U-L)/L = 14.5262%, so the
  1% criterion is not met. Global endpoint optimality is not certified.
- First run including local IRIS generation: 32.96 s (IRIS itself 25.66 s).
  Reusing cache: geometry 0.98 s, stage one 5.36 s, stage two 0.44 s,
  total 6.83 s. These are single measurements, not guaranteed timings.

Outputs: `results/spatial4r/spatial4R_two_stage_path.json` and
`results/spatial4r/spatial4R_two_stage_cached_path.json`. The larger-domain preprocessing-only
report remains in `results/spatial4r/spatial4R_geometry.json`; its counts differ intentionally.

All 79 tests passed, including prior planar tests and 11 new spatial checks:
standard-DH/Jacobian correctness, all 15 reference reliability values, 4D hulls,
partition merging, safe halfspace removal, closed SSM FK equality, stop conditions,
and real 4D GCS cost accounting. Drake tool transforms and collision results
match the analytic model at 100 random configurations. Independent hit-and-run
tests found zero collisions in 5000 points for EACH of the larger and local
caches; these are sampled checks, not continuous certificates.

## Run

```powershell
docker start planar3r-iris-validation
# Existing license has been copied to /tmp/mosek.lic inside this container.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --robot rcmax --iris-regions data/iris/spatial4R_iris_local_regions.npz --generate-iris --output NEW_RCMAX_RESULT.json
# On later runs, omit --generate-iris to load the local cache directly.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.spatial4R_two_stage_gcs_planner --robot rcmax --iris-regions data/iris/spatial4R_iris_local_regions.npz --output NEW_RESULT.json
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial4R_validate_iris --robot rcmax --iris-regions data/iris/spatial4R_iris_local_regions.npz --output NEW_VALIDATION.json
docker stop planar3r-iris-validation
```

On another machine, copy the license privately into its container; never commit
the license. `--solver clarabel` is available, but the larger case's resource
issues have not been resolved by this prototype. Generation options include
`--iris-radius`, `--iris-regions`; planning accepts `--q` (4 angles), `--goal`
(3 coordinates), `--geometric-epsilon`, `--ssm-arc-spacing`, and both rounding
budgets. Custom tasks may need new IRIS seeds/coverage; the default seed
corridor and reference configuration are specific to this tested scene.
