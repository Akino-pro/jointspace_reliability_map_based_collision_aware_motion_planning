# Spatial 4R pre-failure planning

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

Files: spatial4R_ftwmax_paths1000_trials1500.json,
spatial4R_ftwmax_paths200_trials300.json, spatial4R_ftwmax_iris_local_regions.npz,
spatial4R_ftwmax_iris_validation.json.

```powershell
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u spatial4R_two_stage_gcs_planner.py --robot ftwmax --generate-iris --max-rounded-paths 1000 --max-rounding-trials 1500 --output NEW_FTWMAX_RESULT.json
```

The sections below retain the historical second-robot (rcmax) results. To
reproduce them, explicitly pass `--robot rcmax --iris-regions spatial4R_iris_local_regions.npz`
and a NEW output filename. The old unprefixed rcmax cache is preserved.

The implemented workflow is fixed-start, position-only planning through 4D
collision-free IRIS regions, joint-reliability subdivision, adaptive SSM target
selection, and a final fixed-goal GCS solve. It retains the selected stage-one
route as an additional convex restriction in stage two. Spatial post-failure
planning is not implemented by this addition.

## Reference robot

`spatial4R_model.py` reads the selected top-level constant assignments from
`spatial_4r_reliability_maps_proposed_method.py` using Python AST. It does not
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
  their maximum conditional reliability is used. All 15 nonempty-condition
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

Main files: `spatial4R_model.py`, `spatial4R_iris.py`,
`spatial4R_two_stage_gcs_planner.py`, `spatial4R_validate_iris.py`.
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

An initial larger-domain IRIS cache `spatial4R_iris_regions.npz` generated one
region in 90.41 s. Its reliability graph had 207 regions and 3890 directed edges,
with 20 targets. Full optimization did not complete: a run was killed by the
container OOM mechanism (about 7.57 GiB RAM available). A preceding numerical
attempt returned no path for its first target; this is not an infeasibility
proof. The larger cache is preserved; no completed full-chart result is claimed.

The successful first prototype instead uses local IRIS inflation domains of
seed +/-0.9 radians, clipped to the coordinate chart. The cache is
`spatial4R_iris_local_regions.npz`. These are FINITE COVERAGE domains for region
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

Outputs: `spatial4R_two_stage_path.json` and
`spatial4R_two_stage_cached_path.json`. The larger-domain preprocessing-only
report remains in `spatial4R_geometry.json`; its counts differ intentionally.

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
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u spatial4R_two_stage_gcs_planner.py --robot rcmax --iris-regions spatial4R_iris_local_regions.npz --generate-iris --output NEW_RCMAX_RESULT.json
# On later runs, omit --generate-iris to load the local cache directly.
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u spatial4R_two_stage_gcs_planner.py --robot rcmax --iris-regions spatial4R_iris_local_regions.npz --output NEW_RESULT.json
docker exec -w /workspace planar3r-iris-validation /usr/bin/python3 spatial4R_validate_iris.py --robot rcmax --iris-regions spatial4R_iris_local_regions.npz --output NEW_VALIDATION.json
docker stop planar3r-iris-validation
```

On another machine, copy the license privately into its container; never commit
the license. `--solver clarabel` is available, but the larger case's resource
issues have not been resolved by this prototype. Generation options include
`--iris-radius`, `--iris-regions`; planning accepts `--q` (4 angles), `--goal`
(3 coordinates), `--geometric-epsilon`, `--ssm-arc-spacing`, and both rounding
budgets. Custom tasks may need new IRIS seeds/coverage; the default seed
corridor and reference configuration are specific to this tested scene.
