# IRIS + reliability + GCS

The original pre-failure planning workflow is: generate collision-free IRIS polytopes,
intersect them with SSM bounding-box reliability partitions, filter by
connectivity, solve with Drake GCS, and independently validate the path.
Grid region generation and the fallback shortest-path planner have been removed.

The additional fixed-start two-stage planner reuses these IRIS regions and cost
partitions with adaptive SSM target hulls. See
[docs/TWO_STAGE_SSM_WORKFLOW.md](TWO_STAGE_SSM_WORKFLOW.md).

## Verified state

- Docker image: `robotlocomotion/drake`; its digest is pinned in `config/Dockerfile.drake`.
- Container Python: `/usr/bin/python3`, version 3.12; Drake version `0.0.20260601`.
- Current checkpoint `data/iris/planar3R_iris_regions.npz`: six IRIS regions, with 6.41 seconds spent generating successful regions. This excludes SSM computation, failed seeds, and process startup.
- Current generation settings: five iterations, 100 consecutive collision-infeasible optimization attempts, configuration-space margin 0.02, and link radius 0.02.
- Hit-and-run sampling of region interiors: no collisions detected in 30,000 configurations. Results are in `results/planar3r/iris_region_validation.json`. This is numerical sampling, not a continuous collision-free proof.
- The full current-checkpoint solve completed on 2026-10-13. The latest `results/planar3r/iris_path.json` uses the current generation settings: six original regions -> 82 reliability regions -> 43 solve regions. The filtered graph has 824 directed region-adjacency edges.
- This single run took 30.33 seconds in total: 0.067 seconds for reliability intersections, 0.951 seconds for adjacency construction, and 27.36 seconds for the selected candidate's GCS construction, solve, and path validation. One start/goal candidate pair was used; the cost was 0.582765.
- No collisions were detected in 668 sampled path configurations; endpoint error was `2.936e-13`. These are numerical checks, not a continuous collision-free proof.
- Earlier region sampling found two colliding configurations. The checkpoint was regenerated with stronger settings and replaced. The old-settings run used 51 regions and took approximately 72.8 seconds; this is historical information, not a repeatable benchmark for the current settings.
- Grid testing was stopped at the user's request; no further grid comparisons are performed.

## Using the configured container

The configured container is named `planar3r-iris-validation`. It is stopped,
retains its installed dependencies, and mounts the project at `/workspace`.
Run these commands in PowerShell:

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_validate_iris
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_gcs_planner
docker exec planar3r-iris-validation /usr/bin/python3 -m unittest discover -s tests -v
docker stop planar3r-iris-validation
```

By default, planning reads the IRIS checkpoint, uses one start/goal candidate
pair, and writes `results/planar3r/iris_path.json`. Use `--candidate-limit 3` to consider more
combinations at additional runtime cost. `--sx/--sy/--gx/--gy` set workspace
endpoints; `--reliability-source start|goal|none` selects the partition source.

The current robot is assumed to have unrestricted revolute joints. No extra
joint-angle box is imposed on endpoints, regions, or paths. The IRIS checkpoint
uses `[-pi,pi]^3` as a finite coordinate chart for periodic angles, not as a
mechanical limit. Opposite chart boundaries are not connected in the current
graph. Workspace endpoints outside the robot's radial reach are rejected before
SSM tracing or GCS construction.

Resume generation, reusing completed seeds without overwriting a checkpoint
whose settings do not match:

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_drake_iris --resume
```

The current checkpoint has processed all 22 candidate seeds. Increasing
`--max-regions` alone will therefore not produce additional regions. Change the
seed strategy and select a new output file to expand coverage. Changed generation
settings also require a new `--output`.

## Rebuilding the runtime from the image

Open PowerShell in the project root:

```powershell
docker build -f config/Dockerfile.drake -t planar3r-drake .
docker run --rm -it -v "${PWD}:/workspace" planar3r-drake
```

Then run inside the container:

```sh
/usr/bin/python3 -m scripts.planar3R_gcs_planner
/usr/bin/python3 -m scripts.planar3R_validate_iris
```

The Windows PyCharm Python 3.10 environment contains an unrelated package also
named `pydrake`, which cannot run robotics Drake. Use Docker for this workflow.

## Implementation details

IRIS retains its original halfspace constraints and adds the six inequalities
of each reliability box directly. Slanted boundaries are not replaced by
enclosing boxes. A Chebyshev LP removes empty or numerically flat intersections.
Adjacency includes overlaps and boundary contacts.

Candidate configurations are refined to remove workspace drift from SSM
tracing, then selected from common connected components. Each start/goal pair
uses only its relevant regions. Native GCS rounding runs in a single call to
avoid solving the full convex relaxation twice.

The robot base is welded to world, and link collision geometry covers the full
link lengths. The circular obstacle is inflated by the link radius. The current
model supports the project's single circular obstacle. Robot self-collisions
are filtered to match the original analytic checker.

Reliability retains the approximation based on each SSM branch's joint-angle
bounding box, with joint reliabilities 0.7, 0.5, and 0.3. The workspace background
map uses different reliabilities and is not the planner's actual cost map.

Independent path checks follow straight GCS segments in joint space, with
sampling intervals of at most 0.01 rad. Angles are not wrapped along their
shortest direction, since that would change the optimized path. The old animation
script still uses hardcoded historical waypoints and does not read the JSON path;
it cannot display the current IRIS solution.

Reference implementations are located at
`D:/gcs-science-robotics/reproduction/bimanual/bimanual_iiwa_example.ipynb`,
`reproduction/prm_comparison/prm_comparison.ipynb`, and `gcs/linear.py`.
Current Drake renamed the old `IrisInConfigurationSpace` interface to `IrisNp`;
the generator supports both names.
