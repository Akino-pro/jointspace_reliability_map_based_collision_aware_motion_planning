# File organization

The 2026-10-22 cleanup moved 106 root files into category folders. Models,
planning settings, numeric experiment results and cached IRIS geometry were
preserved. Imports, command examples, defaults and saved report path references
were updated together. The full old-root-name to new-relative-path map is in
[`config/file_locations.json`](../config/file_locations.json).

```text
repository/
  README.md
  scripts/                 Python models, workflows, viewers and diagnostics
  gcs/                     Shared GCS implementation
  tests/                   Unit and integration tests
  docs/                    Workflow guides and conversation handoff
  data/
    iris/                  IRIS .npz checkpoints, including historical attempts
    ssm/                   Saved SSM .npz traces
    seeds/                 Input seed JSON files
  results/
    planar3r/              Planar experiment JSON reports
    spatial4r/             Spatial experiment JSON reports
    spatial7r/             Seven-joint full-pose experiment JSON reports
    diagnostics/           Solver/environment reports and layout verification
  visualizations/          Scene-specific GIF/PNG/XML/replay bundles
    planar3R_legacy/       Previously loose root images and animation
  logs/                    Solver diagnosis logs
  config/
    Dockerfile.drake
    requirements-mujoco.txt
    file_locations.json
  licenses/                Local, Git-ignored MOSEK license
```

IDE and Git metadata retain their existing locations. Python `__pycache__`
directories are generated and ignored. Scene-specific visualization folders
remain self-contained: their XML model, pictures, replay report and viewer logs
belong together.

## Starting scripts

Use the repository root as the working directory. All formerly root-level
Python modules now live in the `scripts` package:

```powershell
python -m scripts.planar3R_mujoco_viewer
python -m scripts.spatial4R_mujoco_viewer
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_post_failure_path.json
```

In the existing headless Drake container, the same root is `/workspace`:

```powershell
docker start planar3r-iris-validation
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.planar3R_two_stage_gcs_planner
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial4R_two_stage_gcs_planner
docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -m scripts.spatial4R_post_failure_planner --failed-joint 4
```

The current precomputed IRIS defaults are in `data/iris/`. Future default JSON
outputs are placed under the corresponding `results/` subfolder. An explicit
`--output` is still honored; provide the desired directory yourself. Do not run
the old root filenames or `python scripts/FILE.py`: module execution supplies
the repository package context used by all imports.

The historical reference scripts are retained in `scripts/`, but this cleanup
did not run their grids. Standard workflows do not run those grid computations.

## Environment files

```powershell
docker build -f config/Dockerfile.drake -t planar3r-drake .
# Only if the optional native visualization dependencies are needed:
python -m pip install -r config/requirements-mujoco.txt
# When the container needs a refreshed license copy:
docker cp licenses/mosek.lic planar3r-iris-validation:/tmp/mosek.lic
```

No packages were installed or Docker images rebuilt during cleanup. The existing
container's `/tmp/mosek.lic` remains valid; its internal path did not change.

## Verification after moving

The Docker suite ran 110 tests: 104 passed and six MuJoCo-dependent tests skipped
in the headless environment. All ten host visualization tests passed, including
those six. Eight command entry points passed module/argument-parser checks.
The actual cached 3R/4R preprocessing produced the same 43/56 free regions and
20/13 effective targets. The actual 4R joint-4 replanner again produced cost
0.7353883585602623 and exact locked-angle preservation. See
[`results/diagnostics/layout_validation.json`](../results/diagnostics/layout_validation.json).

No commit or push was performed. Keep the file moves and import/path updates
together when uploading the project to another device.

The file audit also confirmed all 106 mapped targets exist, all local Markdown
links resolve, seven tracked binary artifacts are byte-identical, and 41 tracked
JSON reports retain their numeric structures. Both default native viewer inputs
and the successful spatial post-failure report load from their new locations.
See [`layout_file_audit.json`](../results/diagnostics/layout_file_audit.json).
