# Planar 3R and spatial 4R MuJoCo visualization

`scripts/planar3R_mujoco_viewer.py` loads a successful planar planning JSON and replays
the saved configurations in a native MuJoCo window. It does not run GCS again.
Pre-failure and single-locked-joint post-failure reports are supported. Four-joint
spatial paths are rejected by this planar viewer.

## Start

Run on the Windows host using the project interpreter with MuJoCo installed,
not inside the headless Drake container:

```powershell
python -m scripts.planar3R_mujoco_viewer
python -m scripts.planar3R_mujoco_viewer --path results/planar3r/two_stage_reverse_multibranch_merged_mosek.json
python -m scripts.planar3R_mujoco_viewer --path results/planar3r/post_failure_success.json --speed 0.1
```

The default path is `results/planar3r/two_stage_all_paths200_trials300.json`, resolved relative
to the script so the default also works when launched from another directory.
In the current IDE, the tested interpreter is
`C:\Users\14118\AppData\Local\Programs\Python\Python310\python.exe`, with
MuJoCo 3.4.0 already installed. No packages were installed for this addition.
The optional visualization dependencies are listed in config/requirements-mujoco.txt.
The Python/Drake planning environment is separate and does not need MuJoCo.

Controls:

- Space: pause/resume.
- R: restart the saved path.
- L: toggle looping. `--once` initially holds the goal after the first pass.
- T: show/hide the complete end-effector trace.
- Mouse: MuJoCo camera rotation/panning/zoom; Esc or window close exits.
- `--speed 0.5`: half-speed playback; `--angular-speed` sets the nominal
  joint-space arc speed (default 0.65 rad/s).

## Scene and interpretation

The viewer reads link lengths, collision radius, obstacles, goal and waypoints
from the report. Defaults for missing legacy length/radius fields are [1,1,1]
and 0.02; missing obstacles mean an empty scene. Joint hinges have no angle
limits. The three capsule links are blue, teal and orange. Circular obstacles
are extruded as red cylinders through the arm plane; green marks the goal,
cyan marks the initial tip position, and a pale blue curve shows the full
planned end-effector trace. The failed joint marker is red for post-failure
reports, and that coordinate must remain constant in the input trajectory.

Circle/capsule collision geometry matches the existing planar analytic model.
Box/AABB obstacles can also be shown at their supplied dimensions; the planner's
axis-wise radius inflation is conservative relative to capsule-box corners.
Ground, markers, mount and trace are visual-only. Robot self-collision is
disabled, matching the planner; only arm-obstacle contacts are enabled.

Playback directly sets qpos and calls mj_forward. It is prescribed kinematic
visualization, not a dynamics simulation or actuator/controller demonstration.
The actual GCS joint-space segments are linearly interpolated WITHOUT wrapping
joint differences: winding numbers and any rotations exceeding pi are retained.
Looping resets from goal to start; that reset is not an additional planned path.
Duplicate waypoints and singleton already-at-goal paths are supported.

## Export previews and model

```powershell
python -m scripts.planar3R_mujoco_viewer --render
python -m scripts.planar3R_mujoco_viewer --path results/planar3r/post_failure_success.json --render --output-dir visualizations/planar3R_mujoco/post_failure
```

The default output directory is `visualizations/planar3R_mujoco/`:

- `planar3R_scene.xml`: MuJoCo MJCF with arm, obstacles, markers and trajectory.
- `planar3R_mujoco.gif`: offscreen MuJoCo-rendered animation.
- `planar3R_mujoco_start.png`, `planar3R_mujoco_middle.png`,
  `planar3R_mujoco_goal.png`: rendered snapshots.
- `replay_report.json`: source, MuJoCo version, duration and sampled path checks.

Exports default to 960x720 and 24 fps; use `--width`, `--height` and `--fps`
to adjust. GIF timing is rounded to the format's centisecond resolution;
very short paths are stretched to at least one second for visibility.
`--viewer-seconds 3` automatically closes an interactive smoke test.

## Verification

Six dedicated tests pass on the native host. They check analytic/MuJoCo FK
agreement at 50 configurations with nondefault link lengths, contact agreement
with the circular-obstacle checker at 150 random configurations (including
colliding and free samples), winding-preserving interpolation, singleton and
duplicate handling, spatial-path rejection, and locked-joint preservation.
An actual passive viewer opened and closed successfully in a three-second
smoke test. No interactive key/mouse automation is claimed.

The default existing GCS result produced 228 GIF frames and passed 567 sampled
analytic collision checks with zero collisions; endpoint error 1.76e-13.
The existing results/planar3r/post_failure_success.json also rendered successfully with joint 3
locked and no sampled collisions. These are playback checks, not new planning
experiments or continuous collision certificates. Source trajectory JSON files
were not changed.

MuJoCo's official Python API is documented at
https://mujoco.readthedocs.io/en/stable/python.html.

## Spatial 4R viewer

Spatial single-locked-joint reports are also supported:
`python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_post_failure_path.json`.
The failed joint marker is red, and loading rejects a changing locked column.
See [docs/SPATIAL4R_POST_FAILURE_WORKFLOW.md](SPATIAL4R_POST_FAILURE_WORKFLOW.md).

```powershell
python -m scripts.spatial4R_mujoco_viewer
python -m scripts.spatial4R_mujoco_viewer --render
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_ftwmax_paths1000_trials1500.json
python -m scripts.spatial4R_mujoco_viewer --path results/spatial4r/spatial4R_two_stage_cached_path.json
```

The default input is `results/spatial4r/spatial4R_ftwmax_midpoint_two_stage_path.json`, the first
reference robot's successful 200/300 result with a sphere centered at the
Cartesian start/goal midpoint (cost 0.0780798317). The historical input
`results/spatial4r/spatial4R_ftwmax_paths200_trials300.json` retains the old obstacle location
and can still be selected explicitly with `--path`.
No new GCS solve is performed. Both robots are supported by reading the actual
`robot.a`, `robot.d`, and `robot.alpha` arrays from the selected report, rather
than relying on whichever robot the planning module currently selects.
The viewer requires four joint angles per waypoint, an XYZ goal, positive
link radius and spherical obstacles. Reference CA limits are not applied.

The model implements standard DH `Rz(q) Tz(d) Tx(a) Rx(alpha)` with four
unlimited hinges. Every DH offset and link is a capsule with the saved radius;
zero-length segments are spheres. Self-collision is disabled, matching the
planner. The floor, mount, start/goal markers and full 3D tip trace are visual
only; the floor is placed below the sampled arm envelope. Red spheres mark
obstacles; blue, teal, orange and purple distinguish the four links. The
camera starts at an oblique angle and can be rotated/panned/zoomed with the mouse.

Controls and export options are the same as for 3R. Default playback speed
is 0.5 so the short saved motion is easier to inspect; looping resets from
goal to start. This is prescribed kinematic playback, not dynamic tracking.
The viewer preserves the saved unwrapped joint-space segments.

Artifacts default to `visualizations/spatial4R_midpoint_mujoco/`: `spatial4R_scene.xml`,
`spatial4R_mujoco.gif`, start/middle/goal PNGs, and `replay_report.json`.
The report includes sampled analytic collisions, MuJoCo contacts, and maximum
MuJoCo-versus-DH tip position error. The current midpoint result produced 71
frames at half speed, with 88 checked configurations, zero analytic collisions,
zero MuJoCo contacts, FK agreement within 1.75e-15, and endpoint error 9.16e-16.
The nominal motion duration is 1.444 s; playback takes approximately 2.89 s.
Historical renders remain in `visualizations/spatial4R_mujoco/`.

All nine viewer tests passed on the host: the six planar tests and three new
spatial tests. Both spatial robots were checked at 300 random configurations
each for tool position/orientation, all eight capsule endpoints, and collision
agreement against the planning model, with both colliding and free poses.
Saved-model loading, malformed reports and winding preservation are also tested.
These sampled checks do not establish continuous collision-free certification.
