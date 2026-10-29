# Saved planar 3R primary-method MuJoCo visualization

This replays saved experimental data; no new planning or timing benchmark ran.
Stage one comes from
`results/planar3r/per_target_budget_sweep_20261029/paths100_trials150.json`.
Stage two comes from the saved fixed-endpoint 100/150 replay
`results/diagnostics/stage2_matched_budget_independent_20261029/3r_independent_100_150.json`.
Their graph/task hashes were verified equal. The original source used stage
two 200/300; the matched-budget replay used here returns the identical path
and cost. The visualization input preserves these two sources explicitly.

Primary independent-target method:43 regions,20 target intersections;
stage one100/150 PER TARGET and stage two100/150. Saved weighted cost:
0.3817177694444778. Three unit-length links, capsule radius0.02m.
Tool start=(0.5,0.5),goal=(1.5,1.5). Circle obstacle centre=(0.75,0.75),radius0.15m.
The planar task constrains endpoint position; it does not prescribe orientation.

MuJoCo3.4.0,228 rendered GIF frames. Dense replay check567 configurations:
zero analytic collision samples and zero negative MuJoCo contact samples;
endpoint error2.22e-16m. Original unwrapped joint segments are preserved.
This is prescribed kinematic replay, not dynamic controller tracking.

Artifacts: `planar3R_start_goal_overview.png` (left start,right goal),
`planar3R_mujoco.gif`, individual start/middle/goal PNGs, `replay_report.json`,
`replay_input.json`,and `planar3R_scene.xml`.
Blue marker:start;green marker:goal;red:obstacle;light blue:tool trajectory.

Reopen interactive playback at half speed from the repository root:

```sh
python -m scripts.planar3R_mujoco_viewer --path visualizations/planar3R_100_150_mujoco_20261029/replay_input.json --output-dir visualizations/planar3R_100_150_mujoco_20261029/interactive --speed 0.5
```

Space pauses/resumes,R restarts,L toggles looping,T toggles trace,and mouse
rotates/zooms. Interactive outputs use a separate subdirectory to preserve
the checked export report. Viewer PID and logs are in this directory.
