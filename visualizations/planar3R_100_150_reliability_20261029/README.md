# Saved 3R path on its reliability map

`theta1_theta2_reliability_path.png` and SVG are the requested 2D projection.
The left panel projects all actual 3D polytope vertices onto theta1/theta2,
then fills each projected convex polygon in ascending R order. Its background
therefore represents the maximum R available over feasible theta3 within
the covered regions,not the unique reliability of a 2D point. White means
outside projected saved-region coverage,not a collision certificate. The right
panel shows only the outlines of traversed-region projections. In both panels,
black arrows follow the saved path and segment labels retain actual3D route
reliabilities0.70,0.91,1.00;they may differ from the projected maximum.

Reproduce: `python -m scripts.planar3R_plot_saved_reliability_path`.
Add `--show` on the native host to open a rotatable 3D Matplotlib figure and
the workspace/reliability-profile figure. PNG and SVG exports are standalone.
Optional Plotly HTML is generated only when Plotly is already available;
it was not available in the configured native interpreter for this export.

The graph is reconstructed from the source report's original IRIS checkpoint
and goal-conditioned reliability partition, including start-connected filtering.
Its SHA256 matches the saved stage-two replay. No planning was executed.
This is the same saved primary100/150 stage-one / matched100/150 stage-two
path shown in `../planar3R_100_150_mujoco_20261029/`.

`jointspace_reliability_path.png` and SVG show all43 connected collision-free
polytopes on the left and only traversed regions on the right. Axes are joint
angles in degrees. Black is the original unwrapped path;blue is start;magenta
is goal. The red-yellow-green scale is reliability R in[0,1]. Transparent
polytopes preserve true IRIS halfspaces, not axis-box approximations.

`workspace_and_path_reliability.png` and SVG color the actual end-effector
path by its joint-space segment reliability and show R against cumulative
joint-space arc length. No workspace background reliability is inferred:
different joint configurations at the same XY position can have different R.

| Saved region | Segment reliability R | Joint length(rad) | Weighted cost |
| --- | ---: | ---: | ---: |
|1|0.70|0.3328927877482419|0.103196764201955|
|3|0.91|2.4508724393470835|0.24508724393470857|
|28|1.00|3.3433761307814143|0.03343376130781417|

Cost=sum((0.01+1-R)*segment_length)=0.3817177694444778,matching the saved
result. Every segment's two endpoints were checked against its assigned
convex polytope,ensuring the full straight joint segment lies inside it.
`plot_report.json` records source paths,graph hash,route and full precision.
