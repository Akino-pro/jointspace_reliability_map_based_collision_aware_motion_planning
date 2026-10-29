# Full-graph 7R 60 cm / base-Z 90-degree comparison

Both real workers used the complete original coverage and exact reliability
partition: 2,856 regions, 869,670 directed intersection edges, 23 hull targets,
one detected SSM branch and 54 samples. No corridor cropping or target removal.
Both stages were configured for 200 distinct paths / 300 sampling trials;
the independent method grants the entire budget to each target.

Shared full preprocessing took 264.929300255 s, excluding cached historical
IRIS generation. A hash-checked numeric adjacency cache was reused by both
workers; worker times below include their repeated cached-input preprocessing.

| Method | Attempt wall time (s) | Peak RSS (GiB) | Last phase | Outcome |
| --- | ---: | ---: | --- | --- |
| One graph, all hull targets | 65.288858041 | 6.01450 | GCS model construction | Stopped at 6 GiB RSS protection threshold |
| Independent targets | 62.289031019 | 6.01351 | First target's GCS model construction | Stopped at 6 GiB RSS protection threshold |

Neither worker completed model construction, reached relaxation or rounding,
or produced a first-stage/final cost. No second-stage solve ran. These are
aborted attempt times, not completed solve times or a speed comparison.
The supervisor terminated only its own worker when sampled RSS crossed the
configured threshold; this is not a claim that the OS OOM killer acted.
The available Docker VM memory was approximately 7.57 GiB.

Full precision: `comparison_summary.json`; graph report: `prepare.json`;
real worker output: `union_hulls.log`, `per_target.log`.
Commands: `python -m scripts.spatial7R_full_graph_comparison prepare`, then
`python -m scripts.spatial7R_full_graph_supervisor` in the Drake container.
Regression: 173 tests, 165 passed, eight optional MuJoCo skips.

## Contact-dimension diagnostic

`contact_dimension_diagnostic.json` bounds intersection dimension from axis
intervals, with tolerance 1e-7. It is not an exact affine-rank test. Of all
869,670 directed edges, 831,576 (95.6197%) have coordinate-bound intersection
dimension below six. At most 38,094 edges remain candidates for the original
authors' optional intersection-dimension-at-least-d-minus-one rule; they
still require exact intersection-dimension checks. No such filtering was
applied to either measured worker above.

The authors' optional rule is implemented in
[findEdgesViaFullDimensionOverlaps](https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/gcs/base.py).
It is not Drake's universal default, and preserving connectivity/optimality
when deleting lower-dimensional contacts requires checking the decomposition.
Drake's [preprocessing option](https://drake.mit.edu/doxygen_cxx/structdrake_1_1geometry_1_1optimization_1_1_graph_of_convex_sets_options.html)
removes edges that it can prove cannot lie on a source-target path; our workers
stopped before calling that solver stage.
