# Reliability from recovery sets across SSM branches

The current 3R, 4R and 7R pre-failure planners use
`reliability_rule="union_of_branch_recovery_sets"`. This supersedes the older
`maximum_across_branches` rule. The two-stage target selection, continuous IK
projection, final fixed-goal GCS and post-failure planners are unchanged.

For a common partition cell Q, branch b supplies a set F_b(Q) of joint IDs:
joint j belongs to that set if the cell's j-coordinate interval is inside
the corresponding bounding-box interval of branch b. Define

```text
F(Q) = union over branches b of F_b(Q)
R(Q) = existing_conditional_reliability_lookup[F(Q)]
weight(Q) = 0.01 + 1 - R(Q)
```

For example, if branch A supplies {1} and branch B supplies {2}, Q receives
the reliability of {1,2}, not max(R({1}), R({2})). Joint 1 failure may recover
to A and joint 2 failure may recover to B. Only one joint fails in this model,
so the targets need not be on the same branch. Overlapping sets, duplicate
branches and repeated recovery of the same joint do not add probability twice.
Neither the individual branch probabilities nor their reliabilities are summed.
For joint reliabilities [0.7,0.5,0.3], this example changes the region reliability
from max(0.30,0.42)=0.42 to 0.51. An actual MOSEK/GCS regression verifies the
resulting weight and weighted segment cost after IRIS subdivision.

The conditional formula remains unchanged. For independent joint reliabilities
r_j, let p0 be their product, and p_j the probability that only joint j fails.
For nonempty F, the lookup is

```text
R(F) = (p0 + sum(p_j for j in F)) / (p0 + sum(p_j for all joints j))
```

The no-failure contribution p0 is included exactly once. The existing
empty-set lookup value remains zero; this change deliberately does not redefine
that convention. Single-branch reliability values remain identical.

The shared implementation is
`scripts/planar3R_bbox_reliability_partition.py::aggregate_branch_recovery`.
Planar 3R and spatial 4R use the existing sequential common refinement. Spatial
7R applies the same helper to the existing local common refinement within the
original IRIS cover, preserving its resource limits. Raw cells retain `branch_F`,
individual `branch_reliabilities`, combined `F` and the `recovery_branches` for
each joint. There is no winning branch for the combined probability. Exact
equal-cost box merges retain their original cells as provenance.

This is a union of recovery CONDITIONS, not a convex hull spanning multiple
branches, a geometric union of full bounding boxes, or a new physical SSM.
Local target hulls remain branch-local. Existing IRIS checkpoints can be reused:
they contain original collision geometry, not the derived reliability map.
The raw common-refinement geometry is unchanged for fixed branch bounds, but
new weights can change eligible equal-cost merges and thus final graph counts.
Counts can increase or decrease. An angle-range recovery candidate does not
prove that a locked-joint collision-free path reaches it; actual recovery is
checked by the separate post-failure planner.

Historical multi-branch path costs, region indices and route sequences belong
to the old map and must not be reused as current weighted optimization results.
Rerun pre-failure planning before comparing costs or replaying a retained route
under the new map. Old reports are preserved. The sampled-point baseline always
receives the same current map as the main planner.

Regression coverage includes complementary and overlapping branch sets,
duplicate/empty branches, branch-order invariance, unchanged single-branch
values, unchanged conditional formula and one-time no-failure probability.
Controlled four-branch 4R and eight-branch 7R fixtures check that complementary
branches together recover every joint with reliability one. Convex merging and
original IRIS halfspaces remain checked. The Docker suite ran 142 tests:
136 passed and six optional MuJoCo tests were skipped.

## Actual two-branch geometry validation

The existing reverse planar task, from the configuration
[0.6087380070193158,-0.6279264205318301,1.915910797633629] to workspace
[0.5,0.5], retains two branches, 394 samples and 28 effective hull targets.
The raw partition still has 125 cells. Under the new weights, 75 merged boxes
produce 110 connected free regions and 2,776 directed edges; the historical
max-rule graph had 93 regions and 1,960 edges. The original IRIS cover is reused.

Both the optional sampled-terminal comparison and a separate main-only run
were stopped after more than two minutes without their first solver return,
with Docker using approximately 6.7 GiB of its 7.57 GiB budget. These are
incomplete performance experiments, not solver infeasibility results. No new
final trajectory, cost or gap is claimed for this real two-branch benchmark.
The records are `results/diagnostics/recovery_union_sampled_comparison_stopped.json`
and `results/planar3r/two_stage_reverse_recovery_union.json`, with separate logs.
The model change is validated by the regression tests and actual bounded
GCS examples; large-graph numerical performance remains follow-up work.
