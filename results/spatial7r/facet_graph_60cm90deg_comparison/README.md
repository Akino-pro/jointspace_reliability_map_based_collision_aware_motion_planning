# 7R affine-intersection-dimension adjacency experiment

Task unchanged: 0.60 m translation in the previous direction, +90 degrees
about base Z, 0.08 m sphere exactly at the endpoint-position midpoint.
Both stages are configured for 200 paths / 300 sampling trials, with the
entire first-stage budget available independently to each target.

## Exact graph preparation within numerical tolerance

All 2,856 reliability-weighted regions are retained. Existing 869,670 directed
edges are filtered using intersection affine dimension >= 6 in seven dimensions.
Coordinate bounds screen lower-dimensional pairs; remaining pairs use the
original authors' `gcs.base.polytopeDimension` test with unit-normal halfspaces
and tolerance 1e-7. No polytope geometry or reliability is changed.

| Metric | Result |
| --- | ---: |
| Candidate unordered pairs | 434,835 |
| Pairs rejected by coordinate dimension upper bound | 415,788 |
| Pairs receiving affine-dimension tests | 19,047 |
| Intersections of affine dimension 6 / 7 | 18,204 / 843 |
| Retained directed edges | 38,094 |
| Directed-edge reduction | 95.6197% |
| Connected components after filtering | 1 |
| Retained regions / effective targets | 2,856 / 23 |
| Affine filter time | 219.076407911 s |
| Shared preprocessing including filtering | 238.993671724 s |

The target fingerprint matches the original full graph exactly. Connectivity
is preserved, but this does not prove unchanged optimal cost after removing
lower-dimensional transitions; no full-graph optimum was available.

## Actual solver attempts

Each row is a real sequential worker. Times include rechecking cached geometry
and GCS construction, exclude the shared filtering pass above, and end when
the worker is stopped or exits. They are NOT completed solve times.

| Representation / RSS limit | Method | Attempt seconds | Peak RSS bytes | Result |
| --- | --- | ---: | ---: | --- |
| Original / 6 GiB | Union hulls | 34.080038818 | 6484160512 | Supervisor stopped at RSS threshold |
| Original / 6 GiB | Independent targets | 37.292542511 | 6443188224 | Same, during first target |
| Compact / 6 GiB | Union hulls | 37.463242093 | 6496751616 | Supervisor stopped at RSS threshold |
| Compact / 6 GiB | Independent targets | 41.641634015 | 6479380480 | Same, during first target |
| Compact / 7 GiB | Union hulls | 64.901378139 | 7493988352 | Exited with SIGKILL before threshold action |
| Compact / 7 GiB | Independent targets | 48.451091973 | 7534333952 | Supervisor stopped at RSS threshold |

The 7 GiB union worker's sampled RSS was just below its threshold; exit code
-9 must not be described as a supervisor termination. The container reports
one OOM kill in its cumulative memory events, consistent with memory pressure,
but the event snapshot alone does not identify the killed process.

All workers completed Python-side GCS graph construction and entered native
SolvePath. No worker returned a first relaxation result or a rounded path;
no second-stage solve ran. No feasible path, cost or speed winner is reported.
The VM has approximately 7.57 GiB memory. Increasing the path-sampling budget
cannot address a failure before the first solve result.

## Equivalent constraint compaction

The opt-in `compact_constraints` flag normalizes axis halfspaces, keeps the
tightest identical row, and removes edge membership rows already implied by
the target's exact rows or conservative axis-box support bounds. Regional
sets, edge list, weighted costs and target inputs remain identical. Edge-cost
evaluators with the same source weight/shift are shared.

Regional halfspace rows: 58,848 -> 44,186. Edge membership rows after regional
compaction: 591,087 -> 93,552. A solved weighted-GCS regression verifies fixed
endpoint and feasible-cost parity; geometry tests check scaled-axis redundancy
and preservation of necessary face constraints. Full regression: 178 tests,
170 passed, eight optional MuJoCo skips. The feature defaults to false.

Commands in the configured Drake container:

```sh
python -m scripts.spatial7R_full_graph_comparison prepare --facet-adjacency
python -m scripts.spatial7R_full_graph_supervisor --facet-adjacency
python -m scripts.spatial7R_full_graph_supervisor --facet-adjacency --compact-constraints
python -m scripts.spatial7R_full_graph_supervisor --facet-adjacency --compact-constraints --rss-limit-gib 7 --output-dir results/spatial7r/facet_graph_60cm90deg_compact_rss7_comparison
```

Full precision and logs are in this directory and sibling directories
`facet_graph_60cm90deg_compact_comparison/` and
`facet_graph_60cm90deg_compact_rss7_comparison/`.
