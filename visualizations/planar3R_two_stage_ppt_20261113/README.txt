Two-stage union-hull figures, 3R
Stage 1: select a local target hull and candidate endpoint in one GCS solve.
Stage 2: constrained local IK projects to the task SSM; a second GCS solve replans the full route to this fixed goal.
PNG: 4000 x 2250, white background. SVG/PDF: vector output. English labels for slide use.
Background: code region reliability R, RdYlGn with shared limits 0..1. Overlap shows maximum R over feasible theta3, not the path-specific reliability.
Purple/orange: projected hull/free-region intersections. Dark line: sampled SSM. Magenta arrow: endpoint projection, not an executed connector.
The inset is essential: projection displacement and route changes are small in this actual example. No visual displacement was invented.
Reproduce from repository root: configured Python -m scripts.planar3R_two_stage_ppt
