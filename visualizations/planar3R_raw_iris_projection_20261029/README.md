# Original collision-free IRIS regions: theta1/theta2 projection

All six original 3D regions are loaded directly from
`data/iris/planar3R_iris_regions.npz`. No reliability subdivision, reliability
intersection, start-connected filtering, or path overlay is applied.
Colors and numbers identify original IRIS regions;they do not encode R.

Each projected polygon is the 2D convex hull of all original polytope vertices
after dropping theta3. This is the orthogonal projection of the full polytope,
not a bounding box and not a fixed-theta3 slice. Every projected vertex was
checked against the projected polygon halfspaces. A projected point means
that at least one theta3 is covered,not that every theta3 is collision free.

PNG and SVG:`theta1_theta2_original_collision_free_regions`.
`projection_report.json` preserves the input path,region count and polygons.
Reproduce:`python -m scripts.planar3R_plot_iris_projection`;add`--show` for a window.
