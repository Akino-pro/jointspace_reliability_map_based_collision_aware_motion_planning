"""Plot the saved successful planar 3R recovery run for the framework slide."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, HalfspaceIntersection

from scripts.planar3R_collision import is_collision_free, link_points
from scripts.planar3R_iris_regions import interior_point, load_iris_checkpoint
from scripts.planar3R_post_failure_planner import slice_regions

out = Path(__file__).resolve().parent
record = json.loads(Path("results/planar3r/post_failure_success.json").read_text())
angle = record["locked_angle"]
path = np.asarray(record["waypoints"])
originals, _ = load_iris_checkpoint(record["iris_checkpoint"])
regions = slice_regions(originals, record["failed_joint"], angle)
assert record["success"] and record["failed_joint"] == 3
assert len(regions) == record["slice_region_count"]
assert np.allclose(path[:, 2], angle)
assert np.allclose(path[0], record["failure_configuration"])
assert np.allclose(link_points(path[-1], record["link_lengths"])[-1], record["goal_point"])
for start, end in zip(path[:-1], path[1:]):
    assert any(region.contains(start[:2]) and region.contains(end[:2]) for region in regions)
    assert all(is_collision_free(q, record["obstacles"], record["link_lengths"],
                                 record["link_radius"]) for q in np.linspace(start, end, 101))

def vertices(region):
    point = interior_point(region.A, region.b)
    vertices = HalfspaceIntersection(np.column_stack((region.A, -region.b)), point).intersections
    return vertices[ConvexHull(vertices).vertices]

polygons = [vertices(region) for region in regions]
purple = "#7650AA"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
fig = plt.figure(figsize=(3.4, 2.85))
fig.text(.5, .93, r"Locked slice: $q_3=-0.472$ rad", ha="center", weight="bold", fontsize=14)
ax = fig.add_axes([.08, .20, .78, .67], projection="3d", computed_zorder=False)
limit = np.pi
plane = [[-limit, -limit, angle], [limit, -limit, angle], [limit, limit, angle], [-limit, limit, angle]]
ax.add_collection3d(Poly3DCollection([plane], facecolor="#F4EFFA", edgecolor="#9C7BC4", linewidth=.9, zorder=2))
flat_polygons = [np.column_stack((p, np.full(len(p), angle + .002))) for p in polygons]
ax.add_collection3d(Poly3DCollection(flat_polygons, facecolor="#BEA4EA", edgecolor=purple, linewidth=1.1, zorder=3))
ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
       xlabel=r"$q_1$", ylabel=r"$q_2$", zlabel=r"$q_3$")
ax.set(xticks=[], yticks=[], zticks=[])
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor("#F9F9FC")
ax.view_init(elev=30, azim=-56)
fig.text(.5, .055, "4 reused collision-free regions", ha="center", fontsize=11)
fig.savefig(out / "recovery_locked_slice.png", dpi=300)
plt.close(fig)

fig = plt.figure(figsize=(3.15, 2.85))
fig.text(.53, .93, "Recovery trajectory (zoom)", ha="center", weight="bold", fontsize=13)
ax = fig.add_axes([.21, .25, .71, .57])
ax.set_facecolor("#F7F7FA")
for p in polygons:
    ax.add_patch(Polygon(p, facecolor="#E9DDF8", edgecolor=purple, linewidth=1))
ax.plot(*path[:, :2].T, color="#7300CA", linewidth=3, zorder=4)
ax.scatter(*path[0, :2], c="#35D162", edgecolor="#192131", s=61, zorder=5)
ax.scatter(*path[-1, :2], c="#F04D55", edgecolor="#192131", s=61, zorder=5)
ax.text(path[0, 0], path[0, 1] + .012, "Failure", ha="center", fontsize=10)
ax.text(path[-1, 0], path[-1, 1] + .012, "Goal", ha="center", fontsize=10)
ax.set(xlim=(-.40, -.29), ylim=(1.83, 1.93),
       xticks=[-.40, -.35, -.30], yticks=[1.84, 1.88, 1.92],
       xlabel=r"$q_1$ [rad]", ylabel=r"$q_2$ [rad]")
ax.set_aspect("equal", adjustable="box")
ax.tick_params(labelsize=8, pad=2)
fig.text(.53, .035, f"Path length: {record['path_length']:.2f} rad", ha="center", fontsize=10)
fig.savefig(out / "recovery_trajectory_zoom.png", dpi=300)
plt.close(fig)
