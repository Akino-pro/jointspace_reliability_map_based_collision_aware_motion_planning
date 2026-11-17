"""Build a three-panel slide from the planar 3R checkpoint and goal SSM."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull, HalfspaceIntersection

from scripts.planar3R_bbox_reliability_partition import (
    R1, R2, R3, branch_bounding_box, build_color_lookup, build_reliability_lookup,
    partition_single_branch,
)
from scripts.planar3R_iris_regions import interior_point, load_iris_checkpoint, split_iris_regions
from scripts.planar3R_post_failure_planner import slice_regions

out = Path(__file__).resolve().parent
ssm = np.load(out / "goal_ssm_data.npz")["ssm_0"]
checkpoint, _ = load_iris_checkpoint("data/iris/planar3R_iris_regions.npz")
iris = slice_regions(checkpoint, 3, 0.0)
bounds = branch_bounding_box(ssm)
lookup = build_reliability_lookup(R1, R2, R3)
colors, spectrum = build_color_lookup(lookup)
boxes = [
    dict(part, lower=part["lower"][:2], upper=part["upper"][:2])
    for part in partition_single_branch(bounds, lookup)
    if part["lower"][2] <= 0 < part["upper"][2]
]
children = split_iris_regions(iris, boxes)
assert (len(iris), len(boxes), len(children)) == (5, 9, 23)


def polygon(region):
    point = interior_point(region.A, region.b)
    vertices = HalfspaceIntersection(np.column_stack((region.A, -region.b)), point).intersections
    hull = ConvexHull(vertices)
    return np.degrees(vertices[hull.vertices]), hull.volume


parents = [polygon(region)[0] for region in iris]
pieces = [polygon(region) for region in children]
def box_area(box):
    return np.prod(box["upper"] - box["lower"])


# Highlight a box that the IRIS boundary visibly clips.
selected = max(
    (index for index, region in enumerate(children)
     if .25 < pieces[index][1] / box_area(boxes[region.partition_index]) < .75),
    key=lambda index: pieces[index][1],
)
example = children[selected]
example_box = boxes[example.partition_index]
ink, blue, highlight = "#10284E", "#2764BA", "#7628A6"
plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": ink,
                     "axes.labelcolor": ink, "svg.fonttype": "path"})
fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
fig.text(.5, .925, "Intersecting Collision-Free Regions with the Reliability Map",
         ha="center", fontsize=29, weight="bold")
fig.text(.5, .871, r"Planar 3R  |  Goal $(1.5,1.5)$  |  Joint-space slice $q_3=0$ rad",
         ha="center", fontsize=16, color="#46617C")
fig.add_artist(Rectangle((.24, .842), .52, .005, transform=fig.transFigure, color=blue))

cards = [
    (.035, "1. IRIS collision-free regions", "5 convex regions in the slice"),
    (.353, "2. Joint-space reliability map", "9 cells from the goal SSM bounding box"),
    (.671, "3. Reliability-labeled intersections", "23 nonempty convex pieces"),
]
axes = []
for left, title, subtitle in cards:
    fig.add_artist(Rectangle((left, .208), .294, .591, transform=fig.transFigure,
                             facecolor="white", edgecolor="#A8C6EC", zorder=0))
    fig.add_artist(Rectangle((left, .71), .294, .089, transform=fig.transFigure,
                             facecolor="#DAE9FB", edgecolor="none", zorder=0))
    fig.text(left+.147, .765, title, ha="center", fontsize=16, weight="bold")
    fig.text(left+.147, .731, subtitle, ha="center", fontsize=12, color="#46617C")
    ax = fig.add_axes([left+.042, .305, .205, .376])
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel=r"$q_1$ [deg]", ylabel=r"$q_2$ [deg]",
           xticks=[-180,-90,0,90,180], yticks=[-180,-90,0,90,180])
    ax.set_aspect("equal")
    ax.set_facecolor("#F0F1F4")
    ax.tick_params(labelsize=10)
    axes.append(ax)
fig.text(.341, .49, "∩", ha="center", fontsize=26, color=blue)
fig.text(.659, .49, "→", ha="center", fontsize=26, color=blue)

ax = axes[0]
for vertices in parents:
    ax.add_patch(Polygon(vertices, fc="#DEEDFF", ec=blue, lw=1.1, alpha=.78))
ax.add_patch(Polygon(parents[example.parent_index], fill=False, ec=highlight, lw=2.5, zorder=6))

ax = axes[1]
for box in boxes:
    lo, hi = np.degrees([box["lower"], box["upper"]])
    r = box["reliability"]
    ax.add_patch(Rectangle(lo, *(hi-lo), fc=colors[r], ec="white", lw=1.2))
    center = (lo+hi)/2
    # Keep the central value clear of the SSM curve.
    if len(box["F"]) == 3:
        center += [-29, -69]
    ax.text(*center, f"{r:.2f}", ha="center", va="center", fontsize=11, zorder=5,
            bbox=dict(facecolor=colors[r], edgecolor="none", pad=1.5))
ax.plot(*np.degrees(ssm[:, :2]).T, c="#111111", lw=1.55, zorder=4)
lo, hi = np.degrees([example_box["lower"], example_box["upper"]])
ax.add_patch(Rectangle(lo, *(hi-lo), fill=False, ec=highlight, lw=2.4, zorder=6))
cax = fig.add_axes([.609, .325, .009, .33])
cb = fig.colorbar(spectrum, cax=cax, ticks=[0,.5,1])
cb.ax.tick_params(labelsize=9, length=2)
cb.set_label("R", fontsize=10, labelpad=4)

ax = axes[2]
for region, (vertices, _) in zip(children, pieces):
    ax.add_patch(Polygon(vertices, fc=colors[region.reliability], ec="#4D6B57", lw=.85))
ax.add_patch(Polygon(pieces[selected][0], fill=False, ec=highlight, lw=2.5, zorder=6))

fig.text(.182, .237, "Blue: IRIS coverage  ·  Gray: outside coverage", ha="center", fontsize=10)
fig.text(.500, .237, "Black curve: goal SSM projected onto the slice axes", ha="center", fontsize=10)
fig.text(.818, .237, "Each piece inherits its cell's reliability", ha="center", fontsize=10)
fig.text(.5, .146, "Purple outlines trace one intersection", ha="center",
         fontsize=17, weight="bold", color=highlight)
fig.text(.5, .100,
         rf"$\mathrm{{IRIS\ region}}\ \cap\ \mathrm{{map\ cell}}\ (R={example.reliability:.2f})"
         rf"\ =\ \mathrm{{convex\ piece}}\ (R={example.reliability:.2f})$",
         ha="center", fontsize=19)
for ext in ("png", "svg", "pdf"):
    fig.savefig(out / f"planar3r_intersection_detail.{ext}", dpi=200)
