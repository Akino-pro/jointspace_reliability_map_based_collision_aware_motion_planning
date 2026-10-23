import itertools
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scripts.helper_functions import normalize_and_map_colors, reliability_computation
from scripts.planar3R_two_point_ssm_views import compute_ssms_for_target


TARGET_POINT = (1.5, 1.5)

R1 = 0.7
R2 = 0.5
R3 = 0.3
RELIABILITY_RULE = "union_of_branch_recovery_sets"


def build_reliability_lookup(r1, r2, r3):
    values = reliability_computation(r1, r2, r3)
    return {
        frozenset(): 0.0,
        frozenset({1}): values[0],
        frozenset({2}): values[1],
        frozenset({3}): values[2],
        frozenset({1, 2}): values[3],
        frozenset({1, 3}): values[4],
        frozenset({2, 3}): values[5],
        frozenset({1, 2, 3}): values[6],
    }


def build_color_lookup(reliability_lookup):
    reliability_values = sorted(set(reliability_lookup.values()))
    colors, sm = normalize_and_map_colors(reliability_values)
    color_lookup = dict(zip(reliability_values, colors))
    return color_lookup, sm


def branch_bounding_box(ssm):
    return np.column_stack((np.min(ssm, axis=0), np.max(ssm, axis=0)))


def axis_segments_from_bounds(bounds):
    segments = []
    for theta_min, theta_max in bounds:
        segments.append(
            [
                (-np.pi, theta_min, False),
                (theta_min, theta_max, True),
                (theta_max, np.pi, False),
            ]
        )
    return segments


def partition_single_branch(bounds, reliability_lookup):
    axis_segments = axis_segments_from_bounds(bounds)
    regions = []

    for segment_combo in itertools.product(*axis_segments):
        f_set = frozenset(i + 1 for i, segment in enumerate(segment_combo) if segment[2])
        lower = np.array([segment[0] for segment in segment_combo])
        upper = np.array([segment[1] for segment in segment_combo])
        regions.append(
            {
                "lower": lower,
                "upper": upper,
                "F": f_set,
                "reliability": reliability_lookup[f_set],
            }
        )

    return regions


def aggregate_branch_recovery(f_sets, reliability_lookup):
    """Allow a different target branch for each mutually exclusive joint failure.

    Union joint IDs BEFORE the conditional lookup, so overlapping recovery and
    the no-failure contribution are counted only once. Individual branch values
    remain diagnostics; they are neither maximized nor added. This denotes
    candidate recovery angles, not a proof of collision-free recovery paths.
    """
    f_sets = tuple(frozenset(f) for f in f_sets)
    if not f_sets:
        raise ValueError("At least one branch recovery set is required")
    combined = frozenset().union(*f_sets)
    return dict(F=combined, reliability=reliability_lookup[combined],
                branch_F=f_sets,
                branch_reliabilities=tuple(reliability_lookup[f] for f in f_sets),
                recovery_branches={joint: tuple(i for i, f in enumerate(f_sets) if joint in f)
                                   for joint in sorted(combined)},
                reliability_rule=RELIABILITY_RULE)


def partition_multiple_branches(branch_bounds, reliability_lookup):
    """Common refinement of branch boxes with unioned joint-recovery sets.

    Different single-joint failures may choose different SSM branches. The
    original conditional lookup (including its empty-set convention) is kept.
    Zero-volume contacts are omitted, as in the downstream IRIS subdivision.
    """
    branch_bounds = list(branch_bounds)
    if not branch_bounds:
        raise ValueError("At least one SSM branch is required")
    cells = partition_single_branch(branch_bounds[0], reliability_lookup)
    if len(branch_bounds) == 1:
        return cells
    cells = [dict(cell, **aggregate_branch_recovery([cell["F"]], reliability_lookup))
             for cell in cells]
    for bounds in branch_bounds[1:]:
        refined = []
        other_cells = partition_single_branch(bounds, reliability_lookup)
        for cell in cells:
            for other in other_cells:
                lower = np.maximum(cell["lower"], other["lower"])
                upper = np.minimum(cell["upper"], other["upper"])
                if np.any(upper - lower <= 1e-10):
                    continue
                f_sets = cell["branch_F"] + (other["F"],)
                refined.append({"lower": lower, "upper": upper,
                                **aggregate_branch_recovery(f_sets, reliability_lookup)})
        cells = refined
    return cells


def merge_equal_reliability_boxes(regions):
    """Merge equal-cost boxes only when their union is exactly a box.

    A merge requires an entire shared face and identical extents on the other
    axes. No hull filling, tolerance-based gaps, or nonconvex unions are added.
    Repeat all axes to remove further faces exposed by earlier merges. Keep
    original cells as provenance because branch conditions can vary inside a
    merged box even though its combined reliability is constant.
    """
    cells = list(regions)
    while True:
        previous = cells
        for axis in range(len(cells[0]["lower"]) if cells else 0):
            other_axes = [i for i in range(len(cells[0]["lower"])) if i != axis]
            groups = {}
            for cell in cells:
                key = (float(cell["reliability"]),
                       *np.asarray(cell["lower"])[other_axes],
                       *np.asarray(cell["upper"])[other_axes])
                groups.setdefault(key, []).append(cell)
            merged = []
            for group in groups.values():
                group.sort(key=lambda cell: cell["lower"][axis])
                current = group[0]
                for following in group[1:]:
                    if current["upper"][axis] == following["lower"][axis]:
                        upper = np.asarray(current["upper"]).copy()
                        upper[axis] = following["upper"][axis]
                        sources = (current.get("source_cells", [current])
                                   + following.get("source_cells", [following]))
                        current = {"lower": np.asarray(current["lower"]).copy(),
                                   "upper": upper, "reliability": current["reliability"],
                                   "source_cells": sources}
                    else:
                        merged.append(current)
                        current = following
                merged.append(current)
            cells = merged
        if len(cells) == len(previous):
            return previous


def format_f_set(f_set):
    if not f_set:
        return "empty"
    return "{" + ",".join(str(i) for i in sorted(f_set)) + "}"


def print_branch_summary(branch_index, bounds, regions):
    print(f"branch {branch_index}")
    for joint_index, (theta_min, theta_max) in enumerate(bounds, start=1):
        print(
            f"  theta{joint_index}: "
            f"[{theta_min:.6f}, {theta_max:.6f}] rad "
            f"= [{np.degrees(theta_min):.3f}, {np.degrees(theta_max):.3f}] deg"
        )

    print("  partition regions:")
    for region in regions:
        lower_deg = np.degrees(region["lower"])
        upper_deg = np.degrees(region["upper"])
        print(
            f"    F={format_f_set(region['F']):>7} "
            f"R={region['reliability']:.6f} "
            f"theta1=[{lower_deg[0]:8.3f},{upper_deg[0]:8.3f}] "
            f"theta2=[{lower_deg[1]:8.3f},{upper_deg[1]:8.3f}] "
            f"theta3=[{lower_deg[2]:8.3f},{upper_deg[2]:8.3f}]"
        )


def cuboid_faces(lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    return [
        vertices[[0, 1, 2, 3]],
        vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]],
        vertices[[2, 3, 7, 6]],
        vertices[[1, 2, 6, 5]],
        vertices[[0, 3, 7, 4]],
    ]


def plot_branch_partition(ax, ssm, bounds, regions, branch_label, color_lookup):
    for region in regions:
        faces = cuboid_faces(region["lower"], region["upper"])
        color = color_lookup[region["reliability"]]
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolor=color,
                edgecolor=color,
                linewidth=0.4,
                alpha=0.16,
            )
        )

    ax.plot(ssm[:, 0], ssm[:, 1], ssm[:, 2], color="black", linewidth=2.0)
    ax.scatter(
        [bounds[0, 0], bounds[0, 1]],
        [bounds[1, 0], bounds[1, 1]],
        [bounds[2, 0], bounds[2, 1]],
        color="black",
        s=15,
    )
    ax.set_title(branch_label)
    ax.set_xlabel("theta 1 [rad]")
    ax.set_ylabel("theta 2 [rad]")
    ax.set_zlabel("theta 3 [rad]")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)


def projected_region_cells(regions, axis_pair):
    cells = {}
    for region in regions:
        key = (
            region["lower"][axis_pair[0]],
            region["upper"][axis_pair[0]],
            region["lower"][axis_pair[1]],
            region["upper"][axis_pair[1]],
        )
        cells[key] = max(cells.get(key, 0.0), region["reliability"])
    return cells


def plot_projected_partition(ax, ssm, regions, axis_pair, axis_labels, color_lookup):
    cells = projected_region_cells(regions, axis_pair)

    for (x0, x1, y0, y1), reliability in cells.items():
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=color_lookup[reliability],
                edgecolor="black",
                linewidth=0.35,
                alpha=1.0,
            )
        )

    ax.plot(
        ssm[:, axis_pair[0]],
        ssm[:, axis_pair[1]],
        color="black",
        linewidth=1.8,
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.set_yticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.grid(False)


def plot_branch_2d_projections(branch_data, color_lookup, sm, target):
    axis_specs = [
        ((0, 1), ("theta 1 [rad]", "theta 2 [rad]"), "theta1-theta2"),
        ((1, 2), ("theta 2 [rad]", "theta 3 [rad]"), "theta2-theta3"),
        ((0, 2), ("theta 1 [rad]", "theta 3 [rad]"), "theta1-theta3"),
    ]

    for branch_index, (ssm, _, regions) in enumerate(branch_data, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
        for ax, (axis_pair, axis_labels, title) in zip(axes, axis_specs):
            plot_projected_partition(ax, ssm, regions, axis_pair, axis_labels, color_lookup)
            ax.set_title(title)

        fig.suptitle(
            f"2D Reliability Projections, Branch {branch_index} at "
            f"({target[0]:.3f}, {target[1]:.3f})"
        )
        fig.subplots_adjust(left=0.05, right=0.86, bottom=0.12, top=0.86, wspace=0.28)
        cbar_ax = fig.add_axes([0.90, 0.20, 0.018, 0.60])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Reliability")
        cbar.set_ticks(np.arange(0.0, 1.1, 0.1))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute SSM branches for one planar 3R workspace point and partition "
            "each branch bounding box independently."
        )
    )
    parser.add_argument("--x", type=float, default=TARGET_POINT[0], help="target x coordinate")
    parser.add_argument("--y", type=float, default=TARGET_POINT[1], help="target y coordinate")
    parser.add_argument("--seed", type=int, default=0, help="random seed for IK initial guesses")
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = args.x, args.y
    ssms = compute_ssms_for_target(x, y, seed=args.seed)
    reliability_lookup = build_reliability_lookup(R1, R2, R3)
    color_lookup, sm = build_color_lookup(reliability_lookup)

    print(f"target: ({x:.6f}, {y:.6f})")
    print(f"found {len(ssms)} SSM branch(es)")
    print("each branch is partitioned independently")
    print("reliability lookup:")
    for f_set, value in sorted(reliability_lookup.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        print(f"  F={format_f_set(f_set):>7}: {value:.6f}")

    branch_data = []
    for branch_index, ssm in enumerate(ssms, start=1):
        bounds = branch_bounding_box(ssm)
        regions = partition_single_branch(bounds, reliability_lookup)
        branch_data.append((ssm, bounds, regions))
        print_branch_summary(branch_index, bounds, regions)

    fig = plt.figure(figsize=(6 * max(len(branch_data), 1) + 1.8, 6))
    for plot_index, (ssm, bounds, regions) in enumerate(branch_data, start=1):
        ax = fig.add_subplot(1, len(branch_data), plot_index, projection="3d")
        plot_branch_partition(ax, ssm, bounds, regions, f"Branch {plot_index}", color_lookup)

    fig.suptitle(f"Bounding Box Reliability Partition at ({x:.3f}, {y:.3f})")
    fig.subplots_adjust(left=0.04, right=0.78, bottom=0.08, top=0.90, wspace=0.20)
    cbar_ax = fig.add_axes([0.84, 0.18, 0.025, 0.64])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Reliability")
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))

    plot_branch_2d_projections(branch_data, color_lookup, sm, (x, y))
    plt.show()


if __name__ == "__main__":
    main()
