import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Wedge

from scripts import planar3R_reliability_maps_proposed_method as proposed
from scripts.helper_functions import normalize_and_map_colors


OUTPUT_PNG = 'visualizations/planar3R_legacy/planar3R_workspace_reliability_map.png'
PLANAR3R_L = [1, 1, 1]
FULL_RANGE_CA = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]


def set_planar3r_full_workspace_parameters():
    proposed.L = PLANAR3R_L.copy()
    proposed.CA = [joint_range.copy() for joint_range in FULL_RANGE_CA]
    proposed.C_dot_A = [joint_range.copy() for joint_range in FULL_RANGE_CA]


def add_workspace_reliability_patches(ax, sample_num, alpha=1.0, base_zorder=-10):
    set_planar3r_full_workspace_parameters()
    cr_values = proposed.cr_list.copy()
    if cr_values[-1] != 0:
        cr_values.append(0)
    color_list, sm = normalize_and_map_colors(cr_values)

    indices = proposed.indices
    reach = float(np.sum(proposed.L))
    section_length = reach / sample_num
    x_values = (np.arange(sample_num) + 0.5) * section_length
    ring_width = 2.0 * x_values[0]

    background = patches.Circle(
        (0, 0),
        reach,
        edgecolor=color_list[-1],
        facecolor=color_list[-1],
        linewidth=0,
        alpha=alpha,
        zorder=base_zorder,
    )
    ax.add_patch(background)

    for x in x_values:
        print(f"computing radius sample x={x:.6f}")
        _, reliable_beta_ranges, _, _ = proposed.compute_beta_range(
            x, 0.0, proposed.L, proposed.CA
        )

        level = 0
        for region_index in indices:
            level += 1
            color = color_list[region_index]
            for beta_range in reliable_beta_ranges[region_index]:
                wedge = Wedge(
                    center=(0, 0),
                    r=x + ring_width / 2.0,
                    theta1=np.degrees(beta_range[0]),
                    theta2=np.degrees(beta_range[1]),
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    zorder=base_zorder + level,
                )
                ax.add_patch(wedge)

        prefailure_color = color_list[-2]
        for beta_range in reliable_beta_ranges[-1]:
            wedge = Wedge(
                center=(0, 0),
                r=x + ring_width / 2.0,
                theta1=np.degrees(beta_range[0]),
                theta2=np.degrees(beta_range[1]),
                width=ring_width,
                facecolor=prefailure_color,
                edgecolor=prefailure_color,
                alpha=alpha,
                zorder=base_zorder,
            )
            ax.add_patch(wedge)

    return sm


def draw_workspace_reliability_map(sample_num, output_path):
    set_planar3r_full_workspace_parameters()
    cr_values = proposed.cr_list.copy()
    if cr_values[-1] != 0:
        cr_values.append(0)
    _, sm = normalize_and_map_colors(cr_values)

    reach = float(np.sum(proposed.L))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Planar 3R Workspace Reliability Map")

    sm = add_workspace_reliability_patches(ax, sample_num)

    cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label("Reliability")
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))

    ax.text(
        0.02,
        0.02,
        f"L={PLANAR3R_L}\nCA=[-pi, pi]^3\nsamples={sample_num}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Draw proposed-method workspace reliability map.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", default=OUTPUT_PNG)
    return parser.parse_args()


def main():
    args = parse_args()
    draw_workspace_reliability_map(args.samples, args.output)


if __name__ == "__main__":
    main()
