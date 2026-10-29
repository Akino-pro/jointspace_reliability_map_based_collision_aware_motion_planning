"""Plot every original 3R IRIS region before reliability subdivision or connectivity filtering."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull

from scripts.planar3R_iris_regions import load_iris_checkpoint
from scripts.planar3R_plot_saved_reliability_path import vertices_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-regions", default="data/iris/planar3R_iris_regions.npz")
    parser.add_argument("--output-dir", default="visualizations/planar3R_raw_iris_projection_20261029")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.show:
        plt.switch_backend("TkAgg")
    regions, _ = load_iris_checkpoint(args.iris_regions, expected_dimension=3)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.7, 8), layout="constrained")
    cmap = plt.get_cmap("tab10")
    polygons = []
    for index, region in enumerate(regions):
        vertices, _ = vertices_of(region)
        xy = np.unique(np.rad2deg(vertices[:, :2]), axis=0)
        hull = ConvexHull(xy)
        if np.max(hull.equations[:, :2]@xy.T+hull.equations[:, 2, None]) > 1e-7:
            raise ValueError("Projected hull does not contain the original projected vertices")
        polygon = xy[hull.vertices]
        polygons.append(dict(original_region_index=index, projected_vertices_degrees=polygon.tolist()))
        color = cmap(index % 10)
        ax.add_patch(Polygon(polygon, closed=True, facecolor=(*color[:3], .14),
                             edgecolor=color, linewidth=1.5, label=f"IRIS {index+1}"))
        midpoint = polygon.mean(axis=0)
        ax.text(*midpoint, str(index+1), color=color, fontsize=10, weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel=r"$\theta_1$ (deg)", ylabel=r"$\theta_2$ (deg)",
           title=f"All {len(regions)} original collision-free IRIS regions\n"+
                 r"$\theta_1$-$\theta_2$ projection before reliability subdivision")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.savefig(folder/"theta1_theta2_original_collision_free_regions.png", dpi=200)
    fig.savefig(folder/"theta1_theta2_original_collision_free_regions.svg")
    report = dict(iris_checkpoint=args.iris_regions, original_region_count=len(regions),
                  reliability_subdivision=False, connectivity_filtering=False, path_overlay=False,
                  axes="theta1/theta2 in degrees; theta3 eliminated by orthogonal projection",
                  colors="Categorical original-region identifiers, not reliability", regions=polygons)
    (folder/"projection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Projected all {len(regions)} original IRIS regions; no reliability partition or path overlay.", flush=True)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
