"""Plot the saved planar 3R path in its exact reliability-weighted free-region map."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from scripts.planar3R_iris_regions import interior_point
from scripts.planar3R_collision import link_points
from scripts.replay_saved_stage2_budgets import prepare_graph


def vertices_of(region):
    point = interior_point(region.A, region.b)
    if point is None:
        raise ValueError("Expected a full-dimensional free region")
    vertices = HalfspaceIntersection(np.column_stack((region.A, -region.b)), point).intersections
    return vertices, ConvexHull(vertices).simplices


def interactive_map(folder, meshes, regions, path, route, cmap):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    traces = []
    for index, ((vertices, faces), region) in enumerate(zip(meshes, regions)):
        xyz = np.rad2deg(vertices)
        rgba = cmap(region.reliability)
        color = "rgb(%d,%d,%d)" % tuple(int(255*x) for x in rgba[:3])
        traces.append(go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                               color=color, opacity=.28 if index in route else .07,
                               name=f"Region {index}; R={region.reliability:.6f}",
                               hovertemplate=f"Region {index}<br>R={region.reliability:.6f}<extra></extra>",
                               showlegend=False))
    q = np.rad2deg(path)
    traces += [go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="lines+markers",
                            line=dict(color="black", width=8), marker=dict(size=3, color="black"), name="Saved GCS path"),
               go.Scatter3d(x=[q[0, 0]], y=[q[0, 1]], z=[q[0, 2]], mode="markers+text",
                            marker=dict(size=7, color="royalblue"), text=["Start"], name="Start"),
               go.Scatter3d(x=[q[-1, 0]], y=[q[-1, 1]], z=[q[-1, 2]], mode="markers+text",
                            marker=dict(size=7, color="magenta"), text=["Goal"], name="Goal"),
               go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", showlegend=False,
                            marker=dict(color=[0, 1], cmin=0, cmax=1, showscale=True,
                                        colorscale="RdYlGn", colorbar=dict(title="Reliability R")))]
    all_visible = [True]*len(traces)
    route_visible = [index in route for index in range(len(regions))]+[True]*4
    fig = go.Figure(traces)
    fig.update_layout(title="Planar 3R: reliability map and saved 100/150 path",
                      scene=dict(xaxis_title="theta1 (deg)", yaxis_title="theta2 (deg)",
                                 zaxis_title="theta3 (deg)", aspectmode="cube"),
                      updatemenus=[dict(type="buttons", direction="left", y=1.12,
                                       buttons=[dict(label="All 43 free regions", method="update", args=[dict(visible=all_visible)]),
                                                dict(label="Path regions only", method="update", args=[dict(visible=route_visible)])])],
                      margin=dict(l=20, r=80, b=30, t=110))
    fig.write_html(folder/"reliability_path_interactive.html", include_plotlyjs=True,
                   config=dict(displaylogo=False))
    return True


def theta12_projection(folder, meshes, regions, q, route, reliabilities, cmap, norm):
    """Project actual polytopes; the filled map is max R over all feasible theta3."""
    polygons = []
    for vertices, _ in meshes:
        xy = np.unique(np.rad2deg(vertices[:, :2]), axis=0)
        polygons.append(xy[ConvexHull(xy).vertices])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for index in sorted(range(len(regions)), key=lambda i: regions[i].reliability):
        axes[0].add_patch(Polygon(polygons[index], closed=True,
                                  facecolor=cmap(regions[index].reliability), edgecolor="none"))
    for polygon in polygons:
        axes[0].add_patch(Polygon(polygon, closed=True, fill=False,
                                  edgecolor=(.1, .1, .1, .18), linewidth=.5))
    axes[0].set_title(r"43 free regions projected onto $\theta_1$-$\theta_2$"+"\nBackground: maximum R over feasible theta3")
    for index in dict.fromkeys(route):
        axes[1].add_patch(Polygon(polygons[index], closed=True, fill=False,
                                  edgecolor=cmap(regions[index].reliability), linewidth=2,
                                  label=f"Region {index}: R={regions[index].reliability:.2f}"))
    axes[1].set_title("Only traversed regions: projected outlines\nSegment labels: actual saved path reliability")
    for ax in axes:
        ax.plot(q[:, 0], q[:, 1], color="white", linewidth=5, zorder=5)
        ax.plot(q[:, 0], q[:, 1], color="black", linewidth=2.5, marker="o", markersize=3, zorder=6)
        ax.scatter(*q[0, :2], color="royalblue", edgecolor="white", s=80, label="Start", zorder=8)
        ax.scatter(*q[-1, :2], color="magenta", edgecolor="white", marker="*", s=150, label="Goal", zorder=8)
        for i, (first, second) in enumerate(zip(q[:-1, :2], q[1:, :2])):
            midpoint = (first+second)/2
            ax.annotate(f"R={reliabilities[i]:.2f}", midpoint, xytext=(8, -17 if i == 1 else 12),
                        textcoords="offset points", fontsize=9, zorder=10,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            ax.annotate("", first+.6*(second-first), xytext=first+.4*(second-first),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5), zorder=9)
        ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel=r"$\theta_1$ (deg)", ylabel=r"$\theta_2$ (deg)")
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-180, 181, 60))
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.8, label="Reliability R")
    fig.suptitle("Saved planar 3R GCS path: theta1-theta2 projection (theta3 eliminated)")
    fig.savefig(folder/"theta1_theta2_reliability_path.png", dpi=200)
    fig.savefig(folder/"theta1_theta2_reliability_path.svg")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/planar3r/per_target_budget_sweep_20261029/paths100_trials150.json")
    parser.add_argument("--replay", default="results/diagnostics/stage2_matched_budget_independent_20261029/3r_independent_100_150.json")
    parser.add_argument("--output-dir", default="visualizations/planar3R_100_150_reliability_20261029")
    parser.add_argument("--show", action="store_true", help="Open rotatable Matplotlib figures on the native host")
    args = parser.parse_args()
    if args.show:
        plt.switch_backend("TkAgg")
    source, replay = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.source, args.replay))
    graph = prepare_graph(source, spatial=False)
    if not source["success"] or not replay["success"] or graph["fingerprint"] != replay["graph_task_sha256"]:
        raise ValueError("Saved path and reconstructed reliability graph must match")
    path = np.asarray(replay["waypoints"])-graph["offsets"]
    route, regions = replay["region_route"], graph["regions"]
    if len(path) != len(route)+1:
        raise ValueError("Expected one region per path segment")
    epsilon = source["reliability_epsilon"]
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    reliabilities = np.array([regions[i].reliability for i in route])
    for first, second, index in zip(path[:-1], path[1:], route):
        if not regions[index].contains(first, tol=1e-6) or not regions[index].contains(second, tol=1e-6):
            raise ValueError("Path segment leaves its saved convex free region")
    costs = lengths*(epsilon+1-reliabilities)
    if abs(costs.sum()-replay["cost"]) > 1e-6:
        raise ValueError("Recomputed reliability-weighted cost differs from the saved result")
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    meshes = [vertices_of(r) for r in regions]
    cmap, norm = plt.get_cmap("RdYlGn"), colors.Normalize(0, 1)
    q = np.rad2deg(path)
    projection_figure = theta12_projection(folder, meshes, regions, q, route, reliabilities, cmap, norm)
    if not args.show:
        plt.close(projection_figure)
    fig = plt.figure(figsize=(14, 6.8), layout="constrained")
    for panel, route_only in enumerate((False, True), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for index, ((vertices, faces), region) in enumerate(zip(meshes, regions)):
            if route_only and index not in route:
                continue
            xyz = np.rad2deg(vertices)
            ax.add_collection3d(Poly3DCollection([xyz[face] for face in faces],
                                  facecolor=cmap(region.reliability),
                                  edgecolor=(.15, .15, .15, .18), linewidth=.3,
                                  alpha=.22 if index in route else .035))
        ax.plot(*q.T, color="black", linewidth=3, marker="o", markersize=3, label="Saved GCS path")
        ax.scatter(*q[0], color="royalblue", s=60, depthshade=False, label="Start")
        ax.scatter(*q[-1], color="magenta", marker="*", s=130, depthshade=False, label="Goal")
        for index, midpoint in enumerate((q[:-1]+q[1:])/2):
            ax.text(*midpoint, f" R={reliabilities[index]:.3f}", fontsize=8)
        ax.set(xlim=(-180, 180), ylim=(-180, 180), zlim=(-180, 180),
               xlabel=r"$\theta_1$ (deg)", ylabel=r"$\theta_2$ (deg)", zlabel=r"$\theta_3$ (deg)",
               title="43 connected collision-free regions" if not route_only else "Only regions traversed by the path")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-55)
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=.65, label="Reliability R")
    fig.suptitle(f"Planar 3R reliability map with saved GCS path | weighted cost {costs.sum():.8f}")
    fig.savefig(folder/"jointspace_reliability_path.png", dpi=190)
    fig.savefig(folder/"jointspace_reliability_path.svg")
    if not args.show:
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), layout="constrained")
    ax = axes[0]
    for first, second, reliability in zip(path[:-1], path[1:], reliabilities):
        samples = first+np.linspace(0, 1, 300)[:, None]*(second-first)
        xy = np.array([link_points(p)[-1] for p in samples])
        ax.plot(*xy.T, color="black", linewidth=5)
        ax.plot(*xy.T, color=cmap(reliability), linewidth=3)
    for label, angles, color in (("Start", path[0], "royalblue"), ("Goal", path[-1], "magenta")):
        xy = np.asarray(link_points(angles))
        ax.plot(*xy.T, color=color, alpha=.5, marker="o", linewidth=1.5)
        ax.scatter(*xy[-1], color=color, s=70, label=label, zorder=5)
    for obstacle in source["obstacles"]:
        if obstacle.get("type", "circle") == "circle":
            ax.add_patch(Circle(obstacle["center"], obstacle["radius"], color="tomato", label="Obstacle"))
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel="x (m)", ylabel="y (m)",
           title="Workspace path colored by actual joint-space R")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    distance = np.r_[0, lengths.cumsum()]
    axes[1].stairs(reliabilities, distance, baseline=None, color="black", linewidth=2)
    for first, second, value in zip(distance[:-1], distance[1:], reliabilities):
        axes[1].fill_between([first, second], value, 0, color=cmap(value), alpha=.6)
    axes[1].set(xlim=(0, distance[-1]), ylim=(0, 1.05), xlabel="Cumulative joint-space arc length (rad)",
                ylabel="Assigned region reliability R", title="Reliability along the saved path")
    axes[1].grid(alpha=.2)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.75, label="Reliability R")
    fig.savefig(folder/"workspace_and_path_reliability.png", dpi=190)
    fig.savefig(folder/"workspace_and_path_reliability.svg")
    if not args.show:
        plt.close(fig)
    interactive = interactive_map(folder, meshes, regions, path, route, cmap)
    report = dict(source=args.source, stage2_replay=args.replay, graph_task_sha256=graph["fingerprint"],
                  region_count=len(regions), route=route, path_reliabilities=reliabilities.tolist(),
                  segment_lengths=lengths.tolist(), segment_costs=costs.tolist(),
                  recomputed_cost=float(costs.sum()), epsilon=epsilon, interactive_html=interactive,
                  theta12_projection="Maximum R over feasible theta3 for background; actual route reliability for segment labels",
                  scope="Exact goal-conditioned joint-space reliability; workspace only colors the given path, no IK aggregation",
                  new_planning_executed=False)
    (folder/"plot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
