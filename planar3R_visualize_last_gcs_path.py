import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from planar3R_collision import L, link_points
from planar3R_draw_workspace_reliability import add_workspace_reliability_patches
from planar3R_gcs_planner import GOAL_POINT, OBSTACLES, START_POINT


OUTPUT_GIF = "planar3R_reliability_gcs_path.gif"
RELIABILITY_MAP_SAMPLES = 128

# Waypoints from the latest reliability-aware Drake GCS run.
GCS_WAYPOINTS = np.array(
    [
        [-2.9730, -2.2695, -1.2364],
        [-2.8807, -1.9541, -1.2914],
        [-1.8850, -1.9541, -1.8850],
        [-1.8850, -1.9541, -1.8850],
        [-1.8850, -1.9541, -1.8850],
        [-1.8850, -1.8850, -1.8850],
        [-1.8850, -1.8850, -1.8850],
        [-0.6283, -1.8849, -1.8850],
        [-0.5417, -1.8850, -1.8850],
        [-0.4257, -1.8850, -1.8850],
        [-0.4257, -1.8850, -1.8850],
        [-0.4257, -1.8850, -1.8850],
        [-0.4257, -1.8366, -0.6283],
        [0.1013, -1.2685, 0.6283],
        [0.6283, -0.7004, 1.8850],
        [0.6356, -0.6824, 1.9275],
    ],
    dtype=float,
)


def interpolate_joint_path(waypoints, steps_per_segment=8):
    frames = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
        for alpha in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            frames.append(start + alpha * delta)
    frames.append(waypoints[-1])
    return np.asarray(frames)


def end_effector_points(joint_path):
    return np.asarray([link_points(q)[-1] for q in joint_path])


def draw_obstacles(ax):
    for obstacle in OBSTACLES:
        if obstacle.get("type", "circle") == "circle":
            patch = Circle(
                obstacle["center"],
                obstacle["radius"],
                facecolor="#d95f02",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.95,
                zorder=2,
            )
            ax.add_patch(patch)
        elif obstacle.get("type") in ("box", "aabb"):
            lower = np.asarray(obstacle["lower"], dtype=float)
            upper = np.asarray(obstacle["upper"], dtype=float)
            ax.fill(
                [lower[0], upper[0], upper[0], lower[0]],
                [lower[1], lower[1], upper[1], upper[1]],
                facecolor="#d95f02",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.95,
                zorder=2,
            )


def main():
    joint_path = interpolate_joint_path(GCS_WAYPOINTS)
    ee_path = end_effector_points(joint_path)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    reach = sum(L)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Planar 3R Reliability-Aware GCS Path")
    add_workspace_reliability_patches(
        ax,
        RELIABILITY_MAP_SAMPLES,
        alpha=0.72,
        base_zorder=-20,
    )

    workspace_boundary = Circle(
        (0.0, 0.0),
        reach,
        facecolor="none",
        edgecolor="#999999",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(workspace_boundary)
    draw_obstacles(ax)

    ax.scatter(*START_POINT, color="#1b9e77", s=70, marker="o", label="start", zorder=5)
    ax.scatter(*GOAL_POINT, color="#7570b3", s=80, marker="*", label="goal", zorder=5)
    ax.plot(ee_path[:, 0], ee_path[:, 1], color="#333333", linewidth=1.2, alpha=0.45, zorder=1)

    arm_line, = ax.plot([], [], color="black", linewidth=3.0, marker="o", markersize=5, zorder=4)
    trace_line, = ax.plot([], [], color="#1f78b4", linewidth=2.0, zorder=3)
    ee_dot = ax.scatter([], [], color="#1f78b4", s=35, zorder=6)
    ax.legend(loc="upper right")

    def update(frame_index):
        q = joint_path[frame_index]
        points = np.asarray(link_points(q))
        arm_line.set_data(points[:, 0], points[:, 1])
        trace_line.set_data(ee_path[: frame_index + 1, 0], ee_path[: frame_index + 1, 1])
        ee_dot.set_offsets([ee_path[frame_index]])
        return arm_line, trace_line, ee_dot

    animation = FuncAnimation(
        fig,
        update,
        frames=len(joint_path),
        interval=70,
        blit=True,
    )
    animation.save(OUTPUT_GIF, writer=PillowWriter(fps=14))
    plt.close(fig)
    print(f"saved {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
