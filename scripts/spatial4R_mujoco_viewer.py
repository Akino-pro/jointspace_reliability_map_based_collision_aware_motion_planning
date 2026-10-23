"""MuJoCo kinematic replay of saved spatial 4R paths, using their standard DH model.

No planning, dynamic integration, or reference-script execution is performed.
Both DH offset and link segments have the planner's capsule collision geometry.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scripts.planar3R_mujoco_viewer import (
    _numbers, pose_at, render, set_pose, show_viewer, trajectory,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT / 'results/spatial4r/spatial4R_ftwmax_midpoint_two_stage_path.json'


@dataclass
class SpatialReplayScene:
    waypoints: np.ndarray
    a: np.ndarray
    d: np.ndarray
    alpha: np.ndarray
    obstacles: list
    link_radius: float
    goal: np.ndarray
    source: str = ""
    robot_name: str = "spatial4R"
    failed_joint: int | None = None


def kinematics(scene, q):
    """Return the standard-DH tool transform and all eight capsule segments."""
    transform = np.eye(4)
    segments = []
    for theta, a, d, alpha in zip(q, scene.a, scene.d, scene.alpha):
        origin = transform[:3, 3].copy()
        after_d = origin + d*transform[:3, 2]
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        end = after_d + a*(c*transform[:3, 0] + s*transform[:3, 1])
        segments.extend(((origin, after_d), (after_d, end)))
        transform = transform @ np.array([
            [c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
            [0, sa, ca, d], [0, 0, 0, 1],
        ])
    return transform, segments


def load_scene(path):
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = PROJECT / path
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("success") is False:
        raise ValueError("Expected a successful spatial 4R planning report")
    points = np.asarray(saved["waypoints"], dtype=float)
    robot = saved["robot"]
    a, d, alpha = (np.asarray(robot[name], dtype=float) for name in ("a", "d", "alpha"))
    radius = float(saved["link_radius"])
    goal = np.asarray(saved["goal_point"], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 4 or not len(points)
            or not np.all(np.isfinite(points))
            or any(v.shape != (4,) or not np.all(np.isfinite(v)) for v in (a, d, alpha))
            or goal.shape != (3,) or not np.all(np.isfinite(goal))
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Expected finite 4R waypoints, standard DH arrays, XYZ goal and positive radius")
    obstacles = saved.get("obstacles", [])
    for obstacle in obstacles:
        center = np.asarray(obstacle.get("center"), dtype=float)
        r = float(obstacle.get("radius", np.nan))
        if (obstacle.get("type") != "sphere" or center.shape != (3,)
                or not np.all(np.isfinite(center)) or not np.isfinite(r) or r <= 0):
            raise ValueError("Expected spherical obstacles with finite XYZ centers and positive radii")
    failed = saved.get("failed_joint")
    if failed is not None and (failed not in (1, 2, 3, 4) or np.ptp(points[:, failed-1]) > 1e-8):
        raise ValueError("Post-failure replay must preserve the indicated locked joint")
    return SpatialReplayScene(points, a, d, alpha, obstacles, radius, goal,
                              str(path.resolve()), saved.get("robot_parameter_set", "spatial4R"), failed)


def collision_free(scene, q):
    for first, second in kinematics(scene, q)[1]:
        delta = second-first
        squared = delta @ delta
        for obstacle in scene.obstacles:
            center = np.asarray(obstacle["center"])
            fraction = 0. if squared < 1e-24 else np.clip((center-first) @ delta / squared, 0, 1)
            if np.linalg.norm(first+fraction*delta-center) <= obstacle["radius"]+scene.link_radius:
                return False
    return True


def scene_bounds(scene, points=None, times=None):
    if points is None:
        points, times = trajectory(scene)
    positions = [np.zeros(3), scene.goal]
    for t in np.linspace(0, times[-1], 120):
        positions.extend(p for segment in kinematics(scene, pose_at(points, times, t))[1] for p in segment)
    for obstacle in scene.obstacles:
        center = np.asarray(obstacle["center"])
        positions.extend((center-obstacle["radius"], center+obstacle["radius"]))
    return np.min(positions, axis=0), np.max(positions, axis=0)


def camera_for(scene):
    import mujoco
    lower, upper = scene_bounds(scene)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (lower+upper)/2
    camera.distance = max(1., float(np.linalg.norm(upper-lower))*1.65)
    camera.azimuth, camera.elevation = 135, -25
    return camera


def make_mjcf(scene, points=None, times=None, width=960, height=720):
    if points is None:
        points, times = trajectory(scene)
    lower, upper = scene_bounds(scene, points, times)
    root = ET.Element("mujoco", model="spatial4r_gcs_replay")
    ET.SubElement(root, "compiler", angle="radian", autolimits="false")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="0.002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.7 0.7 0.7")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="floor_texture", type="2d", builtin="checker",
                  rgb1="0.17 0.20 0.24", rgb2="0.21 0.24 0.28", width="128", height="128")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_texture", texrepeat="8 8")
    world = ET.SubElement(root, "worldbody")
    floor_z = float(lower[2]-max(0.15, 3*scene.link_radius))
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 0.1",
                  pos=_numbers([0, 0, floor_z]), material="floor_mat", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos=_numbers([0, -2, upper[2]+4]), dir="0 0 -1")
    ET.SubElement(world, "geom", name="base_mount", type="cylinder", size="0.07 0.035",
                  rgba="0.65 0.7 0.75 1", contype="0", conaffinity="0")
    for i, obstacle in enumerate(scene.obstacles):
        ET.SubElement(world, "geom", name=f"obstacle{i}", type="sphere",
                      pos=_numbers(obstacle["center"]), size=str(obstacle["radius"]),
                      rgba="0.9 0.22 0.12 1", contype="2", conaffinity="1")
    start = kinematics(scene, points[0])[0][:3, 3]
    for name, xyz, color in (("start", start, "0.25 0.75 1 0.85"),
                              ("goal", scene.goal, "0.3 1 0.45 0.9")):
        ET.SubElement(world, "site", name=f"{name}_marker", type="sphere",
                      pos=_numbers(xyz), size="0.035", rgba=color)
    count = min(300, max(2, int(times[-1]*40)+1))
    trace = [kinematics(scene, pose_at(points, times, t))[0][:3, 3]
             for t in np.linspace(0, times[-1], count)]
    for i, (first, second) in enumerate(zip(trace[:-1], trace[1:])):
        if np.linalg.norm(second-first) > 1e-9:
            ET.SubElement(world, "geom", name=f"trace{i}", type="capsule", size="0.006",
                          fromto=_numbers([*first, *second]), rgba="0.5 0.85 1 0.65",
                          contype="0", conaffinity="0", group="3")
    parent = world
    colors = ["0.1 0.55 0.95 1", "0.1 0.8 0.65 1", "1 0.68 0.15 1", "0.7 0.4 0.95 1"]
    for i, (a, d, alpha) in enumerate(zip(scene.a, scene.d, scene.alpha)):
        rotor = ET.SubElement(parent, "body", name=f"rotor{i}")
        ET.SubElement(rotor, "joint", name=f"joint{i+1}", type="hinge", axis="0 0 1", limited="false")
        ET.SubElement(rotor, "geom", name=f"joint_marker{i+1}", type="sphere", size="0.028",
                      rgba="1 0.15 0.15 1" if scene.failed_joint == i+1 else "0.85 0.89 0.94 1",
                      contype="0", conaffinity="0")
        ends = [np.zeros(3), np.array([0., 0., d]), np.array([a, 0., d])]
        for j, (first, second) in enumerate(zip(ends[:-1], ends[1:])):
            attrs = dict(name=f"arm_link{i+1}_{j}", size=str(scene.link_radius),
                         rgba=colors[i], contype="1", conaffinity="2")
            if np.linalg.norm(second-first) <= 1e-12:
                ET.SubElement(rotor, "geom", **attrs, type="sphere", pos=_numbers(first))
            else:
                ET.SubElement(rotor, "geom", **attrs, type="capsule", fromto=_numbers([*first, *second]))
        # Standard DH: Rz(q) Tz(d) Tx(a) Rx(alpha), with a fixed frame after each rotor.
        parent = ET.SubElement(rotor, "body", name=f"dh_frame{i+1}", pos=_numbers([a, 0, d]),
                               quat=_numbers([np.cos(alpha/2), np.sin(alpha/2), 0, 0]))
    ET.SubElement(parent, "site", name="tool_tip", pos="0 0 0", size="0.025", rgba="1 1 0.9 1")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="planned_start", qpos=_numbers(points[0]))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def check_replay(scene, model, data, points, times):
    import mujoco
    count = max(2, int(np.ceil(times[-1]*60))+1)
    collisions = contacts = 0
    fk_error = 0.
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    for t in np.linspace(0, times[-1], count):
        q = pose_at(points, times, t)
        set_pose(model, data, q)
        collisions += not collision_free(scene, q)
        contacts += any(contact.dist < 0 for contact in data.contact)
        fk_error = max(fk_error, float(np.linalg.norm(data.site_xpos[tip]-kinematics(scene, q)[0][:3, 3])))
    set_pose(model, data, points[0])
    return dict(sampled_configurations=count, collision_samples=int(collisions),
                mujoco_contact_samples=int(contacts), maximum_mujoco_fk_error=fk_error,
                endpoint_error=float(np.linalg.norm(kinematics(scene, points[-1])[0][:3, 3]-scene.goal)),
                angular_windings_preserved=True,
                method="prescribed GCS segments; sampled capsule-sphere checks, not dynamic tracking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--render", action="store_true", help="Export PNG/GIF instead of opening a window")
    parser.add_argument("--output-dir", default=str(PROJECT/"visualizations"/"spatial4R_midpoint_mujoco"))
    parser.add_argument("--speed", type=float, default=0.5, help="Playback multiplier; default half speed")
    parser.add_argument("--angular-speed", type=float, default=0.65)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--viewer-seconds", type=float)
    args = parser.parse_args()
    if (not np.isfinite(args.speed) or args.speed <= 0 or args.fps <= 0
            or args.width < 64 or args.height < 64
            or (args.viewer_seconds is not None
                and (not np.isfinite(args.viewer_seconds) or args.viewer_seconds <= 0))):
        parser.error("Use finite positive playback settings and valid image dimensions")
    import mujoco
    scene = load_scene(args.path)
    points, times = trajectory(scene, args.angular_speed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    xml = make_mjcf(scene, points, times, args.width, args.height)
    (output/"spatial4R_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    report = dict(source=scene.source, robot=scene.robot_name, mujoco_version=mujoco.__version__,
                  failed_joint=scene.failed_joint,
                  waypoints=len(scene.waypoints), duration_seconds=float(times[-1]),
                  playback_speed=args.speed, **check_replay(scene, model, data, points, times))
    if report["collision_samples"] or report["mujoco_contact_samples"]:
        print("Warning: sampled collisions in the supplied path; displaying it unchanged.", flush=True)
    if args.render:
        report["rendered_frames"] = render(scene, model, data, points, times, output,
                                            args.speed, args.fps, args.width, args.height,
                                            camera_factory=camera_for, prefix="spatial4R", title="Spatial 4R")
    else:
        show_viewer(scene, model, data, points, times, args.speed, args.fps,
                    not args.once, args.viewer_seconds, camera_factory=camera_for)
    (output/"replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"MuJoCo artifacts: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
