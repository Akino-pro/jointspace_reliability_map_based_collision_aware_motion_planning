"""Replay saved planar 3R GCS paths in MuJoCo; optionally export PNG/GIF previews.

This is prescribed kinematic playback, not a dynamics/controller simulation.
The original, unwrapped joint-space segments are preserved exactly.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET

import numpy as np

from scripts.planar3R_collision import is_collision_free, link_points

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT / 'results/planar3r/two_stage_all_paths200_trials300.json'
ARM_HEIGHT = 0.12


@dataclass
class ReplayScene:
    waypoints: np.ndarray
    lengths: np.ndarray
    obstacles: list
    link_radius: float
    goal: np.ndarray
    source: str = ""
    failed_joint: int | None = None


def load_scene(path):
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = PROJECT / path
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("success") is False:
        raise ValueError("The selected report does not contain a successful planned path")
    points = np.asarray(saved["waypoints"], dtype=float)
    lengths = np.asarray(saved.get("link_lengths", [1, 1, 1]), dtype=float)
    radius = float(saved.get("link_radius", 0.02))
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.all(np.isfinite(points)) or lengths.shape != (3,)
            or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0)
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Expected finite 3R waypoints, positive link lengths and link radius")
    goal = np.asarray(saved.get("goal_point", link_points(points[-1], lengths)[-1]), dtype=float)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("Expected a finite XY goal")
    obstacles = saved.get("obstacles", [])
    for obstacle in obstacles:
        kind = obstacle.get("type", "circle")
        if kind == "circle":
            center = np.asarray(obstacle["center"], dtype=float)
            if center.shape != (2,) or not np.all(np.isfinite(center)) or not np.isfinite(obstacle["radius"]) or obstacle["radius"] <= 0:
                raise ValueError("Invalid circular obstacle")
        elif kind in ("box", "aabb"):
            lower, upper = np.asarray(obstacle["lower"]), np.asarray(obstacle["upper"])
            if lower.shape != (2,) or upper.shape != (2,) or not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
                raise ValueError("Invalid box obstacle")
        else:
            raise ValueError(f"Unsupported planar obstacle: {kind}")
    failed = saved.get("failed_joint")
    if failed is not None:
        if failed not in (1, 2, 3) or np.ptp(points[:, failed-1]) > 1e-8:
            raise ValueError("Post-failure path must keep exactly the indicated joint locked")
    return ReplayScene(points, lengths, obstacles, radius, goal, str(path.resolve()), failed)


def trajectory(scene, angular_speed=0.65):
    if not np.isfinite(angular_speed) or angular_speed <= 0:
        raise ValueError("Angular playback speed must be positive")
    points = scene.waypoints
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    points = points[keep]
    # Do NOT wrap deltas: these are the actual straight segments planned by GCS.
    times = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))/angular_speed]
    return points, times


def pose_at(points, times, elapsed):
    return np.array([np.interp(elapsed, times, points[:, i]) for i in range(points.shape[1])])


def _numbers(values):
    return " ".join(f"{float(v):.16g}" for v in values)


def make_mjcf(scene, points=None, times=None, width=960, height=720):
    if points is None:
        points, times = trajectory(scene)
    root = ET.Element("mujoco", model="planar3r_gcs_replay")
    ET.SubElement(root, "compiler", angle="radian", autolimits="false")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="0.002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.6 0.6 0.6")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="floor_texture", type="2d", builtin="checker",
                  rgb1="0.17 0.20 0.24", rgb2="0.21 0.24 0.28", width="128", height="128")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_texture", texrepeat="8 8",
                  reflectance="0.08")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 0.1", pos="0 0 -0.02",
                  material="floor_mat", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="base_mount", type="cylinder", pos="0 0 0.04",
                  size="0.09 0.06", rgba="0.6 0.65 0.72 1", contype="0", conaffinity="0")
    for i, obstacle in enumerate(scene.obstacles):
        attrs = dict(name=f"obstacle{i}", contype="2", conaffinity="1", rgba="0.9 0.22 0.12 1")
        if obstacle.get("type", "circle") == "circle":
            ET.SubElement(world, "geom", **attrs, type="cylinder",
                          pos=_numbers([*obstacle["center"], 0.14]), size=_numbers([obstacle["radius"], 0.18]))
        else:
            lower, upper = np.asarray(obstacle["lower"]), np.asarray(obstacle["upper"])
            ET.SubElement(world, "geom", **attrs, type="box",
                          pos=_numbers([*((lower+upper)/2), 0.14]), size=_numbers([*((upper-lower)/2), 0.18]))
    start_xy = link_points(scene.waypoints[0], scene.lengths)[-1]
    for name, xy, color in (("start", start_xy, "0.25 0.75 1 0.75"),
                            ("goal", scene.goal, "0.3 1 0.45 0.9")):
        ET.SubElement(world, "site", name=f"{name}_marker", type="sphere",
                      pos=_numbers([*xy, ARM_HEIGHT+0.035]), size="0.045", rgba=color)
    count = min(300, max(2, int(times[-1]*30)+1))
    trace = [link_points(pose_at(points, times, t), scene.lengths)[-1]
             for t in np.linspace(0, times[-1], count)]
    for i, (first, second) in enumerate(zip(trace[:-1], trace[1:])):
        if np.linalg.norm(second-first) > 1e-9:
            ET.SubElement(world, "geom", name=f"trace{i}", type="capsule", size="0.007",
                          fromto=_numbers([*first, 0.04, *second, 0.04]), rgba="0.5 0.85 1 0.65",
                          contype="0", conaffinity="0", group="3")
    parent = world
    colors = ["0.1 0.55 0.95 1", "0.10 0.8 0.65 1", "1 0.68 0.15 1"]
    for i, length in enumerate(scene.lengths):
        pos = [0, 0, ARM_HEIGHT] if i == 0 else [scene.lengths[i-1], 0, 0]
        body = ET.SubElement(parent, "body", name=f"link{i+1}", pos=_numbers(pos))
        ET.SubElement(body, "joint", name=f"joint{i+1}", type="hinge", axis="0 0 1", limited="false")
        ET.SubElement(body, "geom", name=f"arm_link{i+1}", type="capsule", size=str(scene.link_radius),
                      fromto=_numbers([0, 0, 0, length, 0, 0]), rgba=colors[i], contype="1", conaffinity="2")
        color = "1 0.15 0.15 1" if scene.failed_joint == i+1 else "0.85 0.89 0.94 1"
        ET.SubElement(body, "geom", name=f"joint_marker{i+1}", type="cylinder", size="0.04 0.025",
                      rgba=color, contype="0", conaffinity="0")
        parent = body
    ET.SubElement(parent, "site", name="tool_tip", pos=_numbers([scene.lengths[-1], 0, 0]),
                  size="0.03", rgba="1 1 0.9 1")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="planned_start", qpos=_numbers(scene.waypoints[0]))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def camera_for(scene):
    import mujoco
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, ARM_HEIGHT]
    camera.distance = float(sum(scene.lengths)*2.65)
    camera.azimuth = 90
    camera.elevation = -90
    return camera


def set_pose(model, data, q):
    import mujoco
    data.qpos[:] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def check_replay(scene, points, times):
    count = max(2, int(np.ceil(times[-1]*60))+1)
    collisions = 0
    for t in np.linspace(0, times[-1], count):
        q = pose_at(points, times, t)
        collisions += not is_collision_free(q, scene.obstacles, scene.lengths, scene.link_radius)
    return dict(sampled_configurations=count, collision_samples=int(collisions),
                endpoint_error=float(np.linalg.norm(link_points(scene.waypoints[-1], scene.lengths)[-1]-scene.goal)),
                angular_windings_preserved=True,
                method="prescribed GCS segments; sampled analytic checks, not dynamic tracking")


def render(scene, model, data, points, times, output, speed=1, fps=24, width=960, height=720,
           camera_factory=camera_for, prefix="planar3R", title="Planar 3R"):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    duration = max(1., float(times[-1])/speed)
    count = max(2, int(np.ceil(duration*fps))+1)
    camera = camera_factory(scene)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with imageio.get_writer(str(output/f"{prefix}_mujoco.gif"), format="GIF-PIL",
                                mode="I", duration=1/fps, loop=0,
                                palettesize=128, subrectangles=True) as writer:
            for index, elapsed in enumerate(np.linspace(0, times[-1], count)):
                set_pose(model, data, pose_at(points, times, elapsed))
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((12, 12, 405, 62), fill=(24, 30, 39))
                draw.text((23, 22), f"{title} | MuJoCo | GCS trajectory replay", fill=(235, 241, 249))
                draw.text((23, 42), f"t = {elapsed:.2f} s    green: goal    red: obstacle", fill=(182, 209, 231))
                writer.append_data(np.asarray(frame))
                for label, key in (("start", 0), ("middle", count//2), ("goal", count-1)):
                    if index == key:
                        frame.save(output/f"{prefix}_mujoco_{label}.png")
                if index % max(1, fps*2) == 0:
                    print(f"MuJoCo export: {index+1}/{count} frames", flush=True)
    return count


def show_viewer(scene, model, data, points, times, speed=1., fps=60, loop=True, viewer_seconds=None,
                camera_factory=camera_for):
    import mujoco.viewer
    commands = queue.SimpleQueue()
    elapsed, paused = 0., False
    print("MuJoCo controls: Space pause/resume | R restart | L loop | T trace | mouse rotate/zoom | Esc close", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 1
            camera = camera_factory(scene)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        began = previous = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            if viewer_seconds is not None and now-began >= viewer_seconds:
                break
            while not commands.empty():
                key = commands.get_nowait()
                if key == 32:
                    paused = not paused
                elif key in (ord("R"), ord("r")):
                    elapsed = 0.
                elif key in (ord("L"), ord("l")):
                    loop = not loop
                elif key in (ord("T"), ord("t")):
                    with viewer.lock():
                        viewer.opt.geomgroup[3] = not viewer.opt.geomgroup[3]
            if not paused:
                elapsed += (now-previous)*speed
            previous = now
            if loop and times[-1] > 0:
                elapsed %= times[-1]
            else:
                elapsed = min(elapsed, times[-1])
            with viewer.lock():
                set_pose(model, data, pose_at(points, times, elapsed))
                data.time = elapsed
            viewer.sync()
            time.sleep(max(0., 1/fps-(time.perf_counter()-now)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH), help="Successful planar pre/post-failure JSON")
    parser.add_argument("--render", action="store_true", help="Export MuJoCo PNG/GIF instead of opening a window")
    parser.add_argument("--output-dir", default=str(PROJECT/"visualizations"/"planar3R_mujoco"))
    parser.add_argument("--speed", type=float, default=1., help="Playback speed multiplier")
    parser.add_argument("--angular-speed", type=float, default=0.65, help="Nominal joint-space replay speed, rad/s")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--once", action="store_true", help="Stop playback at goal instead of repeating")
    parser.add_argument("--viewer-seconds", type=float, help="Optional automatic viewer close time")
    args = parser.parse_args()
    if (not np.isfinite(args.speed) or args.speed <= 0 or args.fps <= 0
            or args.width < 64 or args.height < 64
            or (args.viewer_seconds is not None and args.viewer_seconds <= 0)):
        parser.error("Use positive playback/frame settings and valid image dimensions")
    import mujoco
    scene = load_scene(args.path)
    points, times = trajectory(scene, args.angular_speed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    xml = make_mjcf(scene, points, times, args.width, args.height)
    (output/"planar3R_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    set_pose(model, data, points[0])
    report = dict(source=scene.source, mujoco_version=mujoco.__version__,
                  waypoints=len(scene.waypoints), duration_seconds=float(times[-1]),
                  failed_joint=scene.failed_joint, **check_replay(scene, points, times))
    if report["collision_samples"]:
        print("Warning: supplied trajectory has sampled collisions; displaying the saved path unchanged.", flush=True)
    if args.render:
        report["rendered_frames"] = render(scene, model, data, points, times, output,
                                            args.speed, args.fps, args.width, args.height)
    else:
        show_viewer(scene, model, data, points, times, args.speed, args.fps,
                    not args.once, args.viewer_seconds)
    (output/"replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"MuJoCo artifacts: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
