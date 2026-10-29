"""Native MuJoCo replay of the saved full-pose 7R experiment and its exact DH geometry."""

import argparse
import json
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET

import numpy as np

from scripts.planar3R_mujoco_viewer import _numbers, render, set_pose, show_viewer, trajectory
from scripts.spatial4R_mujoco_viewer import (
    PROJECT, SpatialReplayScene, camera_for as spatial_camera_for, check_replay, kinematics, load_scene as load_spatial_scene,
    make_mjcf as make_spatial_mjcf,
)

DEFAULT_PATH = PROJECT / "results/spatial7r/spatial7R_60cm90deg_two_stage_path.json"


def camera_for(scene):
    camera = spatial_camera_for(scene)
    camera.distance *= .72
    return camera


def load_scene(path):
    scene = load_spatial_scene(path, expected_joints=7)
    saved = json.loads(Path(scene.source).read_text(encoding="utf-8"))
    goal = np.asarray(saved["goal_pose"], dtype=float)
    if (goal.shape != (4, 4) or not np.all(np.isfinite(goal))
            or not np.allclose(goal[3], [0, 0, 0, 1])
            or not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(goal[:3, :3]), 1)
            or not np.allclose(goal[:3, 3], scene.goal, atol=1e-9)):
        raise ValueError("Expected a valid full goal pose matching the saved goal position")
    scene.goal_pose = goal
    scene.robot_name = "spatial7R"
    scene.label_sites = True
    return scene


def make_mjcf(scene, points=None, times=None, width=1200, height=900):
    if points is None:
        points, times = trajectory(scene)
    root = ET.fromstring(make_spatial_mjcf(scene, points, times, width, height))
    root.set("model", "spatial7r_full_pose_gcs_replay")
    world = root.find("worldbody")
    if getattr(scene, "pose_preview", False):
        root.set("model", "spatial7r_60cm_baseZ90deg_pose_preview")
        for geom in list(world.findall("geom")):
            if geom.get("name", "").startswith("trace"):
                world.remove(geom)
    # Endpoint coordinate frames show the full-pose task, including orientation.
    for name, pose in (("start", kinematics(scene, points[0])[0]), ("goal", scene.goal_pose)):
        origin = pose[:3, 3]
        for axis, color in enumerate(("1 .25 .25 .9", ".25 1 .25 .9", ".25 .5 1 .9")):
            end = origin + .18*pose[:3, axis]
            ET.SubElement(world, "geom", name=f"{name}_axis{axis}", type="capsule",
                          size=".004", fromto=_numbers([*origin, *end]), rgba=color,
                          contype="0", conaffinity="0", group="3")
            ET.SubElement(world, "site", name=f"{name.upper()}_{'XYZ'[axis]}",
                          type="sphere", size=".003", pos=_numbers(end), rgba=color)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def render_overview(scene, model, data, points, output, width, height):
    """Paired actual MuJoCo renders with a shared camera and exact world coordinates."""
    import mujoco
    from PIL import Image, ImageDraw
    camera = camera_for(scene)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    options.label = mujoco.mjtLabel.mjLABEL_SITE
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for label, q in (("START CONFIGURATION", points[0]), ("GOAL CONFIGURATION", points[-1])):
            set_pose(model, data, q)
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((12, 12, 510, 65), fill=(24, 30, 39))
            draw.text((24, 22), "7R | MuJoCo | " + label, fill="white")
            draw.text((24, 43), "XYZ axes: X red / Y green / Z blue | Red sphere: obstacle", fill=(190, 218, 240))
            frames.append(frame)
        camera.lookat[:] = (kinematics(scene, points[0])[0][:3, 3]+scene.goal)/2
        camera.distance = .95
        set_pose(model, data, points[0])
        renderer.update_scene(data, camera=camera, scene_option=options)
        closeup = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(closeup)
        draw.rectangle((12, 12, 540, 65), fill=(24, 30, 39))
        draw.text((24, 22), "7R | MuJoCo | START / GOAL / OBSTACLE CLOSE-UP", fill="white")
        draw.text((24, 43), "XYZ axes: X red / Y green / Z blue | Red sphere: obstacle", fill=(190, 218, 240))
        closeup.save(output / "spatial7R_goal_obstacle_closeup.png")
    combined = Image.new("RGB", (width*2, height+100), (24, 30, 39))
    for index, frame in enumerate(frames):
        combined.paste(frame, (width*index, 0))
    draw = ImageDraw.Draw(combined)
    start = kinematics(scene, points[0])[0][:3, 3]
    draw.text((22, height+15), f"World XYZ (m) | start: {np.round(start, 4)} | goal: {np.round(scene.goal, 4)}", fill="white")
    for index, obstacle in enumerate(scene.obstacles):
        draw.text((22, height+40+20*index),
                  f"Obstacle {index+1}: center {np.round(obstacle['center'], 4)}, radius {obstacle['radius']:.3f} m",
                  fill=(255, 160, 130))
    combined.save(output / "spatial7R_start_goal_overview.png")
    set_pose(model, data, points[0])


def show_pose_viewer(scene, model, data):
    """Inspect the two endpoint poses without inventing a connecting trajectory."""
    import mujoco
    import mujoco.viewer
    commands = queue.SimpleQueue()
    set_pose(model, data, scene.waypoints[0])
    print("MuJoCo pose preview: S start | G goal | mouse rotate/zoom | Esc close. No planned path.", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put) as viewer:
        with viewer.lock():
            camera = camera_for(scene)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        while viewer.is_running():
            while not commands.empty():
                key = commands.get_nowait()
                if key in (ord("S"), ord("G")):
                    with viewer.lock():
                        set_pose(model, data, scene.waypoints[0 if key == ord("S") else -1])
            viewer.sync()
            time.sleep(1/30)


def preview_scene(args):
    import mujoco
    from scripts import spatial7R_model as planning
    saved = json.loads((PROJECT / "results/spatial7r/spatial7R_60cm90deg_scene.json").read_text())
    scene = SpatialReplayScene(np.array([saved["seed_path"][0], saved["reference"]]),
                               planning.A, planning.D, planning.ALPHA, saved["obstacles"],
                               planning.LINK_RADIUS, np.array(saved["goal"])[:3, 3],
                               robot_name="spatial7R")
    scene.goal_pose = np.array(saved["goal"])
    scene.label_sites, scene.pose_preview = True, True
    points, times = trajectory(scene)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    xml = make_mjcf(scene, points, times, args.width, args.height)
    (output / "spatial7R_pose_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for q in points:
        set_pose(model, data, q)
        if any(c.dist < 0 for c in data.contact):
            raise ValueError("Endpoint pose has a MuJoCo collision")
    report = dict(mode="endpoint_pose_preview_only_no_planned_trajectory",
                  start_pose=kinematics(scene, points[0])[0].tolist(),
                  goal_pose=scene.goal_pose.tolist(), obstacles=scene.obstacles,
                  translation_metres=float(np.linalg.norm(scene.goal-kinematics(scene, points[0])[0][:3, 3])),
                  orientation="R_goal = Rz(+90 degrees about base Z) @ R_start",
                  endpoint_collision_free=True)
    (output / "scene_preview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.render:
        render_overview(scene, model, data, points, output, args.width, args.height)
    else:
        show_pose_viewer(scene, model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--render", action="store_true", help="Export MuJoCo PNG/GIF and a paired overview")
    parser.add_argument("--scene-only", action="store_true", help="Inspect new task endpoints without a planned path")
    parser.add_argument("--output-dir", default=str(PROJECT / "visualizations/spatial7R_60cm90deg_mujoco"))
    parser.add_argument("--speed", type=float, default=.15)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--viewer-seconds", type=float)
    args = parser.parse_args()
    if (not np.isfinite(args.speed) or args.speed <= 0 or args.fps <= 0
            or min(args.width, args.height) < 64
            or (args.viewer_seconds is not None and
                (not np.isfinite(args.viewer_seconds) or args.viewer_seconds <= 0))):
        parser.error("Use finite positive playback settings and valid image dimensions")
    import mujoco
    if args.scene_only:
        preview_scene(args)
        return
    if (args.path == str(DEFAULT_PATH) and
            (not DEFAULT_PATH.exists() or not json.loads(DEFAULT_PATH.read_text()).get("success"))):
        print("Current task has no successful saved path; displaying endpoint poses only.", flush=True)
        preview_scene(args)
        return
    scene = load_scene(args.path)
    points, times = trajectory(scene)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    xml = make_mjcf(scene, points, times, args.width, args.height)
    (output / "spatial7R_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    report = dict(source=scene.source, robot=scene.robot_name, mujoco_version=mujoco.__version__,
                  **check_replay(scene, model, data, points, times))
    final = kinematics(scene, points[-1])[0]
    report.update(start_point=kinematics(scene, points[0])[0][:3, 3].tolist(),
                  start_pose=kinematics(scene, points[0])[0].tolist(), goal_pose=scene.goal_pose.tolist(),
                  orientation_axes="Local tool XYZ axes in world coordinates; X red, Y green, Z blue",
                  goal_point=scene.goal.tolist(), obstacles=scene.obstacles,
                  goal_rotation_matrix_error=float(np.linalg.norm(final[:3, :3]-scene.goal_pose[:3, :3])),
                  playback_speed=args.speed, duration_seconds=float(times[-1]))
    if report["collision_samples"] or report["mujoco_contact_samples"]:
        raise ValueError("Saved path failed MuJoCo capsule-sphere collision checks")
    if args.render:
        render_overview(scene, model, data, points, output, args.width, args.height)
        report["rendered_frames"] = render(
            scene, model, data, points, times, output, args.speed, args.fps,
            args.width, args.height, camera_factory=camera_for, prefix="spatial7R", title="Spatial 7R")
    else:
        (output / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        show_viewer(scene, model, data, points, times, args.speed, args.fps, True,
                    args.viewer_seconds, camera_factory=camera_for)
    (output / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
