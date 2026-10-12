import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from planar3R_collision import L, is_collision_free
from planar3R_gcs_planner import GOAL_POINT, OBSTACLES, START_POINT
from planar3R_two_point_ssm_views import compute_ssms_for_target
from planar3R_iris_regions import ReliablePolytope, load_iris_checkpoint, save_iris_checkpoint


OUTPUT_REGIONS = "planar3R_iris_regions.npz"


def make_planar3r_urdf(link_lengths, obstacle, link_radius=0.02):
    # Full-length thin boxes contain every point of the analytic link segment.
    # Inflate the circular obstacle by the analytic link radius; the thin box
    # adds a small conservative margin without shortening either end of a link.
    link_width = 0.002
    link_height = 0.002
    obstacle_radius = obstacle["radius"] + link_radius
    obstacle_x, obstacle_y = obstacle["center"]

    link_xml = []
    joint_xml = []
    for i, length in enumerate(link_lengths, start=1):
        collision_length = length
        link_xml.append(
            f"""
  <link name="link{i}">
    <inertial>
      <origin xyz="{length / 2.0} 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="{length / 2.0} 0 0"/>
      <geometry>
        <box size="{collision_length} {link_width} {link_height}"/>
      </geometry>
    </collision>
    <visual>
      <origin xyz="{length / 2.0} 0 0"/>
      <geometry>
        <box size="{collision_length} {link_width} {link_height}"/>
      </geometry>
    </visual>
  </link>"""
        )
        parent = "base" if i == 1 else f"link{i - 1}"
        origin = "0 0 0" if i == 1 else f"{link_lengths[i - 2]} 0 0"
        joint_xml.append(
            f"""
  <joint name="joint{i}" type="revolute">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="{origin}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.141592653589793" upper="3.141592653589793" effort="10" velocity="10"/>
  </joint>"""
        )

    return f"""<?xml version="1.0"?>
<robot name="planar3r">
  <link name="base"/>
  {''.join(link_xml)}
  <link name="obstacle">
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="{obstacle_radius}" length="1.0"/>
      </geometry>
    </collision>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="{obstacle_radius}" length="1.0"/>
      </geometry>
    </visual>
  </link>
  {''.join(joint_xml)}
  <joint name="obstacle_fixed" type="fixed">
    <parent link="base"/>
    <child link="obstacle"/>
    <origin xyz="{obstacle_x} {obstacle_y} 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def build_drake_plant(link_lengths, obstacle, link_radius=0.02):
    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    urdf_text = make_planar3r_urdf(link_lengths, obstacle, link_radius)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as urdf_file:
        urdf_file.write(urdf_text)
        urdf_path = urdf_file.name

    try:
        model = parser.AddModels(urdf_path)[0]
    except Exception:
        Path(urdf_path).unlink(missing_ok=True)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", model))
    plant.Finalize()
    if plant.num_positions() != 3:
        Path(urdf_path).unlink(missing_ok=True)
        raise RuntimeError("The planar 3R model must have exactly three joint positions.")
    # The existing analytic collision checker checks arm-obstacle collisions,
    # not arm self-collision. Match that choice explicitly in Drake.
    arm_ids = []
    for i in range(1, 4):
        arm_ids.extend(plant.GetCollisionGeometriesForBody(plant.GetBodyByName(f"link{i}", model)))
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(GeometrySet(arm_ids))
    )
    diagram = builder.Build()
    return diagram, plant, scene_graph, Path(urdf_path)


def candidate_iris_seeds(max_per_point=5, link_radius=0.02):
    # Include the most recent GCS waypoints as useful corridor seeds.
    corridor_waypoints = [
        [-2.9730, -2.2695, -1.2364],
        [-2.7610, -2.2152, -0.6283],
        [-2.3230, -2.1030, 0.6283],
        [-1.8850, -1.9909, 1.8850],
        [-0.6283, -1.8850, 1.8849],
        [0.6283, -0.7224, 1.8850],
        [0.6356, -0.6824, 1.9275],
    ]
    # Prioritize both endpoints and corridor coverage before other SSM samples.
    seeds = [np.asarray(corridor_waypoints[0]), np.asarray(corridor_waypoints[-1])]
    for q in corridor_waypoints[1:-1]:
        q = np.asarray(q, dtype=float)
        seeds.append(q)
    for first, second in zip(corridor_waypoints[:-1], corridor_waypoints[1:]):
        seeds.append(0.5 * (np.asarray(first) + np.asarray(second)))
    for point, seed in [(START_POINT, 0), (GOAL_POINT, 1)]:
        for ssm in compute_ssms_for_target(*point, seed=seed):
            stride = max(1, len(ssm) // max_per_point)
            seeds.extend(ssm[::stride][:max_per_point])

    unique = []
    for q in seeds:
        if not is_collision_free(q, OBSTACLES, link_radius=link_radius):
            continue
        if not any(np.linalg.norm(q - old_q) < 1e-3 for old_q in unique):
            unique.append(q)
    return unique


def generate_iris_regions(max_regions, output_path, resume=False, iteration_limit=5, link_radius=0.02,
                          collision_samples=100, configuration_margin=0.02):
    from pydrake.geometry.optimization import IrisOptions
    try:
        from pydrake.geometry.optimization import IrisInConfigurationSpace as inflate_region
    except ImportError:
        # Current Drake renamed the same plant/context/options API to IrisNp.
        from pydrake.geometry.optimization import IrisNp as inflate_region

    if max_regions < 1 or iteration_limit < 1 or link_radius < 0 or collision_samples < 1 or configuration_margin < 0:
        raise ValueError("Region/iteration limits must be positive and link radius nonnegative.")
    if len(OBSTACLES) != 1 or OBSTACLES[0].get("type", "circle") != "circle":
        raise ValueError("The current IRIS URDF supports exactly one circular obstacle.")
    model = {
        "link_lengths": list(L), "link_radius": link_radius,
        "obstacles": [{"type": "circle", "center": list(OBSTACLES[0]["center"]),
                       "radius": OBSTACLES[0]["radius"]}],
        "geometry_version": 2, "self_collision": False,
    }
    regions = []
    generation_options = {"iteration_limit": iteration_limit, "collision_samples": collision_samples,
                          "configuration_margin": configuration_margin}
    metadata = {"model": model, "processed_seeds": [], "generation_seconds": 0.0,
                "options": generation_options}
    output_path = Path(output_path)
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"{output_path} exists; use --resume or a different --output.")
        regions, metadata = load_iris_checkpoint(output_path, allow_empty=True)
        if metadata.get("model") != model or metadata.get("options") != generation_options:
            raise ValueError("Checkpoint model/options differ; choose a new output path.")
    seeds = candidate_iris_seeds(link_radius=link_radius)
    seed_signature = json.dumps([q.tolist() for q in seeds])
    if resume and metadata.get("seed_signature", seed_signature) != seed_signature:
        raise ValueError("Checkpoint seed sequence differs; choose a new output path.")
    metadata["seed_signature"] = seed_signature
    print(f"candidate seeds: {len(seeds)}, cached regions: {len(regions)}", flush=True)
    if len(regions) >= max_regions:
        return regions
    diagram, plant, _, urdf_path = build_drake_plant(L, OBSTACLES[0], link_radius)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    options = IrisOptions()
    options.require_sample_point_is_contained = True
    options.iteration_limit = iteration_limit
    options.termination_threshold = -1
    options.relative_termination_threshold = 0.02
    options.num_collision_infeasible_samples = collision_samples
    options.configuration_space_margin = configuration_margin
    options.random_seed = 0

    try:
        for seed_index, seed in enumerate(seeds):
            if len(regions) >= max_regions:
                break
            if seed_index in metadata["processed_seeds"]:
                continue
            if any(region.contains(seed, tol=-1e-4) for region in regions):
                metadata["processed_seeds"].append(seed_index)
                save_iris_checkpoint(output_path, regions, metadata)
                print(f"seed {seed_index}: already covered", flush=True)
                continue
            start = time.perf_counter()
            print(f"seed {seed_index}: starting IRIS at {seed.tolist()}", flush=True)
            plant.SetPositions(plant_context, seed)
            try:
                region = inflate_region(plant, plant_context, options)
            except Exception as exc:
                print(f"seed {seed_index} failed: {exc}", flush=True)
                metadata["processed_seeds"].append(seed_index)
                save_iris_checkpoint(output_path, regions, metadata)
                continue
            elapsed = time.perf_counter() - start
            print(f"seed {seed_index}: region {len(regions) + 1}, time={elapsed:.3f}s", flush=True)
            regions.append(ReliablePolytope(region.A(), region.b()))
            metadata["processed_seeds"].append(seed_index)
            metadata["generation_seconds"] += elapsed
            save_iris_checkpoint(output_path, regions, metadata)
    finally:
        urdf_path.unlink(missing_ok=True)

    print(f"saved {len(regions)} IRIS regions to {output_path}", flush=True)
    return regions


def parse_args():
    parser = argparse.ArgumentParser(description="Generate planar 3R IRIS regions in Drake.")
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--output", default=OUTPUT_REGIONS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--link-radius", type=float, default=0.02)
    parser.add_argument("--collision-samples", type=int, default=100)
    parser.add_argument("--configuration-margin", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_iris_regions(args.max_regions, args.output, args.resume, args.iterations, args.link_radius,
                          args.collision_samples, args.configuration_margin)


if __name__ == "__main__":
    main()
