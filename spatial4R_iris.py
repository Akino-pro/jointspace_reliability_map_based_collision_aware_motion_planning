"""Drake standard-DH plant and cached 4D IRIS regions for the spatial 4R arm."""

import time

import numpy as np

import spatial4R_model as model
from planar3R_iris_regions import ReliablePolytope, load_iris_checkpoint, save_iris_checkpoint


def make_urdf(obstacles=model.OBSTACLES, link_radius=model.LINK_RADIUS):
    links = ['<link name="base"/>']
    joints = []
    for i, (a, d, alpha) in enumerate(zip(model.A, model.D, model.ALPHA)):
        collisions = []
        points = [np.zeros(3), np.array([0., 0., d]), np.array([a, 0., d])]
        for first, second in zip(points[:-1], points[1:]):
            length = np.linalg.norm(second-first)
            if length <= 1e-12:
                continue
            center = ' '.join(map(str, (first+second)/2))
            rpy = '0 1.5707963267948966 0' if abs(a) > 0 and abs(second[0]-first[0]) > 0 else '0 0 0'
            collisions.append(f'<collision><origin xyz="{center}" rpy="{rpy}"/><geometry>'
                              f'<cylinder radius="{link_radius}" length="{length}"/></geometry></collision>')
            for endpoint in (first, second):
                xyz = ' '.join(map(str, endpoint))
                collisions.append(f'<collision><origin xyz="{xyz}"/><geometry>'
                                  f'<sphere radius="{link_radius}"/></geometry></collision>')
        links.append(f'<link name="rotor{i}"><inertial><mass value="1"/>'
                     '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>'
                     f'</inertial>{"".join(collisions)}</link>')
        links.append(f'<link name="frame{i+1}"/>')
        parent = 'base' if i == 0 else f'frame{i}'
        joints.append(f'<joint name="joint{i}" type="revolute"><parent link="{parent}"/>'
                      f'<child link="rotor{i}"/><axis xyz="0 0 1"/>'
                      '<limit lower="-3.141592653589793" upper="3.141592653589793" effort="10" velocity="10"/></joint>')
        joints.append(f'<joint name="dh{i}" type="fixed"><parent link="rotor{i}"/>'
                      f'<child link="frame{i+1}"/><origin xyz="{a} 0 {d}" rpy="{alpha} 0 0"/></joint>')
    for i, obstacle in enumerate(obstacles):
        if obstacle.get("type") != "sphere":
            raise ValueError("Spatial 4R IRIS supports spheres")
        xyz = ' '.join(map(str, obstacle['center']))
        links.append(f'<link name="obstacle{i}"><collision><geometry>'
                     f'<sphere radius="{obstacle["radius"]}"/></geometry></collision></link>')
        joints.append(f'<joint name="obstacle_fixed{i}" type="fixed"><parent link="base"/>'
                      f'<child link="obstacle{i}"/><origin xyz="{xyz}"/></joint>')
    return f'<robot name="spatial4r">{"".join(links)}{"".join(joints)}</robot>'


def build_plant(obstacles=model.OBSTACLES, link_radius=model.LINK_RADIUS):
    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    instance = Parser(plant).AddModelsFromString(make_urdf(obstacles, link_radius), "urdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", instance))
    plant.Finalize()
    if plant.num_positions() != 4:
        raise RuntimeError("Expected four spatial joint positions")
    geometries = []
    for i in range(4):
        geometries.extend(plant.GetCollisionGeometriesForBody(plant.GetBodyByName(f"rotor{i}", instance)))
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(GeometrySet(geometries)))
    return builder.Build(), plant, scene_graph


def scene_metadata(obstacles=model.OBSTACLES):
    return dict(dh=model.PARAMETERS, obstacles=obstacles, link_radius=model.LINK_RADIUS,
                geometry="DH offset and link capsules", self_collision=False,
                chart=[-float(np.pi), float(np.pi)], physical_joint_limits=None)


def generate_regions(output, obstacles=model.OBSTACLES, max_regions=6, seeds=None, coverage_radius=None):
    from pathlib import Path
    from pydrake.geometry.optimization import IrisOptions, HPolyhedron
    if (max_regions < 1 or (coverage_radius is not None
                           and (not np.isfinite(coverage_radius) or coverage_radius <= 0))):
        raise ValueError("Use a positive region budget and positive finite optional coverage radius")
    try:
        from pydrake.geometry.optimization import IrisInConfigurationSpace as inflate
    except ImportError:
        from pydrake.geometry.optimization import IrisNp as inflate
    metadata = dict(model=scene_metadata(obstacles), generation_seconds=0., processed_seeds=[])
    metadata["coverage_radius"] = coverage_radius
    if Path(output).exists():
        regions, saved = load_iris_checkpoint(output, expected_dimension=4)
        if saved["model"] != metadata["model"]:
            raise ValueError("Cached spatial IRIS geometry differs from the requested scene")
        if saved.get("coverage_radius") != coverage_radius:
            raise ValueError("Cached IRIS inflation domains differ; use another checkpoint filename")
        return regions, saved
    if seeds is None:
        seeds = [model.START_CONFIGURATION, model.GOAL_REFERENCE_CONFIGURATION]
        seeds += [(1-t)*model.START_CONFIGURATION + t*model.GOAL_REFERENCE_CONFIGURATION
                  for t in (0.25, 0.5, 0.75)]
    diagram, plant, _ = build_plant(obstacles)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    options = IrisOptions()
    options.require_sample_point_is_contained = True
    options.iteration_limit = 5
    options.num_collision_infeasible_samples = 60
    options.configuration_space_margin = 0.015
    options.random_seed = 0
    regions = []
    for index, seed in enumerate(seeds):
        seed = np.asarray(seed)
        if len(regions) >= max_regions:
            break
        if not model.collision_free(seed, obstacles):
            print(f"Spatial IRIS seed {index}: collision, skipped", flush=True)
            continue
        if any(r.contains(seed, tol=-1e-5) for r in regions):
            continue
        began = time.perf_counter()
        print(f"Spatial IRIS seed {index}: inflating 4D region", flush=True)
        plant.SetPositions(plant_context, seed)
        if coverage_radius is not None:
            options.bounding_region = HPolyhedron.MakeBox(
                np.maximum(-np.pi, seed-coverage_radius), np.minimum(np.pi, seed+coverage_radius))
        inflated = inflate(plant, plant_context, options)
        regions.append(ReliablePolytope(inflated.A(), inflated.b()))
        metadata["generation_seconds"] += time.perf_counter()-began
        metadata["processed_seeds"].append(index)
        save_iris_checkpoint(output, regions, metadata)
        print(f"Spatial IRIS: {len(regions)} regions, {time.perf_counter()-began:.2f}s", flush=True)
    if not regions:
        raise RuntimeError("No collision-free IRIS region was generated")
    return regions, metadata
