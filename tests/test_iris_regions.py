import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import networkx as nx

from planar3R_drake_iris import make_planar3r_urdf
from planar3R_two_point_ssm_views import refine_ik_configuration
from planar3R_collision import link_points
from planar3R_iris_gcs_planner import select_connected_candidates, validate_joint_path
from planar3R_iris_regions import (
    ReliablePolytope, load_iris_checkpoint, polytope_adjacency,
    save_iris_checkpoint, split_iris_regions,
)

try:
    from pydrake.geometry.optimization import HPolyhedron
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def box(lower, upper):
    return ReliablePolytope(np.vstack((np.eye(3), -np.eye(3))),
                            np.concatenate((upper, -np.asarray(lower))))


class IrisRegionTests(unittest.TestCase):
    def test_ssm_candidate_refinement_removes_endpoint_drift(self):
        reference = np.array([0.6, -0.7, 1.9])
        target = link_points(reference)[-1]
        corrected = refine_ik_configuration(reference + [0.002, -0.001, 0.001], *target)
        self.assertIsNotNone(corrected)
        self.assertLess(np.linalg.norm(link_points(corrected)[-1] - target), 1e-8)

    def test_candidate_budget_does_not_hide_connected_ssm_samples(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1])
        starts = [(np.zeros(3), [0]), (np.ones(3), [1])]
        goals = [(np.full(3, 2), [1])]
        selected_starts, selected_goals = select_connected_candidates(graph, starts, goals, 1)
        self.assertEqual(selected_starts[0][1], [1])
        self.assertEqual(selected_goals[0][1], [1])

    def test_intersection_preserves_slanted_collision_boundary(self):
        parent = box([-1, -1, -1], [1, 1, 1])
        parent.A = np.vstack((parent.A, [1, 1, 0]))
        parent.b = np.append(parent.b, 0.2)
        partitions = [
            {"lower": [0, 0, -1], "upper": [1, 1, 1], "reliability": 0.7},
            {"lower": [0.5, 0.5, -1], "upper": [1, 1, 1], "reliability": 0.9},
        ]
        children = split_iris_regions([parent], partitions)
        self.assertEqual(len(children), 1)
        self.assertAlmostEqual(children[0].reliability, 0.7)
        self.assertTrue(children[0].contains([0.05, 0.05, 0]))
        self.assertFalse(children[0].contains([0.8, 0.8, 0]))

    def test_adjacency_keeps_face_contacts_and_rejects_gaps(self):
        regions = [box([0, 0, 0], [1, 1, 1]), box([1, 0, 0], [2, 1, 1]),
                   box([2.1, 0, 0], [3, 1, 1])]
        self.assertEqual(polytope_adjacency(regions), [(0, 1), (1, 0)])

    def test_checkpoint_preserves_numeric_regions_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.npz"
            regions = [box([-1, -1, -1], [1, 1, 1])]
            metadata = {"processed_seeds": [0, 2], "model": {"link_lengths": [1, 1, 1]}}
            save_iris_checkpoint(path, regions, metadata)
            loaded, progress = load_iris_checkpoint(path)
            np.testing.assert_array_equal(loaded[0].A, regions[0].A)
            self.assertEqual(progress, metadata)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_path_check_detects_collision_between_free_endpoints(self):
        free, checked = validate_joint_path(
            np.array([[0, 0, 0], [np.pi / 2, 0, 0]]),
            [{"type": "circle", "center": [0.75, 0.75], "radius": 0.15}],
        )
        self.assertFalse(free)
        self.assertGreater(checked, 1)

    def test_path_check_does_not_wrap_the_gcs_angle_chart(self):
        free, _ = validate_joint_path(
            np.array([[-3.0, 0, 0], [3.0, 0, 0]]),
            [{"type": "circle", "center": [1.0, 0.0], "radius": 0.1}],
        )
        self.assertFalse(free)

    def test_urdf_does_not_shorten_collision_links(self):
        root = ET.fromstring(make_planar3r_urdf([1, 1, 1], {"center": [0.75, 0.75], "radius": 0.15}))
        for link in root.findall("link"):
            if link.attrib["name"].startswith("link"):
                geometry = link.find("collision/geometry/box")
                self.assertEqual(float(geometry.attrib["size"].split()[0]), 1.0)


@unittest.skipUnless(HAS_DRAKE, "Requires robotics Drake bindings")
class DrakeIntegrationTests(unittest.TestCase):
    def test_native_rounding_returns_a_connected_continuous_path(self):
        from gcs.linear import LinearGCS
        regions = [box([0, -0.2, -0.2], [1.2, 0.2, 0.2]),
                   box([0.8, -0.2, -0.2], [2, 0.2, 0.2])]
        planner = LinearGCS([r.to_drake() for r in regions], edges=[(0, 1), (1, 0)])
        planner.addSourceTarget(np.array([0, 0, 0]), np.array([2, 0, 0]))
        waypoints, result = planner.SolvePath(rounding=True, preprocessing=True)
        self.assertIsNotNone(waypoints)
        np.testing.assert_allclose(waypoints[:, 0], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(waypoints[:, -1], [2, 0, 0], atol=1e-6)
        self.assertAlmostEqual(result["rounded_cost"], 2.0, places=5)
        self.assertAlmostEqual(result["relaxation_cost"], 2.0, places=5)
        self.assertEqual(result["rounded_path_count"], 1)

    def test_plant_has_three_positions_and_fixed_base(self):
        from planar3R_drake_iris import build_drake_plant
        diagram, plant, _, urdf_path = build_drake_plant(
            [1, 1, 1], {"type": "circle", "center": [0.75, 0.75], "radius": 0.15}
        )
        try:
            self.assertEqual(plant.num_positions(), 3)
            context = diagram.CreateDefaultContext()
            plant_context = plant.GetMyContextFromRoot(context)
            position = plant.GetBodyByName("base").EvalPoseInWorld(plant_context).translation()
            np.testing.assert_allclose(position, [0, 0, 0])
        finally:
            urdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
