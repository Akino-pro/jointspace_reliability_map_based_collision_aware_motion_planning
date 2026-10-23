import numpy as np
import pydot
import time

import pydrake.solvers as drake_solvers
from pydrake.geometry.optimization import (
    Point,
)
from pydrake.solvers import (
    L2NormCost,
    LinearConstraint,
)

from gcs.base import BaseGCS


def _typed_binding(*candidate_names):
    for name in candidate_names:
        if hasattr(drake_solvers, name):
            return getattr(drake_solvers, name)
    raise AttributeError(f"Could not find Drake typed Binding in {candidate_names}.")


try:
    BindingCost = drake_solvers.Binding[drake_solvers.Cost]
    BindingConstraint = drake_solvers.Binding[drake_solvers.Constraint]
except (AttributeError, TypeError, KeyError):
    # Compatibility with the older Drake bindings used by previous runs.
    BindingCost = _typed_binding("Binding𝓣Cost𝓤", "_TemporaryName_N5drake7solvers7BindingINS0_4CostEEE")
    BindingConstraint = _typed_binding("Binding𝓣Constraint𝓤", "_TemporaryName_N5drake7solvers7BindingINS0_10ConstraintEEE")


class LinearGCS(BaseGCS):
    def __init__(
        self,
        regions,
        edges=None,
        path_weights=None,
        full_dim_overlap=False,
        vertex_cost_weights=None,
    ):
        BaseGCS.__init__(self, regions)

        if path_weights is None:
            path_weights = np.ones(self.dimension)
        elif isinstance(path_weights, float) or isinstance(path_weights, int):
            path_weights = path_weights * np.ones(self.dimension)
        assert len(path_weights) == self.dimension
        self.path_weights = np.asarray(path_weights, dtype=float)

        self.edge_cost = L2NormCost(
            np.hstack((np.diag(-path_weights), np.diag(path_weights))),
            np.zeros(self.dimension))
        if vertex_cost_weights is None:
            vertex_cost_weights = np.ones(len(self.regions))
        self.vertex_cost_weights = np.asarray(vertex_cost_weights, dtype=float)
        assert len(self.vertex_cost_weights) == len(self.regions)

        for i, r in enumerate(self.regions):
            self.gcs.AddVertex(r, name = self.names[i] if not self.names is None else '')

        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        vertices = self.gcs.Vertices()
        self.edge_shifts = {}
        for edge_index, region_edge in enumerate(edges):
            ii, jj = region_edge
            shift = np.asarray(getattr(region_edge, "shift", np.zeros(self.dimension)))
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})_{edge_index}")
            self.edge_shifts[edge.name()] = shift
            weighted_edge_cost = L2NormCost(
                np.hstack((
                    np.diag(-self.path_weights * self.vertex_cost_weights[ii]),
                    np.diag(self.path_weights * self.vertex_cost_weights[ii]),
                )),
                shift * self.path_weights * self.vertex_cost_weights[ii]
            )

            edge.AddCost(BindingCost(
                weighted_edge_cost, np.append(u.x(), v.x())))

            # Constrain point in v to be in u
            edge.AddConstraint(BindingConstraint(
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()-u.set().A()@shift),
                v.x()))

    def addSourceTarget(self, source, target, edges=None):
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        for edge in source_edges:
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

        for edge in target_edges:
            source_name = edge.u().name()
            source_index = self.names.index(source_name)
            weighted_edge_cost = L2NormCost(
                np.hstack((
                    np.diag(-self.path_weights * self.vertex_cost_weights[source_index]),
                    np.diag(self.path_weights * self.vertex_cost_weights[source_index]),
                )),
                np.zeros(self.dimension)
            )
            edge.AddCost(BindingCost(
                weighted_edge_cost, np.append(edge.xu(), edge.xv())))


    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # Extract trajectory
        from scripts.periodic_graph import lift_path
        points, _ = lift_path(best_path, best_result, self.edge_shifts, self.dimension)
        waypoints = points.T

        return waypoints, results_dict
