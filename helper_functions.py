import heapq

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
import trimesh
import scipy.integrate as spi
from functools import reduce
import operator
from shapely.geometry import Polygon, MultiPolygon


step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size


def union_ranges(ranges):
    """
    Computes the union of a list of ranges, with special handling for ranges that wrap around
    [-π, π]. If such ranges are detected, they are replaced by modified ranges.

    Parameters:
        ranges (list of tuples): List of ranges, where each range is represented as (a, b).

    Returns:
        list of tuples: The merged union of ranges as a list of non-overlapping ranges.
    """
    if not ranges:
        return []

    # Optional: Validate input ranges
    for a, b in ranges:
        if a > b:
            raise ValueError(f"Invalid range: ({a}, {b}). Start must be <= end.")

    # Sort the ranges by their starting point (and by end if start points are the same)
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

    # Initialize the merged ranges with the first range
    merged = [ranges[0]]
    merge_threshold = terminate_threshold
    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1] or (current[0] - last[1] <= merge_threshold):  # Handle small gaps
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged


def plot_shifted_arcs_on_sphere(angles, theta_ranges_per_point, samples_per_arc=50):
    """
    Plot arcs on the unit sphere by 'shifting' each point's theta over
    one or more intervals. The phi (colatitude) for each point remains fixed.

    Args:
        angles (list or np.ndarray): shape (n, 2), each entry is (theta_i, phi_i).
                                     - theta_i in [0, 2π)
                                     - phi_i in [0, π]
        theta_ranges_per_point (list of list of tuples): shape (n, ), where each element
            is a list of intervals [(tmin1, tmax1), (tmin2, tmax2), ...].
            Example: [
                [(0,1), (2,3)],  # intervals for point #1
                [(0, 2*np.pi)],  # intervals for point #2
                ...
            ]
        samples_per_arc (int): number of sample points when drawing each arc.

    Returns:
        None. Displays a 3D matplotlib figure with the arcs plotted on a unit sphere.
    """
    angles = np.asarray(angles)
    n_points = angles.shape[0]

    if len(theta_ranges_per_point) != n_points:
        raise ValueError("Length of theta_ranges_per_point must match number of angles.")

    # 1) Create a reference sphere mesh
    phi_s = np.linspace(0, np.pi, 40)
    theta_s = np.linspace(0, 2 * np.pi, 40)
    phi_s, theta_s = np.meshgrid(phi_s, theta_s)
    x_s = np.sin(phi_s) * np.cos(theta_s)
    y_s = np.sin(phi_s) * np.sin(theta_s)
    z_s = np.cos(phi_s)

    # 2) Prepare the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.2, linewidth=0)

    # 3) Plot arcs for each point
    # colors = plt.cm.rainbow(np.linspace(0, 1, n_points))  # for variety
    for i in range(n_points):
        theta_i, phi_i = angles[i]
        # We'll also plot the original point as a reference
        # x0, y0, z0 = spherical_to_cartesian(theta_i, phi_i)
        # ax.scatter([x0], [y0], [z0], color='b', s=40)

        intervals = theta_ranges_per_point[i]
        for (tmin, tmax) in intervals:
            # Sample points along [tmin, tmax]
            tvals = np.linspace(tmin, tmax, samples_per_arc)
            arc_x, arc_y, arc_z = [], [], []
            for t in tvals:
                xA, yA, zA = spherical_to_cartesian(t, phi_i)
                arc_x.append(xA)
                arc_y.append(yA)
                arc_z.append(zA)

            # Plot the arc
            ax.plot(arc_x, arc_y, arc_z, color='b', lw=6)

    # 4) Final touches
    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1]);
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Shifted Arcs on Unit Sphere")
    plt.show()


def fibonacci_sphere_angles(n):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)
    phi_angles = np.arcsin(z)  # in [-π/2, π/2]
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)
    result = []
    for i in range(n):
        result.append([theta_angles[i], phi_angles[i]])
    return result


def build_truncated_wedge(region_vertices, rmin, rmax, R_outer=1.0):
    """
    Given 'region_vertices' on a sphere of radius R_outer,
    build triangular faces for the wedge from r = rmin to r = rmax.
    Returns a list of triangular faces, each face is [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
    """
    poly_rmin = region_vertices * (rmin / R_outer)
    poly_rmax = region_vertices * (rmax / R_outer)

    faces = []

    # --- Top Face (at rmax) ---
    centroid_rmax = np.mean(poly_rmax, axis=0)
    for i in range(len(poly_rmax)):
        v1 = poly_rmax[i]
        v2 = poly_rmax[(i + 1) % len(poly_rmax)]
        faces.append([centroid_rmax, v1, v2])

    # --- Bottom Face (at rmin) ---
    if rmin > 0:
        centroid_rmin = np.mean(poly_rmin, axis=0)
        for i in range(len(poly_rmin)):
            v1 = poly_rmin[i]
            v2 = poly_rmin[(i + 1) % len(poly_rmin)]
            # Reverse to keep consistent normals
            faces.append([centroid_rmin, v2, v1])

    # --- Side Faces ---
    for i in range(len(region_vertices)):
        i_next = (i + 1) % len(region_vertices)
        vminA = poly_rmin[i]
        vminB = poly_rmin[i_next]
        vmaxA = poly_rmax[i]
        vmaxB = poly_rmax[i_next]
        # Two triangles for the side quadrilateral
        faces.append([vminA, vminB, vmaxB])
        faces.append([vminA, vmaxB, vmaxA])

    return faces


def find_voronoi_region(sv, p):
    """
    Find which Voronoi region (index) the point p (on the unit sphere)
    belongs to, by checking which generator in 'sv.points' is closest.

    Inputs:
        sv: SphericalVoronoi object
        p:  A 3D point on the sphere (x, y, z) ~ unit radius

    Returns:
        region_index: the index of the region in sv.regions to which p belongs.
                      This is effectively the index of the closest generator.
    """
    # Brute force: find which generator is closest in Euclidean distance
    # (for n=100 or so, this is perfectly fast enough.)
    points = sv.points  # shape (n, 3)
    # Distances from p to each generator
    dists = np.linalg.norm(points - p, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx


"""
def plot_extruded_regions_covered_by_arcs(theta_phi_list,
                                          theta_ranges_all,
                                          samples_per_arc=50,
                                          extrude_radius=2 * np.pi,
                                          facecolor='b',
                                          alpha=1.0):

    # 1) Convert angles to 3D points on the sphere
    points = []
    for (theta, phi) in theta_phi_list:
        x, y, z = spherical_to_cartesian(theta, phi)
        points.append([x, y, z])
    points = np.array(points)

    # 2) Build spherical Voronoi
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # We'll keep a set of region indices that are "visited" by any arc
    visited_region_indices = set()

    # 3) For each point i, arcs in theta_ranges_all[i]
    for i, intervals in enumerate(theta_ranges_all):
        # If no arcs are given for this point, skip
        if not intervals:
            continue

        # We'll use the phi_i from the generator
        theta_i, phi_i = theta_phi_list[i]

        for (tmin, tmax) in intervals:
            # sample the arc in small steps
            thetas = np.linspace(tmin, tmax, samples_per_arc)

            for t in thetas:
                # (x_arc, y_arc, z_arc) on the sphere
                x_arc, y_arc, z_arc = spherical_to_cartesian(t, phi_i)
                p_arc = np.array([x_arc, y_arc, z_arc])
                # find which region p_arc belongs to
                r_idx = find_voronoi_region(sv, p_arc)
                visited_region_indices.add(r_idx)

    # 4) Plot only visited regions, extruded out to radius=extrude_radius
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for r_idx in visited_region_indices:
        region = sv.regions[r_idx]
        region_vertices = sv.vertices[region]  # Nx3 on the unit sphere

        # Here we extrude from r=0 to r=extrude_radius
        wedge_faces = build_truncated_wedge(region_vertices,
                                            rmin=0,
                                            rmax=extrude_radius,
                                            R_outer=1.0)
        wedge_poly = Poly3DCollection(
            wedge_faces,
            alpha=alpha,
            edgecolor='b',
            facecolor=facecolor
        )
        ax.add_collection3d(wedge_poly)

    # Optionally, show the original sphere points
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, c='r', label='Generators')

    # Format the axes
    ax.set_box_aspect((1, 1, 1))
    # If extrude_radius=2π, let's keep them symmetrical
    R = extrude_radius
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Voronoi Regions Covered by Arcs, Extruded to r={extrude_radius}")
    plt.show()
    return ax
"""


def plot_voronoi_regions_on_sphere(theta_phi_list,
                                   region_colors,
                                   sm_ori
                                   ):
    """
       Plot spherical Voronoi regions on the unit sphere using an assigned color per region.

       - Edges are drawn in black (edgecolor='k').
       - Regions are filled with the corresponding color from region_colors[r_idx].

       Args:
           theta_phi_list (list of (float, float)):
               Spherical coordinates (theta, phi) for each generator on the unit sphere.
               (In radians, typically theta in [0, π], phi in [0, 2π].)
           region_colors (list of color-like values):
               A list of colors (strings, RGB tuples, etc.) of length = number of regions
               (the same as the number of generator points). region_colors[i] is used to
               color region i. Example: ['red', 'blue', 'green', ...].

       Returns:
           ax (matplotlib.axes._subplots.Axes3DSubplot):
               The Matplotlib 3D axes object with the plotted Voronoi regions.
       """

    # --------------------------------------------------------------------------
    # 1) Convert (theta, phi) -> (x,y,z) on the unit sphere
    #    (Adjust if your spherical coordinate convention differs.)
    # --------------------------------------------------------------------------

    # Convert generator angles to 3D points on the unit sphere
    points = []
    for (theta, phi) in theta_phi_list:
        x, y, z = spherical_to_cartesian(theta, phi)
        points.append([x, y, z])
    points = np.array(points)

    # --------------------------------------------------------------------------
    # 2) Build the SphericalVoronoi
    # --------------------------------------------------------------------------
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # Check the length of region_colors vs. the number of regions
    n_regions = len(sv.regions)
    if len(region_colors) != n_regions:
        raise ValueError(f"region_colors must have length = {n_regions}, but got {len(region_colors)}")

    # --------------------------------------------------------------------------
    # 3) Plot the regions on the unit sphere
    # --------------------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r_idx, region in enumerate(sv.regions):
        region_vertices = sv.vertices[region]
        fc = region_colors[r_idx]  # Assign color based on index
        alpha = 1 # Transparency level

        poly = Poly3DCollection([region_vertices],
                                facecolor=fc,
                                edgecolor='k',
                                alpha=alpha)
        ax.add_collection3d(poly)

    # Set up the axes for a unit sphere
    ax.set_box_aspect((1, 1, 1))
    limit = 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    #cbar = plt.colorbar(sm_ori, ax=ax)
    tick_values = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    tick_labels = [f'{int(t * 100)}%' for t in tick_values]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    cbar = plt.colorbar(sm_ori, ticks=tick_values,ax=ax)
    cbar.ax.tick_params(labelsize=20, length=10)
    plt.show()


def spherical_to_cartesian(theta, phi):
    """
    Using the 'latitude' convention:
       theta: azimuth in [0, 2π)
       phi:   latitude in [-π/2, π/2]
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    return x, y, z


def build_truncated_wedge(region_vertices, rmin, rmax, R_outer=1.0):
    """
    Build triangular faces that form a 'wedge' from r=rmin to r=rmax
    around the polygon given by region_vertices (on sphere of radius=R_outer).
    Returns a list of triangular faces: each face is [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
    """
    poly_rmin = region_vertices * (rmin / R_outer)
    poly_rmax = region_vertices * (rmax / R_outer)

    faces = []

    # Top Face (at rmax)
    centroid_rmax = np.mean(poly_rmax, axis=0)
    for i in range(len(poly_rmax)):
        v1 = poly_rmax[i]
        v2 = poly_rmax[(i + 1) % len(poly_rmax)]
        faces.append([centroid_rmax, v1, v2])

    # Bottom Face (at rmin)
    if rmin > 0:
        centroid_rmin = np.mean(poly_rmin, axis=0)
        for i in range(len(poly_rmin)):
            v1 = poly_rmin[i]
            v2 = poly_rmin[(i + 1) % len(poly_rmin)]
            # Reverse to keep consistent normals
            faces.append([centroid_rmin, v2, v1])

    # Side Faces
    for i in range(len(region_vertices)):
        i_next = (i + 1) % len(region_vertices)
        vminA = poly_rmin[i]
        vminB = poly_rmin[i_next]
        vmaxA = poly_rmax[i]
        vmaxB = poly_rmax[i_next]
        # Two triangles for the side quadrilateral
        faces.append([vminA, vminB, vmaxB])
        faces.append([vminA, vmaxB, vmaxA])

    return faces


def get_extruded_wedges(theta_phi_list,
                        alpha_lists,
                        extrude_radius=2 * np.pi,
                        do_plot=False,
                        color='b'):
    """
    Computes the extruded wedges for all arcs and returns them as a single list of triangular faces.
    If `do_plot` is True, also plots these wedges in a 3D figure.

    Returns:
        all_wedge_faces : list of triangles (each is 3 points in [x, y, z])
    """
    # 1) Convert angles to 3D points on the sphere
    points = []
    for (theta, phi) in theta_phi_list:
        x, y, z = spherical_to_cartesian(theta, phi)
        points.append([x, y, z])
    points = np.array(points)

    # 2) Build spherical Voronoi
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # We'll keep a set of region indices that are visited by any arc
    visited_region_indices = set()
    """
    # 3) For each point i, arcs in theta_ranges_all[i]
    for i, intervals in enumerate(theta_ranges_all):
        if not intervals:
            continue
        # We'll use phi_i from the i-th point
        theta_i, phi_i = theta_phi_list[i]

        for (tmin, tmax) in intervals:
            # Sample points along the arc in theta
            thetas = np.linspace(tmin, tmax, samples_per_arc)
            # Track visited regions for this arc so we only union once per region
            visited_for_this_arc = set()

            for t in thetas:
                x_arc, y_arc, z_arc = spherical_to_cartesian(t, phi_i)
                p_arc = np.array([x_arc, y_arc, z_arc])
                # Find which Voronoi region p_arc belongs to
                r_idx = find_voronoi_region(sv, p_arc)
                visited_for_this_arc.add(r_idx)

            # Union alpha_lists[r_idx] with alpha_lists[i] for each visited region
            for r_idx in visited_for_this_arc:
                visited_region_indices.add(r_idx)
                # Concatenate existing intervals from alpha_lists[r_idx] and alpha_lists[i]
                combined = alpha_lists[r_idx] + alpha_lists[i]
                # Use the union_ranges(...) function to merge them
                alpha_lists[r_idx] = union_ranges(combined)
    """
    # 3) Identify regions with non-empty alpha lists
    valid_region_indices = {i for i, alpha in enumerate(alpha_lists) if alpha}
    # 4) Build the wedge faces for each visited region
    all_wedge_faces = []
    for r_idx in valid_region_indices:
        region = sv.regions[r_idx]
        region_vertices = sv.vertices[region]  # Nx3 on the unit sphere

        wedge_faces = build_truncated_wedge(region_vertices,
                                            rmin=0,
                                            rmax=extrude_radius,
                                            R_outer=1.0)
        all_wedge_faces.extend(wedge_faces)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Add polygon collections
        wedge_poly = Poly3DCollection(all_wedge_faces,
                                      alpha=1.0,
                                      edgecolor='k',
                                      facecolor=color)
        ax.add_collection3d(wedge_poly)

        # Format the axes
        R = extrude_radius
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim([-R, R])
        ax.set_ylim([-R, R])
        ax.set_zlim([-R, R])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        #ax.set_title(f"Voronoi Regions Covered by Arcs, Extruded to r={extrude_radius}")
        plt.show()
    return all_wedge_faces


def wedge_faces_to_binary_volume(all_wedge_faces, NX=50, NY=50, NZ=50):
    """
    Given a list of triangular wedge faces (each face is [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]),
    build a single 3D mesh with trimesh, then produce an (NX, NY, NZ) occupancy grid.

    Returns:
        binary_volume: np.ndarray of shape (NX, NY, NZ), dtype=uint8
        xs, ys, zs   : 1D coordinate arrays for each axis
    """
    unique_vertices = []
    vertex_map = {}
    faces = []

    for tri in all_wedge_faces:
        face_indices = []
        for v in tri:
            v_tuple = (v[0], v[1], v[2])
            if v_tuple not in vertex_map:
                vertex_map[v_tuple] = len(unique_vertices)
                unique_vertices.append(v_tuple)
            face_indices.append(vertex_map[v_tuple])
        faces.append(face_indices)

    vertices = np.array(unique_vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    # Build and clean the trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # Build axis-aligned 3D grid from bounding box
    bounds = mesh.bounds  # shape (2,3)
    min_corner = bounds[0]
    max_corner = bounds[1]

    xs = np.linspace(min_corner[0], max_corner[0], NX)
    ys = np.linspace(min_corner[1], max_corner[1], NY)
    zs = np.linspace(min_corner[2], max_corner[2], NZ)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='xy')
    points_to_check = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    # Check which points are inside the mesh
    inside_mask = mesh.contains(points_to_check)
    inside_mask_3d = inside_mask.reshape((NX, NY, NZ))

    # Convert to a binary occupancy grid
    binary_volume = inside_mask_3d.astype(np.uint8)

    return binary_volume


def track_top_5():
    top_5 = []  # Min heap to keep track of top-5 numbers with indices and properties

    def update(num, index, property_value1, property_value2):
        if len(top_5) < 5:
            heapq.heappush(top_5, (num, index, property_value1, property_value2))
        else:
            heapq.heappushpop(top_5, (num, index, property_value1, property_value2))  # Maintain only top-5

    def get_top_5():
        return sorted(top_5, key=lambda x: x[0], reverse=True)  # Return sorted top-5 list

    return update, get_top_5

import matplotlib as mpl

"""
def normalize_and_map_colors(values, cmap_name='rainbow'):

    values = np.array(values)
    cmap = plt.get_cmap(cmap_name)

    # Map directly since values are normalized
    colors = [cmap(val) for val in values]

    # Create mappable for external colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return colors, sm

"""
def normalize_and_map_colors(values):
    values = np.array(values)

    base_cmap = plt.get_cmap('rainbow')

    # Take the green->yellow->red half of rainbow, then reverse it
    sampled = base_cmap(np.linspace(0.5, 1.0, 256))[::-1]
    cmap = LinearSegmentedColormap.from_list('rainbow_red_to_green', sampled)

    # Map directly since values are already normalized
    colors = [cmap(val) for val in values]

    # Create mappable for external colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return colors, sm

def make_red_yellow_green_cmap():
    base_cmap = plt.get_cmap("rainbow")

    # take upper half of rainbow, then reverse:
    # high value -> red, low value -> green
    sampled = base_cmap(np.linspace(0.5, 1.0, 256))[::-1]
    cmap = LinearSegmentedColormap.from_list("rainbow_red_to_green", sampled)
    return cmap


"""
from matplotlib.colors import ListedColormap
def normalize_and_map_colors(values, cmap_name='rainbow', shallow_factor=0.5, snap_eps=1e-8):
    
    n_levels = 256  # fixed constant for colorbar resolution

    values = np.asarray(values, dtype=float)
    cmap = plt.get_cmap(cmap_name)

    # Build matching colorbar colormap
    xs = np.linspace(0, 1, n_levels)
    arr = cmap(xs)
    mid_violet = cmap(0.0)
    strong_red = (1.0, 0.0, 0.0, 1.0)

    arr[0]  = mid_violet
    arr[-1] = strong_red
    white = np.array([1, 1, 1, 1.0])
    arr[1:-1] = arr[1:-1]*(1 - shallow_factor) + white*shallow_factor

    custom_cmap = ListedColormap(arr)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    # Colors for data
    colors = []
    for v in values:
        if v <= 0 + snap_eps:
            colors.append(mid_violet)
        elif v >= 1 - snap_eps:
            colors.append(strong_red)
        else:
            base = np.array(cmap(v))
            lighter = base*(1 - shallow_factor) + white*shallow_factor
            colors.append(tuple(lighter))

    return colors, sm
"""

def normalize_and_map_greyscale(values, snap_eps=1e-8):
    """
    Map normalized reliability values [0,1] to grayscale colors:
      - v=0 → white
      - v=1 → black
    Returns:
        colors: list of RGBA tuples
        sm: ScalarMappable for colorbar
    """

    values = np.asarray(values, dtype=float)

    # ----- Build grayscale spectrum -----
    # 256 levels from white→black
    greys = np.linspace(1.0, 0.0, 256)  # 1 = white, 0 = black
    arr = np.zeros((256, 4))
    arr[:, 0] = greys
    arr[:, 1] = greys
    arr[:, 2] = greys
    arr[:, 3] = 1.0  # alpha

    custom_cmap = mcolors.ListedColormap(arr)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # scalar mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    # ----- Assign individual colors -----
    colors = []
    for v in values:
        if v <= 0 + snap_eps:
            gray = 1.0  # white
        elif v >= 1 - snap_eps:
            gray = 0.0  # black
        else:
            gray = 1.0 - v  # linear mapping

        colors.append((gray, gray, gray, 1.0))

    return colors, sm


def normalize_and_map_colors_green(values, cmap_name='rainbow', vmin=0, vmax=2*np.pi):
    """
    Maps unnormalized values to colors using a colormap.

    Parameters:
        values (list or np.array): Raw values (not normalized).
        cmap_name (str or Colormap): Colormap instance or name.
        vmin (float): Minimum value for normalization.
        vmax (float): Maximum value for normalization.

    Returns:
        colors (list): RGBA colors.
        ScalarMappable: For external use in colorbar.
    """
    values = np.array(values)
    cmap = plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(val)) for val in values]

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return colors, sm


"""
### Example Usage:
values = [10, 20, 30, 40, 50, 60, 70]
colors, sm = normalize_and_map_colors(values)

fig, ax = plt.subplots()
sc = ax.scatter(values, np.zeros_like(values), c=colors, s=100)
cbar = plt.colorbar(sm, ax=ax, label='Value Spectrum')  # Ensure colorbar is linked to the mappable

plt.show()
"""

def create_shallow_to_deep_green_colormap():
    green_colors = ['#ccffcc', '#66cc66', '#339933', '#006600']  # light → deep green
    return LinearSegmentedColormap.from_list('shallow_deep_green', green_colors, N=256)

def compute_length_of_ranges(ranges_list):
    shallow_deep_green_cmap = create_shallow_to_deep_green_colormap()
    length_list = []
    for ranges in ranges_list:
        if len(ranges) == 0:
            length_list.append(0)
        else:
            list_sum = 0
            for single_range in ranges:
                list_sum += (single_range[1] - single_range[0])
            length_list.append(list_sum)
    color_list, sm = normalize_and_map_colors_green(length_list, cmap_name=shallow_deep_green_cmap)
    return color_list, sm


def update_or_add_square(ax2, square, color, alpha_level):
    """
    Check if a square already exists in ax2.
    - If found, update its color and transparency.
    - If not found, add a new Poly3DCollection to ax2.
    """
    square_array = np.array(square)  # Convert to NumPy array for comparison

    for collection in ax2.collections:
        if isinstance(collection, Poly3DCollection):
            # Extract vertices of existing collections
            for path in collection.get_paths():
                existing_verts = np.array(path.vertices)  # Convert to NumPy array

                # Check if the square already exists in ax2
                if existing_verts.shape == square_array.shape and np.allclose(existing_verts, square_array):
                    collection.set_facecolor(color)  # Update color
                    collection.set_alpha(alpha_level)  # Update transparency
                    plt.draw()  # Refresh the plot
                    return  # Stop function since update was successful

    # If not found, add the new square
    square_poly = Poly3DCollection([square], color=color, alpha=alpha_level)
    ax2.add_collection3d(square_poly)


def update_or_add_square_2d(ax2, square, color, alpha_level, index, index_dict):
    """
    Check if a square with the given index exists in ax2.
    - If found, update its color and transparency.
    - If not found, add a new Polygon to ax2 and store its reference.

    Parameters:
    - ax2: Matplotlib 2D Axes object.
    - square: List of (x, z) tuples representing the square's corners.
    - color: Color of the square.
    - alpha_level: Transparency level (0 = fully transparent, 1 = opaque).
    - index: Unique identifier for the square.
    - index_dict: Dictionary mapping indices to existing patches in ax2.
    """
    square_array = np.array(square)  # Convert input square to NumPy array

    # Check if the index already exists in the dictionary
    if index in index_dict:
        patch = index_dict[index]
        if isinstance(patch, patches.Polygon):
            existing_verts = np.array(patch.get_xy())  # Get existing polygon vertices

            # Ensure correct shape comparison (ignore the extra closing vertex)
            if existing_verts.shape[0] == 5 and existing_verts[:-1].shape == square_array.shape:
                if np.allclose(existing_verts[:-1], square_array):
                    if color == 'w':
                        return  # Skip updating but don't exit the function

                    patch.set_facecolor(color)  # Update color
                    patch.set_alpha(alpha_level)  # Ensure transparency is updated
                    plt.draw()  # Refresh the plot
                    return  # Stop function since update was successful

    # If the square is not found or index is new, add the new square
    square_poly = patches.Polygon(
        square,
        facecolor=color,
        edgecolor="black",  # Keep edges visible
        alpha=alpha_level,
        linewidth=0.3
    )
    ax2.add_patch(square_poly)
    index_dict[index] = square_poly  # Store the new patch reference in the dictionary
    plt.draw()


def sorted_indices(lst):
    return [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1])]


def union_ranges(ranges):
    #ranges = [sublist for sublist in ranges if sublist]
    if not ranges:
        return []

    # Optional: Validate input ranges
    for a, b in ranges:
        if a > b:
            raise ValueError(f"Invalid range: ({a}, {b}). Start must be <= end.")

    # Sort the ranges by their starting point (and by end if start points are the same)
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

    # Initialize the merged ranges with the first range
    merged = [ranges[0]]
    merge_threshold = 9.0 / 5.0 * 0.01
    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1] or (current[0] - last[1] <= merge_threshold):  # Handle small gaps
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged

def computing_6d_volume(alpha_ranges,beta_ranges,x,No,Sx,list_ft):
    VF=0
    for i in range(No):
        if list_ft[i]==1:
            non_overlapping_beta_ranges=beta_ranges[i]
            vx=0
            for beta_range in non_overlapping_beta_ranges:
                vx+=x[0]*Sx*(beta_range[1]-beta_range[0])
            non_overlapping_alpha_ranges=alpha_ranges[i]
            vo=0
            for alpha_range in non_overlapping_alpha_ranges:
                vo += 4*np.pi*(np.power(alpha_range[1],3)-np.power(alpha_range[0],3))/(3*No)
            VF+=(vx*vo)
    return VF

def plot_workspace(workspace, color='k'):
    """
    Plot the workspace in 2D using solid points.

    Parameters:
    workspace : List of (x, y) positions representing the workspace
    title : Title of the plot
    color : Color of the workspace points
    """
    plt.figure(figsize=(8, 8))
    # if not hasattr(thread_local, 'plt_fig'):
    #    thread_local.plt_fig = plt.figure(figsize=(8, 8))
    plt.plot(workspace[:, 0], workspace[:, 1], '.', color=color,
             markersize=15)  # Solid points instead of scatter
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.axis('off')
    plt.draw()  # Ensure the plot is rendered

    # Get the RGBA buffer from the figure
    buf = np.frombuffer(plt.gcf().canvas.buffer_rgba(), dtype=np.uint8)
    img_data = buf.reshape(plt.gcf().canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)

    # Assuming binary image: check if R, G, and B channels are either 0 or 255
    grayscale_matrix = (img_data[..., 0] == 255).astype(np.uint8) * 255  # Convert white (255) to 255, black to 0

    plt.close()  # Close the plot
    return grayscale_matrix

def compute_reliability(ion_list,reliability_list):
    total=0.0
    current=0.0
    full_product = reduce(operator.mul, reliability_list, 1)
    for index, r in enumerate(reliability_list):
        total+=(1.0-r)*full_product/r
        if ion_list[index]:
            current+=(1.0-r)*full_product/r
    return current / total if total != 0 else 0.0


def compute_reliability_by_f_list(F_list,reliability_list):
    total=0.0
    current=0.0
    full_product = reduce(operator.mul, reliability_list, 1)
    if 0 in F_list:
        current += full_product  # prefailure point
    for index, r in enumerate(reliability_list):
        total+=(1.0-r)*full_product/r
        if index+1 in F_list:
            current+=(1.0-r)*full_product/r
    return current / (total+full_product) if total != 0 else 0.0

import matplotlib.cm as cm
import matplotlib.colors as mcolors
def color_by_reliability(values):
    """
    Map a list of values in [0, 1] to colors using the 'rainbow' colormap.

    Parameters:
        values (list or array-like): List of float values in the range [0, 1].

    Returns:
        color_list (list): List of RGBA tuples from the rainbow colormap.
        sm (ScalarMappable): ScalarMappable object for colorbar usage.
    """
    # Normalize values to 0-1 range
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('rainbow')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    color_list = [cmap(norm(v)) for v in values]

    return color_list, sm


import math

def plot_alpha_beta_ranges(theta_phi_list, alpha_ranges_list, beta_ranges_list):
    """
    Plots a bar graph with equally spaced individual bars for each alpha and beta range
    across all (theta, phi) tuples. Alpha bars are light blue; Beta bars are light purple.
    Only 20 (theta, phi) labels are shown on the x-axis for clarity.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 0.4
    alpha_color = '#a6c8ff'  # light blue
    beta_color = '#d1b3ff'   # light purple

    num_pairs = len(theta_phi_list)
    total_bars = num_pairs * 2
    x_positions = np.arange(total_bars)

    # Determine step for sparse labeling (show only 20 tuple labels)
    max_labels = 20
    label_step = max(1, math.ceil(num_pairs / max_labels))

    label_positions = []
    label_texts = []
    for i, (theta, phi) in enumerate(theta_phi_list):
        if i % label_step == 0:
            label_positions.append(i * 2 + 0.5)
            label_texts.append(f"({theta:.2f}, {phi:.2f})")

    for i in range(num_pairs):
        alpha_idx = i * 2
        beta_idx = i * 2 + 1

        # Alpha bar
        if alpha_ranges_list[i]:
            for start, end in alpha_ranges_list[i]:
                ax.bar(alpha_idx, end - start, bottom=start, width=bar_width,
                       color=alpha_color, edgecolor='blue')
        else:
            ax.bar(alpha_idx, 0, bottom=0, width=bar_width, color='white')

        # Beta bar
        if beta_ranges_list[i]:
            for start, end in beta_ranges_list[i]:
                ax.bar(beta_idx, end - start, bottom=start, width=bar_width,
                       color=beta_color, edgecolor='purple')
        else:
            ax.bar(beta_idx, 0, bottom=0, width=bar_width, color='white')

    # Labeling
    ax.set_ylabel("Failure Tolerant Rotation Angles of Alpha and Beta")
    ax.set_xlabel("Theta, Phi Tuples")
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=45, ha='right')

    # Y-ticks
    tick_positions = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

     #Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=alpha_color, edgecolor='blue', label='α Range'),
        Patch(facecolor=beta_color, edgecolor='purple', label='β Range')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Axis labels
    ax.set_ylabel("Failure Tolerant Rotation Angles of α and β", fontsize=27)  # 3x size
    ax.set_xlabel(r"φ, ψ Tuples", fontsize=27)

    # Tick parameters (2x larger)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)

    # Y-tick labels (redefine with fontsize)
    ax.set_yticklabels(tick_labels, fontsize=18)

    # X-tick labels (already set, now add fontsize)
    ax.set_xticklabels(label_texts, rotation=45, ha='right', fontsize=18)

    # Legend (2x larger)
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        bbox_to_anchor=(1, 0.97),  # Slightly above the axes, right-aligned
        fontsize=20,
        frameon=False
    )

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    #plt.subplots_adjust(top=0.85, right=0.85)
    plt.show()


# ---------- interval helpers ----------
def _normalize_ranges(ranges, lo=-math.pi, hi=math.pi):
    out = []
    for a, b in ranges:
        if a == b:  # zero height -> ignore
            continue
        if a > b:
            a, b = b, a
        a = max(a, lo)
        b = min(b, hi)
        if b > a:
            out.append((a, b))
    return out

def _merge_intervals(iv):
    if not iv:
        return []
    iv = sorted(iv)
    merged = [iv[0]]
    for a, b in iv[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def _subtract_intervals(new_iv, existing_iv):
    """Return portions of new_iv not covered by existing_iv."""
    existing_iv = _merge_intervals(existing_iv)
    result = []
    for a, b in new_iv:
        segs = [(a, b)]
        for ea, eb in existing_iv:
            nxt = []
            for sa, sb in segs:
                if eb <= sa or ea >= sb:
                    nxt.append((sa, sb))              # no overlap
                else:
                    if sa < ea: nxt.append((sa, ea))  # left piece
                    if sb > eb: nxt.append((eb, sb))  # right piece
            segs = nxt
            if not segs:
                break
        result.extend(segs)
    return _merge_intervals(result)

# --- helpers for keys / intervals ---
def key3(p, ndigits=8):
    """Convert a 3D point (np array or tuple/list) to a rounded tuple key."""
    p = np.asarray(p, dtype=float)
    return (round(p[0], ndigits), round(p[1], ndigits), round(p[2], ndigits))
# --- UPDATED updater ---
def update_beta_bar_multicolor(
    ax,
    point_xyz,
    beta_ranges,
    *,
    color="#d1b3ff",        # light purple (matches your beta_color)
    edgecolor="purple",     # matches your beta edgecolor
    bar_width=0.4,          # matches the beta bar width you used
    zorder=3,
    x_index_map=None,       # dict with tuple-keys created via key3
    points_xyz=None,        # list/array of points (fallback if no map)
    policy="stack",         # "stack" or "no-overlap"
    ndigits=8,              # rounding used in key3
    x_offset=0.0,           # optional horizontal shift (e.g., +1.0 to mimic odd slots)
    clip_to_pi=True,        # keep bars within [-pi, pi] like your plot
):
    """
    Append colored β segments for a single point. Repeated calls create a multicolor bar.
    Bars match the style from your plot_alpha_beta_ranges() beta bars:
      - color = '#d1b3ff'
      - edgecolor = 'purple'
      - width = 0.4
    """

    # --- resolve x-index (prefer fast dict lookup) ---
    if x_index_map is not None:
        k = key3(point_xyz, ndigits)
        if k not in x_index_map:
            raise ValueError(f"point_xyz {point_xyz} not found in x_index_map.")
        xi = x_index_map[k]
    elif points_xyz is not None:
        # Slow fallback: build keys from points_xyz and find index
        if isinstance(points_xyz, np.ndarray):
            pts = [key3(p, ndigits) for p in points_xyz]
        else:
            pts = [key3(p, ndigits) for p in points_xyz]
        k = key3(point_xyz, ndigits)
        try:
            xi = pts.index(k)
        except ValueError:
            raise ValueError("point_xyz not found in points_xyz.")
    else:
        raise ValueError("Provide x_index_map or points_xyz to locate the bar column.")

    x_center = float(xi) + float(x_offset)

    # --- registry for drawn segments per x-slot ---
    if not hasattr(ax, "_beta_registry"):
        ax._beta_registry = {}  # {xi_effective: [ {'start','end','rect','color','zorder'} ]}
    if x_center not in ax._beta_registry:
        ax._beta_registry[x_center] = []

    # --- normalize new intervals ---
    new_iv = _normalize_ranges(beta_ranges)  # expects list of (start,end), start<end

    # --- optional clipping to [-pi, pi] to mirror your y-limits/ticks ---
    if clip_to_pi and len(new_iv) > 0:
        clipped = []
        lo, hi = -np.pi, np.pi
        for s, e in new_iv:
            cs, ce = max(s, lo), min(e, hi)
            if ce > cs:
                clipped.append((cs, ce))
        new_iv = clipped

    # If nothing to draw, return early (skip white zero-height bars)
    if not new_iv:
        return []

    # --- optional overlap policy ---
    if policy == "no-overlap":
        existing_iv = [(d['start'], d['end']) for d in ax._beta_registry[x_center]]
        new_iv = _subtract_intervals(new_iv, existing_iv)
        if not new_iv:
            return []

    # --- draw segments and record ---
    drawn = []
    for start, end in new_iv:
        height = end - start
        if height <= 0:
            continue
        bars = ax.bar(
            x_center,
            height,
            bottom=start,
            width=bar_width,
            color=color,
            edgecolor=edgecolor,
            linewidth=0,
            zorder=zorder,
            align="center",
        )
        rect = bars.patches[0]
        seg = {'start': start, 'end': end, 'rect': rect, 'color': color, 'zorder': zorder}
        ax._beta_registry[x_center].append(seg)
        drawn.append(seg)

    return drawn



def set_sparse_xyz_labels(ax, points_xyz, max_labels=20, ndigits=2):
    """
    Set x-axis ticks/labels for only `max_labels` evenly spaced points from points_xyz.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    points_xyz : list/ndarray of shape (N, 3)
        The full set of 3D points in plotting order.
    max_labels : int
        Maximum number of labels to display.
    ndigits : int
        Number of decimal places in labels.
    """
    num_points = len(points_xyz)
    step = max(1, math.ceil(num_points / max_labels))

    label_positions = []
    label_texts = []

    for i, pt in enumerate(points_xyz):
        if i % step == 0:
            label_positions.append(i)
            label_texts.append(f"({pt[0]:.{ndigits}f}, {pt[2]:.{ndigits}f})")

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=45, ha='right', fontsize=12)

def draw_cube(ax, center, d, color):
    x, y, z = center
    r = d / 2
    # 8 vertices
    v = np.array([
        [x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],
        [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r],
    ])
    # 6 faces
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]
    coll = Poly3DCollection(faces, facecolors=[color]*6, edgecolor='none', linewidths=0)
    ax.add_collection3d(coll)

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

def draw_sphere(ax, center, d, color,
                resolution=24, edgecolor='none', linewidth=0,
                zsort='average', shade=True):
    """
    Draw a sphere centered at `center` with diameter `d` and `color` (RGB or RGBA).
    Returns the Poly3DCollection handle.
    """
    x0, y0, z0 = center
    r = d / 2.0

    # UV sphere mesh
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, max(8, resolution // 2 + 1))
    uu, vv = np.meshgrid(u, v, indexing='ij')
    X = x0 + r * np.cos(uu) * np.sin(vv)
    Y = y0 + r * np.sin(uu) * np.sin(vv)
    Z = z0 + r * np.cos(vv)

    # handle RGB vs RGBA
    if isinstance(color, (list, tuple)) and len(color) == 4:
        base_color = color[:3]
        alpha = color[3]
    else:
        base_color = color
        alpha = None

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        color=base_color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        antialiased=True,
        shade=shade,
    )

    if alpha is not None:
        surf.set_alpha(alpha)
    # 3D face sorting (if available)
    if hasattr(surf, "set_zsort"):
        surf.set_zsort(zsort)

    return surf

def wedge_polygon(center, r_outer, r_inner, theta1_deg, theta2_deg, n=128):
    cx, cy = center
    th1, th2 = np.radians([theta1_deg, theta2_deg])
    # sample arcs
    outer_t = np.linspace(th1, th2, n, endpoint=True)
    inner_t = np.linspace(th2, th1, n, endpoint=True)  # reverse for inner arc
    outer_arc = np.c_[cx + r_outer*np.cos(outer_t), cy + r_outer*np.sin(outer_t)]
    inner_arc = np.c_[cx + r_inner*np.cos(inner_t), cy + r_inner*np.sin(inner_t)]
    pts = np.vstack([outer_arc, inner_arc])
    return Polygon(pts)

def plot_exterior_boundary(ax, geom, **kw):
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, **kw,zorder=999)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, **kw,zorder=999)


def sample_line(p1, p2, n_between=40):
    """
    Uniformly sample points along the line segment between p1 and p2.

    Parameters:
        p1 (tuple or list): First point (x1, y1)
        p2 (tuple or list): Second point (x2, y2)
        n_between (int): Number of points between p1 and p2 (default=40)

    Returns:
        list of tuples: (n_between + 2) points including endpoints
    """
    x_vals = np.linspace(p1[0], p2[0], n_between + 2)
    y_vals = np.linspace(p1[1], p2[1], n_between + 2)
    return list(zip(x_vals, y_vals))


def exclusive_areas_4(inter: []):
    """
    Input (length 16):
      0..14 : [1,2,3,4,12,13,14,23,24,34,123,124,134,234,1234]
      15    : universal area U

    Output (length 16, SAME ORDER):
      exclusive areas for
      [1,2,3,4,12,13,14,23,24,34,123,124,134,234,1234,U_only]
    """
    if len(inter) != 16:
        raise ValueError("Expected 16 areas")

    idx_to_mask = [
        0b0001,  # 1
        0b0010,  # 2
        0b0100,  # 3
        0b1000,  # 4
        0b0011,  # 12
        0b0101,  # 13
        0b1001,  # 14
        0b0110,  # 23
        0b1010,  # 24
        0b1100,  # 34
        0b0111,  # 123
        0b1011,  # 124
        0b1101,  # 134
        0b1110,  # 234
        0b1111,  # 1234
    ]

    inter_by_mask = [0.0] * 16
    inter_by_mask[0] = float(inter[15])  # universe
    for i, m in enumerate(idx_to_mask):
        inter_by_mask[m] = float(inter[i])

    excl_by_mask = [0.0] * 16
    all_mask = 0b1111

    for s in range(16):
        extra = (~s) & all_mask
        total = 0.0
        sub = extra
        while True:
            t = s | sub
            total += (-1.0 if (sub.bit_count() & 1) else 1.0) * inter_by_mask[t]
            if sub == 0:
                break
            sub = (sub - 1) & extra
        excl_by_mask[s] = total

    result = [excl_by_mask[m] for m in idx_to_mask]
    result.append(excl_by_mask[0])  # U_only

    return result

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def to_pi_intervals(a, b):
    # returns 1 or 2 intervals in [-pi, pi] with start<=end
    a = wrap_to_pi(a)
    b = wrap_to_pi(b)
    if a <= b:
        return [[a, b]]
    else:
        return [[a, np.pi], [-np.pi, b]]

def forward_kinematics_3R(theta, L):
    """
    Computes the forward kinematics for a 3R planar robot arm.

    Parameters:
    theta1, theta2, theta3 : Joint angles in radians
    L : List containing the lengths of the three links

    Returns:
    (x, y) : Tuple containing the end-effector position (x, y)
    """
    theta = theta.flatten().tolist()
    x1 = L[0] * np.cos(theta[0])
    y1 = L[0] * np.sin(theta[0])

    x2 = x1 + L[1] * np.cos(theta[0] + theta[1])
    y2 = y1 + L[1] * np.sin(theta[0] + theta[1])

    x3 = x2 + L[2] * np.cos(theta[0] + theta[1] + theta[2])
    y3 = y2 + L[2] * np.sin(theta[0] + theta[1] + theta[2])

    return np.array([x3, y3]).T.reshape((2, 1))

def Jacobian_3R(theta, L):
    theta = theta.flatten().tolist()
    J = np.zeros((2, 3))
    J[0][0] = -L[0] * np.sin(theta[0]) - L[1] * np.sin(theta[0] + theta[1]) - L[2] * np.sin(
        theta[0] + theta[1] + theta[2])
    J[0][1] = - L[1] * np.sin(theta[0] + theta[1]) - L[2] * np.sin(theta[0] + theta[1] + theta[2])
    J[0][2] = - L[2] * np.sin(theta[0] + theta[1] + theta[2])
    J[1][0] = L[0] * np.cos(theta[0]) + L[1] * np.cos(theta[0] + theta[1]) + L[2] * np.cos(
        theta[0] + theta[1] + theta[2])
    J[1][1] = L[1] * np.cos(theta[0] + theta[1]) + L[2] * np.cos(theta[0] + theta[1] + theta[2])
    J[1][2] = L[2] * np.cos(theta[0] + theta[1] + theta[2])
    return J

def reliability_computation(r1, r2, r3):
    reliability_list = [r2 * r3, r1 * r3, r1 * r2, r2 * r3 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r2 * r3 - r1 * r2 * r3, r1 * r2 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3]
    conditional_reliability_list = []
    for p in reliability_list:
        conditional_reliability_list.append(
            (p / (r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3)))

    return conditional_reliability_list

def dls(x_target, initial_config,L):
    q = np.array(initial_config).T.reshape((3, 1))
    step_num = 0
    error_threshold = 1e-6
    error = 10
    while error > error_threshold:
        x_current = forward_kinematics_3R(q, L)
        delta_x = x_target - x_current
        error = np.linalg.norm(x_target - x_current)
        J = Jacobian_3R(q, L)
        correction = J.T @ np.linalg.inv(J @ J.T + 0.5 ** 2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9: return []
        q = q + 0.1 * correction

        for i in range(len(q)):
            q[i] %= 2 * np.pi
            if q[i] > np.pi: q[i] -= 2 * np.pi
            if q[i] < -np.pi: q[i] += 2 * np.pi

        step_num += 1
        previous_x = x_current
        x_current = forward_kinematics_3R(q, L)
        consecutive_delta_x = np.linalg.norm(previous_x - x_current)
        if consecutive_delta_x < 1e-8:
            return []
    sol = q
    return sol

def fold_offset(La, Lb, theta_locked):
    # orientation offset of the rigid pair relative to the first link
    return np.arctan2(Lb*np.sin(theta_locked), La + Lb*np.cos(theta_locked))


def union_ranges(ranges):
    """
    Computes the union of a list of ranges, with special handling for ranges that wrap around
    [-π, π]. If such ranges are detected, they are replaced by modified ranges.

    Parameters:
        ranges (list of tuples): List of ranges, where each range is represented as (a, b).

    Returns:
        list of tuples: The merged union of ranges as a list of non-overlapping ranges.
    """
    if not ranges:
        return []

    # Optional: Validate input ranges
    for a, b in ranges:
        if a > b:
            raise ValueError(f"Invalid range: ({a}, {b}). Start must be <= end.")

    # Sort the ranges by their starting point (and by end if start points are the same)
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

    # Initialize the merged ranges with the first range
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1]:  # Overlap or adjacent
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged

def lock_joint_1(theta_1,L):
    new_base_point=(L[0]*np.cos(theta_1),L[0]*np.sin(theta_1))
    return [L[1],L[2]],new_base_point
def lock_joint_2(theta_2,L):
    new_L1=np.sqrt(L[0]**2+L[1]**2-2.0*L[0]*L[1]*np.cos(np.pi-np.abs(theta_2)))
    return [new_L1,L[2]],(0,0)
def lock_joint_3(theta_3,L):
    new_L1=np.sqrt(L[1]**2+L[2]**2-2.0*L[1]*L[2]*np.cos(np.pi-np.abs(theta_3)))
    return [L[0],new_L1],(0,0)

def plot_alpha_ranges(theta_phi_list, ranges_list):
    """
    Plots a transposed bar graph where the vertical axis represents (theta, phi) tuples,
    and the horizontal bars represent disjoint ranges from -π to π, using the same color.

    Parameters:
    - theta_phi_list: List of tuples (theta, phi) defining categories.
    - ranges_list: List of corresponding range lists in [[[x1, x2], [y1, y2]], ...] format.

    Example:
    theta_phi_list = [(1, 0.5), (2, 1), (3, 1.5), ...]
    ranges_list = [[[1, 2], [3, 4]], [[0, np.pi], [np.pi, 2*np.pi]], ...]
    plot_bar_graph_transposed_same_color(theta_phi_list, ranges_list)
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    num_columns = len(theta_phi_list)

    # Control labeling density
    max_labels = 20
    step = max(1, math.ceil(num_columns / max_labels))  # Avoid division by zero

    # Generate x-labels with controlled density
    x_labels = [f"({theta:.2f}, {phi:.2f})" if i % step == 0 else "" for i, (theta, phi) in enumerate(theta_phi_list)]
    spacing_factor = 2
    x_positions = np.arange(0, num_columns * spacing_factor, spacing_factor)  # Positions of bars on x-axis

    for i, (x_pos, ranges) in enumerate(zip(x_positions, ranges_list)):
        if ranges:  # Plot bars for valid ranges
            for start, end in ranges:
                ax.bar(x_pos, end - start, bottom=start, align='center', color='blue', edgecolor='blue')
        else:  # Draw an empty column as a placeholder
            ax.bar(x_pos, 0, bottom=0, align='center', color='white', edgecolor='white')

    ax.set_ylabel("Alpha ranges")
    ax.set_xlabel("Theta, Phi Tuples")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')  # Angled labels for better visibility
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
    #ax.set_title("Transposed Bar Graph of (Theta, Phi) Mapping to Ranges (Single Color)")

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_beta_ranges(theta_phi_list, ranges_list):
    """
    Plots a transposed bar graph where the vertical axis represents (theta, phi) tuples,
    and the horizontal bars represent disjoint ranges from -π to π, using the same color.

    Parameters:
    - theta_phi_list: List of tuples (theta, phi) defining categories.
    - ranges_list: List of corresponding range lists in [[[x1, x2], [y1, y2]], ...] format.

    Example:
    theta_phi_list = [(1, 0.5), (2, 1), (3, 1.5), ...]
    ranges_list = [[[1, 2], [3, 4]], [[0, np.pi], [np.pi, 2*np.pi]], ...]
    plot_bar_graph_transposed_same_color(theta_phi_list, ranges_list)
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    num_columns = len(theta_phi_list)

    # Control labeling density
    max_labels = 20
    step = max(1, math.ceil(num_columns / max_labels))  # Avoid division by zero

    # Generate x-labels with controlled density
    x_labels = [f"({theta:.2f}, {phi:.2f})" if i % step == 0 else "" for i, (theta, phi) in enumerate(theta_phi_list)]
    spacing_factor = 2
    x_positions = np.arange(0, num_columns * spacing_factor, spacing_factor)   # Positions of bars on x-axis

    for i, (x_pos, ranges) in enumerate(zip(x_positions, ranges_list)):
        if ranges:  # Plot bars for valid ranges
            for start, end in ranges:
                ax.bar(x_pos, end - start, bottom=start, align='center', color='purple', edgecolor='purple')
        else:  # Draw an empty column as a placeholder
            ax.bar(x_pos, 0, bottom=0, align='center', color='white', edgecolor='white')

    ax.set_ylabel("Beta ranges")
    ax.set_xlabel("Theta, Phi Tuples")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')  # Angled labels for better visibility
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
    #ax.set_title("Transposed Bar Graph of (Theta, Phi) Mapping to Ranges (Single Color)")

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Function to generate the corner points of a square grid in x-z plane
def generate_square_grid(n_x, n_z, x_range, z_range):
    """Generate a grid of squares in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    squares = []

    for i in range(n_x):
        for j in range(n_z):
            # Define four corners of the square
            bottom_left = [x_vals[i], 0, z_vals[j]]
            bottom_right = [x_vals[i + 1], 0, z_vals[j]]
            top_right = [x_vals[i + 1], 0, z_vals[j + 1]]
            top_left = [x_vals[i], 0, z_vals[j + 1]]
            squares.append([bottom_left, bottom_right, top_right, top_left])

    return squares


def generate_square_grid_centers(n_x, n_z, x_range, z_range):
    """Generate grid of square centers in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    centers = []

    for i in range(n_x):
        for j in range(n_z):
            center_x = (x_vals[i] + x_vals[i + 1]) / 2
            center_z = (z_vals[j] + z_vals[j + 1]) / 2
            centers.append([center_x, 0, center_z])

    return centers


def generate_2D_square_grid(n_x, n_z, x_range, z_range):
    """Generate a grid of squares in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    squares = []

    for i in range(n_x):
        for j in range(n_z):
            # Define four corners of the square
            bottom_left = [x_vals[i], z_vals[j]]
            bottom_right = [x_vals[i + 1], z_vals[j]]
            top_right = [x_vals[i + 1], z_vals[j + 1]]
            top_left = [x_vals[i], z_vals[j + 1]]
            squares.append([bottom_left, bottom_right, top_right, top_left])

    return squares


# Function to rotate a set of vertices around the Z-axis
def rotate_around_z(vertices, angle):
    """Rotate vertices around the Z-axis by 'angle' (in radians)."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    return np.dot(vertices, rotation_matrix.T)



def draw_rotated_grid(ax, square, angle_range, color, deg_step=1.0):
    """
    Rotate a 4-vertex square around +Z and draw only when the rotation angle
    falls inside the allowed windows:
      - if all vertices have z >= 0  → draw for angles in [-π/2, 0]
      - if any vertex has z < 0      → draw for angles in (-π, 0] ∪ [0, π/2]
    Assumes angle_range = (lo, hi) with -π ≤ lo < hi ≤ π.
    """
    lo, hi = angle_range
    if hi <= lo:
        return

    # use a tiny tolerance in case vertices sit numerically on z=0
    has_z_neg = any(v[2] < -1e-12 for v in square)

    allowed = [(-np.pi/2, 0.0)] if not has_z_neg else [(-np.pi, 0.0), (0.0, np.pi/2)]

    # intersect [lo, hi] with allowed windows
    ranges = []
    for w0, w1 in allowed:
        l, h = max(lo, w0), min(hi, w1)
        if l < h:
            ranges.append((l, h))
    if not ranges:
        return

    step = np.deg2rad(deg_step)
    square_np = np.asarray(square)

    for l, h in ranges:
        n = max(2, int(np.ceil((h - l) / step)) + 1)
        for ang in np.linspace(l, h, n):
            rotated = rotate_around_z(square_np, ang)  # your helper
            face = [rotated[j] for j in range(4)]
            ax.add_collection3d(
                Poly3DCollection([face], facecolor=color, edgecolor=color, linewidth=0.1)
            )


def draw_rotated_grid_by_range_only(ax, square, angle_range, color):
    intersected_lower = angle_range[0]
    intersected_upper = angle_range[1]
    angles = np.linspace(intersected_lower, intersected_upper, int((angle_range[1]-angle_range[0])/(np.pi/180))*3)

    # Loop through each angle to rotate the square around the Z-axis
    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)

        # Create the rotated face (polygon) for the square
        face = [rotated_vertices[j] for j in range(4)]

        # Add the polygon to the plot with a single color
        ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5))

def draw_rotated_grid_by_range_only_transparent(ax, square, angle_range, color, center):
    intersected_lower = angle_range[0]
    intersected_upper = angle_range[1]

    angles = np.linspace(intersected_lower, intersected_upper, int((angle_range[1]-angle_range[0])/(np.pi/180))*3)
    # Loop through each angle to rotate the square around the Z-axis

    rotated_vertices = rotate_around_z(np.array(square), intersected_lower)
    face = [rotated_vertices[j] for j in range(4)]
    ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5,alpha=0.2))
    rotated_vertices = rotate_around_z(np.array(square), intersected_upper)
    face = [rotated_vertices[j] for j in range(4)]
    ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5,alpha=0.2))

    # Prepare curves for vertices and center
    vertex_curves = [[], [], [], []]
    center_curve = []

    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)
        for idx in range(4):
            vertex_curves[idx].append(rotated_vertices[idx])
        rotated_center = rotate_around_z(np.array([center]), angle)[0]
        center_curve.append(rotated_center)

    # Plot the curves for each vertex
    for vertex_curve in vertex_curves:
        vertex_curve = np.array(vertex_curve)
        ax.plot(vertex_curve[:, 0], vertex_curve[:, 1], vertex_curve[:, 2], color='black', linewidth=0.8)

    # Plot the curve for the center
    center_curve = np.array(center_curve)
    ax.plot(center_curve[:, 0], center_curve[:, 1], center_curve[:, 2], color='black', linewidth=2.0)


def generate_grid_centers(n_x, n_z, num_points, x_range, z_range):
    """Generate the center points of each grid in the x-z plane with correct order."""
    # Ensure that n_x * n_z equals the number of points (or close)
    if n_x * n_z != num_points:
        raise ValueError(f"Number of points ({num_points}) does not match grid layout ({n_x} * {n_z})")

    # Calculate the step size for each square in the grid
    x_step = (x_range[1] - x_range[0]) / n_x
    z_step = (z_range[1] - z_range[0]) / n_z

    # Initialize the list to store centers in the correct order
    centers = []

    # Generate center points for each square by iterating in the same order as `generate_square_grid`
    for i in range(n_x):
        for j in range(n_z):
            # Calculate the center of each square
            x_center = x_range[0] + (i + 0.5) * x_step
            z_center = z_range[0] + (j + 0.5) * z_step
            centers.append([x_center, 0, z_center])  # y = 0 as we're in the x-z plane

    # Convert list to numpy array for compatibility
    return np.array(centers)


def sample_square_points(vertices, num_points=10):
    """Sample points within the square defined by four vertices."""
    points = []
    for i in range(4):
        start, end = vertices[i], vertices[(i + 1) % 4]
        # Interpolate between two vertices to get points along the edge
        edge_points = np.linspace(start, end, num_points)
        points.extend(edge_points)
    return np.array(points)


# Generate scatter points for each rotated square
def generate_rotated_scatter_points(grid_squares, angle_range, num_samples=5):
    """Generate scattered points for each square in grid rotated over specified angle range."""
    scatter_points = []
    angles = np.linspace(angle_range[0], angle_range[1], 60)  # Adjust angle steps as needed

    for square in grid_squares:
        for angle in angles:
            # Rotate square vertices and sample points
            rotated_vertices = rotate_around_z(np.array(square), angle)
            square_points = sample_square_points(rotated_vertices, num_points=num_samples)
            scatter_points.append(square_points)

    # Flatten the list of arrays into a single array of points
    return np.vstack(scatter_points)



def generate_binary_matrix(n_x, n_z, x_range, z_range, grid_size, angle_ranges):
    """Generate a binary matrix representing the object based on angle ranges per grid square."""
    # Generate x and z values for the grid in x-z plane
    x_vals_grid = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals_grid = np.linspace(z_range[0], z_range[1], n_z + 1)

    # Calculate the increments
    delta_x = (x_range[1] - x_range[0]) / n_x
    delta_z = (z_range[1] - z_range[0]) / n_z

    # Create the 3D grid (voxel centers)
    x_vals = np.linspace(-x_range[1], x_range[1], grid_size[0])
    y_vals = np.linspace(-x_range[1], x_range[1], grid_size[1])
    z_vals = np.linspace(z_range[0], z_range[1], grid_size[2])
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    # Compute radial distance and angular coordinate from Z-axis
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)  # Theta ranges from -pi to pi

    # Initialize the binary matrix
    binary_matrix = np.zeros(X.shape, dtype=bool)

    # Check that angle_ranges has the correct length
    if len(angle_ranges) != n_x * n_z:
        raise ValueError(f"angle_ranges should have length {n_x * n_z}, but has length {len(angle_ranges)}")

    # Adjusted loop order to match generate_square_grid
    square_index = 0
    for i in range(n_x):
        for j in range(n_z):
            # Define the boundaries of the square in x-z plane
            x_min = x_vals_grid[i]
            x_max = x_vals_grid[i + 1]
            z_min = z_vals_grid[j]
            z_max = z_vals_grid[j + 1]

            # Get the list of angle ranges for this square
            square_angle_ranges = angle_ranges[square_index]
            square_index += 1

            # Skip this square if the angle range list is empty
            if not square_angle_ranges:
                continue  # No angles specified, skip this square

            # Create masks for points within this square after rotation
            within_r = (R >= x_min) & (R <= x_max)
            within_z = (Z >= z_min) & (Z <= z_max)

            # Initialize the angular mask for this square
            angular_mask = np.zeros(X.shape, dtype=bool)

            # Process each angle range for this square
            for theta_min, theta_max in square_angle_ranges:
                # Adjust angle ranges to handle wrapping around -pi and pi
                theta_min_adj = theta_min
                theta_max_adj = theta_max
                if theta_min < -np.pi:
                    theta_min_adj += 2 * np.pi
                if theta_max > np.pi:
                    theta_max_adj -= 2 * np.pi

                # Create mask for this angular range
                if theta_min_adj <= theta_max_adj:
                    within_theta = (Theta >= theta_min_adj) & (Theta <= theta_max_adj)
                else:
                    # Angle range crosses the -pi to pi boundary
                    within_theta = (Theta >= theta_min_adj) | (Theta <= theta_max_adj)

                # Update the angular mask
                angular_mask |= within_theta

            # Combine all conditions
            total_mask = within_r & within_z & angular_mask
            binary_matrix |= total_mask

    # Generate the edges of the grid for plotting
    x_edges = np.linspace(-x_range[1], x_range[1], grid_size[0] + 1)
    y_edges = np.linspace(-x_range[1], x_range[1], grid_size[1] + 1)
    z_edges = np.linspace(z_range[0], z_range[1], grid_size[2] + 1)

    return binary_matrix, x_edges, y_edges, z_edges
