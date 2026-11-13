"""Render two PPT figures from a saved union-hull solve; never rerun GCS."""
import json
from pathlib import Path
import itertools
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import numpy as np
from scipy.spatial import ConvexHull

from scripts.replay_saved_stage2_budgets import prepare_graph
from scripts.planar3R_plot_saved_reliability_path import vertices_of
from scripts.planar3R_two_point_ssm_views import compute_ssms_for_target, refine_ik_configuration
from scripts.planar3R_ssm_envelopes import build_envelopes, intersect_envelopes
from scripts.goal_set_baselines import hull_targets_fingerprint

SOURCE = Path('results/planar3r/union_hull_primary_cli_20261030.json')
OUT = Path('visualizations/planar3R_two_stage_ppt_20261113')
BLUE, ORANGE, GREEN, INK = '#2268B5', '#E58A24', '#17836B', '#23354D'
PROJECTION = '#9C2C79'


def target_vertices(target, samples):
    """Enumerate exact vertices in the sample hull's affine coordinates."""
    origin = samples[0]
    _, singular, basis = np.linalg.svd(samples-origin, full_matrices=True)
    rank = int(np.sum(singular > 1e-10))
    if rank == 0:
        return origin[None, :]
    axes = basis[:rank]
    poly = target['polytope']
    a, b = poly.A @ axes.T, poly.b-poly.A @ origin
    lengths = np.linalg.norm(a, axis=1)
    keep = lengths > 1e-10
    a, b = a[keep]/lengths[keep, None], b[keep]/lengths[keep]
    found = []
    for ids in itertools.combinations(range(len(b)), rank):
        mat = a[list(ids)]
        if abs(np.linalg.det(mat)) < 1e-10:
            continue
        v = np.linalg.solve(mat, b[list(ids)])
        if np.max(a @ v-b) < 2e-7:
            found.append(origin+v @ axes)
    if not found:
        raise ValueError('No target vertices found')
    return np.unique(np.round(found, 12), axis=0)


def polygon(ax, points, color, alpha, lw=.8, zorder=1):
    xy = np.unique(np.asarray(points)[:, :2], axis=0)
    if len(xy) >= 3 and np.linalg.matrix_rank(xy-xy[0]) == 2:
        xy = xy[ConvexHull(xy).vertices]
        ax.add_patch(Polygon(xy, facecolor=color, edgecolor=color, alpha=alpha,
                             linewidth=lw, zorder=zorder))
    else:
        ax.plot(*xy.T, color=color, lw=1.8, alpha=alpha, zorder=zorder)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    saved = source['union_hull_baseline']
    graph = prepare_graph(source, spatial=False)
    assert graph['fingerprint'] == source['graph_task_sha256']
    goal = source['goal_point']
    pieces = []
    for branch in compute_ssms_for_target(*goal, seed=1):
        group = []
        for q in branch:
            refined = refine_ik_configuration(q, *goal)
            if refined is None:
                if group:
                    pieces.append(np.asarray(group))
                    group = []
            else:
                group.append(refined)
        if group:
            pieces.append(np.asarray(group))
    envs = build_envelopes(pieces, epsilon=source['geometric_epsilon'],
                          arc_spacing=source['ssm_arc_spacing'],
                          refine_sample=lambda q: refine_ik_configuration(q, *goal))
    targets = intersect_envelopes(envs, graph['regions'])
    target_hash_matches = hull_targets_fingerprint(targets) == source['hull_targets_sha256']
    # Native Windows and the original Linux solve can differ at floating-point
    # precision. Verify geometry rather than calling a different byte hash equal.
    assert len(targets) == source['target_set_count']
    assert len(envs) == source['envelope_count']
    selected = saved['selected_hull_target']
    assert targets[selected]['polytope'].contains(np.asarray(saved['stage1_waypoints'])[-1], tol=1e-6)
    for key, route_key in [('stage1_waypoints', 'stage1_region_route'), ('waypoints', 'region_route')]:
        path = np.asarray(saved[key])
        for a, b, region_id in zip(path[:-1], path[1:], saved[route_key]):
            assert graph['regions'][region_id].contains(a, tol=1e-6)
            assert graph['regions'][region_id].contains(b, tol=1e-6)
    maps = [np.rad2deg(vertices_of(r)[0]) for r in graph['regions']]
    target_polys = [np.rad2deg(target_vertices(t, envs[t['envelope']].samples)) for t in targets]
    curves = [np.rad2deg(e.samples) for e in envs]
    p1, p2 = [np.rad2deg(saved[key]) for key in ('stage1_waypoints', 'waypoints')]
    qhat, qstar = p1[-1], p2[-1]
    reliabilities = np.asarray([r.reliability for r in graph['regions']])
    cmap, norm = plt.get_cmap('RdYlGn'), Normalize(0, 1)
    assert np.linalg.norm(np.deg2rad(qhat-qstar)) > 0
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'axes.labelcolor': INK,
                         'text.color': INK, 'axes.edgecolor': '#BAC4CF'})
    for stage in (1, 2):
        fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
        ax = fig.add_axes([.15, .16, .45, .68])
        detail = fig.add_axes([.705, .19, .255, .36])
        cax = fig.add_axes([.055, .21, .017, .57])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label('Region reliability R', fontsize=12, labelpad=10)
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')
        cb.set_ticks([0, .2, .4, .6, .8, 1])
        cb.outline.set_linewidth(.6)
        fig.text(.065, .93, f'0{stage}', color=BLUE if stage == 1 else GREEN,
                 size=25, weight='bold')
        fig.text(.12, .93, 'Joint target selection' if stage == 1 else 'Projection and fixed-goal replanning',
                 size=23, weight='bold')
        fig.text(.12, .88, 'Stage 1  |  One GCS solve over the union of local target hulls' if stage == 1
                 else 'Stage 2  |  Project the selected endpoint, then solve GCS again', size=12.5)
        for axis in (ax, detail):
            axis.set_facecolor('#FAFBFD')
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(color='#DFE5ED', lw=.65, zorder=0)
            axis.set_axisbelow(True)
            axis.set_aspect('equal', adjustable='box')
            # Opaque ascending painting implements max R over feasible theta3;
            # transparent blending would not correspond to the numerical scale.
            for index in np.argsort(reliabilities, kind='stable'):
                polygon(axis, maps[index], cmap(norm(reliabilities[index])), 1, lw=0)
            for pts in maps:
                xy = np.unique(pts[:, :2], axis=0)
                axis.add_patch(Polygon(xy[ConvexHull(xy).vertices], fill=False,
                                       edgecolor=INK, alpha=.16, linewidth=.5, zorder=2))
            for i, pts in enumerate(target_polys):
                polygon(axis, pts, ORANGE if i == selected else '#BDA1D6',
                        .42 if i == selected else .22, lw=1, zorder=3)
            for curve in curves:
                axis.plot(*curve[:, :2].T, color=INK, lw=1.4, zorder=4,
                          path_effects=[pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()])
            if axis is ax:
                # These very thin true hulls otherwise disappear at slide scale.
                # Mark their projected centroids, without enlarging the sets.
                centers = np.asarray([v[:, :2].mean(axis=0) for v in target_polys])
                axis.scatter(*centers.T, s=32, facecolor='#AE82CF',
                             edgecolor='white', linewidth=.55, zorder=5)
            if stage == 2:
                axis.plot(*p1[:, :2].T, color=BLUE, lw=2, ls='--', alpha=.45, zorder=5)
            route, color = (p1, BLUE) if stage == 1 else (p2, GREEN)
            axis.plot(*route[:, :2].T, color='white', lw=6, zorder=6)
            axis.plot(*route[:, :2].T, color=color, lw=3.3, zorder=7)
            axis.scatter(*route[0, :2], s=100, c=INK, edgecolors='white', zorder=9)
            axis.scatter(*qhat[:2], s=85, c=ORANGE, edgecolors='white', zorder=10)
            if stage == 2:
                axis.scatter(*qstar[:2], marker='*', s=210, c=GREEN, edgecolors='white', zorder=11)
            axis.set_xlabel(r'$\theta_1$ (deg)')
            axis.set_ylabel(r'$\theta_2$ (deg)')
        ax.set(xlim=(-180, 180), ylim=(-180, 180))
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.annotate(r'$q_s$', p1[0, :2], xytext=(10, -4), textcoords='offset points', size=15)
        ax.annotate(r'$\hat q$', qhat[:2], xytext=(10, -15), textcoords='offset points', size=16, color=ORANGE)
        # The true displacement is small; magnify it without moving either point.
        center = (qhat[:2]+qstar[:2])/2
        radius = max(np.max(np.abs(qhat[:2]-qstar[:2]))*1.8, .20)
        detail.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[1]-radius, center[1]+radius))
        detail.tick_params(labelsize=9)
        detail.set_title('Selected endpoint: magnified view' if stage == 1 else
                         'Endpoint projection: 3D local IK', fontsize=12, pad=12)
        detail.annotate(r'$\hat q$', qhat[:2], xytext=(12, -18), textcoords='offset points', size=16, color=ORANGE)
        if stage == 2:
            detail.annotate('', qstar[:2], xytext=qhat[:2],
                            arrowprops=dict(arrowstyle='-|>', color=PROJECTION, lw=2.4,
                                            mutation_scale=17, shrinkA=0, shrinkB=0), zorder=12)
            detail.annotate(r'$q^\star$', qstar[:2], xytext=(-33, 12), textcoords='offset points', size=16, color=GREEN,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))
            ax.annotate(r'$q^\star$', qstar[:2], xytext=(-27, 17), textcoords='offset points', size=15, color=GREEN,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))
        handles = [Line2D([], [], color='#BDA1D6', lw=3, marker='o', markersize=5, label='Target hulls (centers marked)'),
                   Line2D([], [], color=ORANGE, lw=7, alpha=.65, label='Selected target hull'),
                   Line2D([], [], color=INK, lw=1.8, label='Sampled task SSM'),
                   Line2D([], [], color=BLUE if stage == 1 else GREEN, lw=3,
                          label='Stage-1 path' if stage == 1 else 'Stage-2 path')]
        if stage == 2:
            handles.append(Line2D([], [], color=PROJECTION, lw=2.4, marker='>',
                                  label='Endpoint projection (not a path segment)'))
        fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.69, .59),
                   frameon=False, fontsize=10.5, labelspacing=.65)
        fig.text(.70, .81, r'$q_s\ \longrightarrow\ \bigcup_i(H_i\cap C_i)$' if stage == 1
                 else r'$\hat q\ \longrightarrow\ q^\star$  (local IK)', size=18)
        fig.text(.15, .06, r'Background: maximum $R$ over feasible $\theta_3$.', size=10, color='#68798B')
        fig.text(.15, .025, '3R solution projected to two angles; not a fixed-angle slice.',
                 size=10, color='#68798B')
        fig.text(.70, .095, 'All targets share one virtual sink.' if stage == 1
                 else 'Replan the full route to fixed '+r'$q^\star$'+'.', size=11, color=INK)
        name = f'stage{stage}_'+('union_hull_selection' if stage == 1 else 'fixed_goal_replanning')
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(OUT/f'{name}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)
    report = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  graph_fingerprint_verified=True, target_fingerprint_verified=target_hash_matches,
                  target_geometry_checks='Counts and selected endpoint membership verified; both saved paths verified against all traversed regions. Native reconstruction, not byte-identical target data when hash differs.',
                  axes=[0, 1], units='degrees', selected_target=selected,
                  target_count=len(targets), region_count=len(maps),
                  stage1_endpoint_rad=np.deg2rad(qhat).tolist(),
                  projected_endpoint_rad=np.deg2rad(qstar).tolist(),
                  projection_distance_rad=float(np.linalg.norm(np.deg2rad(qhat-qstar))),
                  region_reliabilities=reliabilities.tolist(),
                  reliability_colormap='RdYlGn', reliability_color_limits=[0, 1],
                  background_overlap_rule='Opaque increasing R: maximum R among projected regions at each point; no color blending.',
                  gcs_rerun=False, note='Orthogonal projection, not a fixed-theta3 slice. Background overlaps do not imply 3D connectivity. Local inset uses unchanged true coordinates.')
    (OUT/'figure_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'README.txt').write_text('Two-stage union-hull figures, 3R\n'
        'Stage 1: select a local target hull and candidate endpoint in one GCS solve.\n'
        'Stage 2: constrained local IK projects to the task SSM; a second GCS solve replans the full route to this fixed goal.\n'
        'PNG: 4000 x 2250, white background. SVG/PDF: vector output. English labels for slide use.\n'
        'Background: code region reliability R, RdYlGn with shared limits 0..1. Overlap shows maximum R over feasible theta3, not the path-specific reliability.\n'
        'Purple/orange: projected hull/free-region intersections. Dark line: sampled SSM. Magenta arrow: endpoint projection, not an executed connector.\n'
        'The inset is essential: projection displacement and route changes are small in this actual example. No visual displacement was invented.\n'
        'Reproduce from repository root: configured Python -m scripts.planar3R_two_stage_ppt\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
