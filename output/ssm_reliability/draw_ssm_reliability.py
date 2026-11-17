"""Slide figure from the planar 3R goal experiment and project color/partition code."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects
import numpy as np
from scripts.planar3R_bbox_reliability_partition import (
    branch_bounding_box, partition_single_branch, build_reliability_lookup,
    build_color_lookup, cuboid_faces, R1, R2, R3,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
ssm = np.load(OUT / 'goal_ssm_data.npz')['ssm_0']
bounds = branch_bounding_box(ssm)
lookup = build_reliability_lookup(R1, R2, R3)
colors, sm = build_color_lookup(lookup)
theta3 = 0.0
cells = [c for c in partition_single_branch(bounds, lookup)
         if c['lower'][2] <= theta3 < c['upper'][2]]
assert len(cells) == 9
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'svg.fonttype': 'none', 'axes.labelcolor': '#23364D',
                     'text.color': '#23364D', 'axes.titleweight': 'bold'})
fig = plt.figure(figsize=(19.2, 10.8), facecolor='white')
fig.text(.045, .93, 'From Goal SSM to Joint-Space Reliability Map', fontsize=30, weight='bold')
fig.text(.045, .884, r'Planar 3R  |  Goal position $(x_g,y_g)=(1.5,1.5)$  |  Link lengths $(1,1,1)$',
         fontsize=17, color='#63758B')
fig.text(.065, .82, '01  Goal SSM and its bounding box', fontsize=19, weight='bold')
fig.text(.535, .82, r'02  Reliability map: $\theta_3=0$ rad', fontsize=19, weight='bold')
ax = fig.add_axes([.025,.255,.40,.53], projection='3d')
faces = cuboid_faces(bounds[:,0], bounds[:,1])
ax.add_collection3d(Poly3DCollection(faces, facecolor='#93B8DE', edgecolor='#3D6E9C',
                                    alpha=.09, linewidth=1.4))
ax.plot(*ssm.T, color='#1D507F', lw=2.8, label='Goal SSM')
ax.set(xlim=(-np.pi,np.pi), ylim=(-np.pi,np.pi), zlim=(-np.pi,np.pi),
       xlabel=r'$\theta_1$ [rad]', ylabel=r'$\theta_2$ [rad]', zlabel=r'$\theta_3$ [rad]')
for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
    axis.set_ticks([-np.pi,0,np.pi]); axis.set_ticklabels([r'$-\pi$','0',r'$\pi$'])
    axis.pane.fill=False
ax.view_init(elev=24, azim=-57)
ax.set_box_aspect((1,1,1))
ax.legend(loc='upper left', bbox_to_anchor=(.10,.94), frameon=False, fontsize=14)
fig.add_artist(FancyArrowPatch((.421,.54),(.501,.54), transform=fig.transFigure,
                              arrowstyle='-|>', mutation_scale=24, lw=2, color='#627D98'))
fig.text(.461,.579,'Partition',ha='center',fontsize=15,weight='bold')
fig.text(.461,.505,'at box limits',ha='center',fontsize=12,color='#63758B')

am = fig.add_axes([.535,.255,.355,.53])
records=[]
for c in cells:
    lower, upper = np.degrees(c['lower']), np.degrees(c['upper'])
    reliability = c['reliability']
    am.add_patch(Rectangle(lower[:2],*(upper-lower)[:2],
                           facecolor=colors[reliability], edgecolor='#FFFFFF', lw=2.1))
    fs = r'\{' + ','.join(map(str,sorted(c['F']))) + r'\}' if c['F'] else r'\varnothing'
    x,y = ((lower+upper)/2)[:2]
    am.text(x,y,rf'$F={fs}$'+'\n'+rf'$R={reliability:.2f}$',
            ha='center',va='center',fontsize=13.5, linespacing=1.5,
            color='#172B46',weight='bold', zorder=5,
            bbox=dict(facecolor=colors[reliability],edgecolor='none',pad=2),
            path_effects=[patheffects.withStroke(linewidth=2.5,foreground='white',alpha=.62)])
    records.append({'theta1_degrees': [lower[0],upper[0]],
                    'theta2_degrees': [lower[1],upper[1]],
                    'F': sorted(c['F']), 'reliability': reliability})
bdeg=np.degrees(bounds)
for x in bdeg[0]: am.axvline(x,color='#23364D',lw=1.3,ls=(0,(5,4)))
for y in bdeg[1]: am.axhline(y,color='#23364D',lw=1.3,ls=(0,(5,4)))
am.plot(*np.degrees(ssm[:, :2]).T, color='black', linewidth=2.0,
        zorder=4, label=r'Goal SSM ($\theta_1$–$\theta_2$ projection)')
am.legend(loc='upper left', bbox_to_anchor=(0, 1.045), fontsize=10,
          frameon=False, borderaxespad=0, handlelength=2.5)
am.set(xlim=(-180,180),ylim=(-180,180),xlabel=r'$\theta_1$ [deg]',ylabel=r'$\theta_2$ [deg]')
am.set_aspect('equal')
am.set_xticks([-180,*bdeg[0],180]); am.set_yticks([-180,*bdeg[1],180])
am.set_xticklabels(['−180',f'{bdeg[0,0]:.1f}',f'{bdeg[0,1]:.1f}','180'])
am.set_yticklabels(['−180',f'{bdeg[1,0]:.1f}',f'{bdeg[1,1]:.1f}','180'])
am.tick_params(labelsize=12)
cax=fig.add_axes([.927,.276,.017,.488])
cb=fig.colorbar(sm,cax=cax,ticks=np.arange(0,1.01,.2))
cb.set_label('Reliability R',fontsize=16,labelpad=13)
cb.outline.set_edgecolor('#D1D9E3')
fig.text(.065,.235,'Bounding box limits [rad]',fontsize=15,weight='bold')
for i in range(3):
    fig.text(.065,.198-i*.033,rf'$\theta_{i+1}\in[{bounds[i,0]:.3f},\ {bounds[i,1]:.3f}]$',fontsize=14)
fig.text(.535,.185,r'$F(\theta)=\{j\mid\theta_j^{\min}\leq\theta_j\leq\theta_j^{\max}\}$',fontsize=19)
fig.text(.535,.139,r'Joint reliabilities: $(r_1,r_2,r_3)=(0.7,0.5,0.3)$',fontsize=15)
fig.text(.065,.058,'Box limits define the regions; the recovery set F determines the color through R.',
         fontsize=16,color='#63758B')
for ext in ['png','svg','pdf']:
    fig.savefig(OUT / f'goal_ssm_to_reliability_map.{ext}', dpi=200, facecolor='white')
metadata={'goal':[1.5,1.5], 'link_lengths':[1,1,1], 'seed':0,
          'ssm_source':'scripts.planar3R_two_point_ssm_views.compute_ssms_for_target',
          'bounds_rad':bounds.tolist(),'theta3_slice_rad':theta3,
          'joint_reliabilities':[R1,R2,R3], 'regions':records,
          'color_source':'scripts.helper_functions.normalize_and_map_colors',
          'black_curve':'Goal SSM projected onto theta1-theta2; background is theta3=0 slice'}
(OUT/'figure_data.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
